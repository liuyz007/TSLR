import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from einops import rearrange
from torch.utils.cpp_extension import load

T_MAX = 512*512 
# LOL384 * 256   MCR 1024 * 1280

wkv_cuda = load(name="wkv", sources=["cuda/wkv_op.cpp", "cuda/wkv_cuda.cu"],
                verbose=True, extra_cuda_cflags=['-res-usage', '--maxrregcount 60', '--use_fast_math', '-O3', '-Xptxas -O3', f'-DTmax={T_MAX}'])

class WKV(torch.autograd.Function):
    @staticmethod
    def forward(ctx, B, T, C, w, u, k, v):
        ctx.B = B
        ctx.T = T
        ctx.C = C
        assert T <= T_MAX
        assert B * C % min(C, 1024) == 0

        half_mode = (w.dtype == torch.half)
        bf_mode = (w.dtype == torch.bfloat16)
        ctx.save_for_backward(w, u, k, v)
        w = w.float().contiguous()
        u = u.float().contiguous()
        k = k.float().contiguous()
        v = v.float().contiguous()
        y = torch.empty((B, T, C), device='cuda', memory_format=torch.contiguous_format)
        wkv_cuda.forward(B, T, C, w, u, k, v, y)
        if half_mode:
            y = y.half()
        elif bf_mode:
            y = y.bfloat16()
        return y

    @staticmethod
    def backward(ctx, gy):
        B = ctx.B
        T = ctx.T
        C = ctx.C
        assert T <= T_MAX
        assert B * C % min(C, 1024) == 0
        w, u, k, v = ctx.saved_tensors
        gw = torch.zeros((B, C), device='cuda').contiguous()
        gu = torch.zeros((B, C), device='cuda').contiguous()
        gk = torch.zeros((B, T, C), device='cuda').contiguous()
        gv = torch.zeros((B, T, C), device='cuda').contiguous()
        half_mode = (w.dtype == torch.half)
        bf_mode = (w.dtype == torch.bfloat16)
        wkv_cuda.backward(B, T, C,
                          w.float().contiguous(),
                          u.float().contiguous(),
                          k.float().contiguous(),
                          v.float().contiguous(),
                          gy.float().contiguous(),
                          gw, gu, gk, gv)
        if half_mode:
            gw = torch.sum(gw.half(), dim=0)
            gu = torch.sum(gu.half(), dim=0)
            return (None, None, None, gw.half(), gu.half(), gk.half(), gv.half())
        elif bf_mode:
            gw = torch.sum(gw.bfloat16(), dim=0)
            gu = torch.sum(gu.bfloat16(), dim=0)
            return (None, None, None, gw.bfloat16(), gu.bfloat16(), gk.bfloat16(), gv.bfloat16())
        else:
            gw = torch.sum(gw, dim=0)
            gu = torch.sum(gu, dim=0)
            return (None, None, None, gw, gu, gk, gv)

def RUN_CUDA(B, T, C, w, u, k, v):
    return WKV.apply(B, T, C, w.cuda(), u.cuda(), k.cuda(), v.cuda())

class OmniShift(nn.Module):
    def __init__(self, dim):
        super(OmniShift, self).__init__()
        # Define the layers for training
        self.conv1x1 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, groups=dim, bias=False)
        self.conv3x3 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=1, groups=dim, bias=False)
        self.conv5x5 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=5, padding=2, groups=dim, bias=False) 
        self.alpha = nn.Parameter(torch.randn(4), requires_grad=True) 

        # Define the layers for testing
        self.conv5x5_reparam = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=5, padding=2, groups=dim, bias = False) 
        self.repram_flag = True

    def forward_train(self, x):
        out1x1 = self.conv1x1(x)
        out3x3 = self.conv3x3(x)
        out5x5 = self.conv5x5(x) 
        # import pdb 
        # pdb.set_trace() 
        
        out = self.alpha[0]*x + self.alpha[1]*out1x1 + self.alpha[2]*out3x3 + self.alpha[3]*out5x5
        return out

    def reparam_5x5(self):
        # Combine the parameters of conv1x1, conv3x3, and conv5x5 to form a single 5x5 depth-wise convolution 
        
        padded_weight_1x1 = F.pad(self.conv1x1.weight, (2, 2, 2, 2)) 
        padded_weight_3x3 = F.pad(self.conv3x3.weight, (1, 1, 1, 1)) 
        
        identity_weight = F.pad(torch.ones_like(self.conv1x1.weight), (2, 2, 2, 2)) 
        
        combined_weight = self.alpha[0]*identity_weight + self.alpha[1]*padded_weight_1x1 + self.alpha[2]*padded_weight_3x3 + self.alpha[3]*self.conv5x5.weight 

        device = self.conv5x5_reparam.weight.device 

        combined_weight = combined_weight.to(device)

        self.conv5x5_reparam.weight = nn.Parameter(combined_weight)


    def forward(self, x): 
        
        if self.training: 
            self.repram_flag = True
            out = self.forward_train(x) 
        elif self.training == False and self.repram_flag == True:
            self.reparam_5x5() 
            self.repram_flag = False 
            out = self.conv5x5_reparam(x)
        elif self.training == False and self.repram_flag == False:
            out = self.conv5x5_reparam(x)

        return out 

class VRWKV_SpatialMix(nn.Module):
    def __init__(self, n_embd,
                 key_norm=False):
        super().__init__()
        self.n_embd = n_embd
        self.device = None
        attn_sz = n_embd
        
        self.dwconv = nn.Conv2d(n_embd, n_embd, kernel_size=3, stride=1, padding=1, groups=n_embd, bias=False)   
        self.recurrence = 2 
        self.omni_shift = OmniShift(dim=n_embd)

        self.key = nn.Linear(n_embd, attn_sz, bias=False)
        self.value = nn.Linear(n_embd, attn_sz, bias=False)
        self.receptance = nn.Linear(n_embd, attn_sz, bias=False)
        if key_norm:
            self.key_norm = nn.LayerNorm(n_embd)
        else:
            self.key_norm = None
        self.output = nn.Linear(attn_sz, n_embd, bias=False) 

        with torch.no_grad():
            self.spatial_decay = nn.Parameter(torch.randn((self.recurrence, self.n_embd))) 
            self.spatial_first = nn.Parameter(torch.randn((self.recurrence, self.n_embd))) 


    def jit_func(self, x, resolution):
        # Mix x with the previous timestep to produce xk, xv, xr

        h, w = resolution

        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = self.omni_shift(x)
        x = rearrange(x, 'b c h w -> b (h w) c')    

        k = self.key(x)
        v = self.value(x)
        r = self.receptance(x)
        sr = torch.sigmoid(r)

        return sr, k, v

    def forward(self, x, resolution):
        B, T, C = x.size()
        self.device = x.device

        sr, k, v = self.jit_func(x, resolution) 
        
        for j in range(self.recurrence): 
            if j%2==0:
                v = RUN_CUDA(B, T, C, self.spatial_decay[j] / T, self.spatial_first[j] / T, k, v) 
            else:
                h, w = resolution 
                k = rearrange(k, 'b (h w) c -> b (w h) c', h=h, w=w) 
                v = rearrange(v, 'b (h w) c -> b (w h) c', h=h, w=w) 
                v = RUN_CUDA(B, T, C, self.spatial_decay[j] / T, self.spatial_first[j] / T, k, v) 
                k = rearrange(k, 'b (w h) c -> b (h w) c', h=h, w=w) 
                v = rearrange(v, 'b (w h) c -> b (h w) c', h=h, w=w) 

        x = v
        if self.key_norm is not None:
            x = self.key_norm(x)
        x = sr * x
        x = self.output(x)
        return x

class VRWKV_ChannelMix(nn.Module):
    def __init__(self, n_embd,hidden_rate=4,
                 key_norm=False):
        super().__init__()

        self.n_embd = n_embd

        hidden_sz = int(hidden_rate * n_embd)
        self.key = nn.Linear(n_embd, hidden_sz, bias=False) 
        
        self.omni_shift = OmniShift(dim=n_embd)
        
        if key_norm:
            self.key_norm = nn.LayerNorm(hidden_sz)
        else:
            self.key_norm = None
        self.receptance = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(hidden_sz, n_embd, bias=False)

    def forward(self, x, resolution):

        h, w = resolution
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = self.omni_shift(x)
        x = rearrange(x, 'b c h w -> b (h w) c')    
        k = self.key(x)
        k = torch.square(torch.relu(k))
        if self.key_norm is not None:
            k = self.key_norm(k)
        kv = self.value(k)
        x = torch.sigmoid(self.receptance(x)) * kv 

        return x

# rwkv block
class Block(nn.Module):
    def __init__(self, n_embd,hidden_rate=4,
               key_norm=False):
    # self, n_embd, n_layer, layer_id, hidden_rate=4,
        super().__init__()

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd) 

        self.att = VRWKV_SpatialMix(n_embd,
                                   key_norm=key_norm)

        self.ffn = VRWKV_ChannelMix(n_embd, hidden_rate,
                                   key_norm=key_norm)

        self.gamma1 = nn.Parameter(torch.ones((n_embd)), requires_grad=True)
        self.gamma2 = nn.Parameter(torch.ones((n_embd)), requires_grad=True)

    def forward(self, x): 
        b, c, h, w = x.shape
        resolution = (h, w)
        # x = self.dwconv1(x) + x
        x = rearrange(x, 'b c h w -> b (h w) c')
        y = self.gamma1 * self.att(self.ln1(x), resolution)#
        x = x + self.gamma1 * self.att(self.ln1(x), resolution) 
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        # x = self.dwconv2(x) + x
        x = rearrange(x, 'b c h w -> b (h w) c')    
        x = x + self.gamma2 * self.ffn(self.ln2(x), resolution) 
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

        return x

##########################################################################

class LayerNormalization(nn.Module):
    def __init__(self, dim):
        super(LayerNormalization, self).__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        # Rearrange the tensor for LayerNorm (B, C, H, W) to (B, H, W, C)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        # Rearrange back to (B, C, H, W)
        return x.permute(0, 3, 1, 2)

class SEBlock(nn.Module):
    def __init__(self, input_channels, reduction_ratio=16):
        super(SEBlock, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(input_channels, input_channels // reduction_ratio)
        self.fc2 = nn.Linear(input_channels // reduction_ratio, input_channels)
        self._init_weights()

    def forward(self, x):
        batch_size, num_channels, _, _ = x.size()
        y = self.pool(x).reshape(batch_size, num_channels)
        y = F.relu(self.fc1(y))
        y = torch.tanh(self.fc2(y))
        y = y.reshape(batch_size, num_channels, 1, 1)
        return x * y
    
    def _init_weights(self):
        init.kaiming_uniform_(self.fc1.weight, a=0, mode='fan_in', nonlinearity='relu')
        init.kaiming_uniform_(self.fc2.weight, a=0, mode='fan_in', nonlinearity='relu')
        init.constant_(self.fc1.bias, 0)
        init.constant_(self.fc2.bias, 0)

class DCFEBlock(nn.Module):
    def __init__(self, filters):
        super(DCFEBlock, self).__init__()
        self.layer_norm = LayerNormalization(filters)
        self.conv = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1)
        self.depthwise_conv = nn.Conv2d(filters, filters, kernel_size=3, padding=1, groups=filters)
        self.se_attn = SEBlock(filters)
        self._init_weights()

    def forward(self, x):
        x_norm = self.layer_norm(x)
        x_norm = self.conv(x_norm)
        x1 = self.depthwise_conv(x_norm)
        x2 = self.se_attn(x_norm)
        x_fused = x1 * x2
        x_out = x_fused + x
        return x_out
    
    def _init_weights(self):
        init.kaiming_uniform_(self.depthwise_conv.weight, a=0, mode='fan_in', nonlinearity='relu')
        init.constant_(self.depthwise_conv.bias, 0)

class TED(nn.Module):
    def __init__(self, num_filters, num_blocks, kernel_size=3, activation='relu'):
        super(TED, self).__init__()
        self.conv1 = nn.Conv2d(1, num_filters, kernel_size=kernel_size, padding=1)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=kernel_size, stride=2, padding=1)
        self.conv3 = nn.Conv2d(num_filters, num_filters, kernel_size=kernel_size, stride=2, padding=1)
        self.conv4 = nn.Conv2d(num_filters, num_filters, kernel_size=kernel_size, stride=2, padding=1)
        self.ted_rwkv_block = Block(n_embd=num_filters)
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up4 = nn.Upsample(scale_factor=2, mode='nearest')
        self.output_layer = nn.Conv2d(1, 1, kernel_size=kernel_size, padding=1)
        self.res_layer = nn.Conv2d(num_filters, 1, kernel_size=kernel_size, padding=1)
        self.activation = getattr(F, activation)
        # activation='relu'
        self._init_weights()
        self.num_blocks = num_blocks

    def forward(self, x):
        num_blocks = self.num_blocks
        x1 = self.activation(self.conv1(x))
        x2 = self.activation(self.conv2(x1))
        x3 = self.activation(self.conv3(x2))
        x4 = self.activation(self.conv4(x3))
        x = self.ted_rwkv_block(x4)
        for i in range(num_blocks[1] - 1):
            x = self.ted_rwkv_block(x4)
        x = self.up4(x)
        x = self.up3(x + x3)
        x = self.up2(x + x2)
        x = x + x1
        x = self.res_layer(x)
        return torch.tanh(self.output_layer(x + x))
    
    def _init_weights(self):
        for layer in [self.conv1, self.conv2, self.conv3, self.conv4, self.output_layer, self.res_layer]:
            init.kaiming_uniform_(layer.weight, a=0, mode='fan_in', nonlinearity='relu')
            if layer.bias is not None:
                init.constant_(layer.bias, 0)

def downshuffle(var, r):
    """
    Down Shuffle function, same as nn.PixelUnshuffle().
    Input: variable of size (1 × H × W)
    Output: down-shuffled var of size (r^2 × H/r × W/r)
    """
    dim_count = var.ndim
    if dim_count == 5:
        b, c, h, w,_ = var.size()
    else:
        b, c, h, w = var.size()
    out_channel = c * (r ** 2)
    out_h = h // r
    out_w = w // r
    return var.contiguous().view(b, c, out_h, r, out_w, r) \
        .permute(0, 1, 3, 5, 2, 4).contiguous().view(b, out_channel, out_h, out_w).contiguous()

class TSLR(nn.Module):
    def __init__(self, filters=32,inp_channels=1,num_blocks = [1,1]):
        super(TSLR, self).__init__()
        Dx = torch.tensor([[1,-1], [1,-1]], dtype=torch.float32) / 2
        Dy = Dx.t()
        fxx = torch.tensor([[1, -2, 1]] * 3, dtype=torch.float32) / 4
        fyy = fxx.t()
        fuu = torch.tensor([[1, 0, 0], [0, -2, 0], [0, 0, 1]], dtype=torch.float32) / 4
        fvv = torch.tensor([[0, 0, 1], [0, -2, 0], [1, 0, 0]], dtype=torch.float32) / 4
        self.Dx = Dx.unsqueeze(0).unsqueeze(0)
        self.Dy = Dy.unsqueeze(0).unsqueeze(0)
        self.fxx = fxx.unsqueeze(0).unsqueeze(0)
        self.fyy = fyy.unsqueeze(0).unsqueeze(0)
        self.fuu = fuu.unsqueeze(0).unsqueeze(0)
        self.fvv = fvv.unsqueeze(0).unsqueeze(0)

        self.process = self._structure_processing_layers(filters)
        self.num_blocks = num_blocks

        self.denoiser = TED(filters // 2 , num_blocks)
        self.RWKV_pool = nn.MaxPool2d(8)
        self.RWKV_block = Block(n_embd=filters)
        self.RWKV_up = nn.Upsample(scale_factor=8, mode='nearest')
        # self.RWKV_conv = nn.Conv2d(filters, filters, kernel_size=1, padding=0)

        self.DCFE = DCFEBlock(filters)

        self.channel_conv = nn.Conv2d(filters * 3, filters, kernel_size=1, padding=0)
        self.final_conv = nn.Conv2d(filters, 3, kernel_size=3, padding=1)
        self._init_weights()

        self.pixelshuffle = nn.PixelShuffle(2)
        self.embedding = nn.Conv2d(inp_channels*4,filters,kernel_size=3, stride=1, padding=1)
        self.conv_out = nn.Conv2d(filters, inp_channels*3, kernel_size=3, stride=1, padding=1)

    def _structure_processing_layers(self, filters):
        return nn.Sequential(
            nn.Conv2d(1, filters, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def _rwkv_integration(self, x):
        num_blocks = self.num_blocks
        x_process_1 = self.RWKV_pool(x)
        x_process_2 = self.RWKV_block(x_process_1)
        for i in range(num_blocks[0] - 1):
            x_process_2 = self.RWKV_block(x_process_2)
        x_process_3 = self.RWKV_up(x_process_2)
        return x + x_process_3

    def forward(self, inputs):
        channels = torch.chunk(inputs, 3, dim=1)
        img_r = channels[0]
        img_g = channels[1]
        img_b = channels[2]
        S_r = self.ssf(img_r)
        S_g = self.ssf(img_g)
        S_b = self.ssf(img_b)
        tex_r = img_r - S_r
        tex_g = img_g - S_g
        tex_b = img_b - S_b

        tex_r = self.denoiser(tex_r) + tex_r
        tex_g = self.denoiser(tex_g) + tex_g
        tex_b = self.denoiser(tex_b) + tex_b

        tex_r_processed = self.process(tex_r)
        tex_g_processed = self.process(tex_g)
        tex_b_processed = self.process(tex_b)

        S_r_processed = self.process(S_r)
        S_g_processed = self.process(S_g)
        S_b_processed = self.process(S_b)
        S_r_rwkv = self._rwkv_integration(S_r_processed)
        S_g_rwkv = self._rwkv_integration(S_g_processed)
        S_b_rwkv = self._rwkv_integration(S_b_processed)

        R = S_r_rwkv + tex_r_processed
        G = S_g_rwkv + tex_g_processed
        B = S_b_rwkv + tex_b_processed

        RGB = torch.cat([R, G, B], dim=1)
        RGB = self.channel_conv(RGB) 
        RGB_enhancement = self.DCFE(RGB)
        output = self.final_conv(RGB_enhancement)
        return torch.sigmoid(output)

    def _init_weights(self):
        for module in self.children():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                init.kaiming_uniform_(module.weight, a=0, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    init.constant_(module.bias, 0)
    
####################################################################
        # Helper function to convert PSF to OTF
    def psf2otf(self,psf, size):
            otf = torch.fft.fft2(torch.fft.fftshift(psf), s=size)
            norm_factor = torch.sqrt(torch.prod(torch.tensor(size, dtype=torch.float32)))
            otf = otf / norm_factor
            return otf

    def ssf(self,img):
        batch, N, M, D = img.shape
        # (10,3,256,384)
        sizeI2D = (M, D)
        Dx = self.Dx
        Dy = self.Dy
        fxx = self.fxx
        fyy = self.fyy
        fuu = self.fuu
        fvv = self.fvv
        Dx, Dy = Dx.to(img.device), Dy.to(img.device)
        fxx, fyy, fuu, fvv = self.fxx.to(img.device), self.fyy.to(img.device), self.fuu.to(img.device), self.fvv.to(img.device)

        iter = 0
        alpha = 0.8
        beta = 0.05
        kappa = 1.5
        tau = 0.8
        iter_max = 50
        lambd_max = 10**2
        lambd = 0.5
        otfDx = self.psf2otf(Dx, sizeI2D) 
        otfDy = self.psf2otf(Dy, sizeI2D) 
        otffxx = self.psf2otf(fxx, sizeI2D)
        otffyy = self.psf2otf(fyy, sizeI2D)
        otffuu = self.psf2otf(fuu, sizeI2D)
        otffvv = self.psf2otf(fvv, sizeI2D)

        # Denoising parameters
        Denormin1 = torch.abs(otfDx)**2 + torch.abs(otfDy)**2
        Denormin2 = torch.abs(otffxx)**2 + torch.abs(otffyy)**2 + torch.abs(otffuu)**2 + torch.abs(otffvv)**2

        S = img.clone() 
        Normin0 = torch.fft.fft2(S)

        # First-order gradients
        gx = F.conv2d(S, Dx)
        gy = F.conv2d(S, Dy)
        gxx = F.conv2d(S, fxx)
        gyy = F.conv2d(S, fyy)
        guu = F.conv2d(S, fuu)
        gvv = F.conv2d(S, fvv)

        t = (gxx**2 + gyy**2 + guu**2 + gvv**2) < (beta / lambd)
        gxx[t] = 0
        gyy[t] = 0
        guu[t] = 0
        gvv[t] = 0
        dx = torch.flip(torch.flip(Dx, dims=[2, 3]), dims=[1, 0])
        dy = torch.flip(torch.flip(Dy, dims=[2, 3]), dims=[1, 0])
        dx = dx.to(img.device)
        dy = dy.to(img.device)
        convolved_gx = F.conv2d(gx, dx, padding = 1)
        convolved_gy = F.conv2d(gy, dy, padding = 1)

        # Gradient of the gradients
        Normin_x = torch.roll(convolved_gx, shifts=(-1, 1), dims=(2, 3))
        Normin_y = torch.roll(convolved_gy, shifts=(-1, 1), dims=(1, 2))
        Normin1 = Normin_x + Normin_y

        while (lambd <= lambd_max) and (iter <= iter_max):
            Normin_xx = F.conv2d(gxx, fxx, padding = 2)
            Normin_yy = F.conv2d(gyy, fyy, padding = 2)
            Normin_uu = F.conv2d(guu, fuu, padding = 2)
            Normin_vv = F.conv2d(gvv, fvv, padding = 2)

            Normin2 = Normin_xx + Normin_yy + Normin_uu + Normin_vv
            Denormin = 1.0 + alpha * Denormin1 + lambd * Denormin2
            x = torch.fft.fft2(Normin1)
            y = torch.fft.fft2(Normin2)
            # y = y.expand(-1, 3, -1, -1)

            FS = (Normin0 + 0.1 * x + 0.2 * y) / Denormin
            S = torch.fft.ifft2(FS).real
            alpha *= tau
            lambd *= kappa
            iter += 1
        return S