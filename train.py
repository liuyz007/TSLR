import os
import random
import argparse
from datetime import datetime
from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.functional import structural_similarity_index_measure


from losses import CombinedLoss
from dataloader import create_dataloaders
from rwkvir_model import TSLR


def parse_args():
    parser = argparse.ArgumentParser(description="Train TSLR for Image Restoration/Enhancement")

    parser.add_argument('--dataset', type=str, default='lolv2-real', 
                        choices=['lolv1', 'lolv2-real', 'lolv2-syn', 'LSRW-Huawei', 'LSRW-Nikon'],
                        help='Dataset version to use')
    parser.add_argument('--data_root', type=str, default='./', 
                        help='Root directory for datasets')
    parser.add_argument('--crop_size', type=int, default=256, help='Crop size for training images')
    parser.add_argument('--save_dir', type=str, default='./result', help='Directory to save results and models')

    parser.add_argument('--batch_size', type=int, default=16, help='Training batch size')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--max_iterations', type=int, default=200000, help='Total maximum iterations')

    return parser.parse_args()


def get_dataset_paths(dataset_version, data_root):
    dataset_dict = {
        'lolv1': {
            'train_low': 'LOLv1/Train/input', 'train_high': 'LOLv1/Train/target',
            'test_low': 'LOLv1/Test/input', 'test_high': 'LOLv1/Test/target'
        },
        'lolv2-real': {
            'train_low': 'LOLv2/Real_captured/Train/Low', 'train_high': 'LOLv2/Real_captured/Train/Normal',
            'test_low': 'LOLv2/Real_captured/Test/Low', 'test_high': 'LOLv2/Real_captured/Test/Normal'
        },
        'lolv2-syn': {
            'train_low': 'LOLv2/Synthetic/Train/Low', 'train_high': 'LOLv2/Synthetic/Train/Normal',
            'test_low': 'LOLv2/Synthetic/Test/Low', 'test_high': 'LOLv2/Synthetic/Test/Normal'
        },
        'LSRW-Huawei': {
            'train_low': 'LSRW/Training data/Huawei/low', 'train_high': 'LSRW/Training data/Huawei/high',
            'test_low': 'LSRW/Eval/Huawei/low', 'test_high': 'LSRW/Eval/Huawei/high'
        },
        'LSRW-Nikon': {
            'train_low': 'LSRW/Training data/Nikon/low', 'train_high': 'LSRW/Training data/Nikon/high',
            'test_low': 'LSRW/Eval/Nikon/low', 'test_high': 'LSRW/Eval/Nikon/high'
        }
    }
    
    if dataset_version not in dataset_dict:
        raise ValueError(f"Dataset {dataset_version} not recognized!")
        
    paths = dataset_dict[dataset_version]
    return {k: os.path.join(data_root, v) if v else '' for k, v in paths.items()}


def calculate_psnr(img1, img2, max_pixel_value=1.0, gt_mean=True):
    """Calculate PSNR (Peak Signal-to-Noise Ratio) between two images."""
    if gt_mean:
        img1_gray = img1.mean(axis=1)
        img2_gray = img2.mean(axis=1)
        mean_restored = img1_gray.mean()
        mean_target = img2_gray.mean()
        img1 = torch.clamp(img1 * (mean_target / mean_restored), 0, 1)
    
    mse = F.mse_loss(img1, img2, reduction='mean')
    if mse == 0:
        return float('inf')
    psnr = 20 * torch.log10(max_pixel_value / torch.sqrt(mse))
    return psnr.item()


def calculate_ssim(img1, img2, max_pixel_value=1.0, gt_mean=True):
    """Calculate SSIM (Structural Similarity Index) between two images."""
    if gt_mean:
        img1_gray = img1.mean(axis=1, keepdim=True)
        img2_gray = img2.mean(axis=1, keepdim=True)
        mean_restored = img1_gray.mean()
        mean_target = img2_gray.mean()
        img1 = torch.clamp(img1 * (mean_target / mean_restored), 0, 1)

    ssim_val = structural_similarity_index_measure(img1, img2, data_range=max_pixel_value)
    return ssim_val.item()


def validate(model, dataloader, device):
    model.eval()
    total_psnr, total_ssim = 0, 0
    
    with torch.no_grad():
        for low, high, _ in dataloader:
            low, high = low.to(device), high.to(device)
            output = model(low)

            total_psnr += calculate_psnr(output, high)
            total_ssim += calculate_ssim(output, high)

    avg_psnr = total_psnr / len(dataloader)
    avg_ssim = total_ssim / len(dataloader)
    return avg_psnr, avg_ssim


def worker_init_fn(worker_id):
    random.seed(1111 + worker_id)


def main():
    args = parse_args()

    time_str = datetime.now().strftime("%Y%m%d_%H%M")
    exp_name = f"{args.dataset}_{time_str}"
    snapshot_path = os.path.join(args.save_dir, exp_name)
    os.makedirs(snapshot_path, exist_ok=True)
    
    print(f"[*] Starting experiment: {exp_name}")
    print(f"[*] Saving logs and models to: {snapshot_path}")

    paths = get_dataset_paths(args.dataset, args.data_root)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Using device: {device}")

    # init model
    model = TSLR(num_blocks=[1, 1]).to(device)

    # init Dataloader
    train_loader, test_loader = create_dataloaders(
        train_low=paths['train_low'], train_high=paths['train_high'],
        test_low=paths['test_low'], test_high=paths['test_high'],
        crop_size=args.crop_size, batch_size=args.batch_size
    )

    num_epochs = args.max_iterations // len(train_loader) + 1
    print(f"[*] Target Max Iterations: {args.max_iterations} | Equivalent Epochs: {num_epochs}")
    print(f"[*] Learning Rate: {args.lr} | Batch Size: {args.batch_size}")

    criterion = CombinedLoss(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    scaler = torch.cuda.amp.GradScaler()
    writer = SummaryWriter(os.path.join(snapshot_path, 'log'))

    best_psnr = 0
    iter_num = 0
    
    print('[-] Training started.')
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', ncols=100)
        
        for batch_idx, batch in enumerate(train_pbar):
            inputs, targets, _ = batch
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
            iter_num += 1
            writer.add_scalar('Train/Loss', loss.item(), iter_num)
            train_pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        # val
        avg_psnr, avg_ssim = validate(model, test_loader, device)
        writer.add_scalar('Eval/PSNR', avg_psnr, epoch)
        writer.add_scalar('Eval/SSIM', avg_ssim, epoch)
        
        print(f"  --> Validation | PSNR: {avg_psnr:.4f} | SSIM: {avg_ssim:.4f}")
        
        scheduler.step()
        writer.add_scalar('Train/LR', scheduler.get_last_lr()[0], epoch)

        # save best model
        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            save_name = f"best_epoch_{epoch}_psnr_{avg_psnr:.4f}_ssim_{avg_ssim:.4f}.pth"
            save_path = os.path.join(snapshot_path, save_name)
            torch.save(model.state_dict(), save_path)
            print(f"  --> [*] New best model saved: {save_name}")

    writer.close()
    print("[-] Training finished.")


if __name__ == '__main__':
    main()