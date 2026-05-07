# TSLR: Lightweight Low-Light Image Enhancement with Texture-Structure Decoupling and Efficient RWKV

## Package Dependencies
This project is built by Python 3.10, CUDA 11.6. For other python package dependencies:
```bash
pip install -r requirements.txt

Download the LOLV1, LOLv2 datasets, and put them under the TSLR directory.
The folders should be like:

Plaintext
TSLR/
├── LOLv1/
│   ├── Train/
│   │   ├── input/
│   │   └── target/
│   └── Test/
├── LOLv2/
│   ├── Real_captured/
│   └── Synthetic/
├── LSRW/
│   ├── Training data/
│   └── Eval/
To train, set the options in train.py, and run:

Bash
python train.py
Customizing the Training
To train on a specific dataset or change hyperparameters, use the command-line arguments:

Bash
CUDA_VISIBLE_DEVICES=0 python train.py \
    --dataset lolv1 \
    --data_root /path/to/your/datasets \
    --batch_size 16 \
    --lr 2e-4 \
    --crop_size 256 \
    --max_iterations 200000
Available Arguments:

--dataset: Choose from ['lolv1', 'lolv2-real', 'lolv2-syn', 'LSRW-Huawei', 'LSRW-Nikon', 'SICE'].

--data_root: Path to the directory containing all dataset folders (default is ./).

--batch_size: Number of images per batch (default: 16).

--lr: Initial learning rate (default: 2e-4).

--crop_size: Patch size for training (default: 256).

--save_dir: Directory to save the checkpoints and TensorBoard logs (default: ./result).

**Acknowledgment:**
This code is based on the [LYT-Net](https://github.com/albrateanu/LYT-Net) and [Restore-RWKV](https://github.com/Yaziwel/Restore-RWKV).

## Contact
If you have any questions or suggestions, please contact liuyz_007@163.com.

## LICENSE
<a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc/4.0/80x15.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/">Creative Commons Attribution-NonCommercial 4.0 International License</a>.
