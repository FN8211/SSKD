# SSKD: Self-Supervision Knowledge Distillation

Knowledge distillation framework using self-supervised learning on CIFAR-100.

## Requirements

- CUDA-capable GPU
- CUDA 11.x or higher
- Python 3.8+

Install PyTorch with CUDA support from [pytorch.org](https://pytorch.org/get-started/locally/), then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

## Pre-trained Teachers

Downloaded pre-trained teacher models and organize as follows:

```
experiments/
    teacher_<arch>_seed0/
        ckpt/
            best.pth
```

Available architectures: `MobileNetV2`, `resnet8x4`, `resnet20`, `resnet56`, `ShuffleV1`, `ShuffleV2`, `wrn_16_2`, `wrn_40_1`

## Training

**Train a teacher:**
```bash
python teacher.py --arch <teacher_model> --gpu-id 0
```

**Train a student:**
```bash
python student.py --t-path <teacher_checkpoint_folder> --s-arch <student_model> --gpu-id 0
```

## Key Arguments

| Argument | Default | Description |
|---|---|---|
| `--epoch` | 240 | Training epochs |
| `--t-epoch` | 60 | Teacher SSP training epochs |
| `--batch-size` | 64 | Batch size |
| `--val-interval` | 1 | Validate every N epochs |
| `--lr` | 0.05 | Learning rate |
| `--gpu-id` | 0 | GPU index |

## Output

Checkpoints are saved to `./experiments/<exp_name>/ckpt/`.
