# Grid2Matrix: Revealing Digital Agnosia in Vision-Language Models
[![arXiv](https://img.shields.io/badge/arXiv-2506.06220-b31b1b.svg)](https://arxiv.org/abs/2604.09687)

Official code release for the paper **Grid2Matrix: Revealing Digital Agnosia in Vision-Language Models**.

This repository contains scripts to **generate** synthetic grid datasets, run **zero-shot VLM inference**, and train **linear probes** on vision-encoder (or projector) features.

## Repository layout

```text
.
├── generate_data.py          # Synthetic images + JSON labels
├── run_vlm_zeroshot.py       # Zero-shot matrix prediction from images
├── run_probe_experiment.py   # Probe training on frozen visual features
├── data_utils.py
├── model_utils.py
├── configs/
│   └── base_config.yaml      # Example hyperparameters (optional reference)
└── requirements.txt

## Setup

Create a Python environment (for example with Conda), then install dependencies. Install **PyTorch** build that matches your CUDA version from [pytorch.org](https://pytorch.org) before the rest:

```bash
pip install -r requirements.txt
```

Optional: for faster attention on supported GPUs, install `flash-attn` (see your hardware vendor’s instructions).

## 1. Generate data

```bash
python generate_data.py \
  --output_dir datasets/dataset_10x10_512 \
  --grid_size 10 10 \
  --image_resolution 512 512 \
  --train_count 8000 \
  --val_count 2000 \
  --test_count 10000 \
  --num_colors 3 \
  --seed 42
```

## 2. Zero-shot VLM inference

Point `--model_name` at a local model directory or a Hugging Face model id. Metrics are computed inside the script; outputs go under `--output_dir`.

```bash
python run_vlm_zeroshot.py \
  --model_name /path/to/your/model \
  --dataset_path ./datasets/dataset_10x10_512 \
  --output_dir ./results/zeroshot_run \
  --split test
```

Optional logging:

```bash
python run_vlm_zeroshot.py ... --wandb --wandb_project your_project
```

## 3. Vision-encoder probe

The dataset path must include `train/` and `test/` (or equivalent splits as expected by `data_utils.GridDataset`).

```bash
python run_probe_experiment.py \
  --model_name /path/to/your/model \
  --dataset_path ./datasets/dataset_64x64_512 \
  --output_dir ./results/probe_run \
  --batch_size 32 \
  --learning_rate 0.01 \
  --target_component projector \
  --mode spatial
```

## Citation

If this code is useful, please cite our paper:

```bibtex
@article{zhang2026grid2matrix,
  title={Grid2Matrix: Revealing Digital Agnosia in Vision-Language Models},
  author={Zhang, Yunkai and Li, Linda and Cui, Yingxin and Ruan, Xiyuan and Zheng, Zeyu and Chen, Kezhen and Zhang, Yi and Yang, Diji},
  journal={arXiv preprint arXiv:2604.09687},
  year={2026}
}
```

## License

The code in this repository is licensed under the [MIT License](LICENSE). 

The Grid2Matrix dataset, accompanying documentation, and the paper are licensed under a [Creative Commons Attribution-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-sa/4.0/).
