# TAIHRI: Task-Aware 3D Human Keypoints Localization for Close-Range Human-Robot Interaction

## 📁 Repository Structure

- demo/
  - demo.py — 2D/3D keypoint inference (MLLM)
- eval/
  - eval.py — evaluation script
- eval_wrapper/ — wrappers, parsers, task definitions, visualization
- demo_script.sh / eval_script.sh — runnable examples

## 🧰 Requirements

- Linux
- Python 3.10+
- CUDA-capable GPU (recommended for vLLM / FlashAttention)

## 📦 Installation

### 1) Setup Pytorch Environment

```
pip install torch==2.8.0 torchdata==0.11.0 torchvision==0.23.0
```

### 2) Install Other Packages

```bash
pip install -r requirements.txt
```

## 💡 Checkpoints

You may download released checkpoints from [huggingface](https://huggingface.co/RammusAoLi/TAIHRI) and put it under `./checkpoints`

> Note: This repository references external modules (e.g., model weights, optional packages). Make sure they are available in your environment.

## 🚀 Quick Start

Run the provided script:

```bash
bash demo_script.sh
```

## 🧪 Evaluation

Run the provided evaluation script:

```bash
bash eval_script.sh
```

## 🔧 Common Arguments

- `--model_path`: local path or HuggingFace repo
- `--backend`: `transformers` or `vllm`
- `--focal_length`, `--princpt_x`, `--princpt_y`: camera intrinsics
- `--input_path`, `--output_path`: input/output paths

## ❤️ Acknowledgements

This project builds on and is inspired by several excellent open-source projects and tools, including:

- [Rex-Omni](https://github.com/IDEA-Research/Rex-Omni) for the code base of Qwen-VL SFT.
- [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL/tree/main) for the code of Qwen3-VL finetuning and inference.
- [vLLM](https://github.com/vllm-project/vllm) for the efficient inference acceleration.
- [SAM-3D-Body](https://github.com/facebookresearch/sam-3d-body) for 3D body mesh recovery and visualization in the demo pipeline.

We also thank the community contributors and dataset providers who make research and evaluation possible.

## 📄 License

See the [LICENSE](LICENSE) file for details.
