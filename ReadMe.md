
# Vision–Language Models (VLM)

Minimal, readable PyTorch implementations to learn core VLM ideas.

## Key Highlights
- End-to-end CLIP, ViT, and BLIP‑2: compact, from-scratch training and demo code.
- Practical training loops and losses: tokenizers, datasets, and InfoNCE/ITC‑ITM‑ITG included.
- Ready-to-run notebooks and scripts: small Flickr8k setup for quick experiments.

## What’s Inside
- CLIP training (contrastive image–text) — `clip/trainer.ipynb`
- ViT supervised baseline — `vit/main.ipynb`
- BLIP‑2 demo / fine‑tune — `blip2/main.ipynb`
- BLIP‑2 Distillation — `blip2/distillation.ipynb`

Configuration lives in:
- CLIP config — `clip/config.py`
- ViT config — `vit/config.py`
- BLIP‑2 config — `blip2/config.py`

## Dataset
- Download Flickr8k captions and images, then set paths in `clip/config.py` and `blip2/config.py`.
- Example download via Kaggle CLI:
```
curl -L -o flickr8k.zip https://www.kaggle.com/api/v1/datasets/download/adityajn105/flickr8k
```

## Quick Start
- Open the notebooks (CLIP/ViT/BLIP‑2/Distillation) and run cells.
- Or use the BLIP‑2 trainer script for fine‑tuning: `python blip2/main.py`.

## Docker (optional)
If you prefer an isolated environment, build a simple image with CUDA/cuDNN as a base and mount this repo. Example run (assuming an image tagged `vlm:dev`):
```bash
docker run --rm -it \
  --gpus all \
  -v "$PWD":/workspace -w /workspace \
  -p 8888:8888 \
  -e SDL_VIDEODRIVER=dummy \
  -e MPLBACKEND=Agg \
  vlm:dev bash
```

## Requirements
- Python 3.10+; PyTorch + CUDA (for GPU)
- torchvision, timm, transformers, tiktoken, pillow, matplotlib, wandb

## BLIP‑2 Inference Example

Prompt: "Question: describe the image. Answer:"

<img width="981" height="517" alt="Screenshot from 2025-09-05 11-49-45" src="https://github.com/user-attachments/assets/c44fbeaa-2402-492b-86b8-b3cd3d3b2d27" />
<img width="981" height="517" alt="Screenshot from 2025-09-05 11-51-04" src="https://github.com/user-attachments/assets/cf29a2ac-ab7c-4f17-8a91-3cb3fe4578de" />
<img width="981" height="517" alt="Screenshot from 2025-09-05 11-46-15" src="https://github.com/user-attachments/assets/c18a483c-6f44-452b-9662-7509bc146212" />
<img width="981" height="517" alt="Screenshot from 2025-09-05 11-51-32" src="https://github.com/user-attachments/assets/878a3961-400d-4918-ad80-a2694a4352d2" />
<img width="981" height="517" alt="Screenshot from 2025-09-05 11-51-45" src="https://github.com/user-attachments/assets/aadea73b-8a6f-4909-9882-7fc44bdf633d" />

