
# Vision–Language Models (VLM)

Basic, containerized examples for training various vision and vision-language models:

- CLIP training (contrastive image–text) — clip/trainer.ipynb 

- ViT supervised baseline — clip/VIT/VIT.ipynb 

- BLIP-2 demo / fine-tune — blip2/main.ipynb 

- BLIP-2 Distillation — blip2/distillation.ipynb 

The repository ships with a Dev Container for a one-click GPU-ready environment:

- Dockerfile: .devcontainer/Dockerfile 

- Devcontainer config: .devcontainer/devcontainer.json


### Download the Flickr8k dataset using the following command:
```
curl -L -o flickr8k.zip  https://www.kaggle.com/api/v1/datasets/download/adityajn105/flickr8k
```

### 🚀 Quick Start (VS Code Dev Container – recommended)

Open the repo in VS Code → Reopen in Container (Dev Containers extension).

The container will build from .devcontainer/Dockerfile and inherit GPU args/env from .devcontainer/devcontainer.json.

Once inside the container:

Open any notebook and run it (CLIP/ViT/BLIP-2/Distillation).

### 🐳 Docker CLI (no local Python needed)

Build once:
``` bash
docker build -t vlm:dev -f .devcontainer/Dockerfile .
```

Run (GPU + headless-friendly):
``` bash
docker run --rm -it \
  --gpus all \
  -v "$PWD":/workspace -w /workspace \
  -p 8888:8888 \
  -e SDL_VIDEODRIVER=dummy \
  -e MPLBACKEND=Agg \
  vlm:dev bash
```
