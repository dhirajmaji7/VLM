
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
### BLIP 2 Inference Example

#### Prompt : "Question: describe the image. Answer: "
<img width="981" height="517" alt="Screenshot from 2025-09-05 11-49-45" src="https://github.com/user-attachments/assets/c44fbeaa-2402-492b-86b8-b3cd3d3b2d27" />
<img width="981" height="517" alt="Screenshot from 2025-09-05 11-51-04" src="https://github.com/user-attachments/assets/cf29a2ac-ab7c-4f17-8a91-3cb3fe4578de" />

<img width="981" height="517" alt="Screenshot from 2025-09-05 11-46-15" src="https://github.com/user-attachments/assets/c18a483c-6f44-452b-9662-7509bc146212" />

<img width="981" height="517" alt="Screenshot from 2025-09-05 11-51-32" src="https://github.com/user-attachments/assets/878a3961-400d-4918-ad80-a2694a4352d2" />
<img width="981" height="517" alt="Screenshot from 2025-09-05 11-51-45" src="https://github.com/user-attachments/assets/aadea73b-8a6f-4909-9882-7fc44bdf633d" />
