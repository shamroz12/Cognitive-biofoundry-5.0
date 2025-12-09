# Deploy README - Cognitive Biofoundry (Full Deployment)

## Files included
- Dockerfile           : CPU Dockerfile (installs PyTorch CPU)
- Dockerfile.gpu       : GPU Dockerfile (CUDA base image)
- docker-compose.yml   : quick local compose for running the app
- requirements.txt     : Python deps (SB3, Gymnasium, Streamlit)
- .github/workflows/docker-publish.yml : GitHub Actions for build & push
- .dockerignore        : files to ignore in Docker context

## Local build & run (CPU)
1. Build the image:
   ```bash
   docker build -t cognitive-biofoundry:latest .
   ```
2. Run the container:
   ```bash
   docker run --rm -p 8501:8501 -v $(pwd)/models:/app/models cognitive-biofoundry:latest
   ```
3. Visit http://localhost:8501

## Using docker-compose
```bash
docker compose up --build
```

## Push to Docker Hub (CI will also push on commit)
1. Tag and push manually (optional):
   ```bash
   docker tag cognitive-biofoundry:latest yourhubuser/cognitive-biofoundry:latest
   docker push yourhubuser/cognitive-biofoundry:latest
   ```
2. CI: add Docker Hub credentials to GitHub Secrets:
   - DOCKERHUB_USERNAME
   - DOCKERHUB_TOKEN

## GPU build (on machine with NVIDIA drivers)
1. Build using Dockerfile.gpu:
   ```bash
   docker build -t cognitive-biofoundry:gpu -f Dockerfile.gpu .
   ```
2. Run with NVIDIA runtime:
   ```bash
   docker run --gpus all -p 8501:8501 cognitive-biofoundry:gpu
   ```

## Notes
- Streamlit Cloud free tier does not support custom Docker builds; use a VM or platform that supports container images (Render, DigitalOcean App Platform, AWS ECS, GCP Cloud Run with Docker).
- For production, consider adding monitoring, log aggregation, and resource limits.
