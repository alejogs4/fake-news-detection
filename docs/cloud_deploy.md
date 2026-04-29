# Google Cloud Deployment Guide

This project is designed to be easily deployed to Google Cloud Platform (GCP).

## Option 1: Vertex AI (Recommended for ML)
1. **Containerize:** Use the provided `Dockerfile` to build an image.
2. **Push to Artifact Registry:** 
   ```bash
   docker build -t gcr.io/[PROJECT-ID]/upfd-model .
   docker push gcr.io/[PROJECT-ID]/upfd-model
   ```
3. **Run Training Job:** Use the Google Cloud Console or `gcloud` to start a Vertex AI Custom Training Job using your container.

## Option 2: Compute Engine / GKE
1. SSH into your instance.
2. Install Miniconda.
3. Clone the repo and run `conda env create -f environment.yml`.
4. Run `python main.py`.

## Tips for GCP
- **Cloud Storage:** Update `src/data.py` to pull datasets from a GCS bucket if you are working with very large files.
- **GPUs:** Ensure you have GPU quota if you plan to use `device: cuda` in `config.yaml`.
