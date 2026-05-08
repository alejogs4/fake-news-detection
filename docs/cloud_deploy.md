# Google Cloud and Google Colab Deployment Guide

This project is designed to be easily deployed to Google Cloud Platform (GCP) and can also be run interactively on Google Colab.

## Option 1: Google Colab (Recommended for Exploration & Quick Training)

Google Colab provides free access to GPUs and is excellent for training graph neural networks on the UPFD dataset.

### Steps to Run on Colab:
1. **Upload the Dataset:** 
   - Compress your `dataset/` folder containing the unzipped `politifact` files (e.g., `dataset/politifact/raw/*`).
   - Upload this zip file to your Google Drive.
2. **Mount Google Drive in Colab:**
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
3. **Clone the Repository & Unzip Data:**
   ```bash
   !git clone https://github.com/YOUR_USERNAME/fake-news-detection.git
   %cd fake-news-detection
   !cp /content/drive/MyDrive/dataset.zip .
   !unzip dataset.zip -d .
   ```
4. **Install Dependencies:**
   ```bash
   # Install PyTorch Geometric and its dependencies
   !pip install torch_geometric
   !pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
   ```
5. **Run Training:**
   ```bash
   !python main.py
   ```

## Option 2: Google Cloud Platform (Vertex AI - Recommended for Production ML)

Vertex AI is GCP's managed machine learning platform. It allows you to run distributed training jobs using custom containers.

### Steps to Run on Vertex AI:
1. **Prepare Google Cloud Storage (GCS):**
   - Create a GCS bucket: `gsutil mb -l us-central1 gs://your-project-ml-bucket`
   - Upload your dataset: `gsutil -m cp -r dataset/ gs://your-project-ml-bucket/dataset/`
2. **Containerize the Application:** Use the provided `Dockerfile` to build your training image.
3. **Push to Artifact Registry:** 
   ```bash
   gcloud auth configure-docker us-central1-docker.pkg.dev
   docker build -t us-central1-docker.pkg.dev/[PROJECT-ID]/[REPO-NAME]/upfd-model:latest .
   docker push us-central1-docker.pkg.dev/[PROJECT-ID]/[REPO-NAME]/upfd-model:latest
   ```
4. **Run Custom Training Job:**
   - Go to Vertex AI in the Google Cloud Console.
   - Navigate to "Training" -> "Create".
   - Select "Custom Training".
   - Provide the container image URI from step 3.
   - Attach your GCS bucket or mount it using Cloud Storage FUSE so the container can access the dataset natively.

## Option 3: Google Cloud Compute Engine / GKE

If you prefer to manage the infrastructure yourself:
1. Spin up a Compute Engine instance with a GPU attached (e.g., NVIDIA T4 or A100).
2. SSH into your instance.
3. Install Miniconda.
4. Clone the repo and run `conda env create -f environment.yml`.
5. Run `python main.py`.

## Tips for Cloud Deployments
- **Cloud Storage:** For very large files, update the data loading logic to stream or copy files from a GCS bucket at the start of your training script.
- **GPUs:** Ensure you have the necessary GPU quota in your GCP region if you plan to use `device: cuda` in `config.yaml`.
