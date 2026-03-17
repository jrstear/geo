# Deployment Guide for Google Cloud Run

This guide explains how to deploy the Elevation Comparison app to Google Cloud Run without affecting local development.

## Prerequisites

1. Google Cloud account
2. `gcloud` CLI installed and authenticated
3. Docker installed (for local testing)

## Local Development (Unchanged)

The app still works exactly as before:
```bash
python app.py
# or
python3 app.py
```

No changes needed to your local workflow!

## Deployment Steps

### 1. Set up Google Cloud Project

```bash
# Create a new project (or use existing)
gcloud projects create your-project-id --name="Elevation Comparison"

# Set as active project
gcloud config set project your-project-id

# Enable required APIs
gcloud services enable run.googleapis.com
gcloud services enable cloudbuild.googleapis.com
```

### 2. Build and Deploy

```bash
# Build the container image
gcloud builds submit --tag gcr.io/your-project-id/elevation-comparison

# Deploy to Cloud Run
gcloud run deploy elevation-comparison \
  --image gcr.io/your-project-id/elevation-comparison \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --timeout 300 \
  --max-instances 10 \
  --set-env-vars MAX_REQUEST_SIZE=500MB
```

### 3. Get Your URL

After deployment, you'll get a URL like:
```
https://elevation-comparison-xxxxx-uc.a.run.app
```

### 4. (Optional) Custom Domain

1. Go to Cloud Run console
2. Click on your service
3. Go to "Custom Domains" tab
4. Map your domain

## Environment Variables (Optional)

You can set environment variables in Cloud Run:

```bash
gcloud run services update elevation-comparison \
  --set-env-vars DEBUG=False \
  --region us-central1
```

## Key Points

- **Local development unchanged**: App still runs with `python app.py`
- **Automatic detection**: Code detects if running on Cloud Run vs locally
- **No breaking changes**: All existing functionality preserved
- **File uploads work**: Uses `/tmp` on Cloud Run (ephemeral, but sufficient for processing)

## Cost Estimate

- **Free tier**: 2 million requests/month free
- **After free tier**: ~$0.40 per million requests
- **Compute**: ~$0.00002400 per GB-second
- **Typical usage**: $5-20/month for moderate use

## Troubleshooting

- Check logs: `gcloud run services logs read elevation-comparison --region us-central1`
- View service: `gcloud run services describe elevation-comparison --region us-central1`


