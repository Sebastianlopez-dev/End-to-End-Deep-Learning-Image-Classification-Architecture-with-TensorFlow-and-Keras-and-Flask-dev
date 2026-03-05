# Fast and Cheap Deployment Guide (Render.com)

This guide explains how to deploy your Flask application for free using [Render.com](https://render.com/), which is currently one of the easiest and most cost-effective ways to host a Python web app. 

Later, you can move this to a VPS (Virtual Private Server) like DigitalOcean or AWS EC2, but Render is the perfect starting point to quickly fulfill the "+5 Bonus points" requirement.

## Prerequisites

1. Your project must be pushed to a **GitHub repository**.
2. You need an account on [Render.com](https://render.com/) (you can sign up with GitHub).

## Step-by-Step Instructions

### Step 1: Update `requirements.txt`
Render needs a production web server to run your Flask app. We will use `gunicorn`.
Open your `requirements.txt` file and add this line at the very bottom:
```text
gunicorn==21.2.0
```

### Step 2: Push to GitHub
Make sure all your latest changes, particularly the updated `requirements.txt` and your downloaded best model (in the `models/` folder), are committed and pushed to your GitHub repository.

```bash
git add requirements.txt models/best_model.h5
git commit -m "Prepare for Render deployment"
git push origin main
```
*(Note: If your model file is larger than 100MB, you might need to use Git LFS or upload it differently, but MobileNetV2 should be small enough).*

### Step 3: Create a Web Service on Render
1. Log into your Render dashboard.
2. Click on **New +** and select **Web Service**.
3. Connect your GitHub account and select your project repository.

### Step 4: Configure the Web Service
Fill out the deployment form with the following details:
- **Name:** Choose a name for your app (e.g., `cifar10-classifier-sebastian`).
- **Region:** Choose the region closest to you (e.g., Frankfurt/EU).
- **Branch:** `main` (or whichever branch your code is on).
- **Runtime:** `Python 3`.
- **Build Command:** 
  ```bash
  pip install -r requirements.txt
  ```
- **Start Command:** 
  ```bash
  gunicorn app.app:app
  ```
  *(Explanation: The first `app` is your `app` folder, the second `app` is the `app.py` script, and the third `:app` is the Flask instance named `app` inside that script).*

### Step 5: Choose Instance Type & Deploy
- Select the **Free** instance type ($0/month).
- Click **Create Web Service**.

### Step 6: Wait for Build
Render will now install your dependencies and launch your app. The console output will show you the progress. Once it says "Live", your app is running!

You can access your live app using the URL Render provides at the top of the dashboard (e.g., `https://cifar10-classifier-sebastian.onrender.com`).

---
**Why this is the best first step:**
- It is 100% free.
- It automatically redeploys if you push new code to GitHub.
- It handles SSL certificates (`https://`) out of the box.
