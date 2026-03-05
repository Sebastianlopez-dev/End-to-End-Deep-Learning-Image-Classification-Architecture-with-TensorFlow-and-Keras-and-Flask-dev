# Deploying to Hugging Face Spaces

## Why Did Vercel Fail?
The error in your screenshot from Vercel says:
> `Error: Total dependency size (2271.35 MB) exceeds Lambda ephemeral storage limit (500 MB)`

**Why this happens:**
Vercel is designed for frontend applications (like React, Next.js, Vite) and lightweight backend APIs. When you deploy a Python backend on Vercel, it uses **"Serverless Functions"** (powered by AWS Lambda under the hood). These serverless environments have very strict size limits. 

Vercel has a hard limit of **500 MB** for the entire unzipped size of your application and all its dependencies. 

In your case, you are using Machine Learning libraries like `tensorflow`. TensorFlow alone is massive, and when combined with other libraries (like `numpy`, `flask`, `keras`, `pillow`, etc.), your total dependency size reached **2.27 GB (2271.35 MB)**. This completely blew past Vercel's 500 MB limit, causing the build to fail immediately. 

This is exactly why Machine Learning projects require specialized hosting like **Hugging Face Spaces**. Hugging Face is built to intuitively handle gigabytes of models and heavy ML dependencies for free.

---

## How to Deploy Your App to Hugging Face Spaces

Hugging Face Spaces allows you to host Machine Learning web applications completely for free. Since you are building a custom Flask web app, we will use the **Docker** option. Here are the step-by-step instructions.

### Step 1: Create a Hugging Face Account & Space
1. Go to [Hugging Face](https://huggingface.co/) and click **Sign Up** (or Log In).
2. Once logged in, click on your profile picture in the top right corner and select **New Space**.
3. Fill out the application details:
   - **Space name:** e.g., `cifar-10-classifier`
   - **License:** Open Source (e.g., MIT) or leave blank.
   - **Select the Space SDK:** Choose **Docker**.
   - **Choose a Docker template:** Select **Blank**.
   - **Space hardware:** Choose the **Free (CPU basic - 16GB, 2vCPU)** tier.
4. Click **Create space**.

### Step 2: Prepare Your Files for Deployment
You need to make sure your project is ready for Docker. This requires two specific files in the root of your project: `requirements.txt` and a `Dockerfile`.

#### 1. Requirements File (`requirements.txt`)
Ensure you have a `requirements.txt` file listing all your Python dependencies. Based on your project, it should look something like this:
```txt
Flask==3.0.0
tensorflow==2.15.0
numpy
Pillow
Werkzeug
```
*(Make sure this is accurate to your environment, but do NOT include testing libraries or Jupyter notebook libraries to save space).*

#### 2. Create the `Dockerfile`
In your project root folder (where `app/`, `models/`, and `requirements.txt` are located), create a file named exactly **`Dockerfile`** (no extension) and add the following contents:

```dockerfile
# Use the official Python image
FROM python:3.11-slim

# Set the working directory to /code
WORKDIR /code

# Copy the requirements file and install dependencies
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy all the project files into the Docker container
COPY . .

# Expose port 7860 (Hugging Face Spaces requires apps to run on port 7860)
EXPOSE 7860

# Run the Flask app on port 7860
CMD ["flask", "--app", "app/app.py", "run", "--host=0.0.0.0", "--port=7860"]
```
> **Note:** Hugging Face **requires** your application to run on port `7860` instead of the standard Flask port `5000`. The Dockerfile above enforces this.

### Step 3: Push Your Code to Hugging Face
You can upload your files either through the Hugging Face Web UI or via Git. Since you are already using Git, the command line is the best way.

1. Open your terminal in your project directory (`/Users/sebastianlopez/Desktop/it-studies/ironhack/week_7/day_2`).
2. Add the Hugging Face Space as a remote repository to your local Git:
   ```bash
   git remote add huggingface https://huggingface.co/spaces/YOUR_USERNAME/cifar-10-classifier
   ```
   *(Replace `YOUR_USERNAME` and the space name with your actual Hugging Face details).*

3. Commit your new `Dockerfile` and push your code:
   ```bash
   git add Dockerfile requirements.txt app/ models/
   git commit -m "Add Dockerfile for Hugging Face deployment"
   git push huggingface main
   ```
   *(It will prompt you for your Hugging Face username and a **Read/Write Access Token**. You can generate an access token by going to Hugging Face Settings -> Access Tokens).*

### Step 4: Wait for the Build
Once you push your code, go back to your Hugging Face Space page in your browser. 
You will see the status change to **"Building"**. Hugging Face is reading your `Dockerfile`, installing the 2+ GB of dependencies, and starting your Flask server. 

When it finishes, the status will change to **"Running"**, and your Image Classification App will be live and accessible to anyone natively in the browser!
