# 🚀 Deployment Guide: Vercel + Hugging Face Spaces

Complete guide to deploy your Jewellery Search application with **Vercel (Frontend)** and **Hugging Face Spaces (Backend)**.

---

## 📋 Prerequisites

- [x] GitHub account
- [x] Vercel account (sign up at vercel.com)
- [x] Hugging Face account (sign up at huggingface.co)
- [x] Groq API key (from console.groq.com)

---

## Part 1: Deploy Backend to Hugging Face Spaces

### Step 1: Create Hugging Face Space

1. Go to https://huggingface.co/spaces
2. Click **"Create new Space"**
3. Configure:
   - **Space name**: `jewellery-search-api`
   - **License**: Apache 2.0
   - **Space SDK**: Docker
   - **Visibility**: Public (free tier)
4. Click **"Create Space"**

### Step 2: Prepare Backend Files

Create a `Dockerfile` in your backend directory:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Expose port
EXPOSE 7860

# Run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
```

Create `.dockerignore`:

```
__pycache__/
*.pyc
.env
.git/
.gitignore
*.md
```

Update `requirements.txt` to include all dependencies:

```txt
fastapi==0.104.1
uvicorn[standard]==0.24.0
torch==2.1.0
git+https://github.com/openai/CLIP.git
chromadb==0.4.18
numpy==1.24.3
Pillow==10.1.0
python-multipart==0.0.6
openai==1.3.0
python-dotenv==1.0.0
```

### Step 3: Upload to Hugging Face

**Option A: Using Git (Recommended)**

```bash
cd /home/akash/Jewellary_RAG/backend

# Initialize git if not already
git init

# Add HF remote
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/jewellery-search-api

# Add files
git add Dockerfile requirements.txt app.py .dockerignore
git add data/ chroma_primary/  # Your data and ChromaDB

# Commit
git commit -m "Initial backend deployment"

# Push to HF
git push hf main
```

**Option B: Using Web Interface**

1. Go to your Space page
2. Click **"Files"** tab
3. Upload:
   - `Dockerfile`
   - `requirements.txt`
   - `app.py`
   - `data/` folder
   - `chroma_primary/` folder

### Step 4: Configure Environment Variables

1. Go to your Space **Settings**
2. Scroll to **"Repository secrets"**
3. Add secret:
   - **Name**: `GROQ_API_KEY`
   - **Value**: Your Groq API key
4. Click **"Add secret"**

### Step 5: Wait for Build

- HF will automatically build your Docker container
- Check **"Logs"** tab for build progress
- Build takes ~10-15 minutes first time
- Once complete, you'll see: `Running on http://0.0.0.0:7860`

### Step 6: Test Backend

Your backend will be available at:
```
https://YOUR_USERNAME-jewellery-search-api.hf.space
```

Test it:
```bash
curl https://YOUR_USERNAME-jewellery-search-api.hf.space/docs
```

---

## Part 2: Deploy Frontend to Vercel

### Step 1: Prepare Frontend for Deployment

Update `frontend/src/services/api.js` to use HF backend:

```javascript
import axios from 'axios';

// Use environment variable for backend URL
const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: BACKEND_URL,
  timeout: 60000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Rest of the file stays the same...
```

Create `frontend/.env.production`:

```env
VITE_BACKEND_URL=https://YOUR_USERNAME-jewellery-search-api.hf.space
```

Create `frontend/vercel.json`:

```json
{
  "buildCommand": "npm run build",
  "outputDirectory": "dist",
  "framework": "vite",
  "rewrites": [
    {
      "source": "/(.*)",
      "destination": "/index.html"
    }
  ]
}
```

### Step 2: Push to GitHub

```bash
cd /home/akash/Jewellary_RAG

# Initialize git if not already
git init

# Add all files
git add .

# Create .gitignore
echo "node_modules/
dist/
.env
.env.local
backend/__pycache__/
backend/.env" > .gitignore

# Commit
git commit -m "Initial commit for deployment"

# Create GitHub repo and push
git remote add origin https://github.com/YOUR_USERNAME/jewellery-search.git
git branch -M main
git push -u origin main
```

### Step 3: Deploy to Vercel

1. Go to https://vercel.com
2. Click **"Add New Project"**
3. **Import Git Repository**:
   - Select your GitHub repo
   - Click **"Import"**

4. **Configure Project**:
   - **Framework Preset**: Vite
   - **Root Directory**: `frontend`
   - **Build Command**: `npm run build`
   - **Output Directory**: `dist`

5. **Environment Variables**:
   - Click **"Environment Variables"**
   - Add:
     - **Name**: `VITE_BACKEND_URL`
     - **Value**: `https://YOUR_USERNAME-jewellery-search-api.hf.space`

6. Click **"Deploy"**

### Step 4: Wait for Deployment

- Vercel builds in ~2-3 minutes
- You'll get a URL like: `https://jewellery-search.vercel.app`

---

## Part 3: Configure CORS

Update your backend `app.py` to allow Vercel domain:

```python
# Update CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "https://jewellery-search.vercel.app",  # Add your Vercel URL
        "https://*.vercel.app",  # Allow all Vercel preview deployments
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

Push the update to Hugging Face:

```bash
cd backend
git add app.py
git commit -m "Update CORS for Vercel"
git push hf main
```

---

## 🎉 You're Live!

**Frontend**: `https://jewellery-search.vercel.app`
**Backend API**: `https://YOUR_USERNAME-jewellery-search-api.hf.space`

---

## 📊 Monitoring & Limits

### Hugging Face Free Tier:
- ✅ 16GB RAM
- ✅ 2 CPU cores
- ✅ 50GB storage
- ⚠️ Sleeps after 48h inactivity
- ⚠️ Community tier usage limits

### Vercel Free Tier:
- ✅ 100GB bandwidth/month
- ✅ Unlimited deployments
- ✅ Auto-deploy on git push
- ✅ Free SSL

---

## 🔧 Troubleshooting

### Backend Issues:

**Build fails:**
- Check Dockerfile syntax
- Verify all files are uploaded
- Check HF Space logs

**Out of memory:**
- Reduce ChromaDB data
- Use smaller CLIP model
- Upgrade to HF Pro ($9/month)

**API not responding:**
- Check if Space is sleeping
- Visit Space URL to wake it up
- Check CORS settings

### Frontend Issues:

**API calls fail:**
- Verify `VITE_BACKEND_URL` is correct
- Check CORS configuration
- Test backend URL directly

**Build fails:**
- Check `package.json` dependencies
- Verify build command
- Check Vercel logs

---

## 🚀 Updates & Redeployment

### Update Backend:
```bash
cd backend
# Make changes
git add .
git commit -m "Update message"
git push hf main
```

### Update Frontend:
```bash
cd frontend
# Make changes
git add .
git commit -m "Update message"
git push origin main
# Vercel auto-deploys!
```

---

## 💡 Tips

1. **Keep HF Space Active**: Visit it regularly to prevent sleep
2. **Use Environment Variables**: Never commit API keys
3. **Monitor Usage**: Check HF Space analytics
4. **Test Locally First**: Always test before deploying
5. **Use Preview Deployments**: Vercel creates preview for each PR

---

## 📚 Resources

- [Hugging Face Spaces Docs](https://huggingface.co/docs/hub/spaces)
- [Vercel Documentation](https://vercel.com/docs)
- [FastAPI Deployment](https://fastapi.tiangolo.com/deployment/)
- [Vite Production Build](https://vitejs.dev/guide/build.html)
