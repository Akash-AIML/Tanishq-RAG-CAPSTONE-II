# 🚀 Deploy Frontend to Vercel

## Prerequisites

✅ Backend deployed to HF Spaces: https://akash-dragon-jewellery-search-api.hf.space  
✅ Frontend configured with backend URL in `.env.production`

---

## Step 1: Initialize Git (if not already done)

```bash
cd /home/akash/Jewellary_RAG
git init
git add .
git commit -m "Jewellery search app with image upload and OCR"
```

---

## Step 2: Push to GitHub

```bash
# Create a new repo on GitHub, then:
git remote add origin https://github.com/YOUR_USERNAME/jewellery-search.git
git branch -M main
git push -u origin main
```

---

## Step 3: Deploy to Vercel

1. Go to **https://vercel.com/new**

2. **Import Git Repository**
   - Click "Import" next to your GitHub repo
   - Or paste the repo URL

3. **Configure Project**
   - **Framework Preset**: Vite
   - **Root Directory**: `frontend` ← IMPORTANT!
   - **Build Command**: `npm run build`
   - **Output Directory**: `dist`
   - **Install Command**: `npm install`

4. **Environment Variables**
   - Click "Add Environment Variable"
   - Name: `VITE_BACKEND_URL`
   - Value: `https://akash-dragon-jewellery-search-api.hf.space`
   - Apply to: Production, Preview, Development

5. **Click "Deploy"**

---

## Step 4: Update Backend CORS (After Deployment)

Once you get your Vercel URL (e.g., `https://jewellery-search.vercel.app`):

1. Edit `backend/app.py` around line 106:
```python
allow_origins=[
    "http://localhost:5173",
    "https://jewellery-search.vercel.app",  # Your Vercel URL
    "https://*.vercel.app",
],
```

2. Push to HF Space:
```bash
cd backend
git add app.py
git commit -m "Add Vercel URL to CORS"
git push hf main
```

---

## ✅ You're Live!

- **Frontend**: https://your-app.vercel.app
- **Backend**: https://akash-dragon-jewellery-search-api.hf.space
- **API Docs**: https://akash-dragon-jewellery-search-api.hf.space/docs

---

## 🔄 Future Updates

**Update Frontend:**
```bash
git add .
git commit -m "Update frontend"
git push
# Vercel auto-deploys!
```

**Update Backend:**
```bash
cd backend
git add .
git commit -m "Update backend"
git push hf main
# HF Space auto-rebuilds!
```

---

## 📊 All Features Live

✅ Text search with LLM explanations  
✅ Visual similarity search  
✅ Image upload search  
✅ Handwritten OCR search  
✅ Find similar items  

Enjoy your deployed app! 🎉
