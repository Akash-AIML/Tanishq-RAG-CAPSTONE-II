# Jewellery Multimodal Search - Full Stack Application

## 🎯 Overview

A beautiful, AI-powered jewellery search application featuring:
- **Frontend**: Modern React UI with glassmorphism design, smooth animations, and responsive layout
- **Backend**: FastAPI server with CLIP embeddings and ChromaDB vector database
- **Features**: Text-based search, visual similarity search, detailed item views

---

## 🚀 Quick Start

### Prerequisites

- **Node.js** 18+ (for frontend)
- **Python** 3.8+ (for backend)
- **pip** (Python package manager)

### Backend Setup

1. **Navigate to backend directory:**
   ```bash
   cd /home/akash/Jewellary_RAG/backend
   ```

2. **Set up Groq API Key** (for LLM-powered explanations):
   ```bash
   export GROQ_API_KEY="your_groq_api_key_here"
   ```
   
   Or create a `.env` file:
   ```bash
   echo "GROQ_API_KEY=your_groq_api_key_here" > .env
   ```
   
   Get your free API key from: https://console.groq.com/keys

3. **Activate the ml conda environment** (already has all dependencies):
   ```bash
   conda activate ml
   ```

4. **Install OpenAI library** (for Groq client):
   ```bash
   pip install openai
   ```

5. **Start the backend server:**
   ```bash
   python app.py
   ```
   
   Or with uvicorn directly:
   ```bash
   uvicorn app:app --reload --host 0.0.0.0 --port 8000
   ```

   ✅ Backend will run on `http://localhost:8000`

   > **Note**: The `ml` environment already contains all required packages (torch, CLIP, chromadb, fastapi, etc.) from your Jupyter notebook development.

### Frontend Setup

1. **Navigate to frontend directory:**
   ```bash
   cd /home/akash/Jewellary_RAG/frontend
   ```

2. **Install dependencies** (already done if you followed setup):
   ```bash
   npm install
   ```

3. **Start the development server:**
   ```bash
   npm run dev
   ```

   ✅ Frontend will run on `http://localhost:5173`

4. **Open in browser:**
   ```
   http://localhost:5173
   ```

---

## 📁 Project Structure

```
Jewellary_RAG/
├── backend/
│   └── app.py                 # FastAPI server with CLIP + ChromaDB
├── frontend/
│   ├── src/
│   │   ├── components/        # React components
│   │   │   ├── SearchBar.jsx
│   │   │   ├── ResultsGrid.jsx
│   │   │   ├── ImageModal.jsx
│   │   │   └── LoadingSpinner.jsx
│   │   ├── services/
│   │   │   └── api.js         # API integration layer
│   │   ├── App.jsx            # Main application
│   │   ├── main.jsx           # React entry point
│   │   └── index.css          # Design system
│   ├── index.html
│   ├── vite.config.js         # Vite configuration with API proxy
│   └── package.json
├── chroma_primary/            # ChromaDB vector database
├── data/                      # Jewellery images and metadata
└── embeddings/                # Pre-computed embeddings
```

---

## 🎨 Features

### Text Search
- Search using natural language queries
- Example: "gold necklace with pearls", "diamond ring", "silver earrings"
- AI-powered semantic understanding

### Visual Similarity Search
- Click "Find Similar" on any item
- Discovers visually similar jewellery pieces
- Combines visual embeddings with metadata matching

### Detailed Item View
- Click any image to see full details
- View all similarity scores (visual, metadata, final)
- Quick access to find similar items

### Premium Design
- Dark theme with glassmorphism effects
- Smooth animations and micro-interactions
- Responsive layout (desktop, tablet, mobile)
- Gradient accents and modern typography

---

## 🔧 API Endpoints

### Backend API (Port 8000)

**POST** `/search/text`
```json
{
  "query": "gold necklace",
  "top_k": 5
}
```

**POST** `/search/similar`
```json
{
  "image_id": "image_001.jpg",
  "top_k": 5
}
```

**GET** `/image/{image_id}`
- Returns the jewellery image file

---

## 🎯 Usage Guide

1. **Start Both Servers**
   - Backend on port 8000
   - Frontend on port 5173

2. **Search for Jewellery**
   - Enter a search query (e.g., "pearl necklace")
   - Click "Search" or press Enter
   - Results appear in a responsive grid

3. **Explore Similar Items**
   - Click "Find Similar" on any result card
   - View visually similar jewellery pieces

4. **View Details**
   - Click on any image to open detailed view
   - See full scores and descriptions
   - Find similar items from the modal

---

## 🛠️ Development

### Frontend Development
```bash
cd frontend
npm run dev      # Start dev server with hot reload
npm run build    # Build for production
npm run preview  # Preview production build
```

### Backend Development
```bash
cd backend
python app.py    # Start with auto-reload
```

---

## 🎨 Design System

The frontend uses a comprehensive design system with:

- **CSS Variables** for consistent theming
- **Glassmorphism** effects with backdrop blur
- **Gradient Accents** (purple, gold, teal)
- **Smooth Animations** for all interactions
- **Responsive Grid** system
- **Custom Scrollbars** with gradient styling

---

## 📝 Notes

- **CORS**: Vite proxy handles CORS during development
- **Node Version**: Uses Vite 4.x for Node 18 compatibility
- **Image Paths**: Backend serves images from `/data/tanishq/images/`
- **ChromaDB**: Pre-populated with jewellery embeddings

---

## 🐛 Troubleshooting

**Backend won't start:**
- Ensure all Python dependencies are installed
- Check ChromaDB path exists: `/home/akash/Jewellary_RAG/backend/chroma`
- Verify CLIP model downloads successfully

**Frontend can't connect to backend:**
- Ensure backend is running on port 8000
- Check Vite proxy configuration in `vite.config.js`
- Verify no firewall blocking localhost connections

**Images not loading:**
- Check image directory path in `backend/app.py`
- Verify images exist in `/data/tanishq/images/`

---

## 🚀 Production Deployment

### Backend
```bash
pip install gunicorn
gunicorn app:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Frontend
```bash
npm run build
# Serve the dist/ folder with nginx or any static server
```

---

**Built with ❤️ using React, Vite, FastAPI, CLIP, and ChromaDB**
