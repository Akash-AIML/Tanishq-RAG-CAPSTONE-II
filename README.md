# 💍 Jewellery Multimodal Search - RAG Capstone II

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18.0.0-61DAFB?style=for-the-badge&logo=react&logoColor=black)](https://reactjs.org/)
[![Vite](https://img.shields.io/badge/Vite-4.0.0-646CFF?style=for-the-badge&logo=vite&logoColor=white)](https://vitejs.dev/)

An elegant, AI-powered jewellery search application leveraging **Multimodal RAG** (Retrieval-Augmented Generation). Search through vast collections using text, images, or even handwritten queries with state-of-the-art CLIP embeddings.

---

## 🌟 Core Features

- 🔍 **Natural Language Search**: Find jewellery using descriptive queries like *"gold necklace with emeralds"*.
- 📸 **Visual Similarity Search**: Upload an image or click "Find Similar" to discover related pieces.
- ✍️ **Handwritten OCR Search**: Extract and search using queries from handwritten notes or sketches.
- ✨ **Premium UI/UX**: Modern glassmorphism design with smooth animations and responsive layouts.
- 🧠 **AI Explanations**: LLM-powered insights into why certain results were matched (via Groq).

---

## 🚀 Quick Start

### 📋 Prerequisites

- **Node.js** 18+
- **Python** 3.10+
- **Conda** (recommended for ML dependencies)
- **Groq API Key** (Get it at [console.groq.com](https://console.groq.com/))

### 🛠️ Backend Setup

1. **Navigate to backend:**
   ```bash
   cd backend
   ```

2. **Environment Configuration:**
   Create a `.env` file:
   ```env
   GROQ_API_KEY=your_actual_key_here
   ```

3. **Install Dependencies:**
   If using Conda:
   ```bash
   conda create -n jewellery_rag python=3.10
   conda activate jewellery_rag
   pip install -r requirements.txt
   ```

4. **Start the Server:**
   ```bash
   uvicorn app:app --reload --host 0.0.0.0 --port 8000
   ```
   ✅ API available at: `http://localhost:8000`

### 💻 Frontend Setup

1. **Navigate to frontend:**
   ```bash
   cd frontend
   ```

2. **Install & Start:**
   ```bash
   npm install
   npm run dev
   ```
   ✅ App available at: `http://localhost:5173`

---

## 📂 Project Architecture
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

## 📡 API Reference

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/search/text` | `POST` | Semantic text search using CLIP |
| `/search/similar` | `POST` | Visual similarity search via Image ID |
| `/search/upload-image`| `POST` | Image-to-image search via file upload |
| `/search/ocr-query` | `POST` | Handwritten text extraction & search |
| `/image/{id}` | `GET` | Serve jewellery assets |

---

## 🎨 Design System

The application utilizes a **Soft Gradient Luxury** aesthetic:
- **Tokens**: Consistent HSL color palette for gold and dark themes.
- **Glassmorphism**: Backdrop filters for premium card feel.
- **Animations**: Framer Motion / CSS transitions for micro-interactions.
- **Responsiveness**: Mobile-first grid system.

---



---

## 📝 Troubleshooting

- **CORS Errors**: Ensure the backend `allow_origins` includes your frontend URL.
- **Missing Images**: Verify the `DATA_PATH` in `app.py` matches your local setup.
- **Model Load Delay**: The first search may take longer as CLIP model weights are loaded into memory.

---


