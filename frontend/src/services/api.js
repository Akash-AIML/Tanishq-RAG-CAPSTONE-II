import axios from "axios";

/**
 * ============================================================
 * BACKEND BASE URL CONFIG
 * ============================================================
 * Priority:
 * 1. Vercel / Production → VITE_BACKEND_URL
 * 2. Local development → localhost
 */

const BASE_URL =
  import.meta.env.VITE_BACKEND_URL ||
  "http://localhost:8000";

/**
 * ============================================================
 * AXIOS INSTANCE
 * ============================================================
 */

const api = axios.create({
  baseURL: BASE_URL,
  timeout: 60000, // 60s for LLM / reranking latency
  headers: {
    "Content-Type": "application/json",
  },
});

/**
 * ============================================================
 * TEXT SEARCH
 * ============================================================
 */

export const searchByText = async (
  query,
  filters = [],
  topK = 5
) => {
  try {
    // Convert UI filter array → backend dict
    const filterDict = {};

    filters.forEach((f) => {
      let lowerF = f.toLowerCase();

      // Handle plurals
      if (
        [
          "rings",
          "necklaces",
          "earrings",
          "bracelets",
          "diamonds",
          "emeralds",
          "rubies",
          "sapphires",
          "pearls",
        ].includes(lowerF)
      ) {
        if (lowerF.endsWith("ies")) {
          lowerF = lowerF.replace("ies", "y");
        } else {
          lowerF = lowerF.slice(0, -1);
        }
      }

      // Map to attributes
      if (
        [
          "gold",
          "silver",
          "platinum",
          "rose gold",
          "white gold",
        ].includes(lowerF)
      ) {
        filterDict["metal"] = lowerF;
      } else if (
        [
          "necklace",
          "ring",
          "earring",
          "bracelet",
          "bangle",
          "pendant",
        ].includes(lowerF)
      ) {
        filterDict["category"] = lowerF;
      } else {
        filterDict["primary_stone"] = lowerF;
      }
    });

    const response = await api.post("/search/text", {
      query,
      filters: filterDict,
      top_k: topK,
    });

    return response.data;
  } catch (error) {
    console.error("Text search error:", error);
    throw new Error(
      error.response?.data?.detail ||
        "Failed to search. Please try again."
    );
  }
};

/**
 * ============================================================
 * SIMILAR IMAGE SEARCH
 * ============================================================
 */

export const searchSimilar = async (
  imageId,
  topK = 5
) => {
  try {
    const response = await api.post(
      "/search/similar",
      {
        image_id: imageId,
        top_k: topK,
      }
    );

    return response.data;
  } catch (error) {
    console.error("Similar search error:", error);
    throw new Error(
      error.response?.data?.detail ||
        "Failed to find similar items."
    );
  }
};

/**
 * ============================================================
 * IMAGE UPLOAD SEARCH
 * ============================================================
 */

export const searchByUploadedImage = async (
  imageFile,
  topK = 12
) => {
  try {
    const formData = new FormData();
    formData.append("file", imageFile);
    formData.append("top_k", topK);

    const response = await api.post(
      "/search/upload-image",
      formData,
      {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      }
    );

    return response.data;
  } catch (error) {
    console.error(
      "Image upload search error:",
      error
    );
    throw new Error(
      error.response?.data?.detail ||
        "Failed to search by image."
    );
  }
};

/**
 * ============================================================
 * OCR HANDWRITTEN SEARCH
 * ============================================================
 */

export const searchByHandwrittenQuery =
  async (imageFile, topK = 12) => {
    try {
      const formData = new FormData();
      formData.append("file", imageFile);
      formData.append("top_k", topK);

      const response = await api.post(
        "/search/ocr-query",
        formData,
        {
          headers: {
            "Content-Type":
              "multipart/form-data",
          },
        }
      );

      return response.data;
    } catch (error) {
      console.error("OCR search error:", error);
      throw new Error(
        error.response?.data?.detail ||
          "Failed to extract text from image."
      );
    }
  };

/**
 * ============================================================
 * IMAGE URL BUILDER (CRITICAL FIX)
 * ============================================================
 */

export const getImageUrl = (imageId) => {
  const base =
    import.meta.env.VITE_BACKEND_URL ||
    "http://localhost:8000";

  return `${base}/image/${imageId}`;
};

export default api;
