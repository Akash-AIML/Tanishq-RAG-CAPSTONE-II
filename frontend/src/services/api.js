import axios from 'axios';

// Use environment variable for backend URL (production) or local proxy (development)
const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || '/api';

// Create axios instance with base configuration
const api = axios.create({
    baseURL: BACKEND_URL,
    timeout: 60000, // 60 seconds for LLM-powered responses
    headers: {
        'Content-Type': 'application/json',
    },
});

/**
 * Search for jewellery items using text query
 * @param {string} query - Search query text
 * @param {number} topK - Number of results to return (default: 5)
 * @returns {Promise} - Search results with images and scores
 */
export const searchByText = async (query, topK = 5) => {
    try {
        const response = await api.post('/search/text', {
            query,
            top_k: topK,
        });
        return response.data;
    } catch (error) {
        console.error('Text search error:', error);
        throw new Error(error.response?.data?.detail || 'Failed to search. Please try again.');
    }
};

/**
 * Find similar jewellery items based on an image ID
 * @param {string} imageId - ID of the reference image
 * @param {number} topK - Number of results to return (default: 5)
 * @returns {Promise} - Similar items with scores
 */
export const searchSimilar = async (imageId, topK = 5) => {
    try {
        const response = await api.post('/search/similar', {
            image_id: imageId,
            top_k: topK,
        });
        return response.data;
    } catch (error) {
        console.error('Similar search error:', error);
        throw new Error(error.response?.data?.detail || 'Failed to find similar items.');
    }
};

/**
 * Search by uploading an image file
 * @param {File} imageFile - Image file to upload
 * @param {number} topK - Number of results to return (default: 12)
 * @returns {Promise} - Search results
 */
export const searchByUploadedImage = async (imageFile, topK = 12) => {
    try {
        const formData = new FormData();
        formData.append('file', imageFile);
        formData.append('top_k', topK);

        const response = await api.post('/search/upload-image', formData, {
            headers: { 'Content-Type': 'multipart/form-data' }
        });
        return response.data;
    } catch (error) {
        console.error('Image upload search error:', error);
        throw new Error(error.response?.data?.detail || 'Failed to search by image.');
    }
};

/**
 * Extract text from handwritten image and search
 * @param {File} imageFile - Image file with handwritten text
 * @param {number} topK - Number of results to return (default: 12)
 * @returns {Promise} - Search results with extracted text
 */
export const searchByHandwrittenQuery = async (imageFile, topK = 12) => {
    try {
        const formData = new FormData();
        formData.append('file', imageFile);
        formData.append('top_k', topK);

        const response = await api.post('/search/ocr-query', formData, {
            headers: { 'Content-Type': 'multipart/form-data' }
        });
        return response.data;
    } catch (error) {
        console.error('OCR search error:', error);
        throw new Error(error.response?.data?.detail || 'Failed to extract text from image.');
    }
};

/**
 * Get the URL for an image by its ID
 * @param {string} imageId - Image ID
 * @returns {string} - Full image URL
 */
export const getImageUrl = (imageId) => {
    return `${BACKEND_URL}/image/${imageId}`;
};

export default api;
