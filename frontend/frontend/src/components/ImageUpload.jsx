import React, { useState, useRef } from 'react';
import './ImageUpload.css';

const ImageUpload = ({ onSearch, loading }) => {
    const [selectedFile, setSelectedFile] = useState(null);
    const [previewUrl, setPreviewUrl] = useState(null);
    const [isDragging, setIsDragging] = useState(false);
    const [searchMode, setSearchMode] = useState('visual'); // 'visual' or 'ocr'
    const fileInputRef = useRef(null);

    const handleFileSelect = (file) => {
        if (!file) return;

        // Validate file type
        if (!file.type.startsWith('image/')) {
            alert('Please select an image file');
            return;
        }

        // Validate file size (max 10MB)
        if (file.size > 10 * 1024 * 1024) {
            alert('File size must be less than 10MB');
            return;
        }

        setSelectedFile(file);

        // Create preview URL
        const reader = new FileReader();
        reader.onloadend = () => {
            setPreviewUrl(reader.result);
        };
        reader.readAsDataURL(file);
    };

    const handleDrop = (e) => {
        e.preventDefault();
        setIsDragging(false);

        const file = e.dataTransfer.files[0];
        handleFileSelect(file);
    };

    const handleDragOver = (e) => {
        e.preventDefault();
        setIsDragging(true);
    };

    const handleDragLeave = () => {
        setIsDragging(false);
    };

    const handleFileInputChange = (e) => {
        const file = e.target.files[0];
        handleFileSelect(file);
    };

    const handleSearch = () => {
        if (!selectedFile) {
            alert('Please select an image first');
            return;
        }

        onSearch(selectedFile, searchMode);
    };

    const handleClear = () => {
        setSelectedFile(null);
        setPreviewUrl(null);
        if (fileInputRef.current) {
            fileInputRef.current.value = '';
        }
    };

    return (
        <div className="image-upload-container">
            {/* Mode Toggle */}
            <div className="mode-toggle">
                <button
                    className={`mode-btn ${searchMode === 'visual' ? 'active' : ''}`}
                    onClick={() => setSearchMode('visual')}
                    disabled={loading}
                >
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                        <rect x="3" y="3" width="18" height="18" rx="2" ry="2" />
                        <circle cx="8.5" cy="8.5" r="1.5" />
                        <polyline points="21 15 16 10 5 21" />
                    </svg>
                    Visual Search
                </button>
                <button
                    className={`mode-btn ${searchMode === 'ocr' ? 'active' : ''}`}
                    onClick={() => setSearchMode('ocr')}
                    disabled={loading}
                >
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                        <polyline points="14 2 14 8 20 8" />
                        <line x1="16" y1="13" x2="8" y2="13" />
                        <line x1="16" y1="17" x2="8" y2="17" />
                        <polyline points="10 9 9 9 8 9" />
                    </svg>
                    Handwritten Query
                </button>
            </div>

            {/* Upload Zone */}
            <div
                className={`upload-zone ${isDragging ? 'dragging' : ''} ${previewUrl ? 'has-image' : ''}`}
                onDrop={handleDrop}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onClick={() => !previewUrl && fileInputRef.current?.click()}
            >
                {!previewUrl ? (
                    <div className="upload-placeholder">
                        <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                            <polyline points="17 8 12 3 7 8" />
                            <line x1="12" y1="3" x2="12" y2="15" />
                        </svg>
                        <h3>
                            {searchMode === 'visual'
                                ? 'Upload an image to find similar items'
                                : 'Upload an image with handwritten text'
                            }
                        </h3>
                        <p>Drag & drop or click to browse</p>
                        <span className="file-types">PNG, JPG, WEBP (max 10MB)</span>
                    </div>
                ) : (
                    <div className="image-preview">
                        <img src={previewUrl} alt="Preview" />
                        <div className="preview-overlay">
                            <button className="clear-btn" onClick={(e) => {
                                e.stopPropagation();
                                handleClear();
                            }}>
                                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                                    <line x1="18" y1="6" x2="6" y2="18" />
                                    <line x1="6" y1="6" x2="18" y2="18" />
                                </svg>
                            </button>
                        </div>
                    </div>
                )}
            </div>

            <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                onChange={handleFileInputChange}
                style={{ display: 'none' }}
            />

            {/* Search Button */}
            {selectedFile && (
                <button
                    className="btn btn-primary search-btn"
                    onClick={handleSearch}
                    disabled={loading}
                >
                    {loading ? (
                        <>
                            <span className="spinner"></span>
                            {searchMode === 'visual' ? 'Searching...' : 'Extracting & Searching...'}
                        </>
                    ) : (
                        <>
                            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                                <circle cx="11" cy="11" r="8" />
                                <path d="m21 21-4.35-4.35" />
                            </svg>
                            {searchMode === 'visual' ? 'Find Similar Items' : 'Extract Text & Search'}
                        </>
                    )}
                </button>
            )}
        </div>
    );
};

export default ImageUpload;
