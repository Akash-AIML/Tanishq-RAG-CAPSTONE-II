import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import './SearchBar.css';

const SearchBar = ({ onSearch, onImageSearch, isLoading }) => {
    const [activeTab, setActiveTab] = useState('type');
    const [query, setQuery] = useState('');

    const onDrop = useCallback((acceptedFiles) => {
        if (acceptedFiles?.length > 0) {
            onImageSearch(acceptedFiles[0]);
        }
    }, [onImageSearch]);

    const { getRootProps, getInputProps, isDragActive } = useDropzone({ 
        onDrop,
        accept: {'image/*': []},
        multiple: false
    });

    const handleSubmit = (e) => {
        e.preventDefault();
        if (query.trim() && !isLoading) {
            onSearch(query.trim());
        }
    };

    return (
        <div className="search-component">
            <div className="search-tabs">
                <button 
                    className={`tab-btn ${activeTab === 'type' ? 'active' : ''}`}
                    onClick={() => setActiveTab('type')}
                >
                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line></svg>
                    Type
                </button>
                <button 
                    className={`tab-btn ${activeTab === 'upload' ? 'active' : ''}`}
                    onClick={() => setActiveTab('upload')}
                >
                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="17 8 12 3 7 8"></polyline><line x1="12" y1="3" x2="12" y2="15"></line></svg>
                    Upload photo
                </button>
                <button 
                    className={`tab-btn ${activeTab === 'ocr' ? 'active' : ''}`}
                    onClick={() => setActiveTab('ocr')}
                >
                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 12V7H5a2 2 0 0 1 0-4h14a2 2 0 0 1 2 2v5z"></path><path d="M3 7v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-5"></path></svg>
                    OCR
                </button>
            </div>

            <div className="search-content">
                {activeTab === 'type' && (
                    <form onSubmit={handleSubmit} className="type-search-form">
                        <input
                            type="text"
                            className="text-search-input"
                            placeholder="Try 'Gold Wedding Ring' or 'Diamond Band'"
                            value={query}
                            onChange={(e) => setQuery(e.target.value)}
                            disabled={isLoading}
                        />
                        <button type="submit" className="text-search-submit" disabled={isLoading}>
                            <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon></svg>
                        </button>
                    </form>
                )}

                {(activeTab === 'upload' || activeTab === 'ocr') && (
                    <div {...getRootProps()} className={`dropzone ${isDragActive ? 'active' : ''}`}>
                        <input {...getInputProps()} />
                        <div className="dropzone-content">
                            {isLoading ? (
                                <span className="upload-text">Processing...</span>
                            ) : (
                                <>
                                    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="17 8 12 3 7 8"></polyline><line x1="12" y1="3" x2="12" y2="15"></line></svg>
                                    <span className="upload-text">
                                        {isDragActive ? "Drop image here" : activeTab === 'ocr' ? "Upload handwritten note" : "Upload design image"}
                                    </span>
                                </>
                            )}
                        </div>
                    </div>
                )}

                <button className="main-search-btn" onClick={activeTab === 'type' ? handleSubmit : undefined}>
                    {isLoading ? 'Searching...' : 'Search'}
                </button>
            </div>
        </div>
    );
};

export default SearchBar;
