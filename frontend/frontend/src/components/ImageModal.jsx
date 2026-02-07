import React from 'react';
import './ImageModal.css';

const ImageModal = ({ item, onClose, onSimilarSearch }) => {
    if (!item) return null;

    const imageId = item.image_id || item.id;
    const imageUrl = `/api/image/${imageId}`;
    const explanation = item.explanation || (item.metadata && item.metadata.description) || "No description available.";

    // Parse score
    const visualScore = item.scores?.visual ?? item.scores?.final ?? 0;
    const displayScore = Math.max(70, Math.round(100 - (visualScore * 30)));

    return (
        <div className="modal-overlay" onClick={onClose}>
            <div className="modal-content" onClick={e => e.stopPropagation()}>
                <button className="close-button" onClick={onClose}>
                    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg>
                </button>
                
                <div className="modal-body">
                    <div className="modal-image-section">
                        <img 
                            src={imageUrl} 
                            alt={item.metadata?.product_name || 'Jewellery'} 
                            className="modal-image"
                        />
                    </div>
                    
                    <div className="modal-details-section">
                        <div className="breadcrumb">
                            Results &gt; {item.metadata?.product_type || item.metadata?.category || 'Jewellery'} &gt; {imageId}
                        </div>
                        
                        <h2 className="modal-title">{item.metadata?.product_name || `Item ${imageId}`}</h2>
                        <div className="modal-subtitle">
                           <span>Premium Craftsmanship</span> • <span>Authentic Tanishq</span>
                        </div>
                        
                        <div className="ai-recommendation-box">
                            <div className="ai-header">
                                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 11.5a8.38 8.38 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.38 8.38 0 0 1-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.38 8.38 0 0 1 3.8-.9h.5a8.48 8.48 0 0 1 8 8v.5z"></path></svg>
                                AI Explanation
                            </div>
                            <p className="ai-text">
                                {explanation}
                            </p>
                        </div>
                        
                        <div className="match-stats">
                            <div className="stat-box">
                                <span className="stat-number">{displayScore}%</span>
                                <span className="stat-label">Visual Match</span>
                            </div>
                            <div className="stat-box">
                                <span className="stat-number">High</span>
                                <span className="stat-label">Attributes</span>
                            </div>
                            <div className="stat-box">
                                <span className="stat-number">12%</span>
                                <span className="stat-label">Semantic</span>
                            </div>
                        </div>
                        
                        <div className="action-buttons">
                            <button className="book-tryon-btn">
                                <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><rect x="3" y="4" width="18" height="18" rx="2" ry="2"></rect><line x1="16" y1="2" x2="16" y2="6"></line><line x1="8" y1="2" x2="8" y2="6"></line><line x1="3" y1="10" x2="21" y2="10"></line></svg>
                                Book try-on
                            </button>
                            <button className="icon-action-btn" onClick={onSimilarSearch} title="Find Similar">
                                <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line></svg>
                            </button>
                            <button className="icon-action-btn">
                                <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z"></path></svg>
                            </button>
                        </div>
                        
                        <div className="recommendations-section">
                            <h4>You may also like</h4>
                            <div className="rec-list">
                                <div className="rec-item">
                                    <div className="rec-info">
                                        <h5>Similar Design 182</h5>
                                        <p>A stunning ring crafted in gold...</p>
                                    </div>
                                    <span className="rec-match">95%</span>
                                </div>
                                <div className="rec-item">
                                    <div className="rec-info">
                                        <h5>Similar Design 184</h5>
                                        <p>A stunning ring crafted in gold...</p>
                                    </div>
                                    <span className="rec-match">95%</span>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default ImageModal;
