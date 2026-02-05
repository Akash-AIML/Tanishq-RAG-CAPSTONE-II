import React, { useEffect } from 'react';
import { getImageUrl } from '../services/api';
import './ImageModal.css';

const ImageModal = ({ item, onClose, onFindSimilar }) => {
    useEffect(() => {
        // Prevent body scroll when modal is open
        document.body.style.overflow = 'hidden';

        // Close on Escape key
        const handleEscape = (e) => {
            if (e.key === 'Escape') {
                onClose();
            }
        };

        document.addEventListener('keydown', handleEscape);

        return () => {
            document.body.style.overflow = 'unset';
            document.removeEventListener('keydown', handleEscape);
        };
    }, [onClose]);

    if (!item) return null;

    return (
        <div className="modal-overlay" onClick={onClose}>
            <div className="modal-content glass-card" onClick={(e) => e.stopPropagation()}>
                <button className="modal-close" onClick={onClose} aria-label="Close modal">
                    <svg
                        xmlns="http://www.w3.org/2000/svg"
                        fill="none"
                        viewBox="0 0 24 24"
                        stroke="currentColor"
                        width="24"
                        height="24"
                    >
                        <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            strokeWidth={2}
                            d="M6 18L18 6M6 6l12 12"
                        />
                    </svg>
                </button>

                <div className="modal-body">
                    <div className="modal-image-section">
                        <img
                            src={getImageUrl(item.image_id)}
                            alt={item.explanation || 'Jewellery item'}
                            className="modal-image"
                        />
                    </div>

                    <div className="modal-details">
                        <div className="modal-header">
                            <h3>Item Details</h3>
                            <span className="badge badge-teal">{item.image_id}</span>
                        </div>

                        <div className="detail-section">
                            <h4>Description</h4>
                            <p className="detail-text">{item.explanation}</p>
                        </div>

                        {item.scores && (
                            <div className="detail-section">
                                <h4>Similarity Scores</h4>
                                <div className="scores-grid">
                                    <div className="score-card">
                                        <span className="score-card-label">Visual Score</span>
                                        <span className="score-card-value badge badge-purple">
                                            {item.scores.visual?.toFixed(4) || 'N/A'}
                                        </span>
                                        <span className="score-card-description">
                                            Based on visual similarity
                                        </span>
                                    </div>

                                    <div className="score-card">
                                        <span className="score-card-label">Metadata Boost</span>
                                        <span className="score-card-value badge badge-gold">
                                            {item.scores.metadata?.toFixed(4) || 'N/A'}
                                        </span>
                                        <span className="score-card-description">
                                            Attribute matching bonus
                                        </span>
                                    </div>

                                    <div className="score-card">
                                        <span className="score-card-label">Final Score</span>
                                        <span className="score-card-value badge badge-teal">
                                            {item.scores.final?.toFixed(4) || 'N/A'}
                                        </span>
                                        <span className="score-card-description">
                                            Combined ranking score
                                        </span>
                                    </div>
                                </div>
                            </div>
                        )}

                        <button
                            className="btn btn-primary modal-action-btn"
                            onClick={() => {
                                onFindSimilar(item.image_id);
                                onClose();
                            }}
                        >
                            <svg
                                xmlns="http://www.w3.org/2000/svg"
                                fill="none"
                                viewBox="0 0 24 24"
                                stroke="currentColor"
                                width="20"
                                height="20"
                            >
                                <path
                                    strokeLinecap="round"
                                    strokeLinejoin="round"
                                    strokeWidth={2}
                                    d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
                                />
                            </svg>
                            Find Similar Items
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default ImageModal;
