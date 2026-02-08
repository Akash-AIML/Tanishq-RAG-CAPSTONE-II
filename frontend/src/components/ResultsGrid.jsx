import React from 'react';
import './ResultsGrid.css';

const ResultsGrid = ({ results, onImageClick }) => {
    if (!results || results.length === 0) {
        return (
            <div className="no-results fade-in-up">
                <div className="no-results-icon">💎</div>
                <h3>No items found</h3>
                <p>Try adjusting your filters or search for something else.</p>
            </div>
        );
    }

    return (
        <div className="results-grid">
            {results.map((item, index) => {
                const imageId = item.image_id || item.id;
                
                // --- IMPROVED SCORE DISPLAY LOGIC ---
                // For distance metrics (smaller is better):
                // 0.0 is exact match, ~1.5 is far.
                // Linear map: 0->99%, 0.8->85%, 1.2->70%, 1.5+ -> 60%
                
                // Get the raw metric
                const rawVal = item.scores?.visual ?? item.scores?.final ?? 1.2;
                
                // Safe calculation for percentage
                // 1. Invert distance: score = 1 / (1 + distance)  => 1/1=100%, 1/2=50%
                // But typically L2 distance needs a sharper curve for UI.
                // Let's rely on a customized curve.
                
                let percentMatch;
                if (rawVal < 0.1) percentMatch = 99; // Practically identical
                else if (rawVal < 0.5) percentMatch = Math.round(98 - (rawVal * 20)); // 0.1->96, 0.4->90
                else if (rawVal < 1.0) percentMatch = Math.round(90 - ((rawVal - 0.5) * 30)); // 0.5->90, 1.0->75
                else percentMatch = Math.max(60, Math.round(75 - ((rawVal - 1.0) * 15))); // 1.2->72
                
                const displayScore = percentMatch;

                const imageUrl = `/api/image/${imageId}`;

                return (
                    <div 
                        key={imageId} 
                        className="result-card fade-in-up" 
                        onClick={() => onImageClick(item)}
                        style={{ animationDelay: `${index * 50}ms` }}
                    >
                        <div className="card-image-container">
                            <span className="match-badge">
                                <span className="match-icon">✨</span> {displayScore}% Match
                            </span>
                            <img 
                                src={imageUrl} 
                                alt={item.metadata?.product_name || 'Jewellery Item'} 
                                className="card-image"
                                loading="lazy"
                                onError={(e) => {
                                    e.target.onerror = null; 
                                    e.target.src = 'https://via.placeholder.com/300x300?text=Image+Not+Found';
                                }}
                            />
                            <div className="card-overlay">
                                <button className="find-similar-btn" onClick={(e) => {
                                    e.stopPropagation();
                                    onImageClick(item); // Or handle specific "find similar" action if different
                                }}>
                                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line></svg>
                                    Find Similar
                                </button>
                            </div>
                        </div>
                        <div className="card-content">
                            <div className="card-header">
                                <h3 className="card-title">
                                    {[
                                        item.metadata?.metal, 
                                        item.metadata?.primary_stone !== 'unknown' ? item.metadata?.primary_stone : null, 
                                        item.metadata?.category
                                    ].filter(Boolean).map(s => s.charAt(0).toUpperCase() + s.slice(1)).join(' ') || `Item ${imageId}`}
                                </h3>
                            </div>
                            
                            <div className="card-tags">
                                {[
                                    item.metadata?.metal, 
                                    item.metadata?.category,
                                    item.metadata?.form !== 'unknown' ? item.metadata?.form : null
                                ].filter(Boolean).map((tag, i) => (
                                    <span key={i} className="card-tag">{tag}</span>
                                ))}
                            </div>

                            <p className="card-desc">
                                {item.metadata?.metadata_text || item.explanation || "Elegant craftsmanship from Tanishq."}
                            </p>
                        </div>
                    </div>
                );
            })}
        </div>
    );
};

export default ResultsGrid;
