import React from 'react';
import { getImageUrl } from '../services/api';
import './ResultsGrid.css';

const ResultsGrid = ({ results, onImageClick, onFindSimilar, searchQuery }) => {
    if (!results || results.length === 0) {
        return (
            <div className="empty-state">
                <div className="empty-icon">
                    <svg
                        xmlns="http://www.w3.org/2000/svg"
                        fill="none"
                        viewBox="0 0 24 24"
                        stroke="currentColor"
                        width="80"
                        height="80"
                    >
                        <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            strokeWidth={1.5}
                            d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
                        />
                    </svg>
                </div>
                <h3>No results found</h3>
                <p className="empty-description">
                    {searchQuery
                        ? `No jewellery items match "${searchQuery}". Try a different search term.`
                        : 'Start searching to discover beautiful jewellery pieces.'
                    }
                </p>
            </div>
        );
    }

    return (
        <div className="results-section animate-fade-in">
            <div className="results-header">
                <h2>Search Results</h2>
                <span className="results-count badge badge-purple">
                    {results.length} {results.length === 1 ? 'item' : 'items'}
                </span>
            </div>

            <div className="results-grid grid grid-cols-4">
                {results.map((result, index) => (
                    <div
                        key={result.image_id || index}
                        className="result-card glass-card"
                        style={{ animationDelay: `${index * 0.05}s` }}
                    >
                        <div
                            className="result-image-wrapper"
                            onClick={() => onImageClick(result)}
                        >
                            <img
                                src={getImageUrl(result.image_id)}
                                alt={result.explanation || 'Jewellery item'}
                                className="result-image"
                                loading="lazy"
                            />
                            <div className="image-overlay">
                                <svg
                                    xmlns="http://www.w3.org/2000/svg"
                                    fill="none"
                                    viewBox="0 0 24 24"
                                    stroke="currentColor"
                                    width="40"
                                    height="40"
                                >
                                    <path
                                        strokeLinecap="round"
                                        strokeLinejoin="round"
                                        strokeWidth={2}
                                        d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"
                                    />
                                    <path
                                        strokeLinecap="round"
                                        strokeLinejoin="round"
                                        strokeWidth={2}
                                        d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z"
                                    />
                                </svg>
                            </div>
                        </div>

                        <div className="result-content">
                            <p className="result-explanation">{result.explanation}</p>

                            {result.scores && (
                                <div className="result-scores">
                                    <div className="score-item">
                                        <span className="score-label">Visual</span>
                                        <span className="badge badge-purple">
                                            {result.scores.visual?.toFixed(3) || 'N/A'}
                                        </span>
                                    </div>
                                    <div className="score-item">
                                        <span className="score-label">Final</span>
                                        <span className="badge badge-gold">
                                            {result.scores.final?.toFixed(3) || 'N/A'}
                                        </span>
                                    </div>
                                </div>
                            )}

                            <button
                                className="btn btn-secondary find-similar-btn"
                                onClick={() => onFindSimilar(result.image_id)}
                            >
                                <svg
                                    xmlns="http://www.w3.org/2000/svg"
                                    fill="none"
                                    viewBox="0 0 24 24"
                                    stroke="currentColor"
                                    width="18"
                                    height="18"
                                >
                                    <path
                                        strokeLinecap="round"
                                        strokeLinejoin="round"
                                        strokeWidth={2}
                                        d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
                                    />
                                </svg>
                                Find Similar
                            </button>
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
};

export default ResultsGrid;
