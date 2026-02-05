import React, { useState } from 'react';
import SearchBar from './components/SearchBar';
import ResultsGrid from './components/ResultsGrid';
import ImageModal from './components/ImageModal';
import LoadingSpinner from './components/LoadingSpinner';
import ImageUpload from './components/ImageUpload';
import { searchByText, searchSimilar, searchByUploadedImage, searchByHandwrittenQuery } from './services/api';
import './App.css';

function App() {
    const [results, setResults] = useState([]);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState(null);
    const [selectedItem, setSelectedItem] = useState(null);
    const [currentQuery, setCurrentQuery] = useState('');
    const [extractedText, setExtractedText] = useState(null);

    const handleTextSearch = async (query) => {
        setIsLoading(true);
        setError(null);
        setCurrentQuery(query);

        try {
            const data = await searchByText(query, 12);
            setResults(data.results || []);

            // Scroll to results
            setTimeout(() => {
                document.getElementById('results-section')?.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }, 100);
        } catch (err) {
            setError(err.message);
            setResults([]);
        } finally {
            setIsLoading(false);
        }
    };

    const handleSimilarSearch = async (imageId) => {
        setIsLoading(true);
        setError(null);
        setCurrentQuery(`Similar to: ${imageId}`);

        try {
            const data = await searchSimilar(imageId, 12);
            setResults(data.results || []);

            // Scroll to results
            setTimeout(() => {
                document.getElementById('results-section')?.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }, 100);
        } catch (err) {
            setError(err.message);
            setResults([]);
        } finally {
            setIsLoading(false);
        }
    };

    const handleImageClick = (item) => {
        setSelectedItem(item);
    };

    const handleCloseModal = () => {
        setSelectedItem(null);
    };

    const handleImageUploadSearch = async (imageFile, mode) => {
        setIsLoading(true);
        setError(null);
        setExtractedText(null);

        try {
            let data;
            if (mode === 'visual') {
                setCurrentQuery(`Visual search: ${imageFile.name}`);
                data = await searchByUploadedImage(imageFile, 12);
            } else {
                setCurrentQuery(`Handwritten query from: ${imageFile.name}`);
                data = await searchByHandwrittenQuery(imageFile, 12);
                if (data.extracted_text) {
                    setExtractedText(data.extracted_text);
                }
            }

            setResults(data.results || []);

            // Scroll to results
            setTimeout(() => {
                document.getElementById('results-section')?.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }, 100);
        } catch (err) {
            setError(err.message);
            setResults([]);
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="app">
            {isLoading && <LoadingSpinner />}

            <header className="app-header">
                <div className="container">
                    <div className="header-content">
                        <div className="logo-section">
                            <div className="logo-icon">
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
                                        d="M12 8v13m0-13V6a2 2 0 112 2h-2zm0 0V5.5A2.5 2.5 0 109.5 8H12zm-7 4h14M5 12a2 2 0 110-4h14a2 2 0 110 4M5 12v7a2 2 0 002 2h10a2 2 0 002-2v-7"
                                    />
                                </svg>
                            </div>
                            <div>
                                <h1>Jewellery Search</h1>
                                <p className="tagline">AI-Powered Visual Discovery</p>
                            </div>
                        </div>

                        <div className="header-stats">
                            <div className="stat-item">
                                <span className="stat-value">{results.length}</span>
                                <span className="stat-label">Results</span>
                            </div>
                        </div>
                    </div>
                </div>
            </header>

            <main className="app-main">
                <div className="container">
                    <section className="search-section">
                        <SearchBar onSearch={handleTextSearch} isLoading={isLoading} />

                        <div style={{ margin: '2rem 0', textAlign: 'center', color: 'var(--color-text-muted)' }}>
                            <span>or</span>
                        </div>

                        <ImageUpload onSearch={handleImageUploadSearch} loading={isLoading} />

                        {extractedText && (
                            <div className="glass-card" style={{ marginTop: '1rem', padding: '1rem' }}>
                                <strong style={{ color: 'var(--color-accent-purple)' }}>Extracted Text:</strong>
                                <p style={{ margin: '0.5rem 0 0 0', color: 'var(--color-text-secondary)' }}>"{extractedText}"</p>
                            </div>
                        )}

                        {error && (
                            <div className="error-message glass-card">
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
                                        d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                                    />
                                </svg>
                                <span>{error}</span>
                            </div>
                        )}
                    </section>

                    <section id="results-section">
                        <ResultsGrid
                            results={results}
                            onImageClick={handleImageClick}
                            onFindSimilar={handleSimilarSearch}
                            searchQuery={currentQuery}
                        />
                    </section>
                </div>
            </main>

            <footer className="app-footer">
                <div className="container">
                    <p>Powered by CLIP + ChromaDB • Multimodal AI Search</p>
                </div>
            </footer>

            {selectedItem && (
                <ImageModal
                    item={selectedItem}
                    onClose={handleCloseModal}
                    onFindSimilar={handleSimilarSearch}
                />
            )}
        </div>
    );
}

export default App;
