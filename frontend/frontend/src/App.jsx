import React, { useState, useMemo, useEffect } from 'react';
import '@fontsource/playfair-display';
import '@fontsource/playfair-display/700.css';
import '@fontsource/lato';
import SearchBar from './components/SearchBar';
import FilterBar from './components/FilterBar';
import ResultsGrid from './components/ResultsGrid';
import ImageModal from './components/ImageModal';
import LoadingSpinner from './components/LoadingSpinner';
import { searchByText, searchSimilar, searchByUploadedImage } from './services/api';
import './App.css';
import bgImage from './assets/bg-luxury.jpg';

const ITEMS_PER_PAGE = 12;

function App() {
    const [results, setResults] = useState([]);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState(null);
    const [selectedItem, setSelectedItem] = useState(null);
    const [currentQuery, setCurrentQuery] = useState('');
    const [hasSearched, setHasSearched] = useState(false);
    const [activeFilters, setActiveFilters] = useState([]);
    const [currentPage, setCurrentPage] = useState(1);
    const [history, setHistory] = useState([]);

    // Trigger search when query OR filters change
    const performSearch = async (queryText, filters) => {
        setIsLoading(true);
        setError(null);
        setHasSearched(true);
        setCurrentPage(1);
        
        try {
            const data = await searchByText(queryText, filters, 30);
            setResults(data.results || []);
        } catch (err) {
            setError(err.message);
            setResults([]);
        } finally {
            setIsLoading(false);
        }
    };

    const handleTextSearch = (query) => {
        setHistory([]); // Clear history on new text search
        setCurrentQuery(query);
        performSearch(query, activeFilters);
    };

    const handleReset = () => {
        setHasSearched(false);
        setResults([]);
        setCurrentQuery('');
        setActiveFilters([]);
        setHistory([]);
    };

    const handleFilterToggle = (filter) => {
        if (filter === 'All') {
             setActiveFilters([]);
             performSearch(currentQuery, []);
             return;
        }

        const newFilters = activeFilters.includes(filter)
            ? activeFilters.filter(f => f !== filter)
            : [...activeFilters, filter];
        
        setActiveFilters(newFilters);
        performSearch(currentQuery, newFilters);
    };

    const handleSimilarSearch = async (imageId) => {
        // Save current state to history
        setHistory(prev => [...prev, {
            results,
            currentQuery,
            activeFilters,
            currentPage,
            hasSearched
        }]);

        setIsLoading(true);
        setError(null);
        setHasSearched(true);
        setCurrentPage(1);
        setCurrentQuery(`Similar to: ${imageId}`);

        try {
            const data = await searchSimilar(imageId, 30);
            setResults(data.results || []);
            window.scrollTo({ top: 0, behavior: 'smooth' });
        } catch (err) {
            setError(err.message);
            setResults([]);
        } finally {
            setIsLoading(false);
        }
    };

    const handleBack = () => {
        if (history.length === 0) return;
        
        const lastState = history[history.length - 1];
        setResults(lastState.results);
        setCurrentQuery(lastState.currentQuery);
        setActiveFilters(lastState.activeFilters);
        setCurrentPage(lastState.currentPage);
        setHasSearched(lastState.hasSearched);
        
        // Remove the state we just restored
        setHistory(prev => prev.slice(0, -1));
        window.scrollTo({ top: 0, behavior: 'smooth' });
    };

    const handleImageUploadSearch = async (imageFile) => {
        setHistory([]); // New upload clears history
        setIsLoading(true);
        setError(null);
        setHasSearched(true);
        setCurrentPage(1);
        setCurrentQuery(`Visual search: ${imageFile.name}`);

        try {
            const data = await searchByUploadedImage(imageFile, 30);
            setResults(data.results || []);
        } catch (err) {
            setError(err.message);
            setResults([]);
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className={`app ${hasSearched ? 'results-mode' : 'hero-mode'}`} style={{
            backgroundImage: `linear-gradient(rgba(0, 0, 0, 0.6), rgba(0, 0, 0, 0.8)), url(${bgImage})`
        }}>
            <header className="app-header">
                <div className="header-content container">
                    <div className="left-header-section" style={{display: 'flex', alignItems: 'center', gap: '1rem'}}>
                        {history.length > 0 && (
                            <button onClick={handleBack} className="back-btn header-back-btn" title="Go back to previous results">
                                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="19" y1="12" x2="5" y2="12"></line><polyline points="12 19 5 12 12 5"></polyline></svg>
                                <span>Back</span>
                            </button>
                        )}
                        <div className="logo-section" onClick={handleReset} style={{cursor: 'pointer'}}>
                            <h1 className="brand-logo">Tanishq</h1>
                        </div>
                    </div>
                    <div className="header-actions">
                        <button className="icon-btn search-trigger" onClick={handleReset}>
                            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line></svg>
                        </button>
                        <button className="icon-btn bag-trigger">
                            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M6 2L3 6v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2V6l-3-4z"></path><line x1="3" y1="6" x2="21" y2="6"></line><path d="M16 10a4 4 0 0 1-8 0"></path></svg>
                        </button>
                    </div>
                </div>
            </header>

            <main className="app-main">
                {!hasSearched ? (
                    <div className="hero-section">
                        <div className="hero-background"></div>
                        <div className="hero-content">
                            <div className="hero-text">
                                <h2>Find jewellery as unique as you.</h2>
                                <p>Describe a style or upload a photo—our AI brings the closest matches from the House of Tanishq.</p>
                            </div>
                            <div className="hero-search-container">
                                <SearchBar 
                                    onSearch={handleTextSearch} 
                                    onImageSearch={handleImageUploadSearch}
                                    isLoading={isLoading} 
                                />
                                <div className="popular-searches">
                                    <span>POPULAR SEARCHES</span>
                                    <div className="tags">
                                        {['Diamond ring', 'Gold necklace', 'Pearl earrings', 'Bridal set'].map(tag => (
                                            <button key={tag} onClick={() => handleTextSearch(tag)} className="tag-pill">{tag}</button>
                                        ))}
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                ) : (
                    <div className="results-section container">
                        <div className="results-header">
                            <div className="results-info">
                                <h2>Curated for you</h2>
                                <p>Showing matches for "{currentQuery}"</p>
                            </div>
                            <div className="results-meta">
                                <span className="ai-badge">✨ AI-powered</span>
                            </div>
                        </div>
                        
                        <FilterBar 
                            activeFilters={activeFilters} 
                            onToggle={handleFilterToggle} 
                        />
                        
                        {isLoading ? (
                            <LoadingSpinner />
                        ) : error ? (
                            <div className="error-message">{error}</div>
                        ) : (
                            <>
                                <ResultsGrid 
                                    results={results.slice((currentPage - 1) * ITEMS_PER_PAGE, currentPage * ITEMS_PER_PAGE)} 
                                    onImageClick={setSelectedItem}
                                />
                                {results.length > ITEMS_PER_PAGE && (
                                    <div className="pagination-container">
                                        <button 
                                            disabled={currentPage === 1}
                                            onClick={() => {
                                                setCurrentPage(p => p - 1);
                                                window.scrollTo({ top: 0, behavior: 'smooth' });
                                            }}
                                            className="pagination-btn"
                                        >
                                            Previous
                                        </button>
                                        <span className="page-info">
                                            Page {currentPage} of {Math.ceil(results.length / ITEMS_PER_PAGE)}
                                        </span>
                                        <button 
                                            disabled={currentPage >= Math.ceil(results.length / ITEMS_PER_PAGE)}
                                            onClick={() => {
                                                setCurrentPage(p => p + 1);
                                                window.scrollTo({ top: 0, behavior: 'smooth' });
                                            }}
                                            className="pagination-btn"
                                        >
                                            Next
                                        </button>
                                    </div>
                                )}
                            </>
                        )}
                    </div>
                )}
            </main>

            {selectedItem && (
                <ImageModal 
                    item={selectedItem} 
                    onClose={() => setSelectedItem(null)}
                    onSimilarSearch={() => {
                        handleSimilarSearch(selectedItem.image_id || selectedItem.id);
                        setSelectedItem(null);
                    }}
                />
            )}
        </div>
    );
}

export default App;
