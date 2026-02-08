import React, { useState } from 'react';
import './FilterBar.css';

const PRIMARY_FILTERS = ['All', 'Rings', 'Necklaces', 'Diamond', 'Gold'];
const EXTRA_FILTERS = ['Platinum', 'Emerald', 'Ruby', 'Sapphire', 'Pearl', 'Rose Gold'];

const FilterBar = ({ activeFilters = [], onToggle }) => {
    const [showMore, setShowMore] = useState(false);
    
    // If no filter selected, 'All' is implicitly active for UI purposes
    const isAllActive = activeFilters.length === 0;

    const handleFilterClick = (filter) => {
        if (filter === 'All') {
            onToggle('All'); // Parent should handle clearing
        } else {
            onToggle(filter);
        }
    };
    
    const displayedFilters = showMore ? [...PRIMARY_FILTERS, ...EXTRA_FILTERS] : PRIMARY_FILTERS;

    return (
        <div className="filter-bar-container">
            <div className="filter-pills">
                <button 
                    className={`filter-pill ${isAllActive ? 'active' : ''}`}
                    onClick={() => handleFilterClick('All')} 
                >
                    All
                </button>
                {displayedFilters.slice(1).map(filter => (
                    <button 
                        key={filter}
                        className={`filter-pill ${activeFilters.includes(filter) ? 'active' : ''}`}
                        onClick={() => handleFilterClick(filter)}
                    >
                        {filter}
                    </button>
                ))}
            </div>
            
            <button 
                className="more-filters-btn"
                onClick={() => setShowMore(!showMore)}
            >
                {showMore ? (
                    <>
                        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polyline points="18 15 12 9 6 15"></polyline></svg>
                        Less filters
                    </>
                ) : (
                    <>
                        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polygon points="22 3 2 3 10 12.46 10 19 14 21 14 12.46 22 3"></polygon></svg>
                        More filters
                    </>
                )}
            </button>
        </div>
    );
};

export default FilterBar;
