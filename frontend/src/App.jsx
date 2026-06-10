import { useState, useRef, useCallback } from 'react';
import ProductCard from './components/ProductCard.jsx';
import RoutineBoard from './components/RoutineBoard.jsx';
import { searchText, searchImage, generateBoard } from './api.js';

function SearchIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
      <circle cx="11" cy="11" r="8" />
      <path d="m21 21-4.35-4.35" />
    </svg>
  );
}

function UploadIcon() {
  return (
    <svg className="upload-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round">
      <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4" />
      <polyline points="17 8 12 3 7 8" />
      <line x1="12" y1="3" x2="12" y2="15" />
    </svg>
  );
}

export default function App() {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState(null);
  const [board, setBoard] = useState(null);
  const [loading, setLoading] = useState(false);
  const [view, setView] = useState('home'); // 'home' | 'results' | 'board'
  const [dragging, setDragging] = useState(false);
  const fileRef = useRef(null);
  const resultsRef = useRef(null);

  const handleTextSearch = useCallback(async (e) => {
    e?.preventDefault();
    if (!query.trim()) return;
    setLoading(true);
    setView('results');
    try {
      const data = await searchText(query.trim(), 12);
      setResults(data);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
      setTimeout(() => {
        resultsRef.current?.scrollIntoView({ behavior: 'smooth' });
      }, 100);
    }
  }, [query]);

  const handleImageSearch = useCallback(async (file) => {
    if (!file) return;
    setLoading(true);
    setView('results');
    setQuery(file.name);
    try {
      const data = await searchImage(file, 12);
      setResults(data);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  }, []);

  const handleGenerateBoard = useCallback(async (productId) => {
    setLoading(true);
    setView('board');
    try {
      const data = await generateBoard(productId, 8);
      setBoard(data);
      window.scrollTo({ top: 0, behavior: 'smooth' });
    } catch (err) {
      console.error(err);
      setView('results');
    } finally {
      setLoading(false);
    }
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    setDragging(false);
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
      handleImageSearch(file);
    }
  }, [handleImageSearch]);

  const handleBack = useCallback(() => {
    setView(results ? 'results' : 'home');
    setBoard(null);
  }, [results]);

  return (
    <>
      {/* Navigation */}
      <nav className="nav">
        <div className="nav-inner">
          <a
            className="nav-brand"
            href="#"
            onClick={(e) => {
              e.preventDefault();
              setView('home');
              setBoard(null);
              setResults(null);
              setQuery('');
            }}
          >
            Curate
          </a>
          <ul className="nav-links">
            <li><a href="#" className={view === 'home' ? 'active' : ''} onClick={(e) => { e.preventDefault(); setView('home'); }}>Search</a></li>
            <li><a href="#" className={view === 'board' ? 'active' : ''}>Routines</a></li>
          </ul>
        </div>
      </nav>

      {/* Board View */}
      {view === 'board' && (
        <div style={{ paddingTop: '72px' }}>
          {loading ? (
            <div className="loading">
              <div className="loading-spinner" />
              <div className="loading-text">Assembling your routine...</div>
            </div>
          ) : board ? (
            <RoutineBoard data={board} onBack={handleBack} />
          ) : null}
        </div>
      )}

      {/* Home + Results View */}
      {view !== 'board' && (
        <>
          {/* Hero */}
          <section className="hero">
            <div className="hero-overline">Intelligent Skincare Discovery</div>
            <h1>Find what <em>complements</em> you</h1>
            <p className="hero-subtitle">
              Search by text or image. We will build a complete routine around your product,
              with ingredient safety verified at every step.
            </p>

            <div className="search-container">
              <form onSubmit={handleTextSearch}>
                <div className="search-bar">
                  <input
                    id="search-input"
                    className="search-input"
                    type="text"
                    placeholder="Try 'vitamin C serum for brightening' or 'gentle cleanser'..."
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                  />
                  <button type="submit" className="search-btn" id="search-button" aria-label="Search">
                    <SearchIcon />
                  </button>
                </div>
              </form>

              <div className="search-divider">or upload an image</div>

              <div
                className={`upload-zone ${dragging ? 'dragging' : ''}`}
                onClick={() => fileRef.current?.click()}
                onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
                onDragLeave={() => setDragging(false)}
                onDrop={handleDrop}
              >
                <UploadIcon />
                <p className="upload-text">
                  <strong>Drop an image here</strong> or click to browse
                </p>
                <input
                  ref={fileRef}
                  type="file"
                  accept="image/*"
                  hidden
                  onChange={(e) => handleImageSearch(e.target.files[0])}
                />
              </div>
            </div>
          </section>

          {/* Results */}
          {view === 'results' && (
            <section className="results-section" ref={resultsRef}>
              <div className="container">
                {loading ? (
                  <div className="loading">
                    <div className="loading-spinner" />
                    <div className="loading-text">Searching 3,413 products...</div>
                  </div>
                ) : results && results.results?.length > 0 ? (
                  <>
                    <div className="results-header">
                      <h2>Results</h2>
                      <span className="results-count">
                        Found {results.results.length} top matches
                      </span>
                    </div>
                    <p style={{ marginBottom: '24px', fontSize: '0.875rem' }}>
                      Click any product to generate a complete routine board around it.
                    </p>
                    <div className="product-grid">
                      {results.results.map((product, i) => (
                        <div className={`fade-in stagger-${Math.min(i + 1, 8)}`} key={product.id}>
                          <ProductCard
                            product={product}
                            onGenerateBoard={handleGenerateBoard}
                          />
                        </div>
                      ))}
                    </div>
                  </>
                ) : (
                  <div className="empty-state">
                    <h3>No results found</h3>
                    <p>Try a different search term or upload a product image.</p>
                  </div>
                )}
              </div>
            </section>
          )}
        </>
      )}
    </>
  );
}
