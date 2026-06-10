const STEP_ORDER = ['cleanser', 'toner', 'serum', 'treatment', 'moisturizer', 'spf'];

const STEP_LABELS = {
  cleanser: 'Step 1 \u2014 Cleanse',
  toner: 'Step 2 \u2014 Tone',
  serum: 'Step 3 \u2014 Treat',
  treatment: 'Step 4 \u2014 Target',
  moisturizer: 'Step 5 \u2014 Moisturize',
  spf: 'Step 6 \u2014 Protect',
};

export default function RoutineBoard({ data, onBack }) {
  if (!data) return null;

  const { query_product, board, metrics, narrative } = data;

  const sortedBoard = [...board].sort((a, b) => {
    const ai = STEP_ORDER.indexOf(a.routine_step);
    const bi = STEP_ORDER.indexOf(b.routine_step);
    return (ai === -1 ? 99 : ai) - (bi === -1 ? 99 : bi);
  });

  return (
    <div className="board-section">
      <div className="container">
        <div style={{ marginBottom: '32px' }}>
          <button className="btn btn-ghost" onClick={onBack}>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M19 12H5M12 19l-7-7 7-7" />
            </svg>
            Back to search
          </button>
        </div>

        <div className="board-header">
          <h2>Your Routine</h2>
          <p>A curated board of complementary products built around your anchor, with zero ingredient conflicts.</p>
          {narrative && (
            <div className="fade-in" style={{ marginTop: '24px', padding: '24px', background: 'var(--cream)', borderRadius: 'var(--radius-lg)', textAlign: 'left', fontStyle: 'italic', fontSize: '0.9375rem', lineHeight: '1.6', borderLeft: '4px solid var(--accent)' }}>
              {narrative}
            </div>
          )}
        </div>

        <div className="board-anchor fade-in">
          <div>
            <div className="board-anchor-label">Anchor Product</div>
            <div className="board-anchor-name">{query_product.name}</div>
            <div className="board-anchor-brand">
              {query_product.brand} &middot; {query_product.category}
            </div>
          </div>
        </div>

        <div className="routine-flow">
          {sortedBoard.map((item, i) => (
            <div className={`routine-step fade-in stagger-${i + 1}`} key={item.id}>
              <div className="routine-step-indicator">
                <div className="routine-step-dot" />
                <div className="routine-step-line" />
              </div>
              <div className="routine-step-content">
                <div className="routine-step-label">
                  {STEP_LABELS[item.routine_step] || item.routine_step}
                </div>
                <div className="routine-step-name">{item.name}</div>
                <div className="routine-step-brand">{item.brand}</div>
                {item.key_ingredients && item.key_ingredients.length > 0 && (
                  <div style={{ marginBottom: '8px' }}>
                    {item.key_ingredients.slice(0, 3).map((ing, j) => (
                      <span className="tag" key={j}>
                        {typeof ing === 'string' && ing.length > 30
                          ? ing.substring(0, 30) + '\u2026'
                          : ing}
                      </span>
                    ))}
                  </div>
                )}
                <div className="routine-step-meta">
                  <span className="routine-step-price">${item.price?.toFixed(2)}</span>
                  <span
                    className={`conflict-badge ${
                      item.ingredient_conflicts?.length === 0 ? 'safe' : 'warning'
                    }`}
                  >
                    {item.ingredient_conflicts?.length === 0 ? 'No conflicts' : 'Conflict detected'}
                  </span>
                  <span style={{ color: 'var(--warm-gray)' }}>
                    {(item.complementarity_score * 100).toFixed(0)}% match
                  </span>
                </div>
              </div>
            </div>
          ))}
        </div>

        {metrics && (
          <div className="metrics-bar fade-in stagger-8">
            <div className="metric">
              <div className="metric-value">
                {metrics.routine_coverage}/{metrics.routine_coverage_max}
              </div>
              <div className="metric-label">Routine Coverage</div>
            </div>
            <div className="metric">
              <div className="metric-value">{metrics.conflict_free ? 'Yes' : 'No'}</div>
              <div className="metric-label">Conflict Free</div>
            </div>
            <div className="metric">
              <div className="metric-value">
                ${metrics.price_range?.min?.toFixed(0)} - ${metrics.price_range?.max?.toFixed(0)}
              </div>
              <div className="metric-label">Price Range</div>
            </div>
            <div className="metric">
              <div className="metric-value">{board.length}</div>
              <div className="metric-label">Products</div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
