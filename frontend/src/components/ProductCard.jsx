const CATEGORY_ICONS = {
  Cleanser: '\u{1F9F4}',
  Toner: '\u{1F4A7}',
  Serum: '\u{1F9EA}',
  Treatment: '\u{2728}',
  Moisturizer: '\u{1F338}',
  Sunscreen: '\u2600\uFE0F',
  Mask: '\u{1F3AD}',
  'Lip Care': '\u{1F48B}',
  'Eye Cream': '\u{1F441}\uFE0F',
  'Face Oil': '\u{1FAD2}',
};

export default function ProductCard({ product, onGenerateBoard }) {
  const icon = CATEGORY_ICONS[product.category] || '\u{1F9F4}';
  const score = product.similarity_score ?? product.complementarity_score;

  return (
    <div className="product-card" onClick={() => onGenerateBoard?.(product.id)}>
      <div className="product-card-image">
        <span className="category-icon">{icon}</span>
      </div>
      <div className="product-card-body">
        <div className="product-card-brand">{product.brand}</div>
        <div className="product-card-name">{product.name}</div>
        <div className="product-card-category">{product.category}</div>
        {product.concerns && product.concerns.length > 0 && (
          <div style={{ marginBottom: '12px' }}>
            {product.concerns.slice(0, 3).map((c, i) => (
              <span className="tag" key={i}>{c}</span>
            ))}
          </div>
        )}
        <div className="product-card-footer">
          <span className="product-card-price">${product.price?.toFixed(2)}</span>
          {score != null && (
            <span className="product-card-score">{(score * 100).toFixed(0)}%</span>
          )}
        </div>
      </div>
    </div>
  );
}
