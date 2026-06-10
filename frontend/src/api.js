// In dev: /api is proxied to localhost:8000 by vite.config.js
// In prod (Docker/Render): VITE_API_URL points to the backend service
const API = import.meta.env.VITE_API_URL
  ? import.meta.env.VITE_API_URL
  : '/api';

export async function searchText(query, k = 12) {
  const res = await fetch(
    `${API}/search/text?query=${encodeURIComponent(query)}&k=${k}`,
    { method: 'POST' }
  );
  if (!res.ok) throw new Error(`Search failed: ${res.status}`);
  return res.json();
}

export async function searchImage(file, k = 12) {
  const form = new FormData();
  form.append('file', file);
  const res = await fetch(`${API}/search/image?k=${k}`, {
    method: 'POST',
    body: form,
  });
  if (!res.ok) throw new Error(`Image search failed: ${res.status}`);
  return res.json();
}

export async function generateBoard(productId, boardSize = 8) {
  const res = await fetch(
    `${API}/board/generate?product_id=${encodeURIComponent(productId)}&board_size=${boardSize}&narrate=true`,
    { method: 'POST' }
  );
  if (!res.ok) throw new Error(`Board generation failed: ${res.status}`);
  return res.json();
}

export async function getHealth() {
  const res = await fetch(`${API}/health`);
  return res.json();
}
