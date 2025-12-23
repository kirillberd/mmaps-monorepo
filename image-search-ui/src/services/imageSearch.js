/**
 * Calls backend image-search endpoint.
 * The dev server proxies /image-search -> VITE_BACKEND_URL (default http://localhost:8000)
 */
export async function searchByImage(file, k = 10, { signal } = {}) {
  const form = new FormData()
  form.append('file', file)

  const res = await fetch(`/image-search/search?k=${encodeURIComponent(k)}`, {
    method: 'POST',
    body: form,
    signal
  })

  if (!res.ok) {
    // backend might return json or text; keep it simple
    const text = await res.text().catch(() => '')
    throw new Error(`HTTP ${res.status} ${res.statusText}${text ? `: ${text}` : ''}`)
  }

  return await res.json()
}
