# Image Search UI (Svelte + Tailwind)

Минималистичный фронтенд: загрузка картинки → POST в backend → рендер результатов по `source.url`.

## Требования
- Node.js 18+

## Запуск
```bash
npm install
npm run dev
```

По умолчанию backend ожидается на `http://localhost:8000`.

Чтобы указать другой URL backend (dev proxy):
```bash
# Linux/macOS
export VITE_BACKEND_URL="http://localhost:8000"
npm run dev

# Windows PowerShell
$env:VITE_BACKEND_URL="http://localhost:8000"
npm run dev
```

## API, который ожидается
UI вызывает:
`POST /image-search/search?k=10` (multipart/form-data, поле `file`)

Ожидаемый ответ:
```json
{
  "k": 10,
  "results": [
    { "_id": "123", "score": 0.123, "source": { "url": "https://..." } }
  ]
}
```

## Если в браузере CORS
В dev-режиме CORS обычно не нужен, потому что Vite проксирует запросы.
Если ты открываешь build не через Vite dev server, тогда добавь CORS в backend.
