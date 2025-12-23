<script>
  import { onDestroy } from 'svelte'
  import { searchByImage } from './services/imageSearch.js'

  let file = null
  let previewUrl = ''
  let loading = false
  let error = ''
  let results = [] // { url, id, score }

  let abortController = null

  function onFileChange(e) {
    error = ''
    results = []

    const f = e.currentTarget.files?.[0]
    file = f || null

    if (previewUrl) URL.revokeObjectURL(previewUrl)
    previewUrl = file ? URL.createObjectURL(file) : ''
  }

  async function onSubmit() {
    error = ''
    results = []

    if (!file) {
      error = 'Выбери картинку.'
      return
    }

    loading = true
    abortController?.abort()
    abortController = new AbortController()

    try {
      const data = await searchByImage(file, 10, { signal: abortController.signal })

      // Expected backend shape:
      // { k: number, results: [{ _id, score, source: { url, ... } }, ...] }
      const items = Array.isArray(data?.results) ? data.results : []
      results = items
        .map((r) => ({
          id: r?._id ?? null,
          score: r?.score ?? null,
          url: r?.source?.url ?? null
        }))
        .filter((x) => typeof x.url === 'string' && x.url.length > 0)
    } catch (e) {
      if (e?.name === 'AbortError') return
      error = e?.message || 'Ошибка запроса.'
    } finally {
      loading = false
    }
  }

  function onCancel() {
    abortController?.abort()
    loading = false
  }

  onDestroy(() => {
    abortController?.abort()
    if (previewUrl) URL.revokeObjectURL(previewUrl)
  })
</script>

<main class="mx-auto max-w-4xl p-6">
  <header class="mb-6">
    <h1 class="text-xl font-semibold tracking-tight">Image search</h1>
    <p class="mt-1 text-sm text-slate-600">
      Загрузи картинку, получи список совпадений и отрисуй их по URL.
    </p>
  </header>

  <section class="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
    <form
      class="flex flex-col gap-4 sm:flex-row sm:items-end"
      on:submit|preventDefault={onSubmit}
    >
      <label class="flex-1">
        <span class="mb-2 block text-sm font-medium text-slate-700">Картинка</span>
        <input
          class="block w-full cursor-pointer rounded-xl border border-slate-200 bg-slate-50 p-2 text-sm file:mr-3 file:rounded-lg file:border-0 file:bg-slate-900 file:px-3 file:py-2 file:text-sm file:font-medium file:text-white hover:file:bg-slate-800"
          type="file"
          accept="image/*"
          on:change={onFileChange}
        />
      </label>

      <div class="flex gap-2">
        <button
          class="rounded-xl bg-slate-900 px-4 py-2 text-sm font-medium text-white hover:bg-slate-800 disabled:opacity-50"
          type="submit"
          disabled={loading}
        >
          {loading ? 'Ищу…' : 'Поиск'}
        </button>

        {#if loading}
          <button
            class="rounded-xl border border-slate-200 bg-white px-4 py-2 text-sm font-medium text-slate-700 hover:bg-slate-50"
            type="button"
            on:click={onCancel}
          >
            Стоп
          </button>
        {/if}
      </div>
    </form>

    {#if previewUrl}
      <div class="mt-4">
        <div class="text-xs font-medium text-slate-500">Предпросмотр</div>
        <img
          src={previewUrl}
          alt="preview"
          class="mt-2 max-h-64 w-auto rounded-xl border border-slate-200 object-contain"
        />
      </div>
    {/if}

    {#if error}
      <div class="mt-4 rounded-xl border border-red-200 bg-red-50 p-3 text-sm text-red-700">
        {error}
      </div>
    {/if}
  </section>

  <section class="mt-8">
    <div class="mb-3 flex items-baseline justify-between">
      <h2 class="text-sm font-semibold text-slate-900">Результаты</h2>
      <span class="text-xs text-slate-500">{results.length} шт.</span>
    </div>

    {#if results.length === 0}
      <div class="rounded-2xl border border-dashed border-slate-200 bg-white p-8 text-center text-sm text-slate-500">
        Пока пусто.
      </div>
    {:else}
      <div class="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-4">
        {#each results as r (r.url)}
          <a
            class="group block overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-sm hover:shadow"
            href={r.url}
            target="_blank"
            rel="noreferrer"
            title={r.url}
          >
            <div class="aspect-square bg-slate-50">
              <img
                class="h-full w-full object-cover transition-transform duration-200 group-hover:scale-[1.02]"
                src={r.url}
                alt={r.id ? `id=${r.id}` : 'result'}
                loading="lazy"
              />
            </div>
            <div class="p-2">
              <div class="truncate text-xs text-slate-700">
                {r.id ? `id: ${r.id}` : ' '}
              </div>
              <div class="truncate text-[11px] text-slate-500">
                {typeof r.score === 'number' ? `score: ${r.score.toFixed(4)}` : ' '}
              </div>
            </div>
          </a>
        {/each}
      </div>
    {/if}
  </section>

  <footer class="mt-10 text-xs text-slate-500">
    Dev proxy: <code class="rounded bg-slate-100 px-1 py-0.5">/image-search → VITE_BACKEND_URL</code>
  </footer>
</main>
