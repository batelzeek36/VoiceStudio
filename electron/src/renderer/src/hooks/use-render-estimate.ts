import { useDebouncedValue } from '@tanstack/react-pacer';
import { keepPreviousData, useQuery, type QueryClient } from '@tanstack/react-query';
import {
  fetchRenderEstimate,
  type RenderEstimate,
  type RenderEstimateRequest,
} from '@/lib/api/render-estimate';

export const RENDER_ESTIMATE_QUERY_KEY = ['render-estimate'] as const;

/** Typing pause before asking again; a book-sized body waits a little longer. */
const DEBOUNCE_MS = 600;
const LONG_BODY_DEBOUNCE_MS = 1200;
const LONG_BODY_CHARS = 50_000;

/**
 * The render-time estimate for `request`, re-asked (debounced) as the script
 * or settings change. `null` asks nothing. The request is compared as its JSON
 * text, so rebuilding an equal object on every render never re-queries.
 */
export function useRenderEstimate(request: RenderEstimateRequest | null): RenderEstimate | null {
  const serialized = request ? JSON.stringify(request) : '';
  const [debounced] = useDebouncedValue(serialized, {
    wait: serialized.length > LONG_BODY_CHARS ? LONG_BODY_DEBOUNCE_MS : DEBOUNCE_MS,
  });
  const query = useQuery({
    queryKey: [...RENDER_ESTIMATE_QUERY_KEY, debounced],
    queryFn: ({ signal }) =>
      fetchRenderEstimate(JSON.parse(debounced) as RenderEstimateRequest, signal),
    enabled: debounced !== '',
    staleTime: 60_000,
    // Whether the engine is loaded (or was unloaded for idleness) changes the
    // estimate without any input changing: re-ask while the page is open.
    refetchInterval: 60_000,
    retry: false,
    placeholderData: keepPreviousData,
  });
  // Cleared input shows nothing at once, not the last script's estimate.
  if (!serialized || !debounced) return null;
  return query.data ?? null;
}

/** A finished render added timings: every visible estimate is now stale. */
export function refreshRenderEstimates(client: QueryClient): Promise<void> {
  return client.invalidateQueries({ queryKey: RENDER_ESTIMATE_QUERY_KEY });
}
