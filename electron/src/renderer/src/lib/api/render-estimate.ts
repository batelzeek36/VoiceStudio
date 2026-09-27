import { apiFetch } from './client';
import { sanitizeInstruct } from './generate';
import type { CloneSettings } from '@/lib/store/clone-settings';

/**
 * `POST /render/estimate` (docs/adr/render-time-estimate.md): how long a render
 * will take on THIS machine, from the timings its own renders recorded.
 */
export type RenderEstimateBasis = 'measured' | 'rough' | 'none';
export type RenderEstimateReason = 'cold_start' | 'remote' | 'no_rate';

export interface RenderEstimatePart {
  calls: number;
  /** Planned wall seconds; null when the estimate has no number. */
  seconds: number | null;
}

export interface RenderEstimate {
  basis: RenderEstimateBasis;
  /** Why there is no number; null whenever `seconds` is set. */
  reason: RenderEstimateReason | null;
  seconds: number | null;
  low: number | null;
  high: number | null;
  calls: number;
  samples: number;
  /** One per chapter for longform renders, one for a Voice cloning take. */
  parts: RenderEstimatePart[];
}

export type RenderEstimateRequest = { surface: 'generate' | 'audiobook' | 'longform' } & Record<
  string,
  unknown
>;

const REASONS: readonly RenderEstimateReason[] = ['cold_start', 'remote', 'no_rate'];

function seconds(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) && value >= 0 ? value : null;
}

function count(value: unknown): number {
  return typeof value === 'number' && Number.isInteger(value) && value >= 0 ? value : 0;
}

/**
 * Validate the response. Anything malformed degrades to "no number" rather
 * than a wrong one: an estimate is only worth showing when it is honest.
 */
export function parseRenderEstimate(raw: unknown): RenderEstimate | null {
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) return null;
  const record = raw as Record<string, unknown>;
  const total = seconds(record.seconds);
  const priced = (record.basis === 'measured' || record.basis === 'rough') && total !== null;
  const reason = REASONS.find((value) => value === record.reason) ?? 'cold_start';
  const parts = Array.isArray(record.parts)
    ? record.parts.map((part) => {
        const entry = part && typeof part === 'object' ? (part as Record<string, unknown>) : {};
        return { calls: count(entry.calls), seconds: priced ? seconds(entry.seconds) : null };
      })
    : [];
  return {
    basis: priced ? (record.basis as RenderEstimateBasis) : 'none',
    reason: priced ? null : reason,
    seconds: priced ? total : null,
    low: priced ? (seconds(record.low) ?? total) : null,
    high: priced ? (seconds(record.high) ?? total) : null,
    calls: count(record.calls),
    samples: count(record.samples),
    parts,
  };
}

export async function fetchRenderEstimate(
  request: RenderEstimateRequest,
  signal?: AbortSignal,
): Promise<RenderEstimate | null> {
  const response = await apiFetch('/render/estimate', {
    method: 'POST',
    body: JSON.stringify(request),
    headers: { 'Content-Type': 'application/json' },
    signal,
  });
  return parseRenderEstimate(await response.json());
}

/** Longest explicit take length the estimate accepts (the backend bound). */
const MAX_DURATION_SECONDS = 3600;

/**
 * The estimate request for a Voice cloning take: the same inputs
 * `toGenerateForm` sends to `/generate`. Null when there is nothing to render.
 */
export function cloneEstimateRequest(
  settings: Pick<
    CloneSettings,
    | 'text'
    | 'language'
    | 'refText'
    | 'instruct'
    | 'steps'
    | 'speed'
    | 'duration'
    | 'selectedProfileId'
  >,
  referenceSeconds: number | null,
): RenderEstimateRequest | null {
  if (!settings.text.trim()) return null;
  const duration = Number.parseFloat(settings.duration);
  const profile = settings.selectedProfileId;
  return {
    surface: 'generate',
    text: settings.text,
    language: settings.language || null,
    profile_id: profile ?? null,
    ref_text: profile ? null : settings.refText || null,
    ref_seconds:
      !profile && referenceSeconds !== null && referenceSeconds > 0 ? referenceSeconds : null,
    instruct: settings.instruct ? sanitizeInstruct(settings.instruct).instruct || null : null,
    num_step: settings.steps,
    speed: settings.speed,
    duration:
      settings.duration && duration > 0 && duration <= MAX_DURATION_SECONDS ? duration : null,
  };
}
