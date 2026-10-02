/**
 * Typed client for the AtlasAI API.
 *
 * When `NEXT_PUBLIC_API_URL` is set the browser calls the backend directly;
 * otherwise requests go to a same-origin `/api/...` path that Next rewrites
 * (see next.config.ts).
 *
 * Direct is the default for development, because the rewrite proxy imposes its
 * own socket timeout and an analysis that outlives it is reported as a generic
 * "Request failed (500)" — the API is still working, but the proxy has already
 * hung up and the real cause is invisible to the user. Talking to the API
 * directly means a slow analysis stays slow instead of turning into a lie.
 */

/** Backend origin. Empty string routes through the Next rewrite instead. */
export const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "";

export interface LayerDescriptor {
  key: string;
  name: string;
  kind: "continuous" | "categorical" | "mask" | "rgb";
  colormap: string;
  vmin: number;
  vmax: number;
  unit: string;
  description: string;
  opacity: number;
  classes: { id: number; label: string; color: string }[];
  png_url?: string;
}

export interface ClassArea {
  id: number;
  label: string;
  color: string;
  area_km2: number;
  share: number;
  pixels: number;
}

export interface GridInfo {
  bbox: [number, number, number, number];
  /** Corner coordinates in MapLibre image-source order: TL, TR, BR, BL. */
  corners: [number, number][];
  width: number;
  height: number;
  resolution_m: number;
}

export interface AnalysisResult {
  module: string;
  title: string;
  summary: string;
  metrics: Record<string, unknown>;
  breakdown: ClassArea[];
  layers: LayerDescriptor[];
  sources: { id?: string; collection?: string; datetime?: string; cloud_cover?: number }[];
  caveats: string[];
  confidence: number;
  observed_dates: string[];
  grid: GridInfo;
}

export interface ModuleInfo {
  available: boolean;
  unavailable_reason: string | null;
  id: string;
  title: string;
  question: string;
  description: string;
  sensors: string[];
  needs_baseline: boolean;
  neural_backend: string | null;
}

export interface Place {
  id: string;
  name: string;
  kind: "region" | "city" | "reservoir" | "basin";
  lon: number;
  lat: number;
  bbox: [number, number, number, number];
}

export interface PointReport {
  location: {
    lon: number;
    lat: number;
    radius_m: number;
    bbox: [number, number, number, number];
    nearest_place: (Place & { distance_km: number }) | null;
  };
  window: { start: string; end: string };
  modules: Record<string, PointModuleResult>;
  modules_succeeded: string[];
  narrative: string;
}

export interface PointModuleResult {
  status: "ok" | "no_data" | "error";
  detail?: string;
  hint?: string;
  title?: string;
  summary?: string;
  at_point?: Record<string, { value: number | null; unit?: string; label?: string }>;
  area_metrics?: Record<string, unknown>;
  breakdown?: ClassArea[];
  confidence?: number;
  caveats?: string[];
  observed_dates?: string[];
  result_key?: string;
}

export interface TimeseriesPoint {
  start: string;
  end: string;
  label: string;
  status: "ok" | "no_data";
  value: number | null;
  detail?: string;
  scene_count?: number;
}

export interface Timeseries {
  index: { key: string; name: string; description: string; reference: string; vmin: number; vmax: number };
  bbox: [number, number, number, number];
  interval: string;
  resolution_m: number;
  points: TimeseriesPoint[];
  summary: {
    steps: number;
    steps_with_data: number;
    mean: number | null;
    min: number | null;
    max: number | null;
    trend: string | null;
  };
  caveats: string[];
}

export interface AssistantAnswer {
  answer: string;
  answerable: boolean;
  suggestions?: string[];
  plan?: {
    module: string;
    mode: string;
    places: string[];
    window: { start: string; end: string };
    baseline: { start: string; end: string } | null;
    planner: string;
    reasoning: string;
  };
  module?: { id: string; title: string; question: string };
  results?: AssistantResult[];
  gaps?: { place: Place; status: string; detail: string }[];
}

export interface AssistantResult {
  place: Place;
  status: "ok";
  summary: string;
  metrics: Record<string, unknown>;
  confidence: number;
  caveats: string[];
  observed_dates: string[];
  result_key: string;
  layers: LayerDescriptor[];
  rank?: number;
  ranked_by?: { metric: string; value: number | null };
}

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly hint?: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

/** Analyses read imagery live; a national scan legitimately takes minutes. */
const REQUEST_TIMEOUT_MS = 8 * 60 * 1000;

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

  let response: Response;
  try {
    response = await fetch(`${API_BASE}/api/v1${path}`, {
      ...init,
      signal: controller.signal,
      headers: { "Content-Type": "application/json", ...init?.headers },
    });
  } catch (err) {
    // Distinguish "we gave up waiting" from "the API isn't there" — they need
    // completely different responses from the user.
    if (err instanceof DOMException && err.name === "AbortError") {
      throw new ApiError(
        "The analysis took longer than 8 minutes and was cancelled.",
        408,
        "Choose a smaller area, a coarser resolution, or a narrower date range.",
      );
    }
    throw new ApiError(
      "Could not reach the API.",
      0,
      `Is it running on ${API_BASE || "the same origin"}? Start it with: uvicorn app.main:app --port 8000`,
    );
  } finally {
    clearTimeout(timer);
  }

  if (!response.ok) {
    // The API returns structured detail for the errors worth acting on —
    // an empty sky comes back with a hint about widening the date range.
    let message = `Request failed (${response.status})`;
    let hint: string | undefined;
    try {
      const body = await response.json();
      const detail = body?.detail ?? body;
      if (typeof detail === "string") {
        message = detail;
      } else if (Array.isArray(detail)) {
        message = detail.map((entry: { loc?: string[]; msg?: string }) => `${entry.loc?.join(".") ?? "Request"}: ${entry.msg ?? "Invalid value"}`).join("; ");
      } else if (detail && typeof detail === "object") {
        message = detail.detail ?? detail.error ?? message;
        hint = detail.hint;
      }
    } catch {
      /* non-JSON error body — keep the status message */
    }
    throw new ApiError(message, response.status, hint);
  }

  return response.json() as Promise<T>;
}

const qs = (params: Record<string, string | number | undefined | null>) =>
  Object.entries(params)
    .filter(([, v]) => v !== undefined && v !== null && v !== "")
    .map(([k, v]) => `${encodeURIComponent(k)}=${encodeURIComponent(String(v))}`)
    .join("&");

export interface AnalysisParams {
  bbox: string;
  start: string;
  end: string;
  baseline_start?: string;
  baseline_end?: string;
  resolution?: number;
  max_cloud?: number;
  max_items?: number;
}

export const api = {
  indices: () => request<{ indices: { key: string; name: string }[] }>("/indices"),
  modules: () => request<{ modules: ModuleInfo[]; point_report_modules: string[] }>("/modules"),

  places: (kind?: string) =>
    request<{ places: Place[]; morocco_bbox: [number, number, number, number] }>(
      `/places${kind ? `?kind=${kind}` : ""}`,
    ),

  health: () => request<Record<string, unknown>>("/health"),

  analyse: (moduleId: string, params: AnalysisParams) =>
    request<AnalysisResult>(`/analysis/${moduleId}?${qs({ ...params })}`),

  point: (lon: number, lat: number, radius_m = 1500, start?: string, end?: string) =>
    request<PointReport>(`/point?${qs({ lon, lat, radius_m, start, end })}`),

  timeseries: (params: {
    bbox: string;
    index: string;
    start: string;
    end: string;
    interval?: string;
    resolution?: number;
  }) => request<Timeseries>(`/timeseries?${qs({ ...params })}`),

  ask: (question: string, bbox?: string) =>
    request<AssistantAnswer>("/assistant", {
      method: "POST",
      body: JSON.stringify({ question, bbox }),
    }),

  suggestions: () =>
    request<{ suggestions: string[]; planner: string; note: string }>("/assistant/suggestions"),
};

/** Absolute URL for a rendered layer PNG, given the descriptor's template. */
export const layerImageUrl = (pngUrl: string) => `${API_BASE}${pngUrl}`;

/** Approximate ground area of a lon/lat box, in km². */
export function bboxAreaKm2([west, south, east, north]: [number, number, number, number]): number {
  const meanLat = ((south + north) / 2) * (Math.PI / 180);
  const widthKm = (east - west) * 111.32 * Math.cos(meanLat);
  const heightKm = (north - south) * 110.57;
  return Math.abs(widthKm * heightKm);
}

/**
 * Ground resolution to request for an area, in metres.
 *
 * This is the difference between a responsive dashboard and one that appears
 * broken. Analysis cost scales with pixel *count*, so holding resolution fixed
 * while the area grows is quadratic: Casablanca at 30 m is a 1360x1633 grid,
 * and with several scenes and bands that is tens of millions of pixels pulled
 * over the network before anything renders.
 *
 * The ladder keeps every request near a fixed pixel budget instead. Detail is
 * preserved where it matters — a neighbourhood still gets 10 m — and a region
 * gets the resolution its question actually needs, since nobody reads a
 * provincial drought map at 10 m.
 */
export function resolutionForArea(bbox: [number, number, number, number]): number {
  const area = bboxAreaKm2(bbox);
  if (area < 50) return 10;      // neighbourhood
  if (area < 200) return 20;     // small town
  if (area < 800) return 30;     // city
  if (area < 3_000) return 60;   // metro area
  if (area < 15_000) return 100; // province
  return 200;                    // region
}
