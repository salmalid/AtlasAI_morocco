"use client";

import { useEffect, useState } from "react";
import { api, ApiError, resolutionForArea, type Place, type Timeseries } from "@/lib/api";

export default function HistoryPanel({ place, start, end }: { place: Place | null; start: string; end: string }) {
  const [indices, setIndices] = useState<{ key: string; name: string }[]>([]);
  const [index, setIndex] = useState("ndvi");
  const [interval, setInterval] = useState("month");
  const [result, setResult] = useState<Timeseries | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    void api.indices().then((data) => setIndices(data.indices)).catch(() => setError("Could not load spectral indices."));
  }, []);

  async function run() {
    if (!place || busy) return;
    if (!start || !end || end < start) { setError("Choose a valid observation date range."); return; }
    setBusy(true); setError(null); setResult(null);
    try {
      setResult(await api.timeseries({ bbox: place.bbox.join(","), index, interval, start, end,
        resolution: Math.max(100, resolutionForArea(place.bbox)) }));
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "History could not be loaded.");
    } finally { setBusy(false); }
  }

  const points = result?.points ?? [];
  const values = points.flatMap((p) => p.value === null ? [] : [p.value]);
  const low = values.length ? Math.min(...values) : 0;
  const high = values.length ? Math.max(...values) : 1;
  const x = (i: number) => 30 + i * 260 / Math.max(points.length - 1, 1);
  const y = (value: number) => 125 - (value - low) / Math.max(high - low, 0.01) * 100;

  return <>
    <div className="section">
      <h2>Observation history</h2>
      <p className="hint">{place?.name ?? "Select an area"} · {start} to {end}. Each interval reads satellite imagery; up to 40 intervals per request.</p>
      <div className="field"><label htmlFor="history-index">Spectral index</label>
        <select id="history-index" value={index} disabled={busy} onChange={(e) => setIndex(e.target.value)}>
          {indices.map((entry) => <option key={entry.key} value={entry.key}>{entry.name}</option>)}
        </select>
      </div>
      <div className="field"><label htmlFor="history-interval">Interval</label>
        <select id="history-interval" value={interval} disabled={busy} onChange={(e) => setInterval(e.target.value)}>
          <option value="month">Monthly</option><option value="quarter">Quarterly</option><option value="year">Yearly</option>
        </select>
      </div>
      <button className="primary" disabled={busy || !place || !indices.length} onClick={() => void run()}>{busy ? "Reading history…" : "Load history"}</button>
      {error && <p className="error" role="alert">{error}</p>}
    </div>
    {result && <div className="section">
      <h2>{result.index.name}</h2>
      <p className="hint">{result.summary.steps_with_data} of {result.summary.steps} intervals observed. Trend: {result.summary.trend ?? "insufficient observations"}.</p>
      {values.length > 0 && <svg viewBox="0 0 320 150" role="img" aria-label={`${result.index.name} history; gaps represent missing observations`} style={{ width: "100%" }}>
        <path d="M30 15V125H300" fill="none" stroke="var(--muted)" />
        <text x="28" y="22" textAnchor="end" fill="var(--muted)" fontSize="8">{high.toFixed(2)}</text>
        <text x="28" y="125" textAnchor="end" fill="var(--muted)" fontSize="8">{low.toFixed(2)}</text>
        {points.map((p, i) => p.value === null ? null : <g key={p.start}>
          {i > 0 && points[i - 1].value !== null && <line x1={x(i - 1)} y1={y(points[i - 1].value!)} x2={x(i)} y2={y(p.value)} stroke="var(--accent)" strokeWidth="2" />}
          <circle cx={x(i)} cy={y(p.value)} r="3" fill="var(--accent)"><title>{p.label}: {p.value}</title></circle>
        </g>)}
        <text x="30" y="145" fill="var(--muted)" fontSize="10">{points[0]?.label}</text>
        <text x="295" y="145" textAnchor="end" fill="var(--muted)" fontSize="10">{points.at(-1)?.label}</text>
      </svg>}
      {points.map((p) => <div className="metric" key={p.start}><span className="k">{p.label}</span><span className="v">{p.value === null ? "No data" : p.value.toFixed(4)}</span></div>)}
      {result.caveats.map((caveat) => <p className="caveat" key={caveat}>{caveat}</p>)}
    </div>}
  </>;
}
