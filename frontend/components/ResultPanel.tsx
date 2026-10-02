"use client";

import type { AnalysisResult } from "@/lib/api";

/** Turn `mean_ndvi` into `Mean ndvi` — readable without a hand-maintained map. */
const humanise = (key: string) =>
  key.replace(/_/g, " ").replace(/\bpct\b/i, "%").replace(/\bkm2\b/i, "km²").replace(/^./, (c) => c.toUpperCase());

const formatValue = (value: unknown): string => {
  if (value === null || value === undefined) return "—";
  if (typeof value === "number") {
    if (Number.isInteger(value)) return value.toLocaleString();
    return Math.abs(value) < 0.01 ? value.toExponential(2) : value.toLocaleString(undefined, { maximumFractionDigits: 3 });
  }
  if (typeof value === "boolean") return value ? "yes" : "no";
  if (Array.isArray(value)) return `${value.length} item(s)`;
  if (typeof value === "object") return "—";
  return String(value);
};

const confidenceColor = (value: number) =>
  value >= 0.75 ? "var(--good)" : value >= 0.45 ? "var(--warn)" : "var(--bad)";

export default function ResultPanel({ result }: { result: AnalysisResult }) {
  // Nested objects (like the review-site list) belong in their own view, not
  // squeezed into a two-column metric row.
  const scalarMetrics = Object.entries(result.metrics).filter(
    ([, v]) => v === null || typeof v !== "object",
  );

  return (
    <>
      <div className="section">
        <h2>{result.title}</h2>
        <button onClick={() => {
          const url = URL.createObjectURL(new Blob([JSON.stringify(result, null, 2)], { type: "application/json" }));
          const link = document.createElement("a");
          link.href = url; link.download = `${result.module}-analysis.json`; link.click();
          setTimeout(() => URL.revokeObjectURL(url), 1000);
        }}>Download report</button>
        <p className="summary">{result.summary}</p>
        <div className="confidence" style={{ marginTop: 10 }}>
          <span style={{ color: "var(--muted)" }}>Confidence</span>
          <span className="confidence-bar">
            <span
              className="confidence-fill"
              style={{
                width: `${Math.round(result.confidence * 100)}%`,
                background: confidenceColor(result.confidence),
              }}
            />
          </span>
          <span style={{ fontFamily: "var(--mono)" }}>{Math.round(result.confidence * 100)}%</span>
        </div>
      </div>

      {result.breakdown.length > 0 && (
        <div className="section">
          <h2>{result.module === "roads" ? "Road length breakdown" : "Area breakdown"}</h2>
          {result.breakdown
            .filter((entry) => entry.pixels > 0 || (result.module === "roads" && entry.area_km2 > 0))
            .sort((a, b) => b.share - a.share)
            .map((entry) => (
              <div className="metric" key={entry.id}>
                <span className="k">
                  <span
                    className="swatch"
                    style={{ background: entry.color, display: "inline-block", marginRight: 6, verticalAlign: -2 }}
                  />
                  {entry.label}
                </span>
                <span className="v">
                  {(entry.share * 100).toFixed(1)}% · {entry.area_km2.toLocaleString(undefined, { maximumFractionDigits: 2 })} {result.module === "roads" ? "km" : "km²"}
                </span>
              </div>
            ))}
        </div>
      )}

      {scalarMetrics.length > 0 && (
        <div className="section">
          <h2>Metrics</h2>
          {scalarMetrics.map(([key, value]) => (
            <div className="metric" key={key}>
              <span className="k">{humanise(key)}</span>
              <span className="v">{formatValue(value)}</span>
            </div>
          ))}
        </div>
      )}

      <div className="section">
        <h2>Observations used</h2>
        {result.observed_dates.length > 0 ? (
          <div>
            {result.observed_dates.map((date) => (
              <span className="chip" key={date}>
                {date}
              </span>
            ))}
          </div>
        ) : (
          <p className="hint">Static product — no acquisition date.</p>
        )}
        <p className="hint">
          {result.grid.width}×{result.grid.height} px at {result.grid.resolution_m} m ·{" "}
          {result.sources.length} scene(s)
        </p>
      </div>

      {result.caveats.length > 0 && (
        <div className="section">
          <h2>What this cannot tell you</h2>
          {result.caveats.map((caveat, index) => (
            <p className="caveat" key={index}>
              {caveat}
            </p>
          ))}
        </div>
      )}
    </>
  );
}
