"use client";

import { useEffect, useState } from "react";
import { api, ApiError, type PointReport } from "@/lib/api";

interface Props {
  point: { lon: number; lat: number } | null;
  start: string;
  end: string;
}

export default function Inspector({ point, start, end }: Props) {
  const [report, setReport] = useState<PointReport | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!point) return;
    let cancelled = false;

    setBusy(true);
    setError(null);
    setReport(null);

    api
      .point(point.lon, point.lat, 1500, start, end)
      .then((data) => {
        if (!cancelled) setReport(data);
      })
      .catch((err) => {
        if (!cancelled) setError(err instanceof ApiError ? err.message : "Point report failed.");
      })
      .finally(() => {
        if (!cancelled) setBusy(false);
      });

    // A rapid second click must not let the first response overwrite the second.
    return () => {
      cancelled = true;
    };
  }, [point, start, end]);

  if (!point) {
    return (
      <div className="section">
        <h2>Inspect a location</h2>
        <p className="hint">
          Click anywhere on the map. The platform runs land cover, vegetation, drought, water and
          terrain analysis around that point and reports both the pixel value and its surroundings.
        </p>
      </div>
    );
  }

  return (
    <>
      <div className="section">
        <h2>Location</h2>
        <div className="metric">
          <span className="k">Coordinates</span>
          <span className="v">
            {point.lat.toFixed(5)}, {point.lon.toFixed(5)}
          </span>
        </div>
        {report?.location.nearest_place && (
          <div className="metric">
            <span className="k">Nearest place</span>
            <span className="v">
              {report.location.nearest_place.name} · {report.location.nearest_place.distance_km} km
            </span>
          </div>
        )}
      </div>

      {busy && (
        <div className="section">
          <div className="loading">
            <span className="spinner" /> Running five modules over this location…
          </div>
        </div>
      )}

      {error && (
        <div className="section">
          <p className="error">{error}</p>
        </div>
      )}

      {report && !busy && (
        <>
          <div className="section">
            <h2>Summary</h2>
            <p className="summary">{report.narrative}</p>
            <p className="hint">
              Window {report.window.start} → {report.window.end}
            </p>
          </div>

          {Object.entries(report.modules).map(([moduleId, result]) => (
            <div className="section" key={moduleId}>
              <h2>{result.title ?? moduleId}</h2>

              {result.status !== "ok" ? (
                <>
                  <p className="hint">{result.detail}</p>
                  {result.hint && <p className="hint">{result.hint}</p>}
                </>
              ) : (
                <>
                  <p className="summary" style={{ marginBottom: 8 }}>
                    {result.summary}
                  </p>
                  {result.at_point &&
                    Object.entries(result.at_point).map(([key, entry]) => (
                      <div className="metric" key={key}>
                        <span className="k">{key.replace(/_/g, " ")}</span>
                        <span className="v">
                          {entry.label ??
                            (entry.value === null
                              ? "no data"
                              : `${entry.value}${entry.unit ? ` ${entry.unit}` : ""}`)}
                        </span>
                      </div>
                    ))}
                </>
              )}
            </div>
          ))}
        </>
      )}
    </>
  );
}
