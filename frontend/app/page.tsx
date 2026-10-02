"use client";

import dynamic from "next/dynamic";
import { useCallback, useEffect, useMemo, useState } from "react";
import Assistant from "@/components/Assistant";
import Inspector from "@/components/Inspector";
import Legend from "@/components/Legend";
import ResultPanel from "@/components/ResultPanel";
import HistoryPanel from "@/components/HistoryPanel";
import {
  api,
  ApiError,
  bboxAreaKm2,
  resolutionForArea,
  type AnalysisResult,
  type LayerDescriptor,
  type ModuleInfo,
  type Place,
} from "@/lib/api";

// MapLibre touches `window` at import time, so it must not run during SSR.
const MapCanvas = dynamic(() => import("@/components/MapCanvas"), {
  ssr: false,
  loading: () => <div className="map" style={{ display: "grid", placeItems: "center" }}>Loading map…</div>,
});

const iso = (date: Date) => date.toISOString().slice(0, 10);
const daysAgo = (n: number) => iso(new Date(Date.now() - n * 86_400_000));
const inYear = (date: string, year: number) => {
  const [, month, day] = date.split("-").map(Number);
  const lastDay = new Date(Date.UTC(year, month, 0)).getUTCDate();
  return iso(new Date(Date.UTC(year, month - 1, Math.min(day, lastDay))));
};

/** Sentinel-2 L2A coverage is only dependable from 2017 onward. */
const FIRST_YEAR = 2017;
const CURRENT_YEAR = new Date().getFullYear();

export default function Page() {
  const [modules, setModules] = useState<ModuleInfo[]>([]);
  const [places, setPlaces] = useState<Place[]>([]);
  const [moduleId, setModuleId] = useState("landcover");
  const [placeId, setPlaceId] = useState("marrakech");
  const [start, setStart] = useState(daysAgo(120));
  const [end, setEnd] = useState(iso(new Date()));
  const [baselineYear, setBaselineYear] = useState(CURRENT_YEAR - 5);
  const [maxCloud, setMaxCloud] = useState(30);

  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [activeLayerKey, setActiveLayerKey] = useState<string | null>(null);
  const [opacity, setOpacity] = useState(0.85);
  const [busy, setBusy] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [error, setError] = useState<{ message: string; hint?: string } | null>(null);
  const [tab, setTab] = useState<"result" | "inspect" | "ask" | "history">("result");
  const [catalogError, setCatalogError] = useState<string | null>(null);
  const [catalogAttempt, setCatalogAttempt] = useState(0);
  const [point, setPoint] = useState<{ lon: number; lat: number } | null>(null);

  useEffect(() => {
    let cancelled = false;
    setCatalogError(null);
    void Promise.all([api.modules(), api.places()]).then(([catalog, geography]) => {
      if (cancelled) return;
      setModules(catalog.modules);
      setPlaces(geography.places);
    }).catch(() => {
      if (!cancelled) setCatalogError("Could not load the analysis catalog. Check that the API is running.");
    });
    return () => { cancelled = true; };
  }, [catalogAttempt]);

  const selectedModule = useMemo(
    () => modules.find((m) => m.id === moduleId) ?? null,
    [modules, moduleId],
  );
  const selectedPlace = useMemo(
    () => places.find((p) => p.id === placeId) ?? null,
    [places, placeId],
  );
  const activeLayer = useMemo(
    () => result?.layers.find((l) => l.key === activeLayerKey) ?? null,
    [result, activeLayerKey],
  );

  const areaKm2 = useMemo(
    () => (selectedPlace ? bboxAreaKm2(selectedPlace.bbox) : 0),
    [selectedPlace],
  );
  const resolution = useMemo(
    () => (selectedPlace ? resolutionForArea(selectedPlace.bbox) : 30),
    [selectedPlace],
  );
  // Roughly what the backend will build, so the cost is visible before running.
  const megapixels = useMemo(
    () => (areaKm2 * 1e6) / (resolution * resolution) / 1e6,
    [areaKm2, resolution],
  );

  const run = useCallback(async () => {
    if (!selectedPlace || busy) return;
    if (!start || !end || end < start) {
      setError({ message: "Choose a valid date range with the end on or after the start." });
      setTab("result");
      return;
    }
    setBusy(true);
    setError(null);
    setElapsed(0);
    try {
      const needsBaseline = selectedModule?.needs_baseline ?? false;
      const data = await api.analyse(moduleId, {
        bbox: selectedPlace.bbox.join(","),
        start,
        end,
        // A baseline is only sent when the module can use it — passing one to a
        // single-epoch module would silently double its imagery reads.
        baseline_start: needsBaseline ? inYear(start, baselineYear) : undefined,
        baseline_end: needsBaseline ? inYear(end, baselineYear + Number(end.slice(0, 4)) - Number(start.slice(0, 4))) : undefined,
        max_cloud: maxCloud,
        resolution,
        // Four scenes is enough for a cloud-free median composite; each extra
        // one is roughly five more windowed reads over the network.
        max_items: needsBaseline ? 3 : 4,
      });
      setResult(data);
      setActiveLayerKey(data.layers[0]?.key ?? null);
      setTab("result");
    } catch (err) {
      setResult(null);
      setActiveLayerKey(null);
      setError(
        err instanceof ApiError
          ? { message: err.message, hint: err.hint }
          : { message: "Analysis failed. Is the API running on port 8000?" },
      );
    } finally {
      setBusy(false);
    }
  }, [moduleId, selectedPlace, selectedModule, start, end, baselineYear, maxCloud, resolution, busy]);

  // A progress counter, because a request that reads satellite imagery can run
  // for a minute and a spinner with no number is indistinguishable from a hang.
  useEffect(() => {
    if (!busy) return;
    const started = Date.now();
    const timer = setInterval(() => setElapsed(Math.round((Date.now() - started) / 1000)), 1000);
    return () => clearInterval(timer);
  }, [busy]);

  const onPointClick = useCallback((lon: number, lat: number) => {
    setPoint({ lon, lat });
    setTab("inspect");
  }, []);

  return (
    <div className="shell">
      {/* ---------------- left: controls ---------------- */}
      <aside className="sidebar">
        <div className="brand">
          <h1>AtlasAI Morocco</h1>
          <p>Earth observation from open satellite data. Near-real-time, not live.</p>
        </div>

        <div className="section">
          <h2>Analysis</h2>
          {catalogError && <div role="alert"><p className="error">{catalogError}</p><button onClick={() => setCatalogAttempt((n) => n + 1)}>Retry connection</button></div>}
          <div className="field">
            <label htmlFor="module">Module</label>
            <select id="module" value={moduleId} onChange={(e) => setModuleId(e.target.value)}>
              {modules.map((module) => (
                <option key={module.id} value={module.id} disabled={module.available === false}>
                  {module.title}{module.available === false ? " (unavailable)" : ""}
                </option>
              ))}
            </select>
          </div>
          {selectedModule && <p className="hint">{selectedModule.question}</p>}

          <div className="field" style={{ marginTop: 10 }}>
            <label htmlFor="place">Area</label>
            <select id="place" value={placeId} onChange={(e) => setPlaceId(e.target.value)}>
              {["city", "region", "reservoir"].map((kind) => (
                <optgroup key={kind} label={`${kind[0].toUpperCase()}${kind.slice(1)}s`}>
                  {places
                    .filter((place) => place.kind === kind)
                    .map((place) => (
                      <option key={place.id} value={place.id}>
                        {place.name}
                      </option>
                    ))}
                </optgroup>
              ))}
            </select>
          </div>
        </div>

        <div className="section">
          <h2>Time window</h2>
          <div className="row">
            <div className="field">
              <label htmlFor="start">From</label>
              <input id="start" type="date" value={start} onChange={(e) => setStart(e.target.value)} />
            </div>
            <div className="field">
              <label htmlFor="end">To</label>
              <input id="end" type="date" value={end} onChange={(e) => setEnd(e.target.value)} />
            </div>
          </div>
          {selectedModule?.needs_baseline && (
            <div className="field">
              <label htmlFor="baseline">Compare against year {baselineYear}</label>
              <p className="hint">Uses the same seasonal window in the baseline year.</p>
              <input
                id="baseline"
                type="range"
                min={FIRST_YEAR}
                max={CURRENT_YEAR - 1}
                value={baselineYear}
                onChange={(e) => setBaselineYear(Number(e.target.value))}
              />
            </div>
          )}
          <div className="field">
            <label htmlFor="cloud">Max scene cloud cover: {maxCloud}%</label>
            <input
              id="cloud"
              type="range"
              min={5}
              max={100}
              step={5}
              value={maxCloud}
              onChange={(e) => setMaxCloud(Number(e.target.value))}
            />
          </div>
          <button className="primary" style={{ width: "100%" }} onClick={() => void run()} disabled={busy || !selectedPlace || !selectedModule || selectedModule.available === false}>
            {busy ? `Analysing imagery… ${elapsed}s` : "Run analysis"}
          </button>
          {selectedPlace && (
            <p className="hint" style={{ marginTop: 6 }}>
              {areaKm2.toLocaleString(undefined, { maximumFractionDigits: 0 })} km² at {resolution} m
              → {megapixels.toFixed(1)} megapixel grid.
              {megapixels > 3 && " Large areas are analysed at coarser resolution to stay responsive."}
            </p>
          )}
        </div>

        {result && (
          <div className="section">
            <h2>Layers</h2>
            {result.layers.map((layer: LayerDescriptor) => (
              <div key={layer.key}>
                <label className={`layer ${activeLayerKey === layer.key ? "active" : ""}`}>
                  <input
                    type="radio"
                    name="layer"
                    checked={activeLayerKey === layer.key}
                    onChange={() => setActiveLayerKey(layer.key)}
                  />
                  <span className="layer-name">{layer.name}</span>
                </label>
                {activeLayerKey === layer.key && (
                  <>
                    {layer.description && <p className="layer-desc">{layer.description}</p>}
                    <div style={{ padding: "0 8px 8px 30px" }}>
                      <Legend layer={layer} />
                    </div>
                  </>
                )}
              </div>
            ))}
            <div className="field" style={{ marginTop: 10 }}>
              <label htmlFor="opacity">Overlay opacity {Math.round(opacity * 100)}%</label>
              <input
                id="opacity"
                type="range"
                min={0}
                max={1}
                step={0.05}
                value={opacity}
                onChange={(e) => setOpacity(Number(e.target.value))}
              />
            </div>
          </div>
        )}
      </aside>

      {/* ---------------- centre: map ---------------- */}
      <div className="map-wrap">
        <MapCanvas
          result={result}
          activeLayer={activeLayer}
          opacity={opacity}
          place={selectedPlace}
          onPointClick={onPointClick}
        />
        <div className="timebar">
          <div className="timebar-head">
            <span className="timebar-title">
              {selectedModule?.needs_baseline ? "Baseline year" : "Observation year"}
            </span>
            <span className="timebar-year">
              {selectedModule?.needs_baseline ? baselineYear : end.slice(0, 4)}
            </span>
          </div>
          <input
            type="range"
            min={FIRST_YEAR}
            max={CURRENT_YEAR}
            value={selectedModule?.needs_baseline ? baselineYear : Number(end.slice(0, 4))}
            onChange={(e) => {
              const year = Number(e.target.value);
              if (selectedModule?.needs_baseline) {
                setBaselineYear(Math.min(year, CURRENT_YEAR - 1));
              } else {
                // Keep a whole-season window; a single day rarely has a scene.
                setStart(year === CURRENT_YEAR ? daysAgo(120) : `${year}-05-01`);
                setEnd(year === CURRENT_YEAR ? iso(new Date()) : `${year}-09-30`);
              }
            }}
          />
          <div className="timebar-ticks">
            {Array.from({ length: CURRENT_YEAR - FIRST_YEAR + 1 }, (_, i) => FIRST_YEAR + i)
              .filter((_, i, arr) => arr.length <= 10 || i % 2 === 0)
              .map((year) => (
                <span key={year}>{year}</span>
              ))}
          </div>
        </div>
      </div>

      {/* ---------------- right: results / inspect / ask ---------------- */}
      <aside className="sidebar right">
        <div className="tabs">
          <button className={`tab ${tab === "result" ? "active" : ""}`} onClick={() => setTab("result")}>
            Result
          </button>
          <button className={`tab ${tab === "inspect" ? "active" : ""}`} onClick={() => setTab("inspect")}>
            Inspect
          </button>
          <button className={`tab ${tab === "ask" ? "active" : ""}`} onClick={() => setTab("ask")}>
            Ask AI
          </button>
          <button className={`tab ${tab === "history" ? "active" : ""}`} onClick={() => setTab("history")}>History</button>
        </div>

        <div className="scroll">
          {tab === "result" && (
            <>
              {busy && (
                <div className="section">
                  <div className="loading">
                    <span className="spinner" /> Reading satellite archive… {elapsed}s
                  </div>
                  <p className="hint">
                    Imagery is fetched from the cloud archive and analysed for this request —
                    nothing is precomputed. Typically 5–30 seconds; a large area with a baseline
                    comparison can take a minute or more.
                  </p>
                </div>
              )}
              {error && (
                <div className="section">
                  <p className="error">{error.message}</p>
                  {error.hint && <p className="hint">{error.hint}</p>}
                </div>
              )}
              {result && !busy && <ResultPanel result={result} />}
              {!result && !busy && !error && (
                <div className="section">
                  <h2>No analysis yet</h2>
                  <p className="hint">
                    Choose a module and an area, then run the analysis. Every result is computed from
                    imagery at request time — nothing here is precomputed.
                  </p>
                </div>
              )}
            </>
          )}

          {tab === "inspect" && <Inspector point={point} start={start} end={end} />}
          {tab === "history" && <HistoryPanel key={`${placeId}:${start}:${end}`} place={selectedPlace} start={start} end={end} />}
          {tab === "ask" && <Assistant onFocusPlace={setPlaceId} />}
        </div>
      </aside>
    </div>
  );
}
