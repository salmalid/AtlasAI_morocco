"use client";

import type { LayerDescriptor } from "@/lib/api";

/**
 * Colour ramps mirroring `app/eo/render.py`. They are duplicated rather than
 * fetched because the legend must render before (and independently of) the
 * layer image — a legend that arrives late is a legend nobody reads.
 */
const RAMPS: Record<string, string[]> = {
  vegetation: ["#a6611a", "#dfc27d", "#f5f5dc", "#a6d96a", "#1a9641", "#00441b"],
  water: ["#f0e3c8", "#d9f0f7", "#74c2e1", "#2b7bba", "#08306b"],
  moisture: ["#8c510a", "#dfc27d", "#f6e8c3", "#c7eae5", "#35978f", "#01665e"],
  drought: ["#1a9641", "#a6d96a", "#ffffbf", "#fdae61", "#d7191c", "#7f0000"],
  urban: ["#f7f7f7", "#cccccc", "#969696", "#636363", "#d94801", "#7f2704"],
  soil: ["#004529", "#78c679", "#ffffcc", "#fed976", "#e6550d", "#7f2704"],
  burn: ["#0d0887", "#6a00a8", "#b12a90", "#e16462", "#fca636", "#f0f921"],
  turbidity: ["#08306b", "#4292c6", "#9ecae1", "#fdd0a2", "#e6550d", "#7f2704"],
  algae: ["#08306b", "#2b7bba", "#78c679", "#addd8e", "#238443", "#004529"],
  elevation: ["#2b7bba", "#c7e9b4", "#78c679", "#fed976", "#d95f0e", "#8c2d04", "#ffffff"],
  risk: ["#1a9850", "#a6d96a", "#ffffbf", "#fdae61", "#d73027", "#67001f"],
  change: ["#67001f", "#d6604d", "#f7f7f7", "#4393c3", "#053061"],
  gray: ["#000000", "#ffffff"],
};

export default function Legend({ layer }: { layer: LayerDescriptor }) {
  if (layer.kind === "rgb") {
    return <p className="hint">True-colour composite — no value scale.</p>;
  }

  if (layer.kind === "categorical" && layer.classes.length > 0) {
    return (
      <div className="legend">
        {layer.classes.map((entry) => (
          <div className="legend-row" key={entry.id}>
            <span className="swatch" style={{ background: entry.color }} />
            <span>{entry.label}</span>
          </div>
        ))}
      </div>
    );
  }

  if (layer.kind === "mask") {
    return (
      <div className="legend">
        <div className="legend-row">
          <span className="swatch" style={{ background: "#2b7bba" }} />
          <span>Detected</span>
        </div>
      </div>
    );
  }

  const ramp = RAMPS[layer.colormap] ?? RAMPS.gray;
  return (
    <div className="legend">
      <div
        className="ramp"
        style={{ background: `linear-gradient(to right, ${ramp.join(", ")})` }}
      />
      <div className="ramp-labels">
        <span>
          {layer.vmin}
          {layer.unit ? ` ${layer.unit}` : ""}
        </span>
        <span>
          {layer.vmax}
          {layer.unit ? ` ${layer.unit}` : ""}
        </span>
      </div>
    </div>
  );
}
