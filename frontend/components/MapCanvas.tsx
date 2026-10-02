"use client";

import maplibregl from "maplibre-gl";
import { useEffect, useRef, useState } from "react";
import { layerImageUrl, type AnalysisResult, type LayerDescriptor, type Place } from "@/lib/api";

/** Bounding box of Morocco, used as the initial view and the reset target. */
export const MOROCCO_BOUNDS: [[number, number], [number, number]] = [
  [-17.5, 20.5],
  [-0.8, 36.2],
];

/**
 * Authenticated CARTO basemap. The key is read from the env var so it never
 * lands in source control. Without a key the tiles still load but show a
 * watermark; with one they are clean and rate-limit is 5M tiles/month free.
 */
const CARTO_KEY = process.env.NEXT_PUBLIC_CARTO_KEY ?? "";
const keyParam = CARTO_KEY ? `?key=${CARTO_KEY}` : "";

const BASEMAP_STYLE: maplibregl.StyleSpecification = {
  version: 8,
  sources: {
    carto: {
      type: "raster",
      tiles: [
        `https://a.basemaps.cartocdn.com/rastertiles/dark_all/{z}/{x}/{y}@2x.png${keyParam}`,
        `https://b.basemaps.cartocdn.com/rastertiles/dark_all/{z}/{x}/{y}@2x.png${keyParam}`,
        `https://c.basemaps.cartocdn.com/rastertiles/dark_all/{z}/{x}/{y}@2x.png${keyParam}`,
        `https://d.basemaps.cartocdn.com/rastertiles/dark_all/{z}/{x}/{y}@2x.png${keyParam}`,
      ],
      tileSize: 256,
      attribution: "© <a href='https://www.openstreetmap.org/copyright'>OpenStreetMap</a> contributors © <a href='https://carto.com/attributions'>CARTO</a>",
      maxzoom: 19,
    },
  },
  layers: [{ id: "basemap", type: "raster", source: "carto" }],
};

const OVERLAY_SOURCE = "atlasai-overlay";
const OVERLAY_LAYER = "atlasai-overlay-layer";
const AOI_SOURCE = "atlasai-aoi";

interface Props {
  result: AnalysisResult | null;
  activeLayer: LayerDescriptor | null;
  opacity: number;
  place: Place | null;
  onPointClick: (lon: number, lat: number) => void;
}

export default function MapCanvas({ result, activeLayer, opacity, place, onPointClick }: Props) {
  const container = useRef<HTMLDivElement>(null);
  const map = useRef<maplibregl.Map | null>(null);
  const [ready, setReady] = useState(false);
  const overlayOpacity = useRef(opacity);
  overlayOpacity.current = opacity;
  // The click handler is re-created on every render; a ref keeps the map's
  // listener pointed at the current one without re-binding it each time.
  const clickHandler = useRef(onPointClick);
  clickHandler.current = onPointClick;

  useEffect(() => {
    if (!container.current || map.current) return;

    const instance = new maplibregl.Map({
      container: container.current,
      style: BASEMAP_STYLE,
      bounds: MOROCCO_BOUNDS,
      fitBoundsOptions: { padding: 40 },
      attributionControl: { compact: true },
    });

    instance.addControl(new maplibregl.NavigationControl({ showCompass: false }), "top-right");
    instance.addControl(new maplibregl.ScaleControl({ unit: "metric" }), "bottom-left");

    instance.on("load", () => {
      instance.addSource(AOI_SOURCE, {
        type: "geojson",
        data: { type: "FeatureCollection", features: [] },
      });
      instance.addLayer({
        id: "aoi-outline",
        type: "line",
        source: AOI_SOURCE,
        paint: { "line-color": "#4bb3d4", "line-width": 1.5, "line-dasharray": [3, 2] },
      });
      setReady(true);
    });

    instance.on("click", (event) => {
      clickHandler.current(event.lngLat.lng, event.lngLat.lat);
    });
    instance.getCanvas().style.cursor = "crosshair";

    map.current = instance;
    return () => {
      instance.remove();
      map.current = null;
      setReady(false);
    };
  }, []);

  // Swap the analysis overlay whenever the active layer changes.
  useEffect(() => {
    const instance = map.current;
    if (!instance || !ready) return;

    const removeOverlay = () => {
      if (instance.getLayer(OVERLAY_LAYER)) instance.removeLayer(OVERLAY_LAYER);
      if (instance.getSource(OVERLAY_SOURCE)) instance.removeSource(OVERLAY_SOURCE);
    };

    removeOverlay();
    if (!result || !activeLayer?.png_url) return;

    // The backend returns grid corners already ordered TL, TR, BR, BL — the
    // order MapLibre's image source expects — and in Web Mercator, so the
    // image drops onto the map without reprojection.
    instance.addSource(OVERLAY_SOURCE, {
      type: "image",
      url: layerImageUrl(activeLayer.png_url),
      coordinates: result.grid.corners as [
        [number, number],
        [number, number],
        [number, number],
        [number, number],
      ],
    });
    instance.addLayer({
      id: OVERLAY_LAYER,
      type: "raster",
      source: OVERLAY_SOURCE,
      paint: { "raster-opacity": overlayOpacity.current, "raster-fade-duration": 200 },
    });

    return removeOverlay;
  }, [result, activeLayer, ready]);

  useEffect(() => {
    const instance = map.current;
    if (instance?.getLayer(OVERLAY_LAYER)) {
      instance.setPaintProperty(OVERLAY_LAYER, "raster-opacity", opacity);
    }
  }, [opacity]);

  // Outline and fly to the selected area of interest.
  useEffect(() => {
    const instance = map.current;
    if (!instance || !ready || !place) return;

    const [west, south, east, north] = place.bbox;
    const source = instance.getSource(AOI_SOURCE) as maplibregl.GeoJSONSource | undefined;
    source?.setData({
      type: "Feature",
      properties: {},
      geometry: {
        type: "Polygon",
        coordinates: [
          [
            [west, south],
            [east, south],
            [east, north],
            [west, north],
            [west, south],
          ],
        ],
      },
    });
    instance.fitBounds(
      [
        [west, south],
        [east, north],
      ],
      { padding: 60, duration: 900 },
    );
  }, [place, ready]);

  return <div ref={container} className="map" />;
}
