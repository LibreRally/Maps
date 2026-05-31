// LibreRally Maps — Leaflet tile map (called from Blazor OnAfterRenderAsync)
window.libreRallyMaps = window.libreRallyMaps || {};
window.libreRallyMaps._tileMap = null;
window.libreRallyMaps._tileMarkers = [];
window.libreRallyMaps._tileMapContainer = null;
window.libreRallyMaps._alignmentMap = null;
window.libreRallyMaps._alignmentMapContainer = null;
window.libreRallyMaps._alignmentTileLayer = null;
window.libreRallyMaps._alignmentOsmLayers = [];
window.libreRallyMaps._alignmentSegmentLayers = new Map();

window.libreRallyMaps.disposeTileMap = function () {
    var map = window.libreRallyMaps._tileMap;
    if (map) {
        map.remove();
    }

    window.libreRallyMaps._tileMap = null;
    window.libreRallyMaps._tileMarkers = [];
    window.libreRallyMaps._tileMapContainer = null;
};

window.libreRallyMaps.initTileMap = function (centerLat, centerLon, zoom, markers) {
    if (typeof L === 'undefined') { console.error('Leaflet not loaded'); return; }

    var el = document.getElementById('tiles-map');
    if (!el) return;

    var map = window.libreRallyMaps._tileMap;
    var container = window.libreRallyMaps._tileMapContainer;

    if (map && container !== el) {
        window.libreRallyMaps.disposeTileMap();
        map = null;
    }

    if (!map) {
        map = L.map('tiles-map').setView([centerLat, centerLon], zoom);
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '&copy; OpenStreetMap', maxZoom: 19
        }).addTo(map);
        window.libreRallyMaps._tileMap = map;
        window.libreRallyMaps._tileMapContainer = el;
    }

    window.libreRallyMaps._tileMarkers.forEach(function (m) { map.removeLayer(m); });
    window.libreRallyMaps._tileMarkers = [];

    if (markers && markers.length) {
        markers.forEach(function (m) {
            var marker = L.circleMarker([m.lat, m.lon], {
                radius: 16, fillColor: '#3388ff', fillOpacity: 0.3,
                color: '#3388ff', weight: 1
            }).addTo(map).bindPopup(m.label);
            window.libreRallyMaps._tileMarkers.push(marker);
        });
    }

    setTimeout(function () { map.invalidateSize(); }, 200);
};

window.libreRallyMaps.disposeAlignmentMap = function () {
    var map = window.libreRallyMaps._alignmentMap;
    if (map) {
        map.remove();
    }

    window.libreRallyMaps._alignmentMap = null;
    window.libreRallyMaps._alignmentMapContainer = null;
    window.libreRallyMaps._alignmentTileLayer = null;
    window.libreRallyMaps._alignmentOsmLayers = [];
    window.libreRallyMaps._alignmentSegmentLayers = new Map();
};

window.libreRallyMaps.initAlignmentMap = function (containerId, payload) {
    if (typeof L === 'undefined') { console.error('Leaflet not loaded'); return; }

    var el = document.getElementById(containerId);
    if (!el || !payload) return;

    var map = window.libreRallyMaps._alignmentMap;
    var container = window.libreRallyMaps._alignmentMapContainer;

    if (map && container !== el) {
        window.libreRallyMaps.disposeAlignmentMap();
        map = null;
    }

    if (!map) {
        map = L.map(containerId).setView([payload.centerLat, payload.centerLon], Math.max(payload.zoomLevel, 18));
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '&copy; OpenStreetMap',
            maxZoom: 19
        }).addTo(map);
        window.libreRallyMaps._alignmentMap = map;
        window.libreRallyMaps._alignmentMapContainer = el;
    }

    if (window.libreRallyMaps._alignmentTileLayer) {
        map.removeLayer(window.libreRallyMaps._alignmentTileLayer);
        window.libreRallyMaps._alignmentTileLayer = null;
    }

    window.libreRallyMaps._alignmentOsmLayers.forEach(function (layer) { map.removeLayer(layer); });
    window.libreRallyMaps._alignmentOsmLayers = [];

    window.libreRallyMaps._alignmentSegmentLayers.forEach(function (entry) {
        if (entry.boundsLayer) map.removeLayer(entry.boundsLayer);
        if (entry.markerLayer) map.removeLayer(entry.markerLayer);
    });
    window.libreRallyMaps._alignmentSegmentLayers = new Map();

    var boundsLayers = [];

    if (payload.tileBoundsGeoJson) {
        var tileLayer = L.geoJSON(JSON.parse(payload.tileBoundsGeoJson), {
            style: {
                color: '#3388ff',
                weight: 2,
                dashArray: '6,4',
                fillOpacity: 0
            }
        }).bindPopup('Tile bounds');
        tileLayer.addTo(map);
        window.libreRallyMaps._alignmentTileLayer = tileLayer;
        boundsLayers.push(tileLayer);
    }

    (payload.osmBuildings || []).forEach(function (building) {
        if (!building.geometryJson) return;

        var isMatched = !!building.isMatched;
        var layer = L.geoJSON(JSON.parse(building.geometryJson), {
            style: {
                color: isMatched ? '#c07b39' : '#2b8a8a',
                weight: isMatched ? 2 : 1,
                dashArray: isMatched ? null : '4,4',
                fillColor: isMatched ? '#e6c39a' : '#7ed6d4',
                fillOpacity: isMatched ? 0.18 : 0.06
            }
        }).bindPopup((building.name || building.featureType || 'building') + ' (OSM #' + building.osmId + ', ' + (isMatched ? 'matched' : 'unmatched') + ')');
        layer.addTo(map);
        window.libreRallyMaps._alignmentOsmLayers.push(layer);
        boundsLayers.push(layer);
    });

    (payload.segments || []).forEach(function (segment) {
        var baseColor = segment.hasOsmFill ? '#2f9e44' : '#7c4dff';
        var boundsLayer = null;
        if (segment.boundsGeoJson) {
            boundsLayer = L.geoJSON(JSON.parse(segment.boundsGeoJson), {
                style: {
                    color: baseColor,
                    weight: 2,
                    fillColor: baseColor,
                    fillOpacity: 0.08
                }
            }).bindPopup('#' + segment.localSegmentId + ' ' + segment.label + ' [' + segment.geometrySource + ']');
            boundsLayer.addTo(map);
            boundsLayers.push(boundsLayer);
        }

        var markerLayer = null;
        if (segment.centroidLat !== null && segment.centroidLat !== undefined &&
            segment.centroidLon !== null && segment.centroidLon !== undefined) {
            markerLayer = L.circleMarker([segment.centroidLat, segment.centroidLon], {
                radius: 4,
                color: baseColor,
                weight: 1,
                fillColor: baseColor,
                fillOpacity: 0.85
            }).bindPopup('#' + segment.localSegmentId + ' ' + segment.label + ' (' + segment.osmMatchStatus + ')');
            markerLayer.addTo(map);
            boundsLayers.push(markerLayer);
        }

        window.libreRallyMaps._alignmentSegmentLayers.set(segment.localSegmentId, {
            boundsLayer: boundsLayer,
            markerLayer: markerLayer,
            baseColor: baseColor
        });
    });

    if (boundsLayers.length) {
        var featureGroup = L.featureGroup(boundsLayers);
        map.fitBounds(featureGroup.getBounds().pad(0.15));
    } else {
        map.setView([payload.centerLat, payload.centerLon], Math.max(payload.zoomLevel, 18));
    }

    setTimeout(function () { map.invalidateSize(); }, 200);
};

window.libreRallyMaps.clearAlignmentHighlight = function () {
    window.libreRallyMaps._alignmentSegmentLayers.forEach(function (entry) {
        if (entry.boundsLayer) {
            entry.boundsLayer.setStyle({
                color: entry.baseColor,
                weight: 2,
                fillColor: entry.baseColor,
                fillOpacity: 0.08
            });
        }

        if (entry.markerLayer) {
            entry.markerLayer.setStyle({
                radius: 4,
                color: entry.baseColor,
                fillColor: entry.baseColor,
                fillOpacity: 0.85
            });
        }
    });
};

window.libreRallyMaps.highlightAlignmentSegment = function (localSegmentId) {
    window.libreRallyMaps.clearAlignmentHighlight();

    var map = window.libreRallyMaps._alignmentMap;
    var entry = window.libreRallyMaps._alignmentSegmentLayers.get(Number(localSegmentId));
    if (!map || !entry) return;

    if (entry.boundsLayer) {
        entry.boundsLayer.setStyle({
            color: '#ff1744',
            weight: 3,
            fillColor: '#ff1744',
            fillOpacity: 0.16
        });
        if (entry.boundsLayer.getBounds) {
            map.fitBounds(entry.boundsLayer.getBounds().pad(0.4));
        }
    }

    if (entry.markerLayer) {
        entry.markerLayer.setStyle({
            radius: 6,
            color: '#ff1744',
            fillColor: '#ff1744',
            fillOpacity: 1
        });
        map.panTo(entry.markerLayer.getLatLng());
    }
};
