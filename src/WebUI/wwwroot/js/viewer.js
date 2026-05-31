// LibreRally Maps — 3D Viewer (Three.js)
// GLB mesh viewer + point cloud renderer

window.libreRallyMaps = window.libreRallyMaps || {};

window.libreRallyMaps.viewer = (function () {
    const PREVIEW_SIZE = 30;
    const Z_UP_TO_Y_UP_ROTATION = -Math.PI / 2;
    let scene, camera, renderer, controls, animationId;
    let currentContainer;
    let currentTileRoot = null;
    let currentPointCloud = null;
    let selectedSegmentId = null;
    let segmentMeshes = new Map();
    let currentContentMetrics = null;
    let referenceCenter = null;
    let referenceScale = null;
    let currentAlignmentOverlay = null;
    let currentAlignmentOverlayGroup = null;
    let currentScanPreviewGroup = null;
    let currentOsmPreviewGroup = null;
    let currentSegmentMetadataById = new Map();
    let currentPreviewAlignment = {
        scan: {
            yawDegrees: 0,
            offsetX: 0,
            offsetY: 0
        },
        osm: {
            yawDegrees: 0,
            offsetX: 0,
            offsetY: 0
        }
    };

    async function init(containerId) {
        const container = document.getElementById(containerId);
        if (!container) return;
        currentContainer = container;

        // Clean up previous
        dispose();
        container.innerHTML = '';

        // Import Three.js modules via importmap
        const THREE = await import('three');
        const { OrbitControls } = await import('three/addons/controls/OrbitControls.js');
        const { GLTFLoader } = await import('three/addons/loaders/GLTFLoader.js');

        // Scene
        scene = new THREE.Scene();
        scene.background = new THREE.Color(0x1a1a2e);
        scene.fog = new THREE.Fog(0x1a1a2e, 50, 500);

        // Camera
        const w = container.clientWidth || 600;
        const h = container.clientHeight || 400;
        camera = new THREE.PerspectiveCamera(45, w / h, 0.1, 1000);
        camera.position.set(50, 40, 60);

        // Renderer
        renderer = new THREE.WebGLRenderer({ antialias: true });
        renderer.setSize(w, h);
        renderer.setPixelRatio(window.devicePixelRatio);
        renderer.shadowMap.enabled = true;
        renderer.domElement.style.display = 'block';
        container.appendChild(renderer.domElement);

        // Controls
        controls = new OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.05;
        controls.target.set(0, 10, 0);
        controls.update();

        // Lighting
        const ambient = new THREE.AmbientLight(0x404060, 0.8);
        scene.add(ambient);
        const sun = new THREE.DirectionalLight(0xffffff, 1.2);
        sun.position.set(100, 200, 50);
        sun.castShadow = true;
        sun.shadow.mapSize.set(1024, 1024);
        scene.add(sun);

        // Ground grid
        const grid = new THREE.GridHelper(100, 20, 0x444466, 0x222244);
        scene.add(grid);

        // Store references
        window._viewerTHREE = THREE;
        window._viewerGLTFLoader = GLTFLoader;
        window._viewerControls = controls;

        animate();
    }

    function animate() {
        animationId = requestAnimationFrame(animate);
        if (renderer && scene && camera && window._viewerControls) {
            window._viewerControls.update();
            renderer.render(scene, camera);
        }
    }

    async function loadGlb(url, segments) {
        if (!scene) return;
        const THREE = window._viewerTHREE;
        if (!THREE || !url) return;

        clearLoadedContent();
        currentSegmentMetadataById = buildSegmentMetadataMap(segments);

        try {
            const GLTFLoader = window._viewerGLTFLoader;
            const loader = new GLTFLoader();
            const gltf = await loader.loadAsync(url);

            const model = gltf.scene;

            // Center and scale
            const box = new THREE.Box3().setFromObject(model);
            const size = box.getSize(new THREE.Vector3());
            
            if (!referenceCenter || !referenceScale) {
                referenceCenter = box.getCenter(new THREE.Vector3());
                const maxDim = Math.max(size.x, size.y, size.z, 1);
                referenceScale = PREVIEW_SIZE / maxDim;
            }
            
            const tileRoot = new THREE.Group();
            tileRoot.userData = { isTileMesh: true };
            tileRoot.rotation.x = Z_UP_TO_Y_UP_ROTATION;
            tileRoot.scale.setScalar(referenceScale);
            tileRoot.position.sub(referenceCenter.clone().multiplyScalar(referenceScale));

            const scanPreviewGroup = new THREE.Group();
            scanPreviewGroup.userData = { isScanPreviewGroup: true };
            const osmPreviewGroup = new THREE.Group();
            osmPreviewGroup.userData = { isOsmPreviewGroup: true };

            tileRoot.add(scanPreviewGroup);
            tileRoot.add(osmPreviewGroup);
            tileRoot.add(model);

            scene.add(tileRoot);
            tileRoot.updateMatrixWorld(true);

            const meshes = [];
            const segPattern = /^segment-(\d+)-/i;
            model.traverse(obj => {
                if (obj.isMesh) {
                    // Tag the mesh with its segment ID from the parent group name
                    // (needed because reparenting breaks named-group traversal later)
                    let parent = obj.parent;
                    while (parent) {
                        const m = segPattern.exec(parent.name || '');
                        if (m) {
                            obj.userData.segmentLocalId = Number.parseInt(m[1], 10);
                            break;
                        }
                        parent = parent.parent;
                    }
                    meshes.push(obj);
                }
            });

            if (meshes.length === 0) {
                scanPreviewGroup.add(model);
            } else {
                meshes.forEach(mesh => {
                    const isOsm = shouldRouteMeshToOsmGroup(mesh);
                    const targetGroup = isOsm
                        ? osmPreviewGroup
                        : scanPreviewGroup;
                    mesh.userData.isOsmFill = isOsm;
                    targetGroup.attach(mesh);
                });
                tileRoot.remove(model);
            }

            currentTileRoot = tileRoot;
            currentScanPreviewGroup = scanPreviewGroup;
            currentOsmPreviewGroup = osmPreviewGroup;
            segmentMeshes = buildSegmentMeshMap(tileRoot);
            currentContentMetrics = {
                horizontalWidth: size.x * referenceScale,
                horizontalDepth: size.y * referenceScale,
                verticalHeight: size.z * referenceScale
            };
            applyPreviewAlignment();
            rebuildAlignmentOverlay();
            applySegmentSelection();
            showWholeTile();

        } catch (err) {
            console.error('Failed to load GLB:', url, err);
        }
    }

    async function loadPointCloud(points, colors, segmentIds) {
        // points: flat array of [x,y,z,x,y,z,...]
        // colors: flat array of [r,g,b,r,g,b,...] (0-1 range or 0-255)
        if (!scene) return;
        const THREE = window._viewerTHREE;
        if (!THREE || !points || points.length < 3) return;

        clearLoadedContent();

        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute('position', new THREE.Float32BufferAttribute(points, 3));

        if (colors && colors.length >= points.length) {
            geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors.slice(), 3));
        }

        const material = new THREE.PointsMaterial({
            size: 0.12,
            vertexColors: !!colors,
            transparent: true,
            opacity: 0.92,
            blending: THREE.NormalBlending,
            depthWrite: true,
            depthTest: true,
            sizeAttenuation: true,
        });

        const pc = new THREE.Points(geometry, material);
        pc.userData = {
            isPointCloud: true,
            baseColors: geometry.getAttribute('color') ? Array.from(geometry.getAttribute('color').array) : null,
            segmentIds: Array.isArray(segmentIds) ? segmentIds.slice() : null
        };

        geometry.computeBoundingBox();
        const box = geometry.boundingBox;
        if (box) {
            const size = box.getSize(new THREE.Vector3());
            if (!referenceCenter || !referenceScale) {
                referenceCenter = box.getCenter(new THREE.Vector3());
                const maxDim = Math.max(size.x, size.y, size.z, 1);
                referenceScale = PREVIEW_SIZE / maxDim;
            }
            const pointCloudRoot = new THREE.Group();
            pointCloudRoot.userData = { isPointCloud: true };
            pointCloudRoot.rotation.x = Z_UP_TO_Y_UP_ROTATION;
            pointCloudRoot.scale.setScalar(referenceScale);
            pointCloudRoot.position.sub(referenceCenter.clone().multiplyScalar(referenceScale));

            const scanPreviewGroup = new THREE.Group();
            scanPreviewGroup.userData = { isScanPreviewGroup: true };
            scanPreviewGroup.add(pc);
            pointCloudRoot.add(scanPreviewGroup);

            if (window._viewerControls) {
                window._viewerControls.target.set(0, size.z * referenceScale * 0.35, 0);
                window._viewerControls.update();
            }

            scene.add(pointCloudRoot);
            currentPointCloud = pc;
            currentScanPreviewGroup = scanPreviewGroup;
            currentOsmPreviewGroup = null;
            currentContentMetrics = {
                horizontalWidth: size.x * referenceScale,
                horizontalDepth: size.y * referenceScale,
                verticalHeight: size.z * referenceScale
            };
            applyPreviewAlignment();
            rebuildAlignmentOverlay();
            applySegmentSelection();
            showWholeTile();
            return;
        }

        const pointCloudRoot = new THREE.Group();
        pointCloudRoot.userData = { isPointCloud: true };
        pointCloudRoot.rotation.x = Z_UP_TO_Y_UP_ROTATION;
        const scanPreviewGroup = new THREE.Group();
        scanPreviewGroup.userData = { isScanPreviewGroup: true };
        scanPreviewGroup.add(pc);
        pointCloudRoot.add(scanPreviewGroup);
        scene.add(pointCloudRoot);
        currentPointCloud = pc;
        currentScanPreviewGroup = scanPreviewGroup;
        currentOsmPreviewGroup = null;
        currentContentMetrics = {
            horizontalWidth: PREVIEW_SIZE,
            horizontalDepth: PREVIEW_SIZE,
            verticalHeight: PREVIEW_SIZE
        };
        applyPreviewAlignment();
        rebuildAlignmentOverlay();
        applySegmentSelection();
        showWholeTile();
    }

    function dispose() {
        if (animationId) cancelAnimationFrame(animationId);
        animationId = null;
        if (renderer) {
            renderer.dispose();
            try {
                if (renderer.domElement && renderer.domElement.parentNode) {
                    renderer.domElement.parentNode.removeChild(renderer.domElement);
                }
            } catch (e) { /* already removed */ }
            renderer = null;
        }
        currentTileRoot = null;
        currentPointCloud = null;
        selectedSegmentId = null;
        segmentMeshes = new Map();
        currentContentMetrics = null;
        referenceCenter = null;
        referenceScale = null;
        currentAlignmentOverlay = null;
        currentAlignmentOverlayGroup = null;
        currentScanPreviewGroup = null;
        currentOsmPreviewGroup = null;
        currentSegmentMetadataById = new Map();
        currentPreviewAlignment = {
            scan: {
                yawDegrees: 0,
                offsetX: 0,
                offsetY: 0
            },
            osm: {
                yawDegrees: 0,
                offsetX: 0,
                offsetY: 0
            }
        };
        scene = null; camera = null; controls = null;
        currentContainer = null;
    }

    function resetReferenceBounds() {
        referenceCenter = null;
        referenceScale = null;
    }

    function resize() {
        if (!currentContainer || !renderer || !camera) return;
        const w = currentContainer.clientWidth;
        const h = currentContainer.clientHeight;
        renderer.setSize(w, h);
        camera.aspect = w / h;
        camera.updateProjectionMatrix();
    }

    function clearLoadedContent() {
        if (!scene) return;

        scene.children
            .filter(c => c.userData?.isTileMesh || c.userData?.isPointCloud || c.userData?.isAlignmentOverlay)
            .forEach(c => scene.remove(c));

        currentTileRoot = null;
        currentPointCloud = null;
        segmentMeshes = new Map();
        currentContentMetrics = null;
        currentAlignmentOverlayGroup = null;
        currentScanPreviewGroup = null;
        currentOsmPreviewGroup = null;
        currentSegmentMetadataById = new Map();
    }

    function setAlignmentOverlay(alignmentDebug) {
        currentAlignmentOverlay = alignmentDebug || null;
        rebuildAlignmentOverlay();
    }

    function setPreviewAlignment(alignment) {
        const scanAlignment = alignment?.scan || alignment || {};
        const osmAlignment = alignment?.osm || {};
        currentPreviewAlignment = {
            scan: {
                yawDegrees: Number(scanAlignment?.yawDegrees) || 0,
                offsetX: Number(scanAlignment?.offsetX) || 0,
                offsetY: Number(scanAlignment?.offsetY) || 0
            },
            osm: {
                yawDegrees: Number(osmAlignment?.yawDegrees) || 0,
                offsetX: Number(osmAlignment?.offsetX) || 0,
                offsetY: Number(osmAlignment?.offsetY) || 0
            }
        };
        applyPreviewAlignment();
    }

    function applyPreviewAlignment() {
        const THREE = window._viewerTHREE;
        if (!THREE) {
            return;
        }

        applyAlignmentToGroup(currentScanPreviewGroup, currentPreviewAlignment.scan, THREE);
        applyAlignmentToGroup(currentOsmPreviewGroup, currentPreviewAlignment.osm, THREE);
    }

    function applyAlignmentToGroup(group, alignment, THREE) {
        if (!group) {
            return;
        }

        group.rotation.set(0, 0, THREE.MathUtils.degToRad(alignment?.yawDegrees || 0));
        group.position.set(
            Number(alignment?.offsetX) || 0,
            Number(alignment?.offsetY) || 0,
            0
        );
    }

    function rebuildAlignmentOverlay() {
        if (!scene) return;

        if (currentAlignmentOverlayGroup) {
            scene.remove(currentAlignmentOverlayGroup);
            currentAlignmentOverlayGroup = null;
        }

        if (!currentAlignmentOverlay || !referenceCenter || !referenceScale) {
            return;
        }

        const vicmapProperties = Array.isArray(currentAlignmentOverlay.vicmapProperties)
            ? currentAlignmentOverlay.vicmapProperties
            : [];
        if (vicmapProperties.length === 0) {
            return;
        }

        const THREE = window._viewerTHREE;
        if (!THREE) return;

        const overlayGroup = new THREE.Group();
        overlayGroup.userData = { isAlignmentOverlay: true };
        overlayGroup.rotation.x = Z_UP_TO_Y_UP_ROTATION;
        overlayGroup.scale.setScalar(referenceScale);
        overlayGroup.position.sub(referenceCenter.clone().multiplyScalar(referenceScale));

        vicmapProperties.forEach(property => {
            if (!Array.isArray(property.localExteriorRings)) return;

            property.localExteriorRings.forEach(ring => {
                if (!Array.isArray(ring) || ring.length < 2) return;

                const points = ring.map(point =>
                    new THREE.Vector3(Number(point.x) || 0, Number(point.y) || 0, 0.08));
                if (points.length < 2) return;

                const geometry = new THREE.BufferGeometry().setFromPoints(points);
                const material = new THREE.LineBasicMaterial({
                    color: 0x4dd0e1,
                    transparent: true,
                    opacity: 0.95
                });
                const line = new THREE.LineLoop(geometry, material);
                overlayGroup.add(line);
            });
        });

        if (overlayGroup.children.length === 0) {
            return;
        }

        scene.add(overlayGroup);
        currentAlignmentOverlayGroup = overlayGroup;
    }

    function buildSegmentMeshMap(root) {
        const segments = new Map();
        const registered = new Set();
        const pattern = /^segment-(\d+)-/i;

        root.traverse(obj => {
            // Check the object name OR its userData for segment ID
            let localSegmentId = null;
            let isOsmFill = false;

            // Try name first (for named group objects that still exist)
            const nameMatch = pattern.exec(obj.name || '');
            if (nameMatch) {
                localSegmentId = Number.parseInt(nameMatch[1], 10);
                if (Number.isNaN(localSegmentId)) localSegmentId = null;
                isOsmFill = isOsmFillMesh(obj);
            }

            // Try userData tag (set during loadGlb before reparenting)
            if (localSegmentId === null && obj.userData?.segmentLocalId) {
                localSegmentId = obj.userData.segmentLocalId;
                isOsmFill = !!obj.userData.isOsmFill;
            }

            if (localSegmentId === null) return;

            // If obj itself is a mesh, register it directly
            if (obj.isMesh && !registered.has(obj.uuid)) {
                registered.add(obj.uuid);
                if (!segments.has(localSegmentId)) {
                    segments.set(localSegmentId, []);
                }

                if (Array.isArray(obj.material)) {
                    obj.material = obj.material.map(cloneViewerMaterial);
                } else if (obj.material) {
                    obj.material = cloneViewerMaterial(obj.material);
                }

                segments.get(localSegmentId).push({ mesh: obj, isOsmFill });
                return;
            }

            // Otherwise traverse children for meshes (legacy path for named groups)
            obj.traverse(child => {
                if (!child.isMesh || registered.has(child.uuid)) return;

                registered.add(child.uuid);
                if (!segments.has(localSegmentId)) {
                    segments.set(localSegmentId, []);
                }

                if (Array.isArray(child.material)) {
                    child.material = child.material.map(cloneViewerMaterial);
                } else if (child.material) {
                    child.material = cloneViewerMaterial(child.material);
                }

                const childIsOsm = isOsmFill || !!child.userData.isOsmFill;
                segments.get(localSegmentId).push({ mesh: child, isOsmFill: childIsOsm });
            });
        });

        return segments;
    }

    function buildSegmentMetadataMap(segments) {
        const metadataById = new Map();
        if (!Array.isArray(segments)) {
            return metadataById;
        }

        segments.forEach(segment => {
            const localSegmentId = Number(segment?.localSegmentId);
            if (Number.isNaN(localSegmentId)) {
                return;
            }

            metadataById.set(localSegmentId, {
                hasOsmFill: !!segment?.hasOsmFill,
                geometrySource: String(segment?.geometrySource || '')
            });
        });

        return metadataById;
    }

    function shouldRouteMeshToOsmGroup(mesh) {
        if (isOsmFillMesh(mesh) || isStandaloneOsmMesh(mesh)) {
            return true;
        }

        const localSegmentId = tryGetLocalSegmentId(mesh);
        if (localSegmentId === null) {
            return false;
        }

        const segment = currentSegmentMetadataById.get(localSegmentId);
        if (!segment) {
            return false;
        }

        return !!segment.hasOsmFill || segment.geometrySource.toLowerCase().includes('osm');
    }

    function tryGetLocalSegmentId(target) {
        const name = typeof target === 'string'
            ? target
            : findNamedAncestor(target, candidate => /^segment-(\d+)-/i.test(candidate));
        const match = /^segment-(\d+)-/i.exec(name || '');
        if (!match) {
            return null;
        }

        const localSegmentId = Number.parseInt(match[1], 10);
        return Number.isNaN(localSegmentId) ? null : localSegmentId;
    }

    function isOsmFillMesh(target) {
        if (typeof target === 'string') {
            return /-osmfill$/i.test(target);
        }

        return findNamedAncestor(target, name => /-osmfill$/i.test(name)) !== null;
    }

    function isStandaloneOsmMesh(target) {
        if (typeof target === 'string') {
            return /^osm-building-/i.test(target);
        }

        return findNamedAncestor(target, name => /^osm-building-/i.test(name)) !== null;
    }

    function findNamedAncestor(obj, predicate) {
        let current = obj || null;
        while (current) {
            const name = String(current.name || '');
            if (predicate(name)) {
                return name;
            }

            current = current.parent || null;
        }

        return null;
    }

    function cloneViewerMaterial(material) {
        const THREE = window._viewerTHREE;
        const cloned = material.clone();
        cloned.transparent = true;
        cloned.opacity = 1;
        cloned.depthWrite = true;
        if (THREE) {
            cloned.side = THREE.DoubleSide;
        }
        // Preserve vertex colors — GLTFLoader may have set this from COLOR_0
        // attributes, but clones and our own opacity tweaks can lose it.
        cloned.vertexColors = true;
        return cloned;
    }

    function selectSegment(localSegmentId) {
        if (localSegmentId === null || localSegmentId === undefined) {
            clearSegmentSelection();
            return;
        }

        selectedSegmentId = Number(localSegmentId);
        applySegmentSelection();
    }

    function clearSegmentSelection() {
        selectedSegmentId = null;
        applySegmentSelection();
    }

    function showWholeTile() {
        if (!camera || !window._viewerControls || !currentContentMetrics) return;

        const horizontalSpan = Math.max(
            currentContentMetrics.horizontalWidth,
            currentContentMetrics.horizontalDepth,
            1
        );
        const verticalHeight = Math.max(currentContentMetrics.verticalHeight, 1);
        const fovRadians = (camera.fov * Math.PI) / 180;
        const fitDistance = (horizontalSpan * 0.65) / Math.tan(fovRadians / 2);
        const distance = Math.max(fitDistance, verticalHeight * 3, 25);

        camera.up.set(0, 0, -1);
        camera.position.set(0, distance, 0.001);
        window._viewerControls.target.set(0, 0, 0);
        camera.lookAt(window._viewerControls.target);
        camera.near = Math.max(distance / 500, 0.1);
        camera.far = Math.max(distance * 10, 1000);
        camera.updateProjectionMatrix();
        window._viewerControls.update();
    }

    function applySegmentSelection() {
        applyMeshSelection();
        applyPointCloudSelection();
    }

    function applyMeshSelection() {
        if (!segmentMeshes || segmentMeshes.size === 0) return;

        const hasActiveSelection = selectedSegmentId !== null;

        segmentMeshes.forEach((entries, localSegmentId) => {
            const isSelected = !hasActiveSelection || localSegmentId === selectedSegmentId;
            const hasOsmFill = entries.some(entry => entry.isOsmFill);

            entries.forEach(entry => {
                let opacity = isSelected ? 1.0 : 0.12;
                let visible = true;
                if (hasOsmFill && !entry.isOsmFill) {
                    opacity = isSelected ? 0.14 : 0.03;
                    if (!hasActiveSelection) {
                        opacity = 0;
                        visible = false;
                    }
                }

                entry.mesh.visible = visible;
                applyMeshOpacity(entry.mesh, opacity);
            });
        });
    }

    function applyMeshOpacity(mesh, opacity) {
        const materials = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
        materials.forEach(material => {
            if (!material) return;
            material.opacity = opacity;
            material.transparent = opacity < 0.999;
            material.depthWrite = opacity >= 0.999;
            material.needsUpdate = true;
        });
    }

    function applyPointCloudSelection() {
        if (!currentPointCloud) return;

        const geometry = currentPointCloud.geometry;
        const colorAttr = geometry.getAttribute('color');
        const baseColors = currentPointCloud.userData?.baseColors;
        const pointSegmentIds = currentPointCloud.userData?.segmentIds;
        if (!colorAttr || !baseColors || !pointSegmentIds || baseColors.length !== colorAttr.array.length) {
            return;
        }

        const dimFactor = 0.15;
        for (let i = 0; i < pointSegmentIds.length; i++) {
            const keep = selectedSegmentId === null || pointSegmentIds[i] === selectedSegmentId;
            const factor = keep ? 1.0 : dimFactor;
            const colorIndex = i * 3;
            colorAttr.array[colorIndex] = baseColors[colorIndex] * factor;
            colorAttr.array[colorIndex + 1] = baseColors[colorIndex + 1] * factor;
            colorAttr.array[colorIndex + 2] = baseColors[colorIndex + 2] * factor;
        }

        colorAttr.needsUpdate = true;
    }

    return {
        init,
        loadGlb,
        loadPointCloud,
        dispose,
        resize,
        selectSegment,
        clearSegmentSelection,
        showWholeTile,
        resetReferenceBounds,
        setAlignmentOverlay,
        setPreviewAlignment
    };
})();
