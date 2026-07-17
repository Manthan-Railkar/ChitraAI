import React, { useRef, useEffect, useCallback } from 'react';
import * as THREE from 'three';

interface PosterSpiralProps {
  /** Array of absolute poster image URLs */
  posterUrls: string[];
  /** Optional CSS class for the container */
  className?: string;
}

/**
 * Infinite spiral loop of movie poster images rendered with Three.js WebGL.
 * Supports mouse wheel scroll, drag rotation, touch controls, and auto-scroll.
 */
export const PosterSpiral: React.FC<PosterSpiralProps> = ({ posterUrls, className = '' }) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const cleanupRef = useRef<(() => void) | null>(null);

  const buildSpiral = useCallback(
    (canvas: HTMLCanvasElement, container: HTMLDivElement, urls: string[]) => {
      // ── State ──
      let scrollOffset = 0;
      let isDragging = false;
      let previousMousePosition = { x: 0, y: 0 };
      const dragRotation = { x: 0, z: 0 };
      const baseRotation = { x: -0.18, z: 0.12 };
      let targetVelocity = 0;
      let currentVelocity = 0;
      let isTouching = false;
      let touchLastY = 0;
      let touchVelocity = 0;
      let animationId = 0;
      let disposed = false;

      const imageRatios: number[] = [];
      let originalPositions: {
        x: number;
        y: number;
        z: number;
        offsetX: number;
        offsetY: number;
        offsetZ: number;
      }[] = [];

      const numberOfImages = urls.length;

      const inertia = {
        friction: 0.94,
        strength: 0.8,
        maxSpeed: 0.05,
        directionSmoothing: 0.92,
        scrollSensitivity: 0.0008,
      };

      const config = {
        imageHeight: 8.5,
        curvature: -0.03,
        gapSize: 0,
        spiralRadius: 4.5,
        spiralTurns: 2.8 + (numberOfImages - 21) * 0.1,
        spiralHeight: 15 + (numberOfImages - 21) * 0.25,
        centerX: 0,
        centerY: 4.38,
        centerZ: 0,
      };

      // ── Three.js Setup ──
      const scene = new THREE.Scene();
      scene.background = new THREE.Color(0x000000);

      const camera = new THREE.PerspectiveCamera(
        50,
        container.clientWidth / container.clientHeight,
        0.1,
        1000
      );
      camera.position.set(0, 3.5, 9);

      const renderer = new THREE.WebGLRenderer({
        canvas,
        antialias: true,
        alpha: false,
        powerPreference: 'high-performance',
        preserveDrawingBuffer: true,
      });
      renderer.setSize(container.clientWidth, container.clientHeight);
      renderer.outputColorSpace = THREE.SRGBColorSpace;
      renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

      const ambient = new THREE.AmbientLight(0xffffff, 0.6);
      scene.add(ambient);
      const mainLight = new THREE.DirectionalLight(0xffffff, 0.9);
      mainLight.position.set(5, 8, 5);
      scene.add(mainLight);

      const tiltGroup = new THREE.Group();
      tiltGroup.rotation.x = baseRotation.x;
      tiltGroup.rotation.z = baseRotation.z;
      scene.add(tiltGroup);

      // ── Shader Material ──
      let shaderMaterial: THREE.ShaderMaterial | null = null;
      let spiralMesh: THREE.Mesh | null = null;

      function updateUVOffset() {
        if (!shaderMaterial) return;
        let offset = scrollOffset;
        while (offset >= 1.0) offset -= 1.0;
        while (offset < 0) offset += 1.0;
        shaderMaterial.uniforms.offset.value = offset;
      }

      function rebuildGeometry() {
        if (!spiralMesh) return;

        const totalSlots = imageRatios.length;
        const widths = imageRatios.map((r) => r * config.imageHeight);
        const totalWidth = widths.reduce((a, b) => a + b, 0);
        const segmentsW = 200 + totalSlots * 20;
        const segmentsH = 24;

        const geometry = new THREE.PlaneGeometry(
          totalWidth,
          config.imageHeight,
          segmentsW,
          segmentsH
        );
        const positions = geometry.attributes.position;
        const uvs = geometry.attributes.uv;

        const origX: number[] = [];
        const origY: number[] = [];
        for (let i = 0; i < positions.count; i++) {
          origX.push(positions.getX(i));
          origY.push(positions.getY(i));
        }

        const cumulative = [0];
        for (let i = 0; i < totalSlots; i++) {
          cumulative.push(cumulative[i] + widths[i] / totalWidth);
        }

        const imageRatio = 1 - config.gapSize;

        for (let i = 0; i < uvs.count; i++) {
          let u = uvs.getX(i);
          u = Math.max(0, Math.min(0.999999, u));

          for (let j = 0; j < totalSlots; j++) {
            if (u >= cumulative[j] && u < cumulative[j + 1]) {
              const localU = (u - cumulative[j]) / (cumulative[j + 1] - cumulative[j]);
              if (localU > imageRatio) {
                uvs.setX(i, cumulative[j + 1] - 0.001);
              } else {
                let scaledU = localU / imageRatio;
                const edgeMargin = 0.001;
                scaledU = Math.max(edgeMargin, Math.min(1 - edgeMargin, scaledU));
                const newU = cumulative[j] + scaledU * (cumulative[j + 1] - cumulative[j]);
                uvs.setX(i, newU);
              }
              break;
            }
          }
        }

        // Curvature
        for (let i = 0; i < positions.count; i++) {
          const x = positions.getX(i);
          const nx = x / (totalWidth / 2);
          const curve = config.curvature * 0.4 * (nx * nx - 1);
          positions.setZ(i, -curve);
        }

        // Spiral wrapping
        originalPositions = [];
        for (let i = 0; i < positions.count; i++) {
          const x = origX[i];
          const y = origY[i];
          let t = (x + totalWidth / 2) / totalWidth;
          t = Math.max(0, Math.min(1, t));

          const angle = t * Math.PI * 2 * config.spiralTurns;
          const radius = config.spiralRadius * (1 - t * 0.12);
          let px = Math.sin(angle) * radius;
          let pz = Math.cos(angle) * radius;
          let py = (t - 0.5) * config.spiralHeight + y * 0.35;

          const op = {
            x: px,
            y: py,
            z: pz,
            offsetX: (Math.random() - 0.5) * 0.001,
            offsetY: (Math.random() - 0.5) * 0.001,
            offsetZ: (Math.random() - 0.5) * 0.001,
          };
          originalPositions.push(op);

          px += op.offsetX;
          py += op.offsetY;
          pz += op.offsetZ;

          positions.setXYZ(i, px, py, pz);
        }

        geometry.computeVertexNormals();

        const oldGeo = spiralMesh.geometry;
        spiralMesh.geometry = geometry;
        if (oldGeo) oldGeo.dispose();

        if (shaderMaterial) {
          shaderMaterial.uniforms.gap.value = config.gapSize;
        }
      }

      // ── Create Master Texture ──
      function createMasterTexture(): Promise<THREE.CanvasTexture> {
        return new Promise((resolve) => {
          const offscreen = document.createElement('canvas');
          const ctx = offscreen.getContext('2d')!;
          const baseHeight = 500;
          let loaded = 0;
          const images: { img: HTMLImageElement; width: number; height: number }[] = [];

          urls.forEach((url, idx) => {
            const img = new Image();
            img.crossOrigin = '';

            img.onload = () => {
              const ratio = img.naturalWidth / img.naturalHeight;
              imageRatios[idx] = ratio;
              images[idx] = { img, width: baseHeight * ratio, height: baseHeight };
              loaded++;
              if (loaded === numberOfImages) finalize();
            };

            img.onerror = () => {
              imageRatios[idx] = 0.67; // poster aspect fallback
              // Create a dark placeholder
              const placeholder = document.createElement('canvas');
              placeholder.width = baseHeight * 0.67;
              placeholder.height = baseHeight;
              const pCtx = placeholder.getContext('2d')!;
              pCtx.fillStyle = '#111';
              pCtx.fillRect(0, 0, placeholder.width, placeholder.height);
              pCtx.fillStyle = '#333';
              pCtx.font = '24px sans-serif';
              pCtx.textAlign = 'center';
              pCtx.fillText('🎬', placeholder.width / 2, placeholder.height / 2);

              const placeholderImg = new Image();
              placeholderImg.src = placeholder.toDataURL();
              placeholderImg.onload = () => {
                images[idx] = { img: placeholderImg, width: baseHeight * 0.67, height: baseHeight };
                loaded++;
                if (loaded === numberOfImages) finalize();
              };
            };
            img.decoding = 'async';
            img.loading = 'eager';
            img.src = url;
          });

          function finalize() {
            const totalWidth = images.reduce((sum, i) => sum + i.width, 0);
            offscreen.width = totalWidth;
            offscreen.height = baseHeight;
            ctx.fillStyle = '#000000';
            ctx.fillRect(0, 0, offscreen.width, offscreen.height);

            let offsetX = 0;
            images.forEach((data) => {
              if (data && data.img) {
                ctx.drawImage(data.img, offsetX, 0, data.width, data.height);
              }
              offsetX += data.width;
            });

            const tex = new THREE.CanvasTexture(offscreen);
            tex.needsUpdate = true;
            tex.wrapS = THREE.RepeatWrapping;
            tex.wrapT = THREE.ClampToEdgeWrapping;
            tex.minFilter = THREE.LinearFilter;
            tex.magFilter = THREE.LinearFilter;
            tex.generateMipmaps = false;
            resolve(tex);
          }
        });
      }

      // ── Interaction handlers ──
      let acceleration = 0;

      const onWheel = (e: WheelEvent) => {
        e.preventDefault();
        const rawDelta = e.deltaY * inertia.scrollSensitivity * inertia.strength;
        let deltaAccel = rawDelta - acceleration;
        deltaAccel = Math.max(-0.015, Math.min(0.015, deltaAccel));
        acceleration += deltaAccel;
        acceleration = Math.max(-0.03, Math.min(0.03, acceleration));
        targetVelocity =
          targetVelocity * inertia.directionSmoothing +
          acceleration * (1 - inertia.directionSmoothing);
        targetVelocity = Math.max(-inertia.maxSpeed, Math.min(inertia.maxSpeed, targetVelocity));
      };

      const onMouseDown = (e: MouseEvent) => {
        isDragging = true;
        previousMousePosition = { x: e.clientX, y: e.clientY };
        container.style.cursor = 'grabbing';
        e.preventDefault();
      };

      const onMouseMove = (e: MouseEvent) => {
        if (!isDragging) return;
        const dx = e.clientX - previousMousePosition.x;
        const dy = e.clientY - previousMousePosition.y;
        dragRotation.z += dx * 0.002;
        dragRotation.x -= dy * 0.002;
        dragRotation.x = Math.max(-0.35, Math.min(0.35, dragRotation.x));
        dragRotation.z = Math.max(-0.35, Math.min(0.35, dragRotation.z));
        tiltGroup.rotation.x = baseRotation.x + dragRotation.x;
        tiltGroup.rotation.z = baseRotation.z + dragRotation.z;
        previousMousePosition = { x: e.clientX, y: e.clientY };
      };

      const onMouseUp = () => {
        isDragging = false;
        container.style.cursor = 'grab';
      };

      const onTouchStart = (e: TouchEvent) => {
        e.preventDefault();
        isTouching = true;
        touchLastY = e.touches[0].clientY;
        touchVelocity = 0;
        container.style.cursor = 'grabbing';
      };

      const onTouchMove = (e: TouchEvent) => {
        if (!isTouching) return;
        e.preventDefault();
        const currentY = e.touches[0].clientY;
        const deltaY = currentY - touchLastY;
        const rawVelocity = deltaY * inertia.scrollSensitivity * inertia.strength * 0.5;
        touchVelocity = touchVelocity * 0.7 + rawVelocity * 0.3;
        scrollOffset += deltaY * inertia.scrollSensitivity * inertia.strength * 0.8;
        updateUVOffset();
        touchLastY = currentY;
      };

      const onTouchEnd = (e: TouchEvent) => {
        e.preventDefault();
        isTouching = false;
        container.style.cursor = 'grab';
        if (Math.abs(touchVelocity) > 0.001) {
          targetVelocity = Math.max(
            -inertia.maxSpeed * 1.5,
            Math.min(inertia.maxSpeed * 1.5, touchVelocity * 1.2)
          );
        }
        touchVelocity = 0;
      };

      const onResize = () => {
        if (disposed) return;
        camera.aspect = container.clientWidth / container.clientHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(container.clientWidth, container.clientHeight);
      };

      // ── Animation Loop ──
      function animate() {
        if (disposed) return;
        animationId = requestAnimationFrame(animate);

        // Inertia
        targetVelocity *= inertia.friction;
        currentVelocity = currentVelocity * 0.85 + targetVelocity * 0.15;

        if (Math.abs(currentVelocity) > 0.0001) {
          scrollOffset += currentVelocity;
          updateUVOffset();
        } else {
          currentVelocity = 0;
          targetVelocity = 0;
          acceleration = 0;
        }

        // Touch inertia
        if (!isTouching) {
          touchVelocity *= 0.95;
          if (Math.abs(touchVelocity) > 0.0001) {
            scrollOffset += touchVelocity * 0.5;
            updateUVOffset();
          } else {
            touchVelocity = 0;
          }
        }

        // Subtle auto-scroll when user is not interacting
        if (
          !isDragging &&
          !isTouching &&
          Math.abs(currentVelocity) < 0.0001 &&
          Math.abs(touchVelocity) < 0.0001
        ) {
          scrollOffset += 0.00015;
          updateUVOffset();
        }

        renderer.render(scene, camera);
      }

      // ── Init ──
      async function init() {
        const texture = await createMasterTexture();
        if (disposed) {
          texture.dispose();
          return;
        }

        shaderMaterial = new THREE.ShaderMaterial({
          uniforms: {
            map: { value: texture },
            gap: { value: config.gapSize },
            offset: { value: 0.0 },
          },
          vertexShader: `
          varying vec2 vUv;
          void main() {
            vUv = uv;
            gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
          }
        `,
          fragmentShader: `
          uniform sampler2D map;
          uniform float gap;
          uniform float offset;
          varying vec2 vUv;
          void main() {
            float u = vUv.x + offset;
            if (u >= 1.0) u -= 1.0;
            if (u < 0.0) u += 1.0;
            vec4 color = texture2D(map, vec2(u, vUv.y));
            gl_FragColor = color;
          }
        `,
          transparent: false,
          depthTest: true,
          depthWrite: true,
          side: THREE.DoubleSide,
        });

        spiralMesh = new THREE.Mesh(new THREE.BufferGeometry(), shaderMaterial);
        spiralMesh.position.set(config.centerX, config.centerY, config.centerZ);
        spiralMesh.rotation.x = 0.35;
        tiltGroup.add(spiralMesh);

        rebuildGeometry();
        updateUVOffset();
        renderer.render(scene, camera);

        // Attach event listeners
        container.addEventListener('wheel', onWheel, { passive: false });
        container.addEventListener('mousedown', onMouseDown);
        window.addEventListener('mousemove', onMouseMove);
        window.addEventListener('mouseup', onMouseUp);
        container.addEventListener('touchstart', onTouchStart, { passive: false });
        container.addEventListener('touchmove', onTouchMove, { passive: false });
        container.addEventListener('touchend', onTouchEnd);
        window.addEventListener('resize', onResize);

        container.style.cursor = 'grab';

        animate();
      }

      init();

      // ── Cleanup ──
      return () => {
        disposed = true;
        cancelAnimationFrame(animationId);

        container.removeEventListener('wheel', onWheel);
        container.removeEventListener('mousedown', onMouseDown);
        window.removeEventListener('mousemove', onMouseMove);
        window.removeEventListener('mouseup', onMouseUp);
        container.removeEventListener('touchstart', onTouchStart);
        container.removeEventListener('touchmove', onTouchMove);
        container.removeEventListener('touchend', onTouchEnd);
        window.removeEventListener('resize', onResize);

        if (spiralMesh) {
          spiralMesh.geometry.dispose();
          tiltGroup.remove(spiralMesh);
        }
        if (shaderMaterial) {
          if (shaderMaterial.uniforms.map.value) {
            (shaderMaterial.uniforms.map.value as THREE.Texture).dispose();
          }
          shaderMaterial.dispose();
        }
        renderer.dispose();
      };
    },
    []
  );

  useEffect(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container || posterUrls.length === 0) return;

    // Clean up previous instance if poster URLs changed
    if (cleanupRef.current) {
      cleanupRef.current();
      cleanupRef.current = null;
    }

    const validUrls = posterUrls.filter((url) => typeof url === 'string' && url.startsWith('http'));

    cleanupRef.current = buildSpiral(canvas, container, validUrls);

    return () => {
      if (cleanupRef.current) {
        cleanupRef.current();
        cleanupRef.current = null;
      }
    };
  }, [posterUrls, buildSpiral]);

  return (
    <div ref={containerRef} className={`relative w-full h-full ${className}`}>
      <canvas ref={canvasRef} className="absolute inset-0 w-full h-full outline-none" />
    </div>
  );
};

export default PosterSpiral;
