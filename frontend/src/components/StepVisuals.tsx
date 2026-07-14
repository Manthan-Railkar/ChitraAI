/* eslint-disable */
import React, { useRef } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { Float, MeshDistortMaterial, Sphere } from '@react-three/drei';
import * as THREE from 'three';
import poster1 from '@/assets/posters/poster1.png';
import poster2 from '@/assets/posters/poster2.png';
import poster3 from '@/assets/posters/poster3.jpg';
import poster4 from '@/assets/posters/poster4.jpg';
import poster5 from '@/assets/posters/poster5.jpg';
import poster6 from '@/assets/posters/poster6.png';

/* ─── Step 1: Movie Posters Grid ────────────────────────────────── */
export const PosterGrid: React.FC = () => {
  const posters = [poster1, poster2, poster3, poster4, poster5, poster6];
  return (
    <div className="w-full h-full grid grid-cols-3 gap-3 p-2">
      {posters.map((src, i) => (
        <div
          key={i}
          className="relative rounded-xl overflow-hidden border border-white/[0.06]"
          style={{
            animationDelay: `${i * 0.1}s`,
            transform: i % 2 === 0 ? 'translateY(0px)' : 'translateY(8px)',
          }}
        >
          <img src={src} alt="" className="w-full h-full object-cover opacity-80" />
          <div className="absolute inset-0 bg-gradient-to-t from-black/60 to-transparent" />
        </div>
      ))}
    </div>
  );
};

/* ─── Step 2: AI Brain / Neural Network (Three.js) ──────────────── */
function NeuralNodes() {
  const groupRef = useRef<THREE.Group>(null!);
  const nodes = useRef<{ pos: THREE.Vector3; phase: number }[]>(
    Array.from({ length: 18 }, () => ({
      pos: new THREE.Vector3(
        (Math.random() - 0.5) * 5,
        (Math.random() - 0.5) * 4,
        (Math.random() - 0.5) * 2
      ),
      phase: Math.random() * Math.PI * 2,
    }))
  );

  useFrame(({ clock }) => {
    const t = clock.getElapsedTime();
    if (groupRef.current) groupRef.current.rotation.y = t * 0.08;
  });

  return (
    <group ref={groupRef}>
      {nodes.current.map((n, i) => (
        <mesh key={i} position={n.pos}>
          <sphereGeometry args={[0.08, 12, 12]} />
          <meshStandardMaterial color="#e11d48" emissive="#e11d48" emissiveIntensity={0.6} />
        </mesh>
      ))}
      {/* Connection lines */}
      {nodes.current.slice(0, 14).map((n, i) => {
        const next = nodes.current[(i + 3) % nodes.current.length];
        const mid = n.pos.clone().lerp(next.pos, 0.5);
        const dir = next.pos.clone().sub(n.pos);
        const len = dir.length();
        const geometry = new THREE.CylinderGeometry(0.008, 0.008, len, 4);
        const q = new THREE.Quaternion().setFromUnitVectors(
          new THREE.Vector3(0, 1, 0),
          dir.clone().normalize()
        );
        const rot = new THREE.Euler().setFromQuaternion(q);
        return (
          <mesh key={`l-${i}`} position={mid} rotation={rot}>
            <primitive object={geometry} />
            <meshStandardMaterial color="#e11d48" transparent opacity={0.2} />
          </mesh>
        );
      })}
      <pointLight color="#e11d48" intensity={2} position={[0, 0, 2]} />
      <ambientLight intensity={0.3} />
    </group>
  );
}

export const AIBrainViz: React.FC = () => (
  <Canvas camera={{ position: [0, 0, 6], fov: 50 }}>
    <NeuralNodes />
  </Canvas>
);

/* ─── Step 3: Semantic Search Embeddings ────────────────────────── */
function EmbeddingPoints() {
  const groupRef = useRef<THREE.Group>(null!);
  const clusters = [
    { color: '#e11d48', offset: [1.5, 0.5, 0] },
    { color: '#8b5cf6', offset: [-1.2, -0.8, 0.3] },
    { color: '#06b6d4', offset: [0.2, 1.5, -0.5] },
    { color: '#f59e0b', offset: [-0.5, -1.6, 0.2] },
  ];

  useFrame(({ clock }) => {
    const t = clock.getElapsedTime();
    if (groupRef.current) {
      groupRef.current.rotation.y = t * 0.06;
      groupRef.current.rotation.x = Math.sin(t * 0.3) * 0.1;
    }
  });

  return (
    <group ref={groupRef}>
      {clusters.map((c, ci) =>
        Array.from({ length: 12 }).map((_, i) => (
          <mesh
            key={`${ci}-${i}`}
            position={[
              (c.offset[0] as number) + (Math.random() - 0.5) * 1.2,
              (c.offset[1] as number) + (Math.random() - 0.5) * 1.2,
              (c.offset[2] as number) + (Math.random() - 0.5) * 1.2,
            ]}
          >
            <sphereGeometry args={[0.06, 8, 8]} />
            <meshStandardMaterial color={c.color} emissive={c.color} emissiveIntensity={0.5} />
          </mesh>
        ))
      )}
      <ambientLight intensity={0.4} />
      <pointLight color="#8b5cf6" intensity={3} position={[2, 2, 2]} />
    </group>
  );
}

export const EmbeddingsViz: React.FC = () => (
  <Canvas camera={{ position: [0, 0, 7], fov: 45 }}>
    <EmbeddingPoints />
  </Canvas>
);

/* ─── Step 4: Ranking Graph (Three.js bars) ─────────────────────── */
function RankingBars() {
  const groupRef = useRef<THREE.Group>(null!);
  const heights = [3.2, 2.5, 1.9, 1.4, 1.0, 0.7];
  const colors = ['#e11d48', '#f59e0b', '#8b5cf6', '#06b6d4', '#10b981', '#6b7280'];

  useFrame(({ clock }) => {
    const t = clock.getElapsedTime();
    if (groupRef.current) groupRef.current.rotation.y = Math.sin(t * 0.3) * 0.15;
  });

  return (
    <group ref={groupRef} position={[-2.5, -1.5, 0]}>
      {heights.map((h, i) => (
        <mesh key={i} position={[i * 1.0, h / 2, 0]}>
          <boxGeometry args={[0.6, h, 0.6]} />
          <meshStandardMaterial color={colors[i]} emissive={colors[i]} emissiveIntensity={0.3} />
        </mesh>
      ))}
      <ambientLight intensity={0.4} />
      <pointLight color="#e11d48" intensity={2} position={[2, 4, 3]} />
    </group>
  );
}

export const RankingViz: React.FC = () => (
  <Canvas camera={{ position: [0, 2, 8], fov: 50 }}>
    <RankingBars />
  </Canvas>
);

/* ─── Step 5: Recommendation Cards ──────────────────────────────── */
export const RecommendationCards: React.FC = () => {
  const cards = [
    { poster: poster5, title: 'Interstellar', score: '99%', color: '#06b6d4' },
    { poster: poster1, title: 'Breaking Bad', score: '97%', color: '#e11d48' },
    { poster: poster6, title: 'Stranger Things', score: '94%', color: '#8b5cf6' },
  ];
  return (
    <div className="w-full h-full flex items-center justify-center gap-4 p-4">
      {cards.map((card, i) => (
        <div
          key={i}
          className="relative flex-1 h-full max-h-[280px] rounded-2xl overflow-hidden border border-white/[0.08]"
          style={{
            transform: `translateY(${i === 1 ? '-16px' : '0px'}) rotate(${i === 0 ? '-4deg' : i === 2 ? '4deg' : '0deg'})`,
          }}
        >
          <img src={card.poster} alt={card.title} className="w-full h-full object-cover" />
          <div className="absolute inset-0 bg-gradient-to-t from-black/80 to-transparent" />
          <div className="absolute bottom-0 inset-x-0 p-4">
            <div
              className="inline-block text-[10px] font-black rounded px-2 py-0.5 mb-1"
              style={{ background: card.color + '30', color: card.color }}
            >
              {card.score} MATCH
            </div>
            <p className="text-white text-sm font-bold">{card.title}</p>
          </div>
        </div>
      ))}
    </div>
  );
};

/* ─── Step 6: Feedback Loop (distorted sphere) ───────────────────── */
function FeedbackSphere() {
  const meshRef = useRef<THREE.Mesh>(null!);
  useFrame(({ clock }) => {
    if (meshRef.current) {
      meshRef.current.rotation.x = clock.getElapsedTime() * 0.15;
      meshRef.current.rotation.y = clock.getElapsedTime() * 0.2;
    }
  });

  return (
    <Float speed={2} floatIntensity={1}>
      <Sphere ref={meshRef} args={[1.8, 64, 64]}>
        <MeshDistortMaterial
          color="#e11d48"
          distort={0.4}
          speed={2}
          roughness={0.1}
          metalness={0.8}
          transparent
          opacity={0.85}
        />
      </Sphere>
      <pointLight color="#e11d48" intensity={3} position={[3, 3, 3]} />
      <pointLight color="#8b5cf6" intensity={1.5} position={[-3, -2, 2]} />
      <ambientLight intensity={0.2} />
    </Float>
  );
}

export const FeedbackViz: React.FC = () => (
  <Canvas camera={{ position: [0, 0, 5], fov: 50 }}>
    <FeedbackSphere />
  </Canvas>
);
