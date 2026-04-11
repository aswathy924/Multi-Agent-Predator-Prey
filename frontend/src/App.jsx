import React, { useEffect, useRef, useState, useCallback } from 'react';
import './index.css';

const API_URL = 'http://localhost:8000';

// Map Library
const MAP_TEMPLATES = {
  "empty": [],
  "the_cross": [
    { x: 0.45, y: 0.1, w: 0.1, h: 0.8 },
    { x: 0.1, y: 0.45, w: 0.8, h: 0.1 }
  ],
  "the_corridors": [
    { x: 0.25, y: 0.0, w: 0.05, h: 0.7 },
    { x: 0.7, y: 0.3, w: 0.05, h: 0.7 }
  ]
};

function App() {
  const [scene, setScene] = useState('menu'); // menu, setup, arena

  // Game Config
  const [difficulty, setDifficulty] = useState('medium');
  const [numPredators, setNumPredators] = useState(3);
  const [mapSelection, setMapSelection] = useState('empty');

  // States
  const [metrics, setMetrics] = useState({ training: false, play_mode: false, rewards: [], recent_loss: [], algorithm: 'dqn', predator_epsilon: 1 });
  const [gameState, setGameState] = useState(null);
  const [leaderboard, setLeaderboard] = useState([]);

  const canvasRef = useRef(null);
  const [obstacles, setObstacles] = useState([]);

  // Drawing Walls
  const [drawing, setDrawing] = useState(false);
  const [drawStart, setDrawStart] = useState({ x: 0, y: 0 });
  const [tempRect, setTempRect] = useState(null);

  // Physics Toggles
  const currentAction = useRef(0);
  const [isGameOver, setIsGameOver] = useState(false);
  const survivalRecord = useRef(0);
  const prevWasRunning = useRef(false);

  // Heatmap State
  const [showHeatmap, setShowHeatmap] = useState(false);
  const [heatmapData, setHeatmapData] = useState([]);
  const [heatmapGrid, setHeatmapGrid] = useState(20);

  // --- API Calls ---
  const fetchLeaderboard = async () => {
    try {
      const res = await fetch(`${API_URL}/leaderboard`);
      const data = await res.json();
      setLeaderboard(data.leaderboard || []);
    } catch (e) { }
  };

  useEffect(() => { if (scene === 'menu') fetchLeaderboard(); }, [scene]);

  const submitScore = async (e) => {
    e.preventDefault();
    const name = e.target.playerName.value || "Anonymous";
    await fetch(`${API_URL}/leaderboard/submit`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, score: survivalRecord.current })
    });
    setScene('menu');
  };

  const sendAction = useCallback(async (actionCode) => {
    if (scene !== 'arena' || isGameOver) return;
    if (currentAction.current !== actionCode) {
      currentAction.current = actionCode;
      fetch(`${API_URL}/action`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: actionCode, play_mode: true })
      }).catch(e => { });
    }
  }, [scene, isGameOver]);

  // Keyboard
  useEffect(() => {
    if (!metrics.play_mode) return;
    const handleKeyDown = (e) => {
      switch (e.key.toLowerCase()) {
        case 'w': sendAction(2); break;
        case 's': sendAction(1); break;
        case 'a': sendAction(3); break;
        case 'd': sendAction(4); break;
        case 'h': setShowHeatmap(h => !h); break; // Toggle Heatmap on H
      }
    };
    const handleKeyUp = (e) => {
      const k = e.key.toLowerCase();
      if (['w', 's', 'a', 'd'].includes(k)) sendAction(0);
    };
    window.addEventListener('keydown', handleKeyDown);
    window.addEventListener('keyup', handleKeyUp);
    return () => { window.removeEventListener('keydown', handleKeyDown); window.removeEventListener('keyup', handleKeyUp); };
  }, [sendAction, metrics.play_mode]);

  // Polling Game State
  useEffect(() => {
    if (scene !== 'arena') return;
    let animFrame;
    const fetchState = async () => {
      try {
        const res = await fetch(`${API_URL}/state`);
        const data = await res.json();
        if (data) setGameState(data);
      } catch (e) { }
      setTimeout(() => { animFrame = requestAnimationFrame(fetchState); }, 30);
    };
    fetchState();
    return () => cancelAnimationFrame(animFrame);
  }, [scene]);

  useEffect(() => {
    if (scene !== 'arena') return;
    const interval = setInterval(async () => {
      try {
        const res = await fetch(`${API_URL}/metrics`);
        const data = await res.json();
        setMetrics(data);
      } catch (e) { }
    }, 1000);
    return () => clearInterval(interval);
  }, [scene]);

  // Polling Heatmap
  useEffect(() => {
    if (scene !== 'arena' || !showHeatmap) {
      setHeatmapData([]);
      return;
    }
    const hInterval = setInterval(async () => {
      try {
        const res = await fetch(`${API_URL}/heatmap`);
        const data = await res.json();
        setHeatmapData(data.heatmap);
        setHeatmapGrid(data.grid_dim);
      } catch (e) { }
    }, 1000);
    return () => clearInterval(hInterval);
  }, [scene, showHeatmap]);


  const startGame = async () => {
    let algo = "dqn";
    if (difficulty === "hard") algo = "d3qn";
    if (difficulty === "easy") algo = "dqn"; // High espilon mapped in backend

    await fetch(`${API_URL}/init_game`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ num_predators: parseInt(numPredators), algorithm: algo, difficulty: difficulty })
    });

    const activeObs = MAP_TEMPLATES[mapSelection] || [];
    setObstacles(activeObs);
    await fetch(`${API_URL}/settings`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ obstacles: activeObs })
    });

    setScene('arena');
    setIsGameOver(false);
  };

  const startArenaEngine = async () => {
    await fetch(`${API_URL}/train/start`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ play_mode: true })
    });
    setIsGameOver(false);
  };

  // Wall Drawing
  const handleMouseDown = (e) => {
    if (metrics.training || isGameOver) return;
    const rect = canvasRef.current.getBoundingClientRect();
    setDrawStart({ x: e.clientX - rect.left, y: e.clientY - rect.top });
    setDrawing(true);
  };
  const handleMouseMove = (e) => {
    if (!drawing) return;
    const rect = canvasRef.current.getBoundingClientRect();
    setTempRect({
      x: Math.min(drawStart.x, e.clientX - rect.left),
      y: Math.min(drawStart.y, e.clientY - rect.top),
      w: Math.abs(e.clientX - rect.left - drawStart.x),
      h: Math.abs(e.clientY - rect.top - drawStart.y)
    });
  };
  const handleMouseUp = () => {
    if (drawing && tempRect && tempRect.w > 10 && tempRect.h > 10) {
      const s = canvasRef.current.width / gameState.grid_size;
      const newObs = { x: tempRect.x / s, y: tempRect.y / s, w: tempRect.w / s, h: tempRect.h / s };
      const updated = [...obstacles, newObs];
      setObstacles(updated);
      fetch(`${API_URL}/settings`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ obstacles: updated }) });
    }
    setDrawing(false);
    setTempRect(null);
  };


  // Canvas Rendering Loop
  useEffect(() => {
    if (scene !== 'arena' || !gameState || !canvasRef.current) return;
    const ctx = canvasRef.current.getContext('2d');
    const w = canvasRef.current.width, h = canvasRef.current.height;

    ctx.fillStyle = 'rgba(10, 10, 15, 0.4)';
    ctx.fillRect(0, 0, w, h);

    const s = w / gameState.grid_size;

    // HEATMAP RENDER
    if (showHeatmap && heatmapData.length > 0) {
      const cellW = w / heatmapGrid;
      const cellH = h / heatmapGrid;
      const flatData = heatmapData;
      const maxV = Math.max(...flatData) || 1;
      const minV = Math.min(...flatData) || -1;

      flatData.forEach((val, idx) => {
        const gx = (idx % heatmapGrid) * cellW;
        const gy = Math.floor(idx / heatmapGrid) * cellH;
        let norm = (val - minV) / ((maxV - minV) + 1e-5);
        // Color map: 0 is dark blue, 1 is bright red
        ctx.fillStyle = `hsla(${(1 - norm) * 240}, 100%, 50%, 0.3)`;
        ctx.fillRect(gx, gy, cellW, cellH);
      });
    }

    gameState.obstacles.forEach(o => {
      ctx.fillStyle = 'rgba(60, 60, 80, 0.6)';
      ctx.strokeStyle = '#00ffcc';
      ctx.lineWidth = 1;
      ctx.fillRect(o.x * s, o.y * s, o.w * s, o.h * s);
      ctx.strokeRect(o.x * s, o.y * s, o.w * s, o.h * s);
    });

    if (tempRect) {
      ctx.fillStyle = 'rgba(0, 255, 204, 0.2)';
      ctx.fillRect(tempRect.x, tempRect.y, tempRect.w, tempRect.h);
    }

    let caught = false;
    gameState.predators.forEach(p => {
      const dist = Math.sqrt((p[0] - gameState.prey[0]) ** 2 + (p[1] - gameState.prey[1]) ** 2);
      if (dist < gameState.catch_radius) caught = true;

      ctx.strokeStyle = 'rgba(255, 51, 102, 0.2)';
      ctx.beginPath(); ctx.arc(p[0] * s, p[1] * s, gameState.catch_radius * s, 0, Math.PI * 2); ctx.stroke();

      ctx.fillStyle = '#ff3366';
      ctx.shadowBlur = 15; ctx.shadowColor = '#ff3366';
      ctx.beginPath(); ctx.arc(p[0] * s, p[1] * s, 10, 0, Math.PI * 2); ctx.fill();
    });

    ctx.fillStyle = '#00ffcc';
    ctx.shadowBlur = 20; ctx.shadowColor = '#00ffcc';
    ctx.beginPath(); ctx.arc(gameState.prey[0] * s, gameState.prey[1] * s, 8, 0, Math.PI * 2); ctx.fill();
    ctx.shadowBlur = 0;

    if (prevWasRunning.current && !metrics.training && metrics.play_mode && caught) {
      survivalRecord.current = gameState.survival_time;
      setIsGameOver(true);
    }
    prevWasRunning.current = metrics.training;

  }, [gameState, scene, tempRect, metrics, showHeatmap, heatmapData, heatmapGrid]);


  // VIEW LOGIC
  if (scene === 'menu') {
    return (
      <div className="menu-scene">
        <div className="title-section">
          <div className="subtitle">TACTICAL AI SURVIVAL SIMULATOR</div>
          <h1 className="title">NEURAL<br />PREDATORS</h1>
          <div className="menu-buttons" style={{ marginTop: '40px' }}>
            <button className="btn" onClick={() => setScene('setup')}>Start Pursuit</button>
            <button className="btn" onClick={() => { setDifficulty('hard'); setMapSelection('empty'); startGame(); }}>Auto-Train Demo</button>
          </div>
        </div>

        <div className="leaderboard-panel">
          <h3>GLOBAL LEADERBOARD</h3>
          {leaderboard.length === 0 ? <p style={{ textAlign: 'center', color: '#888' }}>No scores yet. Prove your survival skills!</p> : null}
          {leaderboard.map((l, i) => (
            <div className="lb-row" key={i}>
              <span>#{i + 1} {l.name}</span>
              <span>{l.score.toFixed(1)}s</span>
              <span>{l.algorithm} ({l.predators}x)</span>
            </div>
          ))}
        </div>
      </div>
    );
  }

  if (scene === 'setup') {
    return (
      <div className="setup-scene">
        <div className="setup-card">
          <h2>Arena Setup</h2>
          <div className="form-group">
            <label>Map Template</label>
            <select value={mapSelection} onChange={e => setMapSelection(e.target.value)}>
              <option value="empty">Empty Grid</option>
              <option value="the_cross">The Cross</option>
              <option value="the_corridors">The Corridors</option>
            </select>
          </div>
          <div className="form-group">
            <label>Number of Predators [1-5]</label>
            <select value={numPredators} onChange={e => setNumPredators(e.target.value)}>
              {[1, 2, 3, 4, 5].map(n => <option key={n} value={n}>{n} Agents</option>)}
            </select>
          </div>
          <div className="form-group">
            <label>RL Intelligence Architecture</label>
            <select value={difficulty} onChange={e => setDifficulty(e.target.value)}>
              <option value="easy">Easy (DQN High Expl. / Dumb)</option>
              <option value="medium">Medium (DQN Full Greedy)</option>
              <option value="hard">Hard (State-of-the-Art D3QN)</option>
            </select>
          </div>
          <div style={{ display: 'flex', gap: '10px' }}>
            <button className="btn" onClick={() => setScene('menu')} style={{ background: '#333', color: '#888' }}>Back</button>
            <button className="btn" style={{ background: '#00ffcc', color: '#000', flex: 2 }} onClick={startGame}>Enter Arena</button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="app-container">
      <div className="hud-top">
        <div className="hud-item">
          <span className="hud-label">SURVIVAL TIME</span>
          <span className="hud-val time">{gameState?.survival_time.toFixed(1)}s</span>
        </div>
        <div className="hud-item" style={{ marginLeft: '30px', borderLeft: '1px solid #334', paddingLeft: '30px' }}>
          <span className="hud-label">AI NETWORK</span>
          <span className="hud-val danger" style={{ textTransform: 'uppercase' }}>{metrics.algorithm}</span>
        </div>
      </div>

      <div className="arena-scene">
        <div className="canvas-wrapper">
          <canvas
            ref={canvasRef} width={800} height={800}
            onMouseDown={handleMouseDown} onMouseMove={handleMouseMove} onMouseUp={handleMouseUp} onMouseLeave={handleMouseUp}
          />

          {isGameOver && (
            <div className="game-over-overlay">
              <div className="game-over-text">CAUGHT!</div>
              <div style={{ color: '#fff', fontSize: '24px', marginBottom: '40px' }}>
                You survived for <span style={{ color: '#00ffcc', fontWeight: 'bold' }}>{survivalRecord.current.toFixed(1)}</span> seconds
              </div>
              <form onSubmit={submitScore} style={{ display: 'flex', flexDirection: 'column', gap: '15px', alignItems: 'center' }}>
                <input type="text" name="playerName" placeholder="Enter your Name" required maxLength={12} />
                <button type="submit" className="btn" style={{ width: '100%' }}>Save Score to Database</button>
              </form>
              <button className="btn" style={{ background: 'transparent', color: '#ff3366', marginTop: '15px', border: 'none', fontSize: '14px' }} onClick={() => setScene('menu')}>Skip & Go to Menu</button>
            </div>
          )}

          {!metrics.training && !isGameOver && (
            <div className="game-over-overlay" style={{ background: 'rgba(0,0,0,0.4)', pointerEvents: 'none' }}>
              <div style={{ color: '#00ffcc', fontSize: '24px', marginBottom: '20px', textAlign: 'center', fontWeight: 'bold', textShadow: '0 0 10px #00ffcc' }}>
                [ DRAG MOUSE TO BUILD ADDITIONAL WALLS ]
              </div>
              <div style={{ display: 'flex', gap: '20px', pointerEvents: 'auto' }}>
                <button className="btn" style={{ width: '300px' }} onClick={startArenaEngine}>START PURSUIT (WASD)</button>
                <button className="btn" style={{ background: '#333', color: '#00ffcc' }} onClick={() => setShowHeatmap(!showHeatmap)}>
                  {showHeatmap ? 'Hide PyTorch Heatmap' : 'Show PyTorch Heatmap'}
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;