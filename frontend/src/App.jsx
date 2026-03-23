import { useState, useEffect } from "react";
import axios from "axios";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid
} from "recharts";

const GRID_SIZE = 5;

function App() {
  const [grid, setGrid] = useState(
    Array(GRID_SIZE).fill(null).map(() => Array(GRID_SIZE).fill("."))
  );
  const [running, setRunning] = useState(false);
  const [speed, setSpeed] = useState(500);
  const [steps, setSteps] = useState(0);
  const [episode, setEpisode] = useState(1);
  const [metrics, setMetrics] = useState([]);

  // Convert state → grid
  const createGrid = (state) => {
    const newGrid = Array(GRID_SIZE)
      .fill(null)
      .map(() => Array(GRID_SIZE).fill("."));

    if (!state) return newGrid;

    const [px, py, qx, qy] = state;

    newGrid[px][py] = "🐺";
    newGrid[qx][qy] = "🐇";

    return newGrid;
  };

  const reset = async () => {
    const res = await axios.get("http://127.0.0.1:8000/reset");
    setGrid(createGrid(res.data.state));
    setSteps(0);
  };

  const step = async () => {
  const res = await axios.get("http://127.0.0.1:8000/step");

  const { state, done, steps, episode } = res.data;

  setGrid(createGrid(state));
  setSteps(steps);
  setEpisode(episode);

  if (done) {
    await fetchMetrics();
    alert(`🎯 Prey caught in ${steps} steps!`);
  }
};

const fetchMetrics = async () => {
  const res = await axios.get("http://127.0.0.1:8000/metrics");

  const data = res.data.episodes.map((ep, i) => ({
    episode: ep,
    steps: res.data.steps[i]
  }));

  setMetrics(data);
};

  // Animation loop
  useEffect(() => {
    if (!running) return;

    const interval = setInterval(() => {
      step();
    }, speed);

    return () => clearInterval(interval);
  }, [running, speed]);

  return (
    <div
      style={{
        fontFamily: "Arial",
        background: "#1e1e2f",
        minHeight: "100vh",
        color: "white",
        textAlign: "center",
        padding: "20px",
      }}
    >
      <h1 style={{ marginBottom: "20px" }}>
        Predator vs Prey Simulation
      </h1>

      {/* Controls */}
      <div style={{ marginBottom: "20px" }}>
        <button onClick={reset} style={btnStyle}>Reset</button>
        <button onClick={() => setRunning(true)} style={btnStyle}>▶ Play</button>
        <button onClick={() => setRunning(false)} style={btnStyle}>⏸ Pause</button>
      </div>

      {/* Speed Control */}
      <div style={{ marginBottom: "20px" }}>
        <label>Speed: </label>
        <input
          type="range"
          min="100"
          max="1000"
          value={speed}
          onChange={(e) => setSpeed(Number(e.target.value))}
        />
        <span> {speed} ms</span>
      </div>

      {/* Stats */}
      <h3>Episode: {episode}</h3>
      <h3>Steps: {steps}</h3>
      

      {/* Grid */}
      <div style={{ marginTop: "20px", display: "inline-block" }}>
        {grid.map((row, i) => (
          <div key={i} style={{ display: "flex" }}>
            {row.map((cell, j) => (
              <div
                key={j}
                style={{
                  width: "50px",
                  height: "50px",
                  border: "1px solid #444",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  fontSize: "24px",
                  backgroundColor:
                    cell === "🐺"
                      ? "#ff6b6b"
                      : cell === "🐇"
                      ? "#6bff95"
                      : "#2a2a3d",
                  transition: "all 0.2s ease",
                }}
              >
                {cell}
              </div>
            ))}
          </div>
        ))}
      </div>

      {/* 📊 Learning Graph */}
      <div style={{ marginTop: "40px" }}>
        <h2>Learning Progress</h2>

        <LineChart width={500} height={300} data={metrics}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="episode" />
          <YAxis />
          <Tooltip />
          <Line type="monotone" dataKey="steps" stroke="#4c6ef5" />
        </LineChart>
      </div>
    </div>
  );
}

// Button styling
const btnStyle = {
  margin: "5px",
  padding: "10px 15px",
  fontSize: "16px",
  border: "none",
  borderRadius: "5px",
  cursor: "pointer",
  background: "#4c6ef5",
  color: "white",
};

export default App;