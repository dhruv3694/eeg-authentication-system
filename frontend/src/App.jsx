import React, { useEffect, useRef, useState } from 'react';
import { MONTAGES, highpass, lowpass, notch } from './dsp';
import './index.css';

const FS = 256;
const MAX_HISTORY_SECONDS = 15;
const ELECTRODES = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3", "C3", "Cz", "C4", "T4", "T5", "P3", "Pz", "P4", "T6", "O1", "O2", "A1", "A2"];
const SCALP_ELECTRODES = ELECTRODES.filter(e => e !== "A1" && e !== "A2");

const STATE_PROFILES = ["Awake - eyes closed", "Awake - eyes open", "Drowsy", "N2 sleep", "Generalized 3 Hz spike-wave"];
const PATHOLOGY_TENDENCIES = ["None", "Frontal slowing tendency", "Temporal sharp tendency", "Generalized spike-wave tendency", "Low voltage fast tendency"];
const MEDICATION_EFFECTS = ["None", "Sedative effect", "Stimulant effect", "Antiseizure effect"];

export default function App() {
  const canvasRef = useRef(null);
  const overlayRef = useRef(null);
  const ws = useRef(null);
  
  const [connected, setConnected] = useState(false);
  const [config, setConfig] = useState(null);
  const [users, setUsers] = useState([]);
  const [activeUserId, setActiveUserId] = useState(1);
  const [isRunning, setIsRunning] = useState(true);
  const [dbStats, setDbStats] = useState({ user_count: 0, db_size_mb: 0 });
  
  // Custom User Modal State
  const [showCustomModal, setShowCustomModal] = useState(false);
  const [customUserForm, setCustomUserForm] = useState({
    name: 'Jane Doe',
    age: 30,
    pathology: 'None',
    medication: 'None'
  });
  
  // Settings
  const [montageName, setMontageName] = useState("Longitudinal bipolar");
  const [sensitivity, setSensitivity] = useState(7);
  const [sweep, setSweep] = useState(10);
  const [lff, setLff] = useState(1.0);
  const [hff, setHff] = useState(70);
  const [notchEnabled, setNotchEnabled] = useState(true);
  
  // Profile Edits
  const [editProfile, setEditProfile] = useState({});

  // Calipers
  const [caliper, setCaliper] = useState(null);
  const [isDragging, setIsDragging] = useState(false);

  // Data buffer
  const bufferRef = useRef({
    times: new Float32Array(MAX_HISTORY_SECONDS * FS),
    samples: {},
    head: 0,
    count: 0
  });

  const resetBuffer = () => {
    bufferRef.current.head = 0;
    bufferRef.current.count = 0;
  };

  const fetchDbStats = () => {
    fetch('http://127.0.0.1:8000/api/db/stats')
      .then(r => r.json())
      .then(data => setDbStats(data))
      .catch(e => console.error("Failed to fetch DB stats", e));
  };

  const fetchUsers = () => {
    fetch('http://127.0.0.1:8000/api/users')
      .then(r => r.json())
      .then(data => {
        setUsers(data.users);
        setActiveUserId(data.active);
      });
    fetchDbStats();
  };

  const fetchConfig = () => {
    fetch('http://127.0.0.1:8000/api/config')
      .then(r => r.json())
      .then(data => {
        setConfig(data);
        setEditProfile(data.profile || {});
        if (data.is_running !== undefined) setIsRunning(data.is_running);
      });
  };

  useEffect(() => {
    ELECTRODES.forEach(e => {
      bufferRef.current.samples[e] = new Float32Array(MAX_HISTORY_SECONDS * FS);
    });
    
    ws.current = new WebSocket('ws://127.0.0.1:8000/ws');
    ws.current.onopen = () => setConnected(true);
    ws.current.onclose = () => setConnected(false);
    
    ws.current.onmessage = (event) => {
      const msg = JSON.parse(event.data);
      if (msg.type === "eeg_chunk") {
        const { times, samples } = msg.data;
        const buf = bufferRef.current;
        const maxLen = MAX_HISTORY_SECONDS * FS;
        
        for (let i = 0; i < times.length; i++) {
          buf.times[buf.head] = times[i];
          for (const e of ELECTRODES) {
            buf.samples[e][buf.head] = samples[e][i];
          }
          buf.head = (buf.head + 1) % maxLen;
          if (buf.count < maxLen) buf.count++;
        }
      }
    };

    fetchUsers();
    fetchConfig();

    return () => ws.current?.close();
  }, []);

  // Sync profile when active user changes
  useEffect(() => {
    resetBuffer();
    fetchConfig();
  }, [activeUserId]);

  useEffect(() => {
    let animationId;
    const render = () => {
      drawCanvas();
      animationId = requestAnimationFrame(render);
    };
    render();
    return () => cancelAnimationFrame(animationId);
  }, [montageName, sensitivity, sweep, lff, hff, notchEnabled]);

  const drawCanvas = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const width = canvas.width = canvas.clientWidth * window.devicePixelRatio;
    const height = canvas.height = canvas.clientHeight * window.devicePixelRatio;
    ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
    const cssWidth = canvas.clientWidth;
    const cssHeight = canvas.clientHeight;

    ctx.clearRect(0, 0, cssWidth, cssHeight);
    
    ctx.strokeStyle = '#1e293b';
    ctx.lineWidth = 1;
    ctx.beginPath();
    const minorStep = 0.2;
    for (let t = 0; t <= sweep; t += minorStep) {
      const x = (t / sweep) * cssWidth;
      if (Math.abs(t % 1.0) < 0.01) {
        ctx.strokeStyle = '#334155';
        ctx.stroke();
        ctx.beginPath();
      } else {
        ctx.strokeStyle = '#1e293b';
      }
      ctx.moveTo(x, 0);
      ctx.lineTo(x, cssHeight);
      ctx.stroke();
      ctx.beginPath();
    }

    const buf = bufferRef.current;
    if (buf.count === 0) return;

    const windowSamples = Math.min(buf.count, sweep * FS);
    let orderedData = { times: new Float32Array(windowSamples), samples: {} };
    ELECTRODES.forEach(e => { orderedData.samples[e] = new Float32Array(windowSamples); });
    
    let startIdx = (buf.head - windowSamples + (MAX_HISTORY_SECONDS * FS)) % (MAX_HISTORY_SECONDS * FS);
    for (let i = 0; i < windowSamples; i++) {
      let idx = (startIdx + i) % (MAX_HISTORY_SECONDS * FS);
      orderedData.times[i] = buf.times[idx];
      ELECTRODES.forEach(e => {
        orderedData.samples[e][i] = buf.samples[e][idx];
      });
    }

    const montage = MONTAGES[montageName] || [];
    const channelCount = montage.length;
    const spacing = cssHeight / Math.max(channelCount, 1);
    const pixelsPerMm = Math.max(Math.min(spacing / 11.5, 5.6), 2.2);
    const pixelsPerUv = pixelsPerMm / sensitivity;
    
    let average = null;
    if (montage.some(ch => ch[2] === "AVG")) {
      average = new Float32Array(windowSamples);
      for (let i = 0; i < windowSamples; i++) {
        let sum = 0;
        SCALP_ELECTRODES.forEach(e => sum += orderedData.samples[e][i]);
        average[i] = sum / SCALP_ELECTRODES.length;
      }
    }

    ctx.font = 'bold 12px Consolas';
    ctx.textBaseline = 'middle';
    ctx.lineWidth = 1.5;
    ctx.lineJoin = 'round';

    montage.forEach((ch, index) => {
      const [label, pos, neg] = ch;
      const baseline = spacing * (index + 0.5);
      
      ctx.fillStyle = '#10b981'; 
      ctx.fillText(label, 10, baseline);
      
      let signal = new Float32Array(windowSamples);
      for (let i = 0; i < windowSamples; i++) {
        let vPos = orderedData.samples[pos][i];
        let vNeg = 0;
        if (neg === "AVG") vNeg = average[i];
        else if (neg === "A1A2") vNeg = 0.5 * (orderedData.samples["A1"][i] + orderedData.samples["A2"][i]);
        else if (neg === "NONE") vNeg = 0;
        else vNeg = orderedData.samples[neg][i];
        signal[i] = vPos - vNeg;
      }

      signal = highpass(signal, lff, FS);
      if (notchEnabled) signal = notch(signal, 50, FS);
      signal = lowpass(signal, hff, FS);

      ctx.strokeStyle = '#38bdf8'; 
      ctx.beginPath();
      for (let i = 0; i < windowSamples; i++) {
        const x = (i / (sweep * FS)) * cssWidth;
        const y = baseline + signal[i] * pixelsPerUv; 
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
    });
  };

  const handleMouseDown = (e) => {
    const rect = overlayRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    setCaliper({ startX: x, startY: y, endX: x, endY: y });
    setIsDragging(true);
  };

  const handleMouseMove = (e) => {
    if (!isDragging || !caliper) return;
    const rect = overlayRef.current.getBoundingClientRect();
    setCaliper({ ...caliper, endX: e.clientX - rect.left, endY: e.clientY - rect.top });
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  const sendEvent = (label) => {
    fetch('http://127.0.0.1:8000/api/event', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ label })
    });
  };

  const addUser = () => {
    fetch('http://127.0.0.1:8000/api/users', { method: 'POST' }).then(() => fetchUsers());
  };

  const removeUser = (uid) => {
    fetch(`http://127.0.0.1:8000/api/users/${uid}`, { method: 'DELETE' }).then(() => fetchUsers());
  };

  const addCustomUser = () => {
    fetch('http://127.0.0.1:8000/api/users/custom', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(customUserForm)
    }).then(() => {
      setShowCustomModal(false);
      fetchUsers();
    });
  };

  const toggleRunning = () => {
    fetch('http://127.0.0.1:8000/api/control', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ running: !isRunning })
    }).then(r => r.json()).then(data => setIsRunning(data.is_running));
  };

  const switchUser = (id) => {
    fetch('http://127.0.0.1:8000/api/users/active', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ id })
    }).then(() => fetchUsers());
  };

  const applyProfile = () => {
    fetch(`http://127.0.0.1:8000/api/users/${activeUserId}/profile`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(editProfile)
    }).then(() => fetchUsers());
  };

  const randomizeSignature = () => {
    fetch(`http://127.0.0.1:8000/api/users/${activeUserId}/randomize`, { method: 'POST' })
      .then(() => fetchUsers());
  };

  const exportRaw = () => {
    window.open(`http://127.0.0.1:8000/api/export/raw?montage=${montageName}`);
  };

  const exportFeatures = () => {
    window.open(`http://127.0.0.1:8000/api/export/features?montage=${montageName}`);
  };

  return (
    <div className="app-container">
      <div className="user-tabs">
        {users.map(u => (
          <div 
            key={u.id} 
            className={`user-tab ${u.id === activeUserId ? 'active' : ''}`}
            onClick={() => switchUser(u.id)}
            style={{ display: 'flex', alignItems: 'center', gap: '8px' }}
          >
            {u.name}
            {users.length > 1 && (
              <span 
                className="remove-tab-btn"
                onClick={(e) => { e.stopPropagation(); removeUser(u.id); }}
                style={{ cursor: 'pointer', color: '#94a3b8', fontSize: '14px', fontWeight: 'bold' }}
                title="Remove User"
              >
                &times;
              </span>
            )}
          </div>
        ))}
        <button onClick={addUser} className="add-user-btn">+ Random User</button>
        <button onClick={() => setShowCustomModal(true)} className="add-user-btn" style={{ marginLeft: '4px', backgroundColor: '#6366f1' }}>+ Custom User</button>
      </div>

      <div className="toolbar">
        <button 
          onClick={toggleRunning} 
          style={{
            marginRight: '12px',
            backgroundColor: isRunning ? '#ef4444' : '#22c55e',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            padding: '4px 12px',
            cursor: 'pointer',
            fontWeight: 'bold'
          }}
        >
          {isRunning ? 'Pause' : 'Play'}
        </button>
        <div style={{ color: connected ? '#22c55e' : '#ef4444', fontWeight: 'bold' }}>
          {connected ? "LIVE" : "OFFLINE"}
        </div>
        <div style={{ padding: '0 12px' }}>|</div>
        <label>Montage:
          <select value={montageName} onChange={e => setMontageName(e.target.value)}>
            {Object.keys(MONTAGES).map(m => <option key={m}>{m}</option>)}
          </select>
        </label>
        <label>Sweep:
          <select value={sweep} onChange={e => setSweep(Number(e.target.value))}>
            <option value="5">5 s</option>
            <option value="10">10 s</option>
            <option value="15">15 s</option>
          </select>
        </label>
        <label>Sens:
          <select value={sensitivity} onChange={e => setSensitivity(Number(e.target.value))}>
            <option value="3">3 uV/mm</option>
            <option value="7">7 uV/mm</option>
            <option value="10">10 uV/mm</option>
            <option value="15">15 uV/mm</option>
          </select>
        </label>
        <label>LFF:
          <select value={lff} onChange={e => setLff(Number(e.target.value))}>
            <option value="0.1">0.1 Hz</option>
            <option value="0.5">0.5 Hz</option>
            <option value="1.0">1.0 Hz</option>
          </select>
        </label>
        <label>HFF:
          <select value={hff} onChange={e => setHff(Number(e.target.value))}>
            <option value="35">35 Hz</option>
            <option value="70">70 Hz</option>
          </select>
        </label>
        <label>
          <input type="checkbox" checked={notchEnabled} onChange={e => setNotchEnabled(e.target.checked)} /> Notch
        </label>
        
        <div style={{ flex: 1 }}></div>
        <button onClick={exportRaw} style={{ backgroundColor: '#2563eb', borderColor: '#1d4ed8' }}>Export Raw CSV</button>
        <button onClick={exportFeatures} style={{ backgroundColor: '#8b5cf6', borderColor: '#7c3aed' }}>Export Features</button>
      </div>

      <div className="main-content">
        <div className="canvas-container">
          <canvas ref={canvasRef}></canvas>
          <div 
            ref={overlayRef}
            className="caliper-overlay" 
            style={{ pointerEvents: 'auto' }}
            onMouseDown={handleMouseDown}
            onMouseMove={handleMouseMove}
            onMouseUp={handleMouseUp}
            onMouseLeave={handleMouseUp}
          >
            {caliper && (
              <svg width="100%" height="100%" style={{ position: 'absolute', top: 0, left: 0, pointerEvents: 'none' }}>
                <line 
                  x1={caliper.startX} y1={caliper.startY} 
                  x2={caliper.endX} y2={caliper.endY} 
                  stroke="#eab308" strokeWidth="2" strokeDasharray="4"
                />
                <circle cx={caliper.startX} cy={caliper.startY} r="4" fill="#eab308" />
                <circle cx={caliper.endX} cy={caliper.endY} r="4" fill="#eab308" />
                {Math.abs(caliper.endX - caliper.startX) > 5 && (
                  <text 
                    x={(caliper.startX + caliper.endX) / 2} 
                    y={(caliper.startY + caliper.endY) / 2 - 10} 
                    fill="#eab308" 
                    fontSize="14" 
                    fontWeight="bold"
                    textAnchor="middle"
                  >
                    {Math.round(Math.abs(caliper.endX - caliper.startX) / (overlayRef.current?.clientWidth || 1) * sweep * 1000)} ms
                    {' | '}
                    {Math.round(1000 / (Math.abs(caliper.endX - caliper.startX) / (overlayRef.current?.clientWidth || 1) * sweep * 1000) * 10) / 10} Hz
                  </text>
                )}
              </svg>
            )}
          </div>
        </div>
        
        <div className="side-panel">
          <div className="panel-section">
            <h3 style={{ display: 'flex', justifyContent: 'space-between' }}>
              Profile Editor
            </h3>
            <div className="form-group">
              <label>Name</label>
              <input type="text" value={editProfile.name || ''} onChange={e => setEditProfile({...editProfile, name: e.target.value})} />
            </div>
            <div className="form-group">
              <label>Age</label>
              <input type="number" value={editProfile.age || 0} onChange={e => setEditProfile({...editProfile, age: parseInt(e.target.value)})} />
            </div>
            <div className="form-group">
              <label>State</label>
              <select value={editProfile.state || ''} onChange={e => setEditProfile({...editProfile, state: e.target.value})}>
                {STATE_PROFILES.map(s => <option key={s}>{s}</option>)}
              </select>
            </div>
            <div className="form-group">
              <label>Pathology</label>
              <select value={editProfile.pathology || ''} onChange={e => setEditProfile({...editProfile, pathology: e.target.value})}>
                {PATHOLOGY_TENDENCIES.map(p => <option key={p}>{p}</option>)}
              </select>
            </div>
            <div className="form-group">
              <label>Alertness ({editProfile.alertness})</label>
              <input type="range" min="0" max="1" step="0.05" value={editProfile.alertness || 0} onChange={e => setEditProfile({...editProfile, alertness: parseFloat(e.target.value)})} />
            </div>
            <div style={{ display: 'flex', gap: '8px', marginTop: '12px' }}>
              <button onClick={applyProfile} className="action-btn">Apply Changes</button>
              <button onClick={randomizeSignature} className="action-btn outline">Randomize Auth Signature</button>
            </div>
          </div>
          
          <div className="panel-section">
            <h3>Impedance Check</h3>
            <div className="impedance-head">
              <div className="electrode good" style={{ top: '20%', left: '30%' }} title="Fp1: <5k"></div>
              <div className="electrode good" style={{ top: '20%', left: '70%' }} title="Fp2: <5k"></div>
              <div className="electrode fair" style={{ top: '40%', left: '20%' }} title="F7: 8k"></div>
              <div className="electrode good" style={{ top: '40%', left: '40%' }} title="F3: <5k"></div>
              <div className="electrode good" style={{ top: '40%', left: '60%' }} title="F4: <5k"></div>
              <div className="electrode good" style={{ top: '40%', left: '80%' }} title="F8: <5k"></div>
              <div className="electrode good" style={{ top: '60%', left: '15%' }} title="T3: <5k"></div>
              <div className="electrode good" style={{ top: '60%', left: '35%' }} title="C3: <5k"></div>
              <div className="electrode poor" style={{ top: '60%', left: '50%' }} title="Cz: 25k (Check!)"></div>
              <div className="electrode good" style={{ top: '60%', left: '65%' }} title="C4: <5k"></div>
              <div className="electrode good" style={{ top: '60%', left: '85%' }} title="T4: <5k"></div>
              <div className="electrode good" style={{ top: '80%', left: '30%' }} title="P3: <5k"></div>
              <div className="electrode good" style={{ top: '80%', left: '70%' }} title="P4: <5k"></div>
              <div className="electrode good" style={{ top: '95%', left: '40%' }} title="O1: <5k"></div>
              <div className="electrode good" style={{ top: '95%', left: '60%' }} title="O2: <5k"></div>
            </div>
          </div>

          <div className="panel-section" style={{ flex: 1 }}>
            <h3>Annotations & Events</h3>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px', marginBottom: '16px' }}>
              <button className="event-btn" onClick={() => sendEvent("Photic Stim")} style={{ borderColor: '#db2777', color: '#db2777' }}>Photic Stim</button>
              <button className="event-btn" onClick={() => sendEvent("HV Start")} style={{ borderColor: '#0284c7', color: '#0284c7' }}>HV Timer</button>
            </div>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
              <button className="event-btn" onClick={() => sendEvent("Spike")}>Spike</button>
              <button className="event-btn" onClick={() => sendEvent("Seizure")}>Seizure</button>
              <button className="event-btn" onClick={() => sendEvent("Artifact")}>Artifact</button>
              <button className="event-btn" onClick={() => sendEvent("Movement")}>Movement</button>
            </div>
          </div>
        </div>
      </div>
      
      {showCustomModal && (
        <div className="modal-overlay" style={{
          position: 'fixed', top: 0, left: 0, right: 0, bottom: 0,
          backgroundColor: 'rgba(0,0,0,0.6)', display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 1000
        }}>
          <div className="modal-content" style={{
            backgroundColor: '#1e293b', padding: '24px', borderRadius: '8px', width: '400px', border: '1px solid #334155'
          }}>
            <h2 style={{ marginTop: 0, marginBottom: '16px', color: '#f8fafc' }}>Create Custom Patient</h2>
            <p style={{ fontSize: '13px', color: '#94a3b8', marginBottom: '16px' }}>
              These traits are hashed to generate a mathematically deterministic brainprint. Identical inputs produce identical EEG features!
            </p>
            
            <div className="form-group" style={{ marginBottom: '12px' }}>
              <label>Name</label>
              <input type="text" value={customUserForm.name} onChange={e => setCustomUserForm({...customUserForm, name: e.target.value})} style={{ width: '100%' }} />
            </div>
            <div className="form-group" style={{ marginBottom: '12px' }}>
              <label>Age</label>
              <input type="number" value={customUserForm.age} onChange={e => setCustomUserForm({...customUserForm, age: parseInt(e.target.value)})} style={{ width: '100%' }} />
            </div>
            <div className="form-group" style={{ marginBottom: '12px' }}>
              <label>Pathology Tendency</label>
              <select value={customUserForm.pathology} onChange={e => setCustomUserForm({...customUserForm, pathology: e.target.value})} style={{ width: '100%' }}>
                {PATHOLOGY_TENDENCIES.map(p => <option key={p}>{p}</option>)}
              </select>
            </div>
            <div className="form-group" style={{ marginBottom: '24px' }}>
              <label>Medication</label>
              <select value={customUserForm.medication} onChange={e => setCustomUserForm({...customUserForm, medication: e.target.value})} style={{ width: '100%' }}>
                {MEDICATION_EFFECTS.map(m => <option key={m}>{m}</option>)}
              </select>
            </div>
            
            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '8px' }}>
              <button onClick={() => setShowCustomModal(false)} style={{
                padding: '8px 16px', backgroundColor: 'transparent', border: '1px solid #475569', color: '#f8fafc', borderRadius: '4px', cursor: 'pointer'
              }}>Cancel</button>
              <button onClick={addCustomUser} style={{
                padding: '8px 16px', backgroundColor: '#3b82f6', border: 'none', color: 'white', borderRadius: '4px', cursor: 'pointer', fontWeight: 'bold'
              }}>Create Deterministic User</button>
            </div>
          </div>
        </div>
      )}

      <div className="status-bar">
        <span>Clinical EEG Web Simulator</span>
        <span style={{ color: '#38bdf8', fontWeight: 'bold' }}>
          Database: {dbStats.user_count} Users | {dbStats.db_size_mb} MB stored
        </span>
        <span>EEG-Based Authentication Ready | EDF+ Format Output | 256 Hz</span>
      </div>
    </div>
  );
}
