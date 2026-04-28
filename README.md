# Clinical EEG Authentication System 🧠🔒

A high-performance, clinical-grade platform for EEG-based biometric research. This system simulates realistic brainwave signals and uses them to verify user identity through advanced signal processing and Deep Learning.

## 🚀 Quick Start

### 1. Prerequisites
Ensure you have **Python 3.10+** and **Node.js** installed.

### 2. Setup & Installation
Clone the repository and install dependencies:

```bash
# Install Python dependencies
pip install fastapi uvicorn numpy scipy httpx pydantic

# Install Frontend dependencies
cd frontend
npm install
cd ..
```

### 3. Running the System
You need to run three components simultaneously. Open three terminal windows:

**Terminal 1: EEG Simulator (Backend)**
```bash
python server.py
```

**Terminal 2: Authentication Server**
```bash
python auth_server.py
```

**Terminal 3: Simulator Visualizer (Frontend)**
```bash
cd frontend
npm run dev
```

## 🛠️ System Architecture
- **Simulator (Port 8000):** Generates deterministic multi-channel EEG with clinical artifacts.
- **Auth Web App (Port 8001):** The dashboard for enrollment and identity verification.
- **Visualizer (Port 5173):** Real-time medical-grade waveform monitoring.

## 🔑 Key Features
- **Deterministic Brainprints:** User traits (name, age, etc.) are hashed to generate consistent, unique EEG signatures.
- **Ensemble Matching:** Combines Cosine Similarity, Euclidean Distance, and Feature Ratios.
- **Deep Learning Engine:** 1D CNN + LSTM model for temporal biometric verification.
- **Clinical Artifacts:** Realistic simulation of eye blinks and muscle noise.

## 👥 Access & Collaboration
If you are a collaborator:
1. Clone the repo: `git clone <your-repo-url>`
2. Create a branch for your feature: `git checkout -b feature-name`
3. Push your changes: `git push origin feature-name`


