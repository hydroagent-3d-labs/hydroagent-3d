# HydroAgent 3D

**Predictive urban flash-flood resilience & incident-management engine**
Built for the NVIDIA Open Models Codefest 2026 (NVIDIA · Open Hackathons · Oracle).

Flash floods overwhelm urban drainage faster than civil-protection units can react, and
current municipal warning systems are reactive. HydroAgent 3D gives municipal
civil-protection and disaster-management units a predictive, agentic early-warning engine.

📋 **Full technical plan:** [PROJECT_PLAN.md](PROJECT_PLAN.md) — 
architecture, data strategy, validation protocol, execution roadmap, and verified references.

## How it works

- **Predict** — a physics-informed graph-neural-network surrogate (NVIDIA PhysicsNeMo,
  adapting the HydroGraphNet flood-modeling example from riverine to urban pluvial
  flooding) trained on batched EPA SWMM simulations of the open Bellinge catchment
  dataset (Odense, Denmark). Millisecond-scale inference per scenario.
- **Act** — an NVIDIA Nemotron agent (served via NIM) monitors rainfall nowcasts, queries
  the surrogate, identifies at-risk critical infrastructure, and drafts geo-targeted
  alerts for human approval.
- **Show** — a Three.js/WebGL digital twin renders live risk and water levels for
  non-technical coordinators. A Node.js backend streams replayed real sensor telemetry
  and cached nowcast feeds.

## Repository structure

- `ai-engine-python/` — SWMM scenario generation, surrogate training, inference
- `backend-nodejs-api/` — feed ingestion, caching, sensor replay, WebSocket API
- `frontend-webgl-dashboard/` — 3D digital twin and risk rendering

## Data

Bellinge open urban-drainage dataset (calibrated EPA SWMM model, 10 years of sensor
observations) · Danish open elevation data / Copernicus GLO-30 · radar-based rainfall
nowcasts. All data sources are open.

## Status

Pre-event development. Hackathon: September 9 – October 7, 2026.

## License

Apache License 2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE).
