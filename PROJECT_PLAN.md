# HydroAgent 3D — Technical Design & Development Plan

**Predictive urban flash-flood resilience and incident-management engine**
Developed for the NVIDIA Open Models Codefest 2026 (NVIDIA · Open Hackathons · Oracle) · Apache License 2.0

## 1. Problem statement

Flash floods are one of the few disaster types where cities still operate largely blind. Extreme rainfall overwhelms urban drainage networks faster than civil-protection units can react, and most municipal warning systems are reactive: alerts are issued after streets are already underwater. The consequences — paralyzed transit, damaged infrastructure, preventable casualties — recur every storm season. Classical hydrodynamic solvers can predict urban flooding accurately, but a single scenario takes hours of compute, which makes them unusable for real-time early warning.

HydroAgent 3D addresses this gap: a physics-informed AI surrogate that emulates a hydrodynamic solver at millisecond speed, wrapped in an agentic workflow that turns predictions into actionable civil-protection output, and a 3D digital twin that makes the risk immediately legible to non-technical coordinators.

**Target users:** municipal civil-protection and disaster-management units, and water utilities operating urban drainage networks.

## 2. System capabilities

**Predict.** A graph-neural-network surrogate forecasts water levels and street-scale flood risk across an urban drainage catchment in milliseconds per scenario. Effective lead time is bounded by the rainfall nowcast feeding the model, with ~30 minutes as the working target.

**Act.** An LLM agent built on NVIDIA Nemotron operates as a civil-defense coordinator: it monitors incoming rainfall nowcasts, queries the surrogate when trigger conditions are met, cross-references predicted risk against critical infrastructure (underpasses, subway entrances, hospitals, care homes), and drafts geo-targeted warning alerts and asset-deployment suggestions. All agent output is draft-only: a human coordinator reviews and approves before anything is issued. Human-in-the-loop operation is a deliberate design decision for public-sector deployment.

**Show.** A WebGL digital twin of the pilot catchment renders 3D terrain from real elevation data with the drainage network overlaid. Streets and assets are colored by live risk level, and water levels rise and recede as forecasts update, so the situation is understandable at a glance.

## 3. Pilot site: the Bellinge open dataset

The system is developed against the Bellinge open dataset — a published, open-access urban-drainage research dataset covering a 1.7 km² catchment near Odense, Denmark. It provides roughly ten years (2010–2020) of observations from in-sewer level meters and flow sensors, rain-gauge and weather-radar records, complete network asset data (manholes, pipes, elevations, surface characteristics), and a calibrated EPA SWMM model of the full system.

This choice grounds every layer of the project in reality: the drainage network is real and documented, the "sensor feed" is a replay of genuinely recorded telemetry rather than an invention, and — most importantly — model predictions can be validated against real recorded storm events, not only against synthetic data.

## 4. System architecture

```
              [ Oracle Cloud Infrastructure — GPU instances ]
                                  │
   ┌──────────────────────────────┼──────────────────────────────┐
   ▼                              ▼                              ▼
[ NVIDIA PhysicsNeMo ]     [ NVIDIA NIM / Nemotron ]      [ Node.js backend ]
  GNN/PINN flood surrogate   Agentic coordinator LLM        Nowcast ingestion + cache
  trained on SWMM scenarios  Drafts alerts & actions        Bellinge sensor replay
  ms-scale GPU inference     Queries model on triggers      WebSocket push
   └──────────────────────────────┬──────────────────────────────┘
                                  ▼
                     [ Three.js / WebGL frontend ]
                       3D catchment digital twin
                       Live risk + water rendering
```

**AI engine — NVIDIA PhysicsNeMo (Python 3.11+, PyTorch, CUDA).** PhysicsNeMo is NVIDIA's open-source, Apache 2.0 physics-ML framework (the renamed Modulus). Its HydroGraphNet example — a physics-informed autoregressive GNN for flood forecasting with mass conservation embedded in the loss — serves as the reference implementation. HydroGraphNet was built for riverine flooding on hydrodynamic-solver data; this project's core technical contribution is adapting that architecture to urban pluvial flooding driven by a calibrated SWMM drainage model (see §5).

**Agent layer — NVIDIA Nemotron served through NIM.** During development, the agent runs against NVIDIA's hosted NIM endpoints; for deployment, the same Nemotron model is self-hosted via NIM on OCI GPU instances. The agent is event-driven rather than polling-based: calls are triggered by nowcast intensity thresholds or predicted depth exceedances, keeping inference traffic sparse and auditable.

**Data backend — Node.js (Express, WebSockets).** Three responsibilities: replay the Bellinge sensor time series as a live telemetry stream over WebSockets (architecturally identical to a production IoT feed); poll external rainfall/nowcast APIs on a cached 15-minute cycle so the running system never depends on live third-party endpoints; and serve the frontend and agent through a single API.

**Frontend — Three.js / WebGL.** Terrain mesh of the catchment generated from open elevation data (Danish national elevation model, with Copernicus GLO-30 as fallback), drainage-network overlay, risk heat-mapping on streets and assets, and animated water levels driven by model output arriving over WebSocket.

**Infrastructure — OCI GPU instances** for large training runs, NIM self-hosting, and the deployed demo; a local RTX-class GPU (16 GB VRAM) for day-to-day model iteration.

## 5. Model design

**Graph construction.** The SWMM network maps directly onto a graph: nodes are junctions, manholes, storage units, and outfalls; edges are conduits. Static node features include invert elevation, ground elevation, and node type; static edge features include conduit length, cross-section, slope, and roughness. Dynamic node features carry current water depth and rainfall-derived runoff inflow per timestep.

**Learning task.** The surrogate is an encoder–processor–decoder message-passing GNN rolled out autoregressively (Δt = 5 min) over a 30–60 minute horizon. Targets are per-node water depths; a street-level flood indicator is derived where predicted depth exceeds surcharge/ground level. Following the HydroGraphNet approach, the loss combines a data term with a physics residual penalizing mass-balance violations, which keeps rollouts physically plausible in regimes the training data covers sparsely.

**Training data.** A scenario battery is generated by batch-running the calibrated Bellinge SWMM model (scripted via PySWMM) over synthetic design storms — varying return periods, durations, hyetograph shapes, and spatial rainfall patterns — plus recorded historical events. Scenarios are split by storm (never by timestep) into training, validation, and test sets.

**Inference contract.** `inference.py` ingests the latest rainfall forcing, rolls out the surrogate, and emits a versioned `predictions.json`: timestamp, horizon steps, per-node depths, per-street risk class, and model metadata. This schema is the frozen interface between the AI engine and the rest of the system.

**Precedent.** Published results with this class of approach support the design targets: HydroGraphNet reports a 67% error reduction and 58% critical-success-index improvement over a baseline GNN on riverine data, and an NVIDIA-supported industrial study demonstrated 6-hour basin flood forecasts computed in ~19 ms on a single A100. The working target here — sub-second inference per scenario — is conservative by comparison.

**Degradation path.** If the urban adaptation underperforms within the development window, the system degrades gracefully: a precomputed SWMM scenario library with retrieval over storm descriptors is served through the identical `predictions.json` interface, so the agent and digital twin operate unchanged while the surrogate matures.

## 6. Data sources

All data is open: elevation from the Danish national elevation model or Copernicus GLO-30; drainage network, sensor records, and the calibrated SWMM model from the Bellinge open dataset; training rainfall from Bellinge gauge/radar records plus synthetic design storms; live-mode rainfall from low-latency radar-based nowcast services. Multi-hour-latency satellite QPE products (e.g., GPM IMERG) are used for historical context only — their latency makes them unsuitable for short-lead-time operation, which is a deliberate architectural constraint of the feed design.

## 7. Evaluation protocol

Model skill is measured on two tiers. First, held-out synthetic storms never seen in training, reporting hit rate on flooded-node/street identification, achieved lead time, CSI (critical success index — the standard skill score in flood forecasting), and mass-balance error against the reference SWMM solution. Second, replay of recorded real Bellinge storm events, comparing surrogate output against actually measured sensor levels. Reported claims are limited to what these measurements support.

## 8. Engineering workstreams and interfaces

Development is decomposed into three parallel workstreams with two frozen interfaces:

| Workstream | Stack | Scope |
|---|---|---|
| AI engine (`ai-engine-python/`) | Python 3.11+, PhysicsNeMo, PySWMM | SWMM scenario generation, surrogate training, inference pipeline |
| Backend (`backend-nodejs-api/`) | Node.js, Express, WebSockets | Nowcast ingestion and caching, Bellinge sensor replay, API |
| Frontend (`frontend-webgl-dashboard/`) | Three.js, WebGL, GLSL | Digital twin, risk rendering, scenario playback |

The frozen interfaces are the `predictions.json` schema and the WebSocket message contract. They are fixed early so the workstreams integrate without blocking each other.

## 9. Development roadmap

**August (foundations).** Week 1: environments, Bellinge data acquisition, SWMM batch runner, backend and frontend skeletons. Week 2: interface freeze; scenario generator producing training set v0; sensor replay streaming; terrain rendering from elevation tiles. Week 3: first surrogate trained (local GPU); frontend consuming real `predictions.json`. Week 4: end-to-end integration dry run — storm in → prediction → agent draft → digital twin out.

**September 9 – October 7 (hackathon window).** Mid-September: surrogate retrained at scale on OCI; first hold-out metrics published in-repo. Late September: agent loop operational end-to-end with self-hosted NIM drafting geo-targeted alerts. End of September: integrated digital twin playing full storm scenarios in real time. October 7: complete system — replayed real-event demonstration, published evaluation results, and deployment documentation.

## 10. Beyond the prototype

The same engine extends naturally past the pilot: integration of the agent with municipal alerting and SMS infrastructure; flood-risk simulation services for insurers and urban planners stress-testing developments; and predictive drainage-maintenance analytics for water utilities. The Bellinge pilot serves as the reference deployment for any of these paths.

## 11. References

- NVIDIA PhysicsNeMo (Apache 2.0): https://github.com/NVIDIA/physicsnemo
- HydroGraphNet flood-modeling example: https://github.com/NVIDIA/physicsnemo/tree/main/examples/weather/flood_modeling/hydrographnet
- Taghizadeh et al. (2025), physics-informed GNN flood forecasting, Computer-Aided Civil & Infrastructure Engineering: https://onlinelibrary.wiley.com/doi/full/10.1111/mice.13484
- NVIDIA developer spotlight — AI-based flood models (BRLi / Toulouse INP): https://developer.nvidia.com/blog/spotlight-brli-and-toulouse-inp-develop-ai-based-flood-models-using-nvidia-modulus
- Pedersen et al. (2021), the Bellinge open dataset, Earth System Science Data: https://essd.copernicus.org/articles/13/4779/2021/
- EPA Storm Water Management Model (SWMM): https://www.epa.gov/water-research/storm-water-management-model-swmm
- NASA GPM IMERG product documentation: https://gpm.nasa.gov/data/imerg
- NVIDIA NIM / hosted model endpoints: https://build.nvidia.com

## 12. License

Apache License 2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE).
