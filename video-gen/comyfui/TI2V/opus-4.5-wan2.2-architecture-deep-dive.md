# WAN 2.2 Architecture: Deep Technical Reference

> **Source Documentation**: [DeepWiki - Wan-Video/Wan2.2](https://deepwiki.com/Wan-Video/Wan2.2)
> 
> **GitHub Repository**: [Wan-Video/Wan2.2](https://github.com/Wan-Video/Wan2.2)
> 
> **Last Updated**: December 2025

---

## Table of Contents

### Part 1 — WAN 2.2 in our architecture (business → system → nodes)

1. [Executive Summary (Business + Architecture)](#executive-summary-business--architecture)
2. [Architecture Extension Overview](#architecture-extension-overview)
3. [ASCII Diagrams (multi-level)](#ascii-diagrams-multi-level)
4. [End-to-End Request Walkthrough (Concept → Execution)](#end-to-end-request-walkthrough-concept--execution)
5. [Node & Dependency Breakdown (Technical Detail)](#node--dependency-breakdown-technical-detail)
6. [Variable → Outcome Mapping (Engineer Control Surface)](#variable--outcome-mapping-engineer-control-surface)
7. [Practical tuning and operating guidance](#practical-tuning-and-operating-guidance)

### Part 2 — Appendix: WAN 2.2 deep technical reference (DeepWiki-derived)

8. [High-Level Overview](#high-level-overview)
9. [System Architecture Diagram](#system-architecture-diagram)
10. [Model Variants](#model-variants)
11. [Core Components Deep Dive](#core-components-deep-dive)
12. [Mixture-of-Experts (MoE) Architecture](#mixture-of-experts-moe-architecture)
13. [VAE Architecture](#vae-architecture)
14. [Text Encoder (UMT5-XXL)](#text-encoder-umt5-xxl)
15. [End-to-End Execution Flow](#end-to-end-execution-flow)
16. [Node-by-Node Data Flow](#node-by-node-data-flow)
17. [Variable and Parameter Reference](#variable-and-parameter-reference)
18. [Memory Optimization Strategies](#memory-optimization-strategies)
19. [Key Insights and Modification Guidelines](#key-insights-and-modification-guidelines)

---

## Executive Summary (Business + Architecture)

This document extends **our ComfyUI-based video generation architecture** by adding a clearly owned, engineer-tunable integration point for **WAN 2.2 TI2V-5B**. It is intentionally written from the perspective of *our system*: business intent → workflow selection → ComfyUI execution → outputs + feedback.

**Primary references**:

- WAN architecture + variants (MoE vs dense TI2V-5B): [DeepWiki: Wan-Video/Wan2.2](https://deepwiki.com/Wan-Video/Wan2.2)
- ComfyUI WAN 2.2 TI2V workflow template (the graph we run): [video_wan2_2_5B_ti2v.json](https://raw.githubusercontent.com/Comfy-Org/workflow_templates/refs/heads/main/templates/video_wan2_2_5B_ti2v.json)
- ComfyUI WAN 2.2 tutorial: [docs.comfy.org/tutorials/video/wan/wan2_2](https://docs.comfy.org/tutorials/video/wan/wan2_2)

### 1) Business rationale (why this exists in our stack)

WAN 2.2 helps us ship a **high-quality, controllable 5s @ 720p (24fps)** video capability with a control surface that maps cleanly to business goals:

- **Quality**: better temporal coherence/camera motion with a modern diffusion-video backbone.
- **Speed**: the TI2V-5B variant is designed to be feasible on **24GB-class GPUs** using a high-compression VAE (enabling faster iteration for most workloads).
- **Cost**: TI2V-5B on consumer GPUs lowers infra cost vs. forcing every run onto multi-GPU enterprise setups.
- **Flexibility**: our “workflow catalog” approach lets us route requests (T2V vs TI2V vs I2V) and add adapters/LoRAs later without rewriting the entire system.

Why adopt/extend WAN 2.2 vs alternatives (in our system terms):

- **Operational simplicity**: a single ComfyUI graph with a small number of stability-critical knobs is easier to productionize than highly modular graphs with many moving parts.
- **Scaling path**: WAN 2.2 explicitly offers both **consumer-friendly dense (TI2V-5B)** and **enterprise MoE (A14B)** variants (see [DeepWiki: Wan-Video/Wan2.2](https://deepwiki.com/Wan-Video/Wan2.2)).

### 2) What success looks like (measurable)

We evaluate WAN 2.2 integration in our system on *measurable* outcomes:

- **Quality**
  - % of outputs passing QA checks (no severe flicker, no gross deformation, no subtitles/watermarks if disallowed)
  - human preference / rating lift vs baseline model/workflow
- **Latency**
  - p50/p95 wall-clock for our “default” preset (e.g., 121 frames @ 24fps, 1280×704)
- **Cost**
  - GPU-minutes per successful render
  - failure/OOM rate (wasted GPU time)
- **Iteration speed**
  - time from prompt/config tweak → new result
  - reproducibility rate (same inputs + seed → same output)

---

## Architecture Extension Overview

This section makes the integration boundaries explicit so engineers know **what is durable**, **what is workflow**, and **what is tunable**.

### What remains unchanged (durable systems)

These are system capabilities we keep stable while swapping/adding workflows:

- **Workflow catalog & versioning**: we treat ComfyUI workflow JSON + docs in `video-gen/comyfui/` as versioned assets.
- **Model artifact management**: downloading, storing, and pinning model files.
- **Execution runtime**: a ComfyUI runtime (local or service) that executes graph JSON.
- **Artifact storage**: storing video outputs + metadata (prompt, seed, node params) for reproducibility.
- **Telemetry**: latency/VRAM/failure metrics.

### What is added/augmented by WAN 2.2

- **New workflow(s)**: WAN 2.2 TI2V graph (`video_wan2_2_5B_ti2v.json`) and any internal variants we fork.
- **New model set** (WAN diffusion + WAN VAE + UMT5 encoder), as referenced by the workflow template:
  - diffusion model: `wan2.2_ti2v_5B_fp16.safetensors`
  - VAE: `wan2.2_vae.safetensors`
  - text encoder: `umt5_xxl_fp8_e4m3fn_scaled.safetensors`

### Clear boundaries (ownership + change policy)

- **Durable systems (platform-owned)**
  - job orchestration, GPU allocation, artifact storage, logging/metrics
- **Runtime workflows (workflow-owned)**
  - ComfyUI graph structure (node wiring), model compatibility matrix, default presets
- **Experimental/tunable components (product/research-owned)**
  - prompt templates, negative prompt libraries, “safe iteration” sampler settings, A/B test presets

---

## ASCII Diagrams (multi-level)

### A) Business goals → architectural components

```
BUSINESS GOALS
  ├─ Quality ─────────────────────┬─ Prompt templates + negative library
  │                               ├─ Workflow defaults (CFG/steps/denoise)
  │                               └─ Model variant choice (TI2V-5B vs A14B)
  ├─ Latency (p95) ───────────────┬─ Resolution/frames policy (token budget)
  │                               ├─ Steps / sampler preset
  │                               └─ Model preload + caching
  ├─ Cost per render ─────────────┬─ TI2V-5B on 24GB GPUs (where possible)
  │                               ├─ Failure/OOM guardrails
  │                               └─ Batch/queue strategy
  └─ Flexibility / iteration ─────┬─ Workflow catalog (route by intent)
                                  ├─ Parameterization (safe knobs)
                                  └─ Adapter injection points (future)
```

### B) Architectural components → workflow nodes

```
[Model Store]
  ├─► (37) UNETLoader
  ├─► (38) CLIPLoader
  └─► (39) VAELoader

[Prompt Builder]
  ├─► (6) CLIPTextEncode (positive)
  └─► (7) CLIPTextEncode (negative)

[Input Prep]
  └─► (56) LoadImage ─► (55) Wan22ImageToVideoLatent

[Sampling Engine]
  (48) ModelSamplingSD3 ─► (3) KSampler

[Decode + Packaging]
  (8) VAEDecode ─► (57) CreateVideo ─► (58) SaveVideo

[Metadata/Telemetry]
  Capture: prompts, seed/mode, steps/cfg, width/height/frames/fps, model filenames, workflow revision
```

### C) Workflow nodes → tunable variables (engineer control surface)

```
(6)/(7) CLIPTextEncode
  - positive_prompt / negative_prompt  → semantics, style, artifact avoidance

(55) Wan22ImageToVideoLatent
  - width/height/frames               → VRAM, cost, output length, drift risk

(3) KSampler
  - steps                             → quality vs latency
  - cfg                               → prompt adherence vs motion naturalness
  - denoise                           → start-image adherence vs creativity
  - sampler/scheduler                 → stability/character of motion
  - seed + seed_mode                  → reproducibility

(57) CreateVideo
  - fps                               → playback speed (not generation compute)

(37)/(38)/(39) loaders + (48) ModelSamplingSD3
  - model files + sampling shift      → stability-critical compatibility
```

---

## End-to-End Request Walkthrough (Concept → Execution)

Trace a single request through our architecture (what happens and why):

### Example request (business intent)

> “Generate a 5-second, 720p, cinematic product shot video. Use this reference image as the first frame. The camera should dolly right slowly.”

### Step 1 — Business intent → workflow selection

- If **start image is provided** and the intent is “anchor identity/composition,” we select **WAN TI2V**.
- If no image is provided, we route to **WAN T2V** (different workflow/preset).

**Why this step exists**: picking the right workflow is the highest-leverage lever for cost and quality. TI2V gives stronger first-frame control; T2V is simpler/faster when you don’t need anchoring.

### Step 2 — Input preparation (text, image, conditioning)

- **Prompt construction**
  - Build a structured positive prompt (subject → action → camera → lighting → style).
  - Select a negative prompt “baseline” to reduce common artifacts.
- **Image preparation**
  - Load/crop/resize reference image to the target aspect ratio.
  - Ensure width/height conform to workflow constraints (multiples of 64 in this template).

**Why this step exists**: most “model failures” are actually input mismatches (aspect ratio distortion, contradictory prompt vs image, or overly aggressive negatives).

### Step 3 — WAN 2.2 execution (ComfyUI graph)

- Load models (cached if already resident).
- Encode prompts → conditioning.
- Encode start image → latent initialization.
- Run diffusion sampling loop.
- Decode frames → create video.

**Why this step exists**: this is where we spend GPU time. Everything upstream exists to make this step deterministic, debuggable, and cost-controlled.

### Step 4 — Output handling + feedback loops

- Save video + persist a **reproducibility bundle**:
  - workflow JSON hash/version
  - all node widget values
  - prompts, seed/mode
  - model filenames
- Capture metrics (latency, VRAM, failure reason if any).
- Feed outcomes into:
  - prompt template improvements
  - preset tuning (steps/cfg/denoise)
  - routing policy (when TI2V is worth the cost)

---

## Node & Dependency Breakdown (Technical Detail)

This section is the engineer-facing “node contract.” The canonical TI2V workflow we run is the Comfy-Org template: [video_wan2_2_5B_ti2v.json](https://raw.githubusercontent.com/Comfy-Org/workflow_templates/refs/heads/main/templates/video_wan2_2_5B_ti2v.json)

### (37) UNETLoader

- **Upstream deps**: model files on disk
- **Inputs → processing → outputs**: `MODEL` weights loaded → `MODEL`
- **Downstream**: `(48) ModelSamplingSD3`
- **Business meaning**: selecting the backbone (quality/cost envelope)
- **Stability**: **critical** (must match the WAN workflow family)

### (38) CLIPLoader (WAN UMT5/T5 encoder)

- **Outputs**: `CLIP` encoder handle
- **Downstream**: `(6)/(7) CLIPTextEncode`
- **Business meaning**: prompt understanding quality + multilingual prompt support
- **Stability**: **critical** (encoder must match training)

### (6) CLIPTextEncode (positive)

- **Input**: `text` → **embeddings/conditioning**
- **Output**: `CONDITIONING`
- **Downstream**: `(3) KSampler.positive`
- **Business meaning**: primary semantic/style control
- **Safe knobs**: prompt text, template structure

### (7) CLIPTextEncode (negative)

- **Output**: `CONDITIONING`
- **Downstream**: `(3) KSampler.negative`
- **Business meaning**: artifact suppression + policy constraints (e.g., “no subtitles/watermarks”)
- **Safe knobs**: negative prompt library; beware over-constraining (can reduce motion/detail)

### (39) VAELoader (WAN VAE)

- **Outputs**: `VAE`
- **Downstream**: `(55) Wan22ImageToVideoLatent`, `(8) VAEDecode`
- **Business meaning**: decode fidelity + compatibility; impacts speed/memory through compression behavior
- **Stability**: **critical** (VAE must match the WAN checkpoint)

### (56) LoadImage

- **Outputs**: `IMAGE` (and `MASK`, unused)
- **Downstream**: `(55) Wan22ImageToVideoLatent.start_image`
- **Business meaning**: first-frame anchoring; reduces iteration cost when clients provide a reference frame

### (55) Wan22ImageToVideoLatent

- **Inputs**: `VAE` + `IMAGE`
- **Output**: initial video `LATENT`
- **Downstream**: `(3) KSampler.latent_image`
- **Data shape change (conceptual)**:
  - `IMAGE` → VAE latent (compressed) → spatio-temporal latent sized by `width/height/frames`
- **Business meaning**: this is the **token budget** knob that dominates VRAM/cost
- **Safe knobs**: width/height/frames within known-good presets
- **Stability-critical constraints**:
  - width/height should remain multiples of 64
  - frames often work best as **8n+1** (the template uses 121)

### (48) ModelSamplingSD3

- **Role**: applies model-specific sampling conventions (sigma/noise schedule compatibility)
- **Stability**: **critical** (treat as “do not touch” unless you are matching training/sampler math)

### (3) KSampler

- **Inputs**: `MODEL`, `CONDITIONING` (pos/neg), initial `LATENT`
- **Output**: denoised `LATENT`
- **Business meaning**: primary quality/latency trade-off node
- **Safe knobs**:
  - steps (15–30 typical)
  - cfg (3–7 typical)
  - denoise (0.6–1.0 depending on start-image adherence)
  - seed mode (fixed for reproducibility)
- **Stability caution**: sampler/scheduler changes can alter motion/quality drastically

### (8) VAEDecode

- **Inputs**: denoised `LATENT` + `VAE`
- **Output**: `IMAGE` frames
- **Business meaning**: visual fidelity; failures here usually indicate incompatibility/memory pressure

### (57) CreateVideo

- **Inputs**: `IMAGE` frames (+ optional `AUDIO`, unused)
- **Output**: `VIDEO`
- **Knob**: fps (affects playback speed)

### (58) SaveVideo

- **Input**: `VIDEO`
- **Output**: file written
- **Business meaning**: artifact handling + reproducibility (store path + metadata)

---

## Variable → Outcome Mapping (Engineer Control Surface)

This table enumerates the key tunables *engineers actually touch* in this workflow and maps them to business outcomes.

| Variable (node) | Affects | When to change | Safe range / default | What breaks if wrong |
| --- | --- | --- | --- | --- |
| Positive prompt `(6)` | semantic quality, style, motion intent | first lever for iteration | structured prompt | vague prompt → weak adherence |
| Negative prompt `(7)` | artifact rate, compliance | when artifacts recur | start from baseline | too strong → frozen motion/detail loss |
| Start image path/mode `(56)` | identity + composition anchoring | when using TI2V | match aspect ratio | distortion/jitter if mismatched |
| width/height `(55)` | VRAM, cost, detail | when you need different output size | multiples of 64 (template: 1280×704) | OOM, distortions, invalid shapes |
| frames `(55)` | duration, drift risk, cost | when changing length | prefer 8n+1 (template: 121) | instability/drift/OOM |
| batch `(55)` | throughput vs VRAM | batching experiments | keep 1 unless proven | OOM + queue blowups |
| seed `(3)` | reproducibility | debugging/tuning | fixed during tuning | randomize → can’t compare runs |
| seed mode `(3)` | determinism policy | when iterating | fixed vs randomize | poor experiment hygiene |
| steps `(3)` | quality vs latency | when quality insufficient | 15–30 (template: 20) | too low → noisy/blur; too high → slow |
| cfg `(3)` | adherence vs motion naturalness | prompt ignored / motion unnatural | 3–7 (template: 5) | too high → artifacts/static feel |
| denoise `(3)` | start-image adherence | when start frame ignored | 0.6–1.0 (template: 1.0) | too low → weak motion; too high → drift |
| sampler `(3)` | stability + motion character | only with intent | keep template (`uni_pc`) | regressions/instability |
| scheduler `(3)` | stability/contrast | only with intent | keep template (`simple`) | regressions/instability |
| sampling shift `(48)` | sampling correctness | **do not change casually** | keep template (`8`) | severe degradation/failure |
| diffusion model file `(37)` | quality/capability | upgrades only | pinned versions | mismatch → failure |
| text encoder file/profile `(38)` | prompt understanding | upgrades only | pinned + `wan` profile | mismatch → failure |
| VAE file `(39)` | decode fidelity + compression behavior | upgrades only | pinned versions | mismatch → failure |
| fps `(57)` | playback speed | when adjusting delivery | 12–30 (template: 24) | wrong duration feel |
| output dir/codec `(58)` | artifact handling + ops | deployment | pinned/policy | storage/compat issues |

**Safe iteration parameters**: prompts, steps, cfg, denoise, seed.

**Stability-critical parameters**: model filenames, VAE/text encoder pairing, `ModelSamplingSD3` setting, width/height constraints, sampler/scheduler (unless validated).

---

## Practical tuning and operating guidance

### A reliable iteration loop (engineer workflow)

1. **Fix the seed** and keep resolution/frames at the default preset.
2. Iterate on **prompt structure** (subject/action/camera/lighting/style).
3. Adjust **CFG** (3→7) before increasing steps.
4. Increase **steps** only if you need more detail/stability.
5. Adjust **denoise** to control how strongly the start image anchors identity.
6. Only then change **resolution/frames** (token budget).

### Guardrails (keep the system stable)

- Always persist the **repro bundle** (workflow version + all widget values + prompts + seed + model filenames).
- Maintain a small set of **validated presets** (resolution × frames × sampler/scheduler).
- Route requests to the cheapest workflow that meets the intent (e.g., TI2V only when start-image anchoring matters).


### Engineer intervention checklist (node + variable level)

- **Safe iteration (day-to-day)**
  - Edit prompt + negative prompt in `(6)` / `(7)`.
  - Tune sampler knobs in `(3)` (steps/cfg/denoise/seed) to hit quality/latency targets.
  - Adjust delivery knobs in `(57)` / `(58)` (fps, output settings) to meet product requirements.

- **Preset changes (performance/cost envelope)**
  - Adjust `(55)` width/height/frames only inside a validated preset grid (token budget guardrail).

- **Stability-critical changes (treat as migrations)**
  - Model upgrades in `(37)/(38)/(39)` and sampling convention changes in `(48)` must be tested as a versioned rollout because they can silently change output distribution or fail at runtime.

- **Always capture “repro metadata”**
  - Persist the full workflow JSON (or hash), all node widget values, prompts, seed/mode, and model filenames.


---

## Appendix: WAN 2.2 Deep Technical Reference (DeepWiki-derived)

The remaining sections below are a deep technical reference on WAN 2.2 itself (variants, MoE, VAE, etc.). Use this appendix when you need to reason about *why* the workflow behaves the way it does or when considering model upgrades/variant switches.

**Note for our TI2V deployment:** some generic WAN diagrams annotate VAE spatial compression as `H/16, W/16`. DeepWiki describes **TI2V-5B** as using **higher spatial compression (`H/32, W/32`)** plus additional patchification to enable 720p workloads on 24GB GPUs. Use that mental model when estimating token budgets/VRAM for our presets. (Source: [DeepWiki: Wan-Video/Wan2.2](https://deepwiki.com/Wan-Video/Wan2.2))

## High-Level Overview

WAN 2.2 is an open-source video generation framework developed by Alibaba's DAMO Academy. It implements **Diffusion Transformer (DiT)** models with an innovative **Mixture-of-Experts (MoE)** architecture for high-quality, controllable video synthesis.

### Key Design Goals

| Goal | Implementation |
|------|----------------|
| **High Quality** | 27B total parameters, 14B active per step |
| **Temporal Coherence** | 3D attention (spatial + temporal) |
| **Flexibility** | 5 specialized model variants |
| **Efficiency** | MoE reduces active compute, high-compression VAE |
| **Accessibility** | Consumer GPU support via TI2V-5B |

### Core Capabilities

- **Text-to-Video (T2V)**: Generate video from text descriptions
- **Image-to-Video (I2V)**: Animate a starting image
- **Text+Image-to-Video (TI2V)**: Combine text control with first frame
- **Speech-to-Video (S2V)**: Audio-driven lip-sync generation
- **Animation**: Character animation and replacement

---

## System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              WAN 2.2 SYSTEM ARCHITECTURE                            │
└─────────────────────────────────────────────────────────────────────────────────────┘

                                    ┌─────────────────┐
                                    │   User Inputs   │
                                    └────────┬────────┘
                                             │
              ┌──────────────────────────────┼──────────────────────────────┐
              │                              │                              │
              ▼                              ▼                              ▼
    ┌─────────────────┐           ┌─────────────────┐           ┌─────────────────┐
    │   Text Prompt   │           │  Start Image    │           │  Audio (S2V)    │
    │                 │           │  (optional)     │           │  (optional)     │
    └────────┬────────┘           └────────┬────────┘           └────────┬────────┘
             │                             │                             │
             ▼                             ▼                             ▼
┌────────────────────────────────────────────────────────────────────────────────────┐
│                               ENCODING LAYER                                        │
├────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│  ┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐    │
│  │   T5EncoderModel    │    │   3D VAE Encoder    │    │    Wav2Vec2         │    │
│  │   (umt5-xxl)        │    │   (Wan2_1/2_2_VAE)  │    │   (audio features)  │    │
│  │                     │    │                     │    │                     │    │
│  │  Input: Text        │    │  Input: Image/Video │    │  Input: Audio       │    │
│  │  Output: (B,L,4096) │    │  Output: Latent     │    │  Output: Features   │    │
│  └──────────┬──────────┘    └──────────┬──────────┘    └──────────┬──────────┘    │
│             │                          │                          │               │
│             └──────────────┬───────────┴──────────────────────────┘               │
│                            │                                                       │
│                            ▼                                                       │
│                 ┌─────────────────────┐                                           │
│                 │  Conditioning       │                                           │
│                 │  Preparation        │                                           │
│                 │  (cross-attention   │                                           │
│                 │   contexts)         │                                           │
│                 └──────────┬──────────┘                                           │
│                            │                                                       │
└────────────────────────────┼───────────────────────────────────────────────────────┘
                             │
                             ▼
┌────────────────────────────────────────────────────────────────────────────────────┐
│                          DIFFUSION BACKBONE (DiT)                                   │
├────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│                    ┌───────────────────────────────────┐                           │
│                    │     Noise Schedule / Timesteps    │                           │
│                    │     (Flow Matching / RF)          │                           │
│                    └─────────────────┬─────────────────┘                           │
│                                      │                                             │
│            ┌─────────────────────────┼─────────────────────────┐                   │
│            │                         │                         │                   │
│            ▼                         ▼                         ▼                   │
│  ┌─────────────────────┐   ┌─────────────────────┐   ┌─────────────────────┐      │
│  │  High-Noise Expert  │   │      Boundary       │   │  Low-Noise Expert   │      │
│  │  (t >= boundary)    │   │      Crossing       │   │  (t < boundary)     │      │
│  │                     │   │   (SNR = SNR_min/2) │   │                     │      │
│  │  - Global structure │   │                     │   │  - Fine details     │      │
│  │  - Layout planning  │◄──┤   Switch Point:     ├──►│  - Texture quality  │      │
│  │  - Coarse motion    │   │   config.boundary   │   │  - Edge sharpening  │      │
│  │  - Object placement │   │   × num_timesteps   │   │  - Local coherence  │      │
│  │                     │   │                     │   │                     │      │
│  │  WanModel (14B)     │   └─────────────────────┘   │  WanModel (14B)     │      │
│  └──────────┬──────────┘                             └──────────┬──────────┘      │
│             │                                                   │                  │
│             └─────────────────────┬─────────────────────────────┘                  │
│                                   │                                                │
│                                   ▼                                                │
│                    ┌───────────────────────────────────┐                          │
│                    │      Classifier-Free Guidance     │                          │
│                    │                                   │                          │
│                    │  noise_pred = uncond +            │                          │
│                    │    cfg × (cond - uncond)          │                          │
│                    │                                   │                          │
│                    │  Per-expert guidance scale:       │                          │
│                    │  guide_scale = (low, high)        │                          │
│                    └─────────────────┬─────────────────┘                          │
│                                      │                                             │
│                                      ▼                                             │
│                    ┌───────────────────────────────────┐                          │
│                    │      Scheduler Step Update        │                          │
│                    │      (uni_pc / euler / dpm)       │                          │
│                    │                                   │                          │
│                    │  latent = scheduler.step(         │                          │
│                    │      noise_pred, t, latent)       │                          │
│                    └─────────────────┬─────────────────┘                          │
│                                      │                                             │
│                              (loop until t=0)                                      │
│                                      │                                             │
└──────────────────────────────────────┼─────────────────────────────────────────────┘
                                       │
                                       ▼
┌────────────────────────────────────────────────────────────────────────────────────┐
│                              DECODING LAYER                                         │
├────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│                    ┌───────────────────────────────────┐                           │
│                    │        3D VAE Decoder             │                           │
│                    │                                   │                           │
│                    │  Input: Denoised Latent           │                           │
│                    │  Shape: (B, C, T/4, H/16, W/16)   │                           │
│                    │                                   │                           │
│                    │  Output: Pixel Video              │                           │
│                    │  Shape: (B, T, H, W, 3)           │                           │
│                    └─────────────────┬─────────────────┘                           │
│                                      │                                             │
└──────────────────────────────────────┼─────────────────────────────────────────────┘
                                       │
                                       ▼
                    ┌───────────────────────────────────┐
                    │         Output Video              │
                    │         (720P @ 24fps)            │
                    └───────────────────────────────────┘
```

---

## Model Variants

WAN 2.2 provides **five specialized model variants**, each optimized for different video generation tasks:

| Model Variant | Task Flag | Parameters | Architecture | Primary Task | Min GPU |
|--------------|-----------|------------|--------------|--------------|---------|
| **T2V-A14B** | `--task t2v-A14B` | 27B total, 14B active | MoE (2 experts) | Text-to-Video | 80GB |
| **I2V-A14B** | `--task i2v-A14B` | 27B total, 14B active | MoE (2 experts) | Image-to-Video | 80GB |
| **TI2V-5B** | `--task ti2v-5B` | 5B dense | Dense transformer | T2V+I2V hybrid | 24GB (RTX 4090) |
| **S2V-14B** | `--task s2v-14B` | 14B | Dense + Wav2Vec2 | Speech-to-Video | 80GB |
| **Animate-14B** | `--task animate-14B` | 14B | Dense + Face/CLIP | Animation | 80GB |

### Architecture Patterns

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           MODEL VARIANT ARCHITECTURES                            │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  MoE Models (T2V-A14B, I2V-A14B):                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐        │
│  │  [T5 Encoder] ──► [High-Noise Expert (14B)] ◄──┬──► [Low-Noise Expert (14B)] │
│  │                          ▲                     │            ▲                │
│  │                          │                     │            │                │
│  │                          └─── t >= boundary ───┴─ t < boundary               │
│  │                                                                              │
│  │  Total: 27B params │ Active: 14B params/step │ Temporal switching           │
│  └──────────────────────────────────────────────────────────────────────────────┘
│                                                                                  │
│  Dense Models (TI2V-5B, S2V-14B, Animate-14B):                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐        │
│  │  [Modality Encoder(s)] ──► [Single Diffusion Backbone] ──► [Output]          │
│  │                                                                              │
│  │  TI2V-5B:    T5 + VAE(64×) ──► WanModel(5B) ──► Video                       │
│  │  S2V-14B:    T5 + VAE + Wav2Vec2 ──► WanModel(14B) ──► Video                │
│  │  Animate:    T5 + VAE + CLIP + Face ──► WanModel(14B) ──► Video             │
│  └──────────────────────────────────────────────────────────────────────────────┘
│                                                                                  │
│  Shared Components (all models):                                                │
│  ┌─────────────────────────────────────────────────────────────────────┐        │
│  │  • T5EncoderModel (google/umt5-xxl) for text encoding                       │
│  │  • AutoencoderKLWan2 for VAE encode/decode                                  │
│  │  • Flow Matching training paradigm                                          │
│  └──────────────────────────────────────────────────────────────────────────────┘
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Core Components Deep Dive

### 1. WanModel (Diffusion Backbone)

The `WanModel` class implements the **Diffusion Transformer (DiT)** architecture:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           WanModel INTERNAL STRUCTURE                            │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  Input: (noisy_latent, timestep, text_embeddings)                               │
│                                                                                  │
│  ┌───────────────────────────────────────────────────────────────────────┐      │
│  │                         Transformer Blocks                             │      │
│  │                                                                        │      │
│  │  ┌─────────────────────────────────────────────────────────────────┐  │      │
│  │  │  Block 1..N (repeated D times, typically 28-40 blocks)          │  │      │
│  │  │                                                                  │  │      │
│  │  │  ┌─────────────────────────────────────────────────────────┐    │  │      │
│  │  │  │  1. Spatial Self-Attention                              │    │  │      │
│  │  │  │     - Attention across spatial dimensions (H×W)         │    │  │      │
│  │  │  │     - Learns spatial relationships within frames        │    │  │      │
│  │  │  └─────────────────────────────────────────────────────────┘    │  │      │
│  │  │                          ↓                                       │  │      │
│  │  │  ┌─────────────────────────────────────────────────────────┐    │  │      │
│  │  │  │  2. Temporal Self-Attention                             │    │  │      │
│  │  │  │     - Attention across temporal dimension (T)           │    │  │      │
│  │  │  │     - Learns motion and temporal coherence              │    │  │      │
│  │  │  └─────────────────────────────────────────────────────────┘    │  │      │
│  │  │                          ↓                                       │  │      │
│  │  │  ┌─────────────────────────────────────────────────────────┐    │  │      │
│  │  │  │  3. Cross-Attention (Text Conditioning)                 │    │  │      │
│  │  │  │     - Query: latent features                            │    │  │      │
│  │  │  │     - Key/Value: text embeddings                        │    │  │      │
│  │  │  │     - Injects semantic information from prompt          │    │  │      │
│  │  │  └─────────────────────────────────────────────────────────┘    │  │      │
│  │  │                          ↓                                       │  │      │
│  │  │  ┌─────────────────────────────────────────────────────────┐    │  │      │
│  │  │  │  4. Feed-Forward Network (FFN)                          │    │  │      │
│  │  │  │     - MLP with GELU activation                          │    │  │      │
│  │  │  │     - Channel mixing and feature transformation         │    │  │      │
│  │  │  └─────────────────────────────────────────────────────────┘    │  │      │
│  │  │                          ↓                                       │  │      │
│  │  │  ┌─────────────────────────────────────────────────────────┐    │  │      │
│  │  │  │  5. AdaLN (Adaptive Layer Normalization)                │    │  │      │
│  │  │  │     - Timestep-conditioned normalization                │    │  │      │
│  │  │  │     - Allows model to adapt behavior per noise level    │    │  │      │
│  │  │  └─────────────────────────────────────────────────────────┘    │  │      │
│  │  │                                                                  │  │      │
│  │  └──────────────────────────────────────────────────────────────────┘  │      │
│  │                                                                        │      │
│  └────────────────────────────────────────────────────────────────────────┘      │
│                                                                                  │
│  Output: Predicted noise (same shape as input latent)                           │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 2. Text Encoder (T5EncoderModel)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           T5 ENCODER (umt5-xxl)                                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  Model: google/umt5-xxl                                                         │
│  Parameters: ~4.7B                                                              │
│  Languages: 100+ (multilingual)                                                 │
│                                                                                  │
│  Input:                                                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐        │
│  │  Text Prompt: "A cat walking through a garden..."                    │        │
│  └─────────────────────────────────────────────────────────────────────┘        │
│                                      │                                           │
│                                      ▼                                           │
│                          ┌─────────────────────┐                                │
│                          │     Tokenizer       │                                │
│                          │  (SentencePiece)    │                                │
│                          └──────────┬──────────┘                                │
│                                     │                                            │
│                                     ▼                                            │
│                          ┌─────────────────────┐                                │
│                          │  Token Embeddings   │                                │
│                          │  + Positional       │                                │
│                          └──────────┬──────────┘                                │
│                                     │                                            │
│                                     ▼                                            │
│                          ┌─────────────────────┐                                │
│                          │  Encoder Layers     │                                │
│                          │  (24 transformer    │                                │
│                          │   blocks)           │                                │
│                          └──────────┬──────────┘                                │
│                                     │                                            │
│                                     ▼                                            │
│  Output:                                                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐        │
│  │  Text Embeddings: Tensor(batch, seq_len, 4096)                       │        │
│  │                                                                      │        │
│  │  - Rich semantic representation of prompt                           │        │
│  │  - Used as Key/Value in cross-attention                             │        │
│  │  - Variable sequence length (typically 77-512 tokens)               │        │
│  └─────────────────────────────────────────────────────────────────────┘        │
│                                                                                  │
│  Memory Optimization:                                                           │
│  • --t5_cpu: Keep on CPU, only transfer embeddings to GPU                      │
│  • --t5_fsdp: Shard across multiple GPUs                                       │
│  • fp8 quantization available for reduced memory                               │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Mixture-of-Experts (MoE) Architecture

The MoE design is the key innovation in WAN 2.2's T2V-A14B and I2V-A14B models.

### Temporal Expert Specialization

Unlike traditional MoE (which routes by token type), WAN 2.2 routes by **denoising timestep**:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         MOE TEMPORAL ROUTING MECHANISM                           │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  Denoising Timeline:                                                            │
│                                                                                  │
│   t = num_train_timesteps                              t = 0                    │
│   (Pure Noise)                                         (Clean Signal)           │
│                                                                                  │
│   ├────────────────────────┬───────────────────────────┤                        │
│   │                        │                           │                        │
│   │   HIGH-NOISE EXPERT    │    LOW-NOISE EXPERT       │                        │
│   │   (14B parameters)     │    (14B parameters)       │                        │
│   │                        │                           │                        │
│   │   Specialization:      │    Specialization:        │                        │
│   │   • Global structure   │    • Fine details         │                        │
│   │   • Layout planning    │    • Texture refinement   │                        │
│   │   • Coarse motion      │    • Local coherence      │                        │
│   │   • Object placement   │    • Edge sharpening      │                        │
│   │                        │                           │                        │
│   ├────────────────────────┤───────────────────────────┤                        │
│                            │                                                     │
│                   boundary = config.boundary × num_train_timesteps              │
│                            │                                                     │
│                            ▼                                                     │
│                   ┌─────────────────┐                                           │
│                   │  Switch Point   │                                           │
│                   │  SNR = SNR_min/2│                                           │
│                   └─────────────────┘                                           │
│                                                                                  │
│  Routing Logic (in _prepare_model_for_timestep):                                │
│  ┌─────────────────────────────────────────────────────────────────────┐        │
│  │  if t >= boundary:                                                   │        │
│  │      required_model = 'high_noise_model'                             │        │
│  │      offload_model = 'low_noise_model'                               │        │
│  │  else:                                                               │        │
│  │      required_model = 'low_noise_model'                              │        │
│  │      offload_model = 'high_noise_model'                              │        │
│  └─────────────────────────────────────────────────────────────────────┘        │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### SNR-Based Boundary Calculation

```
Signal-to-Noise Ratio (SNR) determines the switch point:

SNR_min ◄───────────────────────────────────────────────────────► SNR_max
   │                                                                  │
   │                    boundary (switch point)                       │
   │                           │                                      │
   │           ◄───────────────┼──────────────────────────────►       │
   │                           │                                      │
   │      High-noise regime    │      Low-noise regime                │
   │      (structure)          │      (details)                       │
   │                           │                                      │
   ▼                           ▼                                      ▼
t=T (max timestep)     t=boundary          t=0 (clean)

Where:
- boundary = config.boundary × num_train_timesteps
- Typical boundary corresponds to SNR = SNR_min / 2
- Only ONE model switch per generation (efficient!)
```

### Memory Management with MoE

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         MoE MEMORY STATES DURING GENERATION                      │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  Configuration: MoE with offloading enabled (offload_model=True)                │
│                                                                                  │
│  ┌───────────────┐         ┌───────────────┐         ┌───────────────┐          │
│  │   Initial     │         │  High-Noise   │         │   Low-Noise   │          │
│  │   State       │ ──────► │   Stage       │ ──────► │   Stage       │          │
│  └───────────────┘         └───────────────┘         └───────────────┘          │
│                                                                                  │
│        GPU                       GPU                       GPU                   │
│   ┌───────────┐            ┌───────────┐            ┌───────────┐               │
│   │ Both      │            │ high_noise │            │ low_noise │               │
│   │ models    │            │ _model     │            │ _model    │               │
│   │ loading   │            │ (14B)      │            │ (14B)     │               │
│   └───────────┘            └───────────┘            └───────────┘               │
│                                                                                  │
│        CPU                       CPU                       CPU                   │
│   ┌───────────┐            ┌───────────┐            ┌───────────┐               │
│   │           │            │ low_noise  │            │ high_noise│               │
│   │           │            │ _model     │            │ _model    │               │
│   │           │            │ (14B)      │            │ (14B)     │               │
│   └───────────┘            └───────────┘            └───────────┘               │
│                                                                                  │
│  Memory Comparison:                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐        │
│  │  Configuration           │ GPU Memory │ Total Capacity │ Active    │        │
│  │─────────────────────────────────────────────────────────────────────│        │
│  │  Single 14B model        │ ~14B params│ 14B            │ 14B       │        │
│  │  MoE without offloading  │ ~27B params│ 27B            │ 14B       │        │
│  │  MoE with offloading     │ ~14B params│ 27B            │ 14B       │ ✓ Best│
│  └─────────────────────────────────────────────────────────────────────┘        │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## VAE Architecture

WAN 2.2 uses **3D Causal Variational Autoencoders** with different compression ratios:

### Standard VAE (Most Models)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           STANDARD VAE (16×16×4)                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  Compression Ratios:                                                            │
│  • Temporal: 4× (121 frames → 31 latent frames)                                 │
│  • Spatial:  16×16 (1280×704 → 80×44 latent)                                    │
│  • Combined: 1024× total compression                                            │
│                                                                                  │
│  ┌───────────────────┐                      ┌───────────────────┐               │
│  │   Input Video     │                      │   Output Video    │               │
│  │   (B, T, H, W, 3) │                      │   (B, T, H, W, 3) │               │
│  │                   │                      │                   │               │
│  │   T = 121 frames  │                      │   T = 121 frames  │               │
│  │   H = 704 pixels  │                      │   H = 704 pixels  │               │
│  │   W = 1280 pixels │                      │   W = 1280 pixels │               │
│  └─────────┬─────────┘                      └─────────▲─────────┘               │
│            │                                          │                          │
│            ▼                                          │                          │
│  ┌───────────────────┐                      ┌───────────────────┐               │
│  │     Encoder       │                      │     Decoder       │               │
│  │                   │                      │                   │               │
│  │  3D Conv Layers   │                      │  3D Conv Layers   │               │
│  │  (downsample)     │                      │  (upsample)       │               │
│  │                   │                      │                   │               │
│  │  Temporal: ↓4×    │                      │  Temporal: ↑4×    │               │
│  │  Spatial:  ↓16×   │                      │  Spatial:  ↑16×   │               │
│  └─────────┬─────────┘                      └─────────▲─────────┘               │
│            │                                          │                          │
│            ▼                                          │                          │
│  ┌───────────────────────────────────────────────────────────────┐              │
│  │                     Latent Space                              │              │
│  │                                                               │              │
│  │   Shape: (B, C, T/4, H/16, W/16)                             │              │
│  │   Example: (1, 16, 31, 44, 80)                               │              │
│  │                                                               │              │
│  │   C = 16 latent channels                                     │              │
│  │   Compressed representation for diffusion                    │              │
│  └───────────────────────────────────────────────────────────────┘              │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### High-Compression VAE (TI2V-5B)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                      HIGH-COMPRESSION VAE (32×32×4) - TI2V-5B                    │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  Why Higher Compression?                                                        │
│  • Enables 720P generation on 24GB consumer GPUs (RTX 4090)                     │
│  • Smaller latent = less memory for diffusion backbone                          │
│  • Trade-off: Slight quality reduction for massive efficiency gain             │
│                                                                                  │
│  Compression Stack:                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐        │
│  │  Layer              │ Temporal │ Spatial (H×W) │ Combined          │        │
│  │─────────────────────────────────────────────────────────────────────│        │
│  │  Base VAE encoder   │ 4×       │ 16×16         │ 1024×             │        │
│  │  + Patchification   │ 1×       │ 2×2           │ 4×                │        │
│  │─────────────────────────────────────────────────────────────────────│        │
│  │  Total (TI2V-5B)    │ 4×       │ 32×32         │ 4096×             │ ✓      │
│  │  Standard (others)  │ 4×       │ 16×16         │ 1024×             │        │
│  └─────────────────────────────────────────────────────────────────────┘        │
│                                                                                  │
│  Impact on Latent Size:                                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐        │
│  │  Video: 1280×704, 121 frames                                        │        │
│  │                                                                      │        │
│  │  Standard VAE:  (1, 16, 31, 44, 80)  = 1,740,800 elements           │        │
│  │  TI2V-5B VAE:   (1, 16, 31, 22, 40)  =   435,200 elements   (4× smaller)│   │
│  └─────────────────────────────────────────────────────────────────────┘        │
│                                                                                  │
│  Performance on RTX 4090 (24GB VRAM):                                           │
│  • 5-second 720P@24fps video                                                    │
│  • Generation time: <9 minutes                                                  │
│  • No multi-GPU required                                                        │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## End-to-End Execution Flow

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        COMPLETE GENERATION PIPELINE                              │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  Step 1: CLI Parsing (generate.py)                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐        │
│  │  python generate.py --task ti2v-5B --prompt "..." --image input.png │        │
│  │                                                                      │        │
│  │  → Parse arguments                                                   │        │
│  │  → Validate resolution, task compatibility                           │        │
│  │  → Load WAN_CONFIGS[task]                                            │        │
│  └─────────────────────────────────────────────────────────────────────┘        │
│                                      │                                           │
│                                      ▼                                           │
│  Step 2: Model Initialization                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐        │
│  │  WanTI2V/WanI2V/WanT2V.__init__():                                  │        │
│  │                                                                      │        │
│  │  1. Load T5EncoderModel (umt5-xxl)                                  │        │
│  │     - Apply FSDP if t5_fsdp=True                                    │        │
│  │     - Keep on CPU if t5_cpu=True                                    │        │
│  │                                                                      │        │
│  │  2. Load VAE (Wan2_1_VAE or Wan2_2_VAE)                             │        │
│  │                                                                      │        │
│  │  3. Load Diffusion Model(s)                                         │        │
│  │     - MoE: Load both high_noise_model and low_noise_model           │        │
│  │     - Dense: Load single WanModel                                   │        │
│  │                                                                      │        │
│  │  4. Configure models:                                               │        │
│  │     - Apply sequence parallelism (use_sp)                           │        │
│  │     - Apply FSDP sharding (dit_fsdp)                                │        │
│  │     - Convert dtype if needed                                       │        │
│  │     - Set up offloading                                             │        │
│  └─────────────────────────────────────────────────────────────────────┘        │
│                                      │                                           │
│                                      ▼                                           │
│  Step 3: Text Encoding                                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐        │
│  │  prompt_embeds = t5_encoder(prompt)                                  │        │
│  │  null_embeds = t5_encoder("")  # For CFG                            │        │
│  │                                                                      │        │
│  │  Shape: (batch, seq_len, 4096)                                      │        │
│  └─────────────────────────────────────────────────────────────────────┘        │
│                                      │                                           │
│                                      ▼                                           │
│  Step 4: Image Encoding (if I2V/TI2V)                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐        │
│  │  image_latent = vae.encode(start_image)                              │        │
│  │                                                                      │        │
│  │  → First frame latent is fixed                                      │        │
│  │  → Remaining frames initialized with noise                          │        │
│  │                                                                      │        │
│  │  latent = torch.cat([image_latent, random_noise], dim=temporal)     │        │
│  └─────────────────────────────────────────────────────────────────────┘        │
│                                      │                                           │
│                                      ▼                                           │
│  Step 5: Denoising Loop                                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐        │
│  │  for t in timesteps:  # e.g., 1000...0 in 20 steps                  │        │
│  │                                                                      │        │
│  │      # MoE: Select appropriate expert                               │        │
│  │      if moe_mode:                                                    │        │
│  │          model = _prepare_model_for_timestep(t, boundary)           │        │
│  │          cfg_scale = guide_scale[1] if t >= boundary else guide_scale[0]│   │
│  │                                                                      │        │
│  │      # Forward pass (conditional)                                   │        │
│  │      noise_pred_cond = model(latent, t, prompt_embeds)              │        │
│  │                                                                      │        │
│  │      # Forward pass (unconditional for CFG)                         │        │
│  │      noise_pred_uncond = model(latent, t, null_embeds)              │        │
│  │                                                                      │        │
│  │      # Classifier-Free Guidance                                     │        │
│  │      noise_pred = noise_pred_uncond + cfg_scale * (                 │        │
│  │          noise_pred_cond - noise_pred_uncond)                       │        │
│  │                                                                      │        │
│  │      # Scheduler step                                               │        │
│  │      latent = scheduler.step(noise_pred, t, latent)                 │        │
│  │                                                                      │        │
│  └─────────────────────────────────────────────────────────────────────┘        │
│                                      │                                           │
│                                      ▼                                           │
│  Step 6: VAE Decode                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐        │
│  │  video = vae.decode(latent)                                          │        │
│  │                                                                      │        │
│  │  Shape: (batch, frames, height, width, 3)                           │        │
│  │  Example: (1, 121, 704, 1280, 3)                                    │        │
│  └─────────────────────────────────────────────────────────────────────┘        │
│                                      │                                           │
│                                      ▼                                           │
│  Step 7: Output                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐        │
│  │  save_video(video, output_path, fps=24)                              │        │
│  │                                                                      │        │
│  │  → MP4 file at 720P, 24fps                                          │        │
│  │  → ~5 seconds duration                                              │        │
│  └─────────────────────────────────────────────────────────────────────┘        │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Variable and Parameter Reference

### Generation Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `--steps` | 20 | 15-50 | Number of denoising steps |
| `--cfg` / `--guide_scale` | 5.0 | 3.0-8.0 | Classifier-free guidance scale |
| `--seed` | Random | 0-2^64 | Random seed for reproducibility |
| `--size` | Task-dependent | See below | Output resolution |
| `--num_frames` | 121 | 33-241 | Number of output frames |

### Resolution Options by Task

| Task | Supported Sizes | Recommended |
|------|-----------------|-------------|
| T2V-A14B | 480P, 720P | 720P |
| I2V-A14B | 480P, 720P | 720P |
| TI2V-5B | 720P only | 720P (1280×704) |
| S2V-14B | 480P, 720P | 480P |
| Animate-14B | 480P, 720P | 480P |

### Memory Optimization Flags

| Flag | Effect | VRAM Savings | Trade-off |
|------|--------|--------------|-----------|
| `--offload_model` | Move inactive models to CPU | 30-40% | Slower model switching |
| `--convert_model_dtype` | Convert to FP16/BF16 | 50% on weights | Minimal quality impact |
| `--t5_cpu` | Keep T5 encoder on CPU | 10-15GB | Text encoding on CPU |
| `--dit_fsdp` | Shard DiT across GPUs | Distributed | Requires multi-GPU |
| `--t5_fsdp` | Shard T5 across GPUs | Distributed | Requires multi-GPU |
| `--ulysses_size N` | Sequence parallelism | Distributed | Requires N GPUs |

### Model-Specific Configuration

| Config Key | T2V-A14B | I2V-A14B | TI2V-5B | S2V-14B | Animate-14B |
|------------|----------|----------|---------|---------|-------------|
| `boundary` | ~0.5 | ~0.5 | N/A | N/A | N/A |
| `num_train_timesteps` | 1000 | 1000 | 1000 | 1000 | 1000 |
| `vae_type` | Wan2_1 | Wan2_1 | Wan2_2 | Wan2_1 | Wan2_1 |
| `compression` | 1024× | 1024× | 4096× | 1024× | 1024× |

---

## Memory Optimization Strategies

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        MEMORY OPTIMIZATION DECISION TREE                         │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  What GPU(s) do you have?                                                       │
│                                                                                  │
│  ├── Single 24GB GPU (RTX 4090, RTX 3090)                                       │
│  │   │                                                                          │
│  │   └── Use TI2V-5B model                                                      │
│  │       ├── --offload_model                                                    │
│  │       ├── --t5_cpu                                                           │
│  │       └── --convert_model_dtype                                              │
│  │                                                                              │
│  ├── Single 40-48GB GPU (A6000, RTX 6000 Ada)                                   │
│  │   │                                                                          │
│  │   └── TI2V-5B comfortable, larger models with optimization                   │
│  │       ├── --offload_model (for MoE models)                                   │
│  │       └── --convert_model_dtype                                              │
│  │                                                                              │
│  ├── Single 80GB GPU (A100, H100)                                               │
│  │   │                                                                          │
│  │   └── All models supported                                                   │
│  │       └── MoE models may still benefit from --offload_model                  │
│  │                                                                              │
│  └── Multi-GPU Setup                                                            │
│      │                                                                          │
│      ├── 2-4× GPUs                                                              │
│      │   ├── --dit_fsdp --t5_fsdp (model sharding)                             │
│      │   └── torchrun --nproc_per_node=N                                       │
│      │                                                                          │
│      └── 8× GPUs (A100/H100)                                                    │
│          ├── --dit_fsdp --t5_fsdp                                               │
│          ├── --ulysses_size N (sequence parallelism)                           │
│          └── Optimal for 720P with MoE models                                  │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Recommended Configurations

**Consumer GPU (RTX 4090, 24GB):**
```bash
python generate.py \
    --task ti2v-5B \
    --size 720P \
    --prompt "Your prompt here" \
    --image input.png \
    --offload_model \
    --t5_cpu \
    --convert_model_dtype
```

**Enterprise Single GPU (A100 80GB):**
```bash
python generate.py \
    --task t2v-A14B \
    --size 720P \
    --prompt "Your prompt here" \
    --offload_model
```

**Multi-GPU Cluster (8×A100):**
```bash
torchrun --nproc_per_node=8 generate.py \
    --task t2v-A14B \
    --size 720P \
    --prompt "Your prompt here" \
    --dit_fsdp \
    --t5_fsdp \
    --ulysses_size 8
```

---

## Key Insights and Modification Guidelines

### What to Modify (Safe)

| Component | What to Change | Expected Effect |
|-----------|---------------|-----------------|
| **Prompt** | Text description | Direct control over content |
| **CFG Scale** | 3.0-8.0 | Lower = creative, Higher = prompt-adherent |
| **Steps** | 15-30 | Quality vs speed trade-off |
| **Seed** | Any integer | Reproducibility |
| **Resolution** | Supported sizes | Memory/quality trade-off |
| **Frame count** | 33-241 (increments of 8+1) | Video length |

### What NOT to Modify (Unless Expert)

| Component | Why Not | Risk |
|-----------|---------|------|
| **boundary** (MoE) | Model-specific optimized value | Quality degradation |
| **VAE architecture** | Trained together with diffusion | Incompatibility |
| **Timestep schedule** | Matched to training | Poor sampling |
| **T5 model** | Embeddings must match | Complete failure |
| **Attention structure** | Architecture-specific | Crashes |

### Extending the Architecture

**Adding LoRAs (when supported):**
```python
# Future capability - not yet officially supported
# LoRAs would inject at:
# 1. Cross-attention layers (style/content)
# 2. Self-attention layers (structure)
# 3. FFN layers (general adaptation)
```

**Adding ControlNet (when available):**
```python
# Control signals would be injected:
# 1. As additional conditioning to cross-attention
# 2. As residuals to intermediate features
# 3. Per-block or per-layer guidance
```

---

## Training Data & Assumptions

### Inferred Training Data

| Data Type | Estimated Volume | Purpose |
|-----------|------------------|---------|
| Video clips | +83.2% vs Wan2.1 | Motion diversity |
| Images | +65.6% vs Wan2.1 | Semantic coverage |
| Aesthetic labels | Curated subset | Lighting, composition control |
| Cinematic data | Professional | High-quality motion patterns |

### Model Strengths

- ✅ Cinematic camera movements
- ✅ Natural human motion
- ✅ Consistent object identity
- ✅ Temporal coherence over 5s
- ✅ Text-following accuracy
- ✅ Multilingual prompts (100+ languages)

### Model Limitations

- ❌ Text rendering in video
- ❌ Complex multi-person interactions
- ❌ Physics-defying motion
- ❌ Fine hand/finger details
- ❌ Long-duration coherence (>10s)
- ❌ Rapid scene changes

---

## Resources

- **GitHub Repository**: [Wan-Video/Wan2.2](https://github.com/Wan-Video/Wan2.2)
- **DeepWiki Documentation**: [deepwiki.com/Wan-Video/Wan2.2](https://deepwiki.com/Wan-Video/Wan2.2)
- **HuggingFace Models**: [Comfy-Org/Wan_2.2_ComfyUI_Repackaged](https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged)
- **ComfyUI Integration**: [Comfy-Org/workflow_templates](https://github.com/Comfy-Org/workflow_templates)

---

*This document provides a comprehensive technical reference for understanding, using, and modifying the WAN 2.2 video generation architecture.*

