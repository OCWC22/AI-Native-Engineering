# WAN 2.2 Ti2V: Integrated Architecture & Engineering Guide

> **Purpose**: Connect WAN 2.2 video generation to our business objectives, existing architecture, and engineering control surfaces.
> 
> **Audience**: Engineers who need to understand *why* WAN 2.2 exists in our stack, *how* it integrates, and *where* to intervene.

---

## Executive Summary

### Why WAN 2.2 Exists in Our System

| Business Problem | WAN 2.2 Solution | Measurable Outcome |
|-----------------|------------------|-------------------|
| **Manual video production is slow** | Automated text-to-video generation | 5-second video in <9 minutes vs hours of manual work |
| **Creative iteration is expensive** | Seed-based reproducibility + fast preview mode | 10x more iterations per production cycle |
| **Quality inconsistent across vendors** | Deterministic pipeline with tunable parameters | Consistent 720P output quality |
| **Consumer GPU accessibility** | 5B model fits on RTX 4090 (24GB) | No cloud GPU dependency for development |

### What We're Adopting

We are adopting **WAN 2.2 TI2V-5B** via ComfyUI as our primary video generation pipeline because:

1. **Cost**: Runs on consumer hardware we already own
2. **Quality**: State-of-the-art at the 5B scale, sufficient for our use cases
3. **Flexibility**: Both text-to-video and image-to-video in one workflow
4. **Integration**: ComfyUI provides visual workflow editing + API access
5. **Extensibility**: Clear node-based architecture for future LoRA/ControlNet additions

### What Success Looks Like

| Metric | Current Baseline | Target | Measurement |
|--------|-----------------|--------|-------------|
| **Time-to-video** | N/A (manual) | <10 min for 5s clip | ComfyUI execution time |
| **Iteration speed** | 1-2 variants/day | 20+ variants/day | Completed generations |
| **Quality consistency** | Variable | 90%+ usable outputs | Manual review pass rate |
| **GPU utilization** | Idle | 80%+ during generation | nvidia-smi monitoring |

---

## Architecture Extension Overview

### How WAN 2.2 Fits Our Existing Stack

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                          OUR EXISTING ARCHITECTURE                                   │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────────────┐    │
│  │                         ORCHESTRATION LAYER                                  │    │
│  │                         (Conductor / Workflow Management)                    │    │
│  │                                                                              │    │
│  │   • Tracks: Define what video assets to produce                             │    │
│  │   • Specs: Describe each video's requirements                               │    │
│  │   • Plans: Break into generation tasks                                      │    │
│  │                                                                              │    │
│  └───────────────────────────────────┬──────────────────────────────────────────┘    │
│                                      │                                               │
│                                      │ (1) Prompt + Parameters                       │
│                                      ▼                                               │
│  ┌─────────────────────────────────────────────────────────────────────────────┐    │
│  │                         VIDEO GENERATION LAYER                               │    │
│  │                         ◄── NEW: WAN 2.2 Ti2V via ComfyUI ──►               │    │
│  │                                                                              │    │
│  │   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                     │    │
│  │   │  ComfyUI    │───▶│  WAN 2.2    │───▶│   Output    │                     │    │
│  │   │  Workflow   │    │  Pipeline   │    │  Handler    │                     │    │
│  │   └─────────────┘    └─────────────┘    └─────────────┘                     │    │
│  │                                                                              │    │
│  └───────────────────────────────────┬──────────────────────────────────────────┘    │
│                                      │                                               │
│                                      │ (2) Generated Video                           │
│                                      ▼                                               │
│  ┌─────────────────────────────────────────────────────────────────────────────┐    │
│  │                         ASSET MANAGEMENT LAYER                               │    │
│  │                         (Existing: Storage, Versioning, Delivery)            │    │
│  │                                                                              │    │
│  │   • Store generated videos with metadata                                    │    │
│  │   • Link to generation parameters for reproducibility                       │    │
│  │   • Serve to downstream consumers                                           │    │
│  │                                                                              │    │
│  └─────────────────────────────────────────────────────────────────────────────┘    │
│                                                                                      │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

### Component Ownership

| Component | Status | Owner | Changes Required |
|-----------|--------|-------|-----------------|
| **Conductor/Orchestration** | UNCHANGED | Existing | Add video generation track templates |
| **ComfyUI Server** | NEW | This project | Install, configure, maintain |
| **WAN 2.2 Models** | NEW | This project | Download, store, version |
| **Ti2V Workflow** | NEW | This project | Create, tune, document |
| **Output Handler** | AUGMENTED | Existing + new | Add video format support |
| **Asset Storage** | UNCHANGED | Existing | None |

### Boundary Definitions

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                             SYSTEM BOUNDARIES                                        │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│  DURABLE SYSTEMS (Don't touch without approval)                                     │
│  ──────────────────────────────────────────────                                     │
│  │                                                                                  │
│  │  • Model Files: wan2.2_ti2v_5B_fp16.safetensors                                │
│  │  • Text Encoder: umt5_xxl_fp8_e4m3fn_scaled.safetensors                        │
│  │  • VAE: wan2.2_vae.safetensors                                                 │
│  │  • Sampling Config: shift=8 (model-specific, validated)                        │
│  │                                                                                  │
│  RUNTIME WORKFLOWS (Edit per-project)                                               │
│  ─────────────────────────────────────                                              │
│  │                                                                                  │
│  │  • ComfyUI workflow JSON                                                       │
│  │  • Prompt templates                                                            │
│  │  • Resolution/frame configurations                                             │
│  │  • Output paths and formats                                                    │
│  │                                                                                  │
│  EXPERIMENTAL / TUNABLE (Safe to iterate)                                           │
│  ──────────────────────────────────────────                                         │
│  │                                                                                  │
│  │  • CFG scale (3.0-8.0)                                                         │
│  │  • Steps (15-30)                                                               │
│  │  • Seed (any integer)                                                          │
│  │  • Prompt text                                                                 │
│  │  • Frame count (within VRAM limits)                                            │
│  │                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Multi-Level Architecture Diagrams

### Level 1: Business Goals → Architectural Components

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                  BUSINESS GOALS → ARCHITECTURAL COMPONENTS                           │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│   BUSINESS GOAL                              COMPONENT(S) THAT DELIVER IT            │
│   ─────────────                              ───────────────────────────             │
│                                                                                      │
│   ┌───────────────────────┐                  ┌───────────────────────────┐          │
│   │ "Generate videos from │─────────────────▶│ WAN 2.2 Ti2V Diffusion    │          │
│   │  text descriptions"   │                  │ Model (5B parameters)     │          │
│   └───────────────────────┘                  └───────────────────────────┘          │
│                                                                                      │
│   ┌───────────────────────┐                  ┌───────────────────────────┐          │
│   │ "Control video style  │─────────────────▶│ UMT5-XXL Text Encoder     │          │
│   │  with natural lang"   │                  │ (prompt understanding)    │          │
│   └───────────────────────┘                  └───────────────────────────┘          │
│                                                                                      │
│   ┌───────────────────────┐                  ┌───────────────────────────┐          │
│   │ "Start from existing  │─────────────────▶│ Wan22ImageToVideoLatent   │          │
│   │  image as first frame"│                  │ Node (I2V capability)     │          │
│   └───────────────────────┘                  └───────────────────────────┘          │
│                                                                                      │
│   ┌───────────────────────┐                  ┌───────────────────────────┐          │
│   │ "Run on our existing  │─────────────────▶│ High-compression VAE      │          │
│   │  24GB GPUs"           │                  │ (4096× ratio for TI2V-5B) │          │
│   └───────────────────────┘                  └───────────────────────────┘          │
│                                                                                      │
│   ┌───────────────────────┐                  ┌───────────────────────────┐          │
│   │ "Reproducible results │─────────────────▶│ KSampler (seed control)   │          │
│   │  for iteration"       │                  │ + saved workflow JSON     │          │
│   └───────────────────────┘                  └───────────────────────────┘          │
│                                                                                      │
│   ┌───────────────────────┐                  ┌───────────────────────────┐          │
│   │ "Visual workflow      │─────────────────▶│ ComfyUI (node editor +    │          │
│   │  editing for artists" │                  │  API for automation)      │          │
│   └───────────────────────┘                  └───────────────────────────┘          │
│                                                                                      │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

### Level 2: Architectural Components → Workflow Nodes

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│               ARCHITECTURAL COMPONENTS → WORKFLOW NODES                              │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│   COMPONENT                    IMPLEMENTED BY NODES                                  │
│   ─────────                    ────────────────────                                  │
│                                                                                      │
│   ┌─────────────────────────────────────────────────────────────────────────────┐   │
│   │ TEXT UNDERSTANDING                                                           │   │
│   │                                                                              │   │
│   │   CLIPLoader ────────────▶ CLIPTextEncode (Positive) ────┐                  │   │
│   │   (umt5_xxl_fp8)          CLIPTextEncode (Negative) ─────┼─▶ Conditioning   │   │
│   │                                                          │                   │   │
│   │   WHY: UMT5-XXL provides semantic understanding for      │                   │   │
│   │        prompt-to-video alignment. 100+ language support. │                   │   │
│   └──────────────────────────────────────────────────────────┘                   │   │
│                                                                                      │
│   ┌─────────────────────────────────────────────────────────────────────────────┐   │
│   │ IMAGE-TO-VIDEO CAPABILITY                                                    │   │
│   │                                                                              │   │
│   │   LoadImage ─────────────▶ Wan22ImageToVideoLatent ─────▶ Initial Latent    │   │
│   │   (start_image.png)       (encodes first frame)                             │   │
│   │                                                                              │   │
│   │   WHY: Allows control over starting composition.                            │   │
│   │        First frame is fixed, subsequent frames generated.                   │   │
│   └─────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                      │
│   ┌─────────────────────────────────────────────────────────────────────────────┐   │
│   │ DIFFUSION BACKBONE                                                           │   │
│   │                                                                              │   │
│   │   UNETLoader ────────────▶ ModelSamplingSD3 ────────────▶ KSampler          │   │
│   │   (wan2.2_ti2v_5B)        (shift=8)                      (denoising loop)   │   │
│   │                                                                              │   │
│   │   WHY: 5B DiT model with flow-matching. shift=8 is                          │   │
│   │        optimized for video temporal coherence.                              │   │
│   └─────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                      │
│   ┌─────────────────────────────────────────────────────────────────────────────┐   │
│   │ LATENT-TO-PIXEL CONVERSION                                                   │   │
│   │                                                                              │   │
│   │   VAELoader ─────────────▶ VAEDecode ───────────────────▶ Pixel Frames      │   │
│   │   (wan2.2_vae)            (31 latent → 121 frames)                          │   │
│   │                                                                              │   │
│   │   WHY: 3D VAE with 4096× compression enables 720P on 24GB.                  │   │
│   │        Temporal decompression 4× (31→121 frames).                           │   │
│   └─────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                      │
│   ┌─────────────────────────────────────────────────────────────────────────────┐   │
│   │ OUTPUT HANDLING                                                              │   │
│   │                                                                              │   │
│   │   CreateVideo ───────────▶ SaveVideo ───────────────────▶ MP4 File          │   │
│   │   (fps=24)                (output path)                                      │   │
│   │                                                                              │   │
│   │   WHY: Standard video format for downstream consumption.                    │   │
│   │        24fps matches training data assumptions.                             │   │
│   └─────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                      │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

### Level 3: Workflow Nodes → Tunable Variables

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                    WORKFLOW NODES → TUNABLE VARIABLES                                │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│   NODE                    VARIABLES                      BUSINESS IMPACT             │
│   ────                    ─────────                      ───────────────             │
│                                                                                      │
│   CLIPTextEncode          prompt (string)                Content/style control      │
│   (Positive)              ─────────────────────────────────────────────────────     │
│                           "Cinematic. A cat walks..."    Direct creative input      │
│                                                                                      │
│   CLIPTextEncode          negative_prompt (string)       Quality guardrails         │
│   (Negative)              ─────────────────────────────────────────────────────     │
│                           "" (empty default)             Rarely needed              │
│                                                                                      │
│   Wan22ImageTo            width (int)                    Resolution/cost            │
│   VideoLatent             height (int)                   Resolution/cost            │
│                           num_frames (int)               Duration/cost              │
│                           ─────────────────────────────────────────────────────     │
│                           1280, 704, 121                 5s @ 720P (default)        │
│                           960, 544, 49                   2s @ lower res (fast)      │
│                                                                                      │
│   KSampler                seed (int)                     Reproducibility            │
│                           steps (int)                    Quality vs speed           │
│                           cfg (float)                    Prompt adherence           │
│                           sampler_name (enum)            Algorithm choice           │
│                           scheduler (enum)               Noise schedule             │
│                           denoise (float)                Full vs partial gen        │
│                           ─────────────────────────────────────────────────────     │
│                           seed=random, steps=20,         Balanced defaults          │
│                           cfg=5.0, sampler=uni_pc,                                  │
│                           scheduler=simple, denoise=1.0                             │
│                                                                                      │
│   CreateVideo             fps (int)                      Playback speed             │
│                           ─────────────────────────────────────────────────────     │
│                           24                             Standard (matches model)   │
│                                                                                      │
│   ModelSamplingSD3        shift (int)                    ⚠️ DO NOT CHANGE           │
│                           ─────────────────────────────────────────────────────     │
│                           8                              Model-optimized value      │
│                                                                                      │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

---

## End-to-End Request Walkthrough

### Concept → Execution Trace

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                          END-TO-END REQUEST FLOW                                     │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│  PHASE 1: BUSINESS INTENT                                                           │
│  ════════════════════════                                                           │
│                                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐            │
│  │  Business Request:                                                   │            │
│  │  "Generate a 5-second product demo video showing our app in use"     │            │
│  │                                                                      │            │
│  │  Translated to:                                                      │            │
│  │  • Task: Text+Image-to-Video generation                             │            │
│  │  • Input: Product screenshot + descriptive prompt                   │            │
│  │  • Output: 720P video, 5 seconds @ 24fps                            │            │
│  │  • Quality: Production-ready (high steps, moderate CFG)             │            │
│  └─────────────────────────────────────────────────────────────────────┘            │
│                                      │                                               │
│                                      │ WHY: Business intent must map to             │
│                                      │      concrete generation parameters           │
│                                      ▼                                               │
│  PHASE 2: WORKFLOW SELECTION                                                        │
│  ════════════════════════════                                                       │
│                                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐            │
│  │  Decision Tree:                                                      │            │
│  │                                                                      │            │
│  │  Has starting image? ─────▶ YES ─────▶ Use Ti2V workflow             │            │
│  │         │                              (Wan22ImageToVideoLatent)     │            │
│  │         │                                                            │            │
│  │         └─────────────────▶ NO ──────▶ Use T2V workflow              │            │
│  │                                        (EmptyLatent + full denoise)  │            │
│  │                                                                      │            │
│  │  Selected: Ti2V workflow (we have product screenshot)               │            │
│  └─────────────────────────────────────────────────────────────────────┘            │
│                                      │                                               │
│                                      │ WHY: I2V mode gives control over              │
│                                      │      first frame composition                  │
│                                      ▼                                               │
│  PHASE 3: INPUT PREPARATION                                                         │
│  ══════════════════════════                                                         │
│                                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐            │
│  │  3A. PROMPT CONSTRUCTION                                             │            │
│  │      ─────────────────────                                           │            │
│  │      Template: "[Style]. [Scene]. [Subject]. [Motion]. [Camera]."   │            │
│  │                                                                      │            │
│  │      Example:                                                        │            │
│  │      "Clean, modern UI demonstration. A smartphone displays our     │            │
│  │       app interface with smooth transitions. A finger taps through  │            │
│  │       menu options. Camera holds steady on the device. Soft studio  │            │
│  │       lighting, shallow depth of field."                            │            │
│  │                                                                      │            │
│  │  3B. IMAGE PREPARATION                                               │            │
│  │      ─────────────────────                                           │            │
│  │      • Resolution: 1280×704 (match output)                          │            │
│  │      • Format: PNG (lossless)                                        │            │
│  │      • Content: Clear, high-quality product screenshot              │            │
│  │                                                                      │            │
│  │  3C. PARAMETER SELECTION                                             │            │
│  │      ─────────────────────                                           │            │
│  │      • steps: 25 (production quality)                               │            │
│  │      • cfg: 5.5 (moderate prompt adherence)                         │            │
│  │      • num_frames: 121 (5 seconds)                                  │            │
│  │      • seed: 42 (fixed for iteration)                               │            │
│  └─────────────────────────────────────────────────────────────────────┘            │
│                                      │                                               │
│                                      │ WHY: Structured prompt + quality image        │
│                                      │      maximize generation success              │
│                                      ▼                                               │
│  PHASE 4: WAN 2.2 EXECUTION                                                         │
│  ══════════════════════════                                                         │
│                                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐            │
│  │  STEP 4.1: Model Loading (~30 seconds, once)                        │            │
│  │  ─────────────────────────────────────────                          │            │
│  │  • UNETLoader: wan2.2_ti2v_5B_fp16.safetensors → GPU               │            │
│  │  • CLIPLoader: umt5_xxl_fp8_e4m3fn_scaled.safetensors → GPU        │            │
│  │  • VAELoader: wan2.2_vae.safetensors → GPU                         │            │
│  │                                                                      │            │
│  │  WHY EXISTS: Models must be in VRAM before inference                │            │
│  └─────────────────────────────────────────────────────────────────────┘            │
│                                      │                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐            │
│  │  STEP 4.2: Text Encoding (~2 seconds)                               │            │
│  │  ─────────────────────────────────────                              │            │
│  │  • CLIPTextEncode(positive) → context tensor (1, seq_len, 4096)    │            │
│  │  • CLIPTextEncode(negative) → null tensor (1, seq_len, 4096)       │            │
│  │                                                                      │            │
│  │  WHY EXISTS: Text must become embeddings for cross-attention        │            │
│  │                                                                      │            │
│  │  DATA SHAPE: "Clean, modern UI..." → float tensor [1, 128, 4096]   │            │
│  └─────────────────────────────────────────────────────────────────────┘            │
│                                      │                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐            │
│  │  STEP 4.3: Image Encoding + Latent Prep (~5 seconds)                │            │
│  │  ───────────────────────────────────────────────                    │            │
│  │  • LoadImage: PNG → pixel tensor (1, 704, 1280, 3)                 │            │
│  │  • Wan22ImageToVideoLatent:                                         │            │
│  │    - VAE encodes first frame → latent (1, 16, 1, 44, 80)           │            │
│  │    - Allocate noise for remaining frames                           │            │
│  │    - Concatenate → latent (1, 16, 31, 44, 80)                      │            │
│  │                                                                      │            │
│  │  WHY EXISTS: Diffusion operates in compressed latent space          │            │
│  │              First frame is fixed, others will be denoised          │            │
│  │                                                                      │            │
│  │  DATA SHAPE: 1280×704 image → (1, 16, 31, 44, 80) latent           │            │
│  │              [batch, channels, frames, height, width]               │            │
│  └─────────────────────────────────────────────────────────────────────┘            │
│                                      │                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐            │
│  │  STEP 4.4: Denoising Loop (~6-8 minutes @ 25 steps)                 │            │
│  │  ──────────────────────────────────────────────────                 │            │
│  │  FOR t in [1000, 960, 920, ... 40, 0]:  (25 steps)                 │            │
│  │      │                                                              │            │
│  │      ├─▶ Conditional forward pass                                  │            │
│  │      │   model(latent, t, positive_context) → noise_pred_cond      │            │
│  │      │                                                              │            │
│  │      ├─▶ Unconditional forward pass                                │            │
│  │      │   model(latent, t, null_context) → noise_pred_uncond        │            │
│  │      │                                                              │            │
│  │      ├─▶ Classifier-Free Guidance                                  │            │
│  │      │   noise = uncond + cfg × (cond - uncond)                    │            │
│  │      │                                                              │            │
│  │      └─▶ Scheduler step                                            │            │
│  │          latent = scheduler.step(noise, t, latent)                 │            │
│  │                                                                      │            │
│  │  WHY EXISTS: Iterative denoising is how diffusion models generate  │            │
│  │              CFG steers output toward prompt                        │            │
│  │                                                                      │            │
│  │  DATA SHAPE: (1, 16, 31, 44, 80) → (1, 16, 31, 44, 80)            │            │
│  │              [unchanged shape, but noise→signal]                   │            │
│  └─────────────────────────────────────────────────────────────────────┘            │
│                                      │                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐            │
│  │  STEP 4.5: VAE Decode (~30 seconds)                                 │            │
│  │  ──────────────────────────────────                                 │            │
│  │  • VAEDecode: latent (1, 16, 31, 44, 80) → frames                  │            │
│  │  • Temporal upsample: 31 → 121 frames                              │            │
│  │  • Spatial upsample: 44×80 → 704×1280                              │            │
│  │                                                                      │            │
│  │  WHY EXISTS: Convert compressed latent back to viewable pixels     │            │
│  │                                                                      │            │
│  │  DATA SHAPE: (1, 16, 31, 44, 80) → (121, 704, 1280, 3)            │            │
│  └─────────────────────────────────────────────────────────────────────┘            │
│                                      │                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐            │
│  │  STEP 4.6: Video Assembly + Save (~5 seconds)                       │            │
│  │  ────────────────────────────────────────────                       │            │
│  │  • CreateVideo: frames → video object @ 24fps                      │            │
│  │  • SaveVideo: video → output/product_demo_seed42.mp4               │            │
│  │                                                                      │            │
│  │  WHY EXISTS: Final deliverable must be playable video file         │            │
│  └─────────────────────────────────────────────────────────────────────┘            │
│                                      │                                               │
│                                      ▼                                               │
│  PHASE 5: OUTPUT & FEEDBACK                                                         │
│  ══════════════════════════                                                         │
│                                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐            │
│  │  5A. ASSET REGISTRATION                                              │            │
│  │      ─────────────────────                                           │            │
│  │      Store with metadata:                                            │            │
│  │      {                                                               │            │
│  │        "file": "product_demo_seed42.mp4",                           │            │
│  │        "workflow": "ti2v_v1.json",                                  │            │
│  │        "params": { "seed": 42, "steps": 25, "cfg": 5.5 },          │            │
│  │        "prompt": "Clean, modern UI demonstration...",               │            │
│  │        "input_image": "screenshot.png"                              │            │
│  │      }                                                               │            │
│  │                                                                      │            │
│  │  5B. QUALITY REVIEW                                                  │            │
│  │      ──────────────────                                              │            │
│  │      Reviewer checks:                                                │            │
│  │      • Motion coherence: Does it flow naturally?                    │            │
│  │      • Prompt adherence: Does it match description?                 │            │
│  │      • Artifacts: Any flickering, blurring, distortion?            │            │
│  │                                                                      │            │
│  │  5C. ITERATION (if needed)                                           │            │
│  │      ─────────────────────                                           │            │
│  │      • Adjust prompt for content changes                            │            │
│  │      • Adjust CFG for style changes                                 │            │
│  │      • Try new seed for variety                                     │            │
│  │      • Re-run with same params for comparison                       │            │
│  └─────────────────────────────────────────────────────────────────────┘            │
│                                                                                      │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Node & Dependency Breakdown

### Complete Node Graph with Dependencies

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                         NODE DEPENDENCY GRAPH                                        │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│  LEGEND:                                                                            │
│  ────────                                                                           │
│  [NODE] = ComfyUI node                                                              │
│  ─────▶ = Data dependency (downstream needs upstream output)                        │
│  (type) = Data type on connection                                                   │
│                                                                                      │
│                                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────────────┐   │
│  │  INDEPENDENT LOADERS (no upstream dependencies)                              │   │
│  │                                                                              │   │
│  │  [UNETLoader]           [CLIPLoader]           [VAELoader]    [LoadImage]   │   │
│  │  (wan2.2_ti2v           (umt5_xxl_fp8)         (wan2.2_vae)   (start.png)   │   │
│  │   _5B_fp16)                                                                  │   │
│  │       │                     │                      │              │          │   │
│  │       │                     │                      │              │          │   │
│  │   (MODEL)               (CLIP)                 (VAE)          (IMAGE)       │   │
│  └───────┼─────────────────────┼──────────────────────┼──────────────┼──────────┘   │
│          │                     │                      │              │              │
│          ▼                     │                      │              │              │
│  ┌────────────────┐            │                      │              │              │
│  │ModelSamplingSD3│            │                      │              │              │
│  │  (shift=8)     │            │                      │              │              │
│  └───────┬────────┘            │                      │              │              │
│          │                     │                      │              │              │
│      (MODEL)                   │                      │              │              │
│          │         ┌───────────┼───────────┐          │              │              │
│          │         │           │           │          │              │              │
│          │         ▼           ▼           │          ▼              ▼              │
│          │  ┌────────────┐ ┌────────────┐  │   ┌─────────────────────────────┐     │
│          │  │CLIPText    │ │CLIPText    │  │   │    Wan22ImageToVideoLatent  │     │
│          │  │Encode      │ │Encode      │  │   │                             │     │
│          │  │(positive)  │ │(negative)  │  │   │ width=1280, height=704,     │     │
│          │  │            │ │            │  │   │ num_frames=121, batch=1     │     │
│          │  └─────┬──────┘ └─────┬──────┘  │   └────────────┬────────────────┘     │
│          │        │              │         │                │                       │
│          │   (CONDITIONING) (CONDITIONING) │           (LATENT)                     │
│          │        │              │         │                │                       │
│          │        └──────────────┼─────────┼────────────────┘                       │
│          │                       │         │                                        │
│          ▼                       ▼         ▼                                        │
│  ┌───────────────────────────────────────────────────────────────────────────┐     │
│  │                              KSampler                                      │     │
│  │                                                                            │     │
│  │  INPUTS:                          PARAMETERS:                              │     │
│  │  ├─ model ◀── MODEL               ├─ seed: random                         │     │
│  │  ├─ positive ◀── CONDITIONING     ├─ steps: 20                            │     │
│  │  ├─ negative ◀── CONDITIONING     ├─ cfg: 5.0                             │     │
│  │  └─ latent_image ◀── LATENT       ├─ sampler_name: uni_pc                 │     │
│  │                                    ├─ scheduler: simple                    │     │
│  │  OUTPUT:                           └─ denoise: 1.0                         │     │
│  │  └─ LATENT ─────────────────────────────────────────────────────────────▶ │     │
│  └───────────────────────────────────────────────────────────────────────┬───┘     │
│                                                                          │         │
│                                                                      (LATENT)      │
│                                                                          │         │
│          ┌───────────────────────────────────────────────────────────────┘         │
│          │                                                                          │
│          ▼                                                                          │
│  ┌────────────────┐                                                                │
│  │   VAEDecode    │◀──────── (VAE) ◀── [VAELoader]                                 │
│  │                │                                                                 │
│  └───────┬────────┘                                                                │
│          │                                                                          │
│      (IMAGE)                                                                        │
│          │                                                                          │
│          ▼                                                                          │
│  ┌────────────────┐                                                                │
│  │  CreateVideo   │                                                                 │
│  │  (fps=24)      │                                                                 │
│  └───────┬────────┘                                                                │
│          │                                                                          │
│      (VIDEO)                                                                        │
│          │                                                                          │
│          ▼                                                                          │
│  ┌────────────────┐                                                                │
│  │   SaveVideo    │ ──────────────────────────────────────▶ output.mp4             │
│  │                │                                                                 │
│  └────────────────┘                                                                │
│                                                                                      │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

### Node-by-Node Detail Table

| Node | Upstream | Transformation | Downstream | Shape Change |
|------|----------|---------------|------------|--------------|
| **UNETLoader** | None (reads file) | Load weights → GPU tensor | ModelSamplingSD3 | None → (model weights ~10GB) |
| **CLIPLoader** | None (reads file) | Load weights → GPU tensor | CLIPTextEncode (×2) | None → (encoder weights ~5GB) |
| **VAELoader** | None (reads file) | Load weights → GPU tensor | Wan22ImageToVideoLatent, VAEDecode | None → (VAE weights ~200MB) |
| **LoadImage** | None (reads file) | PNG → pixel tensor | Wan22ImageToVideoLatent | file → (1, H, W, 3) |
| **ModelSamplingSD3** | UNETLoader | Configure flow-matching schedule | KSampler | MODEL → MODEL (with shift config) |
| **CLIPTextEncode** | CLIPLoader | Tokenize + embed prompt | KSampler | string → (1, seq_len, 4096) |
| **Wan22ImageToVideoLatent** | VAELoader, LoadImage | VAE encode + allocate latent | KSampler | (1, H, W, 3) → (1, 16, T/4, H/16, W/16) |
| **KSampler** | All above | Iterative denoising loop | VAEDecode | LATENT → LATENT (noise→signal) |
| **VAEDecode** | KSampler, VAELoader | Latent → pixel space | CreateVideo | (1, 16, 31, 44, 80) → (121, 704, 1280, 3) |
| **CreateVideo** | VAEDecode | Frame tensor → video object | SaveVideo | frames → video |
| **SaveVideo** | CreateVideo | Video → file write | None | video → .mp4 file |

---

## Variable → Outcome Mapping

### Complete Control Surface

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                    VARIABLE → BUSINESS OUTCOME MAP                                   │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│  VARIABLE          AFFECTS              WHEN TO ADJUST          IF MISCONFIGURED    │
│  ════════          ═══════              ══════════════          ═════════════════   │
│                                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────────────┐   │
│  │  SAFE ITERATION PARAMETERS (experiment freely)                               │   │
│  └─────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                      │
│  prompt            Video content        Every generation        Wrong content       │
│  (string)          and style                                    (fixable: re-prompt)│
│                                                                                      │
│  seed              Specific output      Reproducing or          None (just noise)   │
│  (0-2^64)          variation            varying results                             │
│                                                                                      │
│  steps             Quality/speed        Slow? Reduce.           <15: flickering     │
│  (15-30)           tradeoff             Need polish? Increase.  >30: diminishing    │
│                                                                                      │
│  cfg               Prompt adherence     Too creative? ↑         <3: ignores prompt  │
│  (3.0-8.0)         vs creativity        Too rigid? ↓            >8: oversaturated   │
│                                                                                      │
│  num_frames        Video duration       Match content needs     OOM if too high     │
│  (49/73/97/121)    and memory use       (formula: sec×24+1)     (reduce or crash)   │
│                                                                                      │
│  fps               Playback speed       Always 24 for this      Mismatch: weird     │
│  (24)              perception           model                   motion speed        │
│                                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────────────┐   │
│  │  STABILITY-CRITICAL PARAMETERS (change only with understanding)             │   │
│  └─────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                      │
│  width/height      Resolution and       When aspect ratio       Not ÷16: artifacts  │
│  (÷16)             memory use           needs change            Too high: OOM       │
│                                                                                      │
│  sampler_name      Sampling algorithm   If uni_pc has issues,   Wrong: quality drop │
│  (uni_pc/euler)    characteristics      try euler               or failure          │
│                                                                                      │
│  scheduler         Noise schedule       Rarely needed;          Wrong: major        │
│  (simple/normal)   curve                simple for flow models  artifacts           │
│                                                                                      │
│  denoise           Partial vs full      Only for img2img        <1.0 with no image: │
│  (0.0-1.0)         regeneration         variations              weird output        │
│                                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────────────┐   │
│  │  DO NOT TOUCH (model-specific, validated values)                             │   │
│  └─────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                      │
│  shift             Flow-matching        NEVER                   Completely breaks   │
│  (8)               schedule tuning                              generation          │
│                                                                                      │
│  model files       Core weights         NEVER (version          Incompatible:       │
│  (*.safetensors)                        controlled separately)  crash or garbage    │
│                                                                                      │
│  VAE model         Encode/decode        NEVER (matched to       Mismatch: corrupt   │
│  (wan2.2_vae)      capability           diffusion model)        output              │
│                                                                                      │
│  text encoder      Prompt embedding     NEVER (embeddings       Mismatch: prompt    │
│  (umt5_xxl_fp8)    format               format must match)      ignored             │
│                                                                                      │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

### Decision Matrix: What to Adjust First

| Problem | First Try | Second Try | Escalate To |
|---------|-----------|------------|-------------|
| **Video doesn't match prompt** | Rewrite prompt more specifically | Increase cfg to 6-7 | Different seed batch |
| **Motion too chaotic** | Decrease cfg to 4-4.5 | Describe motion in prompt | Add motion keywords |
| **Output too blurry** | Increase steps to 25-28 | Check input image quality | Try higher resolution |
| **Generation too slow** | Decrease steps to 15 | Reduce num_frames | Lower resolution |
| **Out of memory (OOM)** | Reduce num_frames first | Reduce resolution | Enable ComfyUI lowvram |
| **Temporal flickering** | Increase steps to 20+ | Check if cfg is too high | Re-run with new seed |
| **First frame ignored** | Ensure denoise=1.0 | Check image format/size | Verify I2V node config |

---

## Practical Operating Guidance

### Preset Configurations

#### Fast Preview (Iteration)

```json
{
  "purpose": "Quick previews during creative iteration",
  "steps": 15,
  "cfg": 4.5,
  "num_frames": 49,
  "width": 960,
  "height": 544,
  "estimated_time": "2-3 minutes",
  "estimated_vram": "~16GB"
}
```

**When to use**: Testing new prompts, exploring concepts, quick feedback loops.

#### Balanced Production

```json
{
  "purpose": "Standard quality for most use cases",
  "steps": 20,
  "cfg": 5.0,
  "num_frames": 121,
  "width": 1280,
  "height": 704,
  "estimated_time": "8-9 minutes",
  "estimated_vram": "~24GB"
}
```

**When to use**: Final assets, client deliverables, production content.

#### Maximum Quality

```json
{
  "purpose": "Highest quality for hero content",
  "steps": 28,
  "cfg": 5.5,
  "num_frames": 121,
  "width": 1280,
  "height": 704,
  "estimated_time": "12-15 minutes",
  "estimated_vram": "~24GB"
}
```

**When to use**: Featured content, marketing materials, final exports.

### Operational Checklist

#### Before Generation

- [ ] Models downloaded to correct directories
- [ ] ComfyUI workflow loaded
- [ ] Input image matches target resolution (if using I2V)
- [ ] Prompt follows `[Style]. [Scene]. [Subject]. [Motion]. [Camera].` structure
- [ ] VRAM availability confirmed (nvidia-smi)

#### During Generation

- [ ] Monitor GPU memory usage
- [ ] Note any error messages
- [ ] Track execution time for capacity planning

#### After Generation

- [ ] Review output for quality issues
- [ ] Log generation parameters with output
- [ ] Iterate on prompt if needed
- [ ] Archive seed and workflow for reproducibility

### Troubleshooting Quick Reference

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| ComfyUI hangs at start | Models loading | Wait 30-60 seconds first run |
| "CUDA out of memory" | num_frames or resolution too high | Reduce frames, try 960×544 |
| Black/blank output | VAE mismatch or corrupt model | Re-download model files |
| Prompt completely ignored | Text encoder issue or cfg=0 | Verify CLIPLoader, set cfg>0 |
| Video plays too fast/slow | fps mismatch | Set CreateVideo fps=24 |
| First frame doesn't match input | Wrong I2V node or denoise<1 | Use Wan22ImageToVideoLatent |

---

## Summary

### Why WAN 2.2 Exists in Our System

1. **Business need**: Automated, high-quality video generation from text/image inputs
2. **Cost efficiency**: Runs on existing 24GB consumer GPUs
3. **Quality**: State-of-the-art at accessible parameter count
4. **Reproducibility**: Seed-based control for iteration and versioning

### How It Extends Our Architecture

1. **Orchestration layer**: Unchanged (Conductor can queue video tasks)
2. **Execution layer**: NEW—ComfyUI + WAN 2.2 pipeline
3. **Storage layer**: Augmented with video format + generation metadata

### Where Engineers Intervene

| Level | What | Where | Risk |
|-------|------|-------|------|
| **Content** | Prompt, input image | CLIPTextEncode, LoadImage | None (creative) |
| **Quality** | steps, cfg, seed | KSampler | Low (tune safely) |
| **Resources** | resolution, frames | Wan22ImageToVideoLatent | Medium (OOM possible) |
| **Architecture** | model files, shift | Loaders, ModelSampling | High (don't touch) |

### Confidence Checklist

After reading this document, you should be able to:

- [ ] Explain why WAN 2.2 solves our video generation needs
- [ ] Trace a request from business intent → video output
- [ ] Identify which node performs which transformation
- [ ] Adjust safe parameters without breaking the pipeline
- [ ] Know which parameters must never change
- [ ] Troubleshoot common issues independently

---

*This document connects business objectives through architecture to implementation details. For isolated technical reference, see `wan2.2-ti2v-workflow-guide.md`.*

