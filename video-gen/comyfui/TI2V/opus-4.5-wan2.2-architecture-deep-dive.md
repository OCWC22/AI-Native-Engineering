# WAN 2.2 Architecture: Deep Technical Reference

> **Source Documentation**: [DeepWiki - Wan-Video/Wan2.2](https://deepwiki.com/Wan-Video/Wan2.2)
> 
> **GitHub Repository**: [Wan-Video/Wan2.2](https://github.com/Wan-Video/Wan2.2)
> 
> **Last Updated**: December 2024

---

## Table of Contents

1. [High-Level Overview](#high-level-overview)
2. [System Architecture Diagram](#system-architecture-diagram)
3. [Model Variants](#model-variants)
4. [Core Components Deep Dive](#core-components-deep-dive)
5. [Mixture-of-Experts (MoE) Architecture](#mixture-of-experts-moe-architecture)
6. [VAE Architecture](#vae-architecture)
7. [Text Encoder (UMT5-XXL)](#text-encoder-umt5-xxl)
8. [End-to-End Execution Flow](#end-to-end-execution-flow)
9. [Node-by-Node Data Flow](#node-by-node-data-flow)
10. [Variable and Parameter Reference](#variable-and-parameter-reference)
11. [Memory Optimization Strategies](#memory-optimization-strategies)
12. [Key Insights and Modification Guidelines](#key-insights-and-modification-guidelines)

---

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

