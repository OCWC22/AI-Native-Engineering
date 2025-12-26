# ComfyUI WAN 2.2 Ti2V (Text-to-Image-to-Video) Workflow: Deep Dive

> **Workflow Source**: [Comfy-Org Official Template](https://raw.githubusercontent.com/Comfy-Org/workflow_templates/refs/heads/main/templates/video_wan2_2_5B_ti2v.json)
> 
> **Last Updated**: December 2024

---

## Table of Contents

1. [High-Level Overview](#high-level-overview)
2. [Architecture Diagram](#architecture-diagram)
3. [Node-by-Node Breakdown](#node-by-node-breakdown)
4. [Underlying Architecture of WAN 2.2](#underlying-architecture-of-wan-22)
5. [Data Flow: Text + Image → Video](#data-flow-text--image--video)
6. [Variable & Parameter Reference](#variable--parameter-reference)
7. [Prompt Engineering Guide](#prompt-engineering-guide)
8. [Training Data & Assumptions](#training-data--assumptions)
9. [Tuning Guide](#tuning-guide)
10. [Common Pitfalls & Best Practices](#common-pitfalls--best-practices)
11. [Why This Workflow?](#why-this-workflow)
12. [Summary](#summary)

---

## High-Level Overview

The WAN 2.2 workflow is designed for **Text-to-Image-to-Video (Ti2V)** generation using the WAN 2.2 5B parameter model. This workflow:

1. Takes a **text prompt** describing the desired video content
2. Optionally takes a **starting image** as the first frame
3. Generates a coherent video sequence that follows the text description while maintaining temporal consistency

### Key Capabilities

| Capability | Value |
|------------|-------|
| **Resolution** | 1280×704 (16:9 aspect ratio) |
| **Frame Count** | 121 frames (~5 seconds at 24fps) |
| **Model Size** | 5 billion parameters (fp16) |
| **Architecture** | Diffusion Transformer (DiT) based |

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                           STEP 1: MODEL LOADING                                      │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│  ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐                │
│  │   UNETLoader    │     │   CLIPLoader    │     │   VAELoader     │                │
│  │  (wan2.2_ti2v   │     │ (umt5_xxl_fp8)  │     │ (wan2.2_vae)    │                │
│  │   _5B_fp16)     │     │                 │     │                 │                │
│  └────────┬────────┘     └────────┬────────┘     └────────┬────────┘                │
│           │                       │                       │                          │
│           ▼                       │                       │                          │
│  ┌─────────────────┐              │                       │                          │
│  │ ModelSamplingSD3│              │                       │                          │
│  │   (shift=8)     │              │                       │                          │
│  └────────┬────────┘              │                       │                          │
│           │                       │                       │                          │
└───────────┼───────────────────────┼───────────────────────┼──────────────────────────┘
            │                       │                       │
            ▼                       ▼                       ▼
┌───────────────────────────────────────────────────────────────────────────────────────┐
│                           STEP 2: CONDITIONING                                        │
├───────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                       │
│           ┌───────────────────────┴───────────────────────┐                          │
│           │                                               │                          │
│           ▼                                               ▼                          │
│  ┌─────────────────┐                             ┌─────────────────┐                 │
│  │ CLIPTextEncode  │                             │ CLIPTextEncode  │                 │
│  │   (POSITIVE)    │                             │   (NEGATIVE)    │                 │
│  │                 │                             │                 │                 │
│  │ "Low contrast.  │                             │ (empty)         │                 │
│  │  In a retro..." │                             │                 │                 │
│  └────────┬────────┘                             └────────┬────────┘                 │
│           │                                               │                          │
└───────────┼───────────────────────────────────────────────┼──────────────────────────┘
            │                                               │
            ▼                                               ▼
┌───────────────────────────────────────────────────────────────────────────────────────┐
│                           STEP 3: LATENT PREPARATION (Optional I2V)                   │
├───────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                       │
│  ┌─────────────────┐     ┌─────────────────────────┐                                 │
│  │   LoadImage     │────▶│ Wan22ImageToVideoLatent │                                 │
│  │ (start_image)   │     │                         │                                 │
│  └─────────────────┘     │  width: 1280            │                                 │
│                          │  height: 704            │                                 │
│           ┌──────────────│  num_frames: 121        │                                 │
│           │              │  batch_size: 1          │                                 │
│           ▼              └───────────┬─────────────┘                                 │
│       [VAE] ─────────────────────────┤                                               │
│                                      │                                               │
└──────────────────────────────────────┼───────────────────────────────────────────────┘
                                       │
                                       ▼
┌───────────────────────────────────────────────────────────────────────────────────────┐
│                           STEP 4: SAMPLING (DENOISING)                                │
├───────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                       │
│                          ┌─────────────────────────┐                                 │
│       MODEL ────────────▶│       KSampler          │                                 │
│       POSITIVE ─────────▶│                         │                                 │
│       NEGATIVE ─────────▶│  seed: random           │                                 │
│       LATENT ───────────▶│  steps: 20              │                                 │
│                          │  cfg: 5                 │                                 │
│                          │  sampler: uni_pc        │                                 │
│                          │  scheduler: simple      │                                 │
│                          │  denoise: 1.0           │                                 │
│                          └───────────┬─────────────┘                                 │
│                                      │                                               │
└──────────────────────────────────────┼───────────────────────────────────────────────┘
                                       │
                                       ▼
┌───────────────────────────────────────────────────────────────────────────────────────┐
│                           STEP 5: DECODE & OUTPUT                                     │
├───────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                       │
│  ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐                 │
│  │    VAEDecode    │────▶│   CreateVideo   │────▶│    SaveVideo    │                 │
│  │                 │     │   (fps: 24)     │     │                 │                 │
│  └─────────────────┘     └─────────────────┘     └─────────────────┘                 │
│           ▲                                                                          │
│           │                                                                          │
│       [VAE] ◀────────────────────────────────────────────────────────────────────────│
│                                                                                       │
└───────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Node-by-Node Breakdown

### 1. Model Loading Nodes

| Node | ID | Purpose | Output |
|------|-----|---------|--------|
| **UNETLoader** | 37 | Loads the WAN 2.2 5B diffusion model (fp16) | MODEL |
| **CLIPLoader** | 38 | Loads UMT5-XXL text encoder (fp8 quantized) | CLIP |
| **VAELoader** | 39 | Loads WAN 2.2 specific VAE | VAE |

**Model Files Required:**

```
ComfyUI/
├── models/
│   ├── diffusion_models/
│   │   └── wan2.2_ti2v_5B_fp16.safetensors   # ~10GB
│   ├── text_encoders/
│   │   └── umt5_xxl_fp8_e4m3fn_scaled.safetensors  # ~5GB
│   └── vae/
│       └── wan2.2_vae.safetensors  # ~200MB
```

**Download Links:**
- [wan2.2_ti2v_5B_fp16.safetensors](https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/diffusion_models/wan2.2_ti2v_5B_fp16.safetensors)
- [umt5_xxl_fp8_e4m3fn_scaled.safetensors](https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors)
- [wan2.2_vae.safetensors](https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/vae/wan2.2_vae.safetensors)

### 2. Model Sampling Configuration

| Node | ID | Purpose | Key Parameter |
|------|-----|---------|---------------|
| **ModelSamplingSD3** | 48 | Configures flow-matching sampling | shift=8 |

**Why shift=8?**
- WAN 2.2 uses a **flow-matching** training paradigm (like SD3/Flux)
- The shift parameter controls the noise schedule
- Higher shift = more emphasis on high-frequency details early in denoising
- Value of 8 is optimized for video temporal coherence

### 3. Text Encoding Nodes

| Node | ID | Purpose | Content |
|------|-----|---------|---------|
| **CLIPTextEncode (Positive)** | 6 | Encodes the descriptive prompt | Full scene description |
| **CLIPTextEncode (Negative)** | 7 | Encodes what to avoid | Empty by default |

**Example Positive Prompt (from workflow):**

```
Low contrast. In a retro 1970s-style subway station, a street musician 
plays in dim colors and rough textures. He wears an old jacket, playing 
guitar with focus. Commuters hurry by, and a small crowd gathers to listen. 
The camera slowly moves right, capturing the blend of music and city noise, 
with old subway signs and mottled walls in the background.
```

### 4. Latent Preparation (Image-to-Video)

| Node | ID | Purpose | Parameters |
|------|-----|---------|------------|
| **LoadImage** | 56 | Loads starting frame image | User-provided image |
| **Wan22ImageToVideoLatent** | 55 | Prepares latent space for I2V | width, height, num_frames |

**Parameters:**
- `width`: 1280 pixels
- `height`: 704 pixels  
- `num_frames`: 121 frames
- `batch_size`: 1

### 5. Sampling (Core Generation)

| Node | ID | Purpose |
|------|-----|---------|
| **KSampler** | 3 | Main diffusion sampling loop |

**Sampler Parameters:**

| Parameter | Value | Description |
|-----------|-------|-------------|
| `seed` | Random | Controls generation randomness |
| `steps` | 20 | Number of denoising steps |
| `cfg` | 5.0 | Classifier-free guidance scale |
| `sampler_name` | uni_pc | Fast predictor-corrector sampler |
| `scheduler` | simple | Linear noise schedule |
| `denoise` | 1.0 | Full denoising (0=none, 1=full) |

### 6. Output Nodes

| Node | ID | Purpose | Parameters |
|------|-----|---------|------------|
| **VAEDecode** | 8 | Converts latent → pixel space | Uses VAE |
| **CreateVideo** | 57 | Assembles frames into video | fps=24 |
| **SaveVideo** | 58 | Writes video file to disk | Format: auto |

---

## Underlying Architecture of WAN 2.2

### Model Architecture

WAN 2.2 is built on a **Diffusion Transformer (DiT)** architecture:

```
┌─────────────────────────────────────────────────────────────────┐
│                     WAN 2.2 Architecture                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐                                           │
│  │   UMT5-XXL      │  Text Encoder                             │
│  │   (Multilingual)│  - 4.7B parameters                        │
│  │                 │  - Supports 100+ languages                │
│  └────────┬────────┘  - Deep semantic understanding            │
│           │                                                     │
│           ▼                                                     │
│  ┌─────────────────────────────────────────────────┐           │
│  │           3D VAE (Spatial + Temporal)           │           │
│  │                                                 │           │
│  │  Image/Video → Latent (8x compression spatial) │           │
│  │                (4x compression temporal)        │           │
│  └────────┬────────────────────────────────────────┘           │
│           │                                                     │
│           ▼                                                     │
│  ┌─────────────────────────────────────────────────┐           │
│  │     Diffusion Transformer (DiT) - 5B params     │           │
│  │                                                 │           │
│  │  ┌─────────────────────────────────────────┐   │           │
│  │  │  3D Attention Blocks                    │   │           │
│  │  │  - Spatial self-attention               │   │           │
│  │  │  - Temporal self-attention              │   │           │
│  │  │  - Cross-attention (text conditioning)  │   │           │
│  │  └─────────────────────────────────────────┘   │           │
│  │                                                 │           │
│  │  ┌─────────────────────────────────────────┐   │           │
│  │  │  Flow Matching Training                 │   │           │
│  │  │  - Continuous time diffusion            │   │           │
│  │  │  - Better convergence than DDPM         │   │           │
│  │  └─────────────────────────────────────────┘   │           │
│  └─────────────────────────────────────────────────┘           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Key Architectural Choices

| Component | Choice | Why |
|-----------|--------|-----|
| **Text Encoder** | UMT5-XXL | Multilingual, better semantic understanding than CLIP |
| **Backbone** | DiT (not UNet) | Better scaling, more coherent long-range dependencies |
| **Training** | Flow Matching | Faster convergence, better sample quality |
| **VAE** | 3D Causal VAE | Temporal compression + causality for video |
| **Precision** | fp16 model, fp8 text encoder | Memory efficiency with minimal quality loss |

### How Ti2V (Text+Image to Video) Works

1. **Image Encoding**: Starting image encoded to latent via 3D VAE
2. **Latent Initialization**: First frame latent is fixed, remaining frames initialized with noise
3. **Denoising**: Model learns to denoise while maintaining consistency with first frame
4. **Temporal Attention**: Cross-frame attention ensures motion coherence

---

## Data Flow: Text + Image → Video

```
INPUT PHASE:
────────────────────────────────────────────────────────────
Text Prompt ──────────────────┐
"Low contrast. In a retro..." │
                              ▼
                    ┌─────────────────┐
                    │ UMT5-XXL Encode │
                    │ (77 → 512 tokens│
                    │  embeddings)    │
                    └────────┬────────┘
                             │
                             ▼
                    [Conditioning Tensor]
                    Shape: (1, seq_len, 4096)


Start Image ──────────────────┐
(1280x704 RGB)                │
                              ▼
                    ┌─────────────────┐
                    │   3D VAE Encode │
                    │  (spatial 8x,   │
                    │   temporal 4x)  │
                    └────────┬────────┘
                             │
                             ▼
                    [Latent Tensor]
                    Shape: (1, 16, 31, 160, 88)
                    (batch, channels, frames, h, w)


GENERATION PHASE:
────────────────────────────────────────────────────────────
                    ┌─────────────────────────┐
                    │       KSampler          │
                    │                         │
Timestep t=T ──────▶│  For t = T...0:        │
                    │    1. Predict noise     │
                    │    2. Apply guidance    │
                    │    3. Update latent     │
                    │                         │
                    │  CFG = 5.0              │
                    │  x_t = x_t - ε_θ(...)   │
                    └────────┬────────────────┘
                             │
                             │ (20 iterations)
                             ▼
                    [Denoised Latent]
                    Shape: (1, 16, 31, 160, 88)


OUTPUT PHASE:
────────────────────────────────────────────────────────────
                    ┌─────────────────┐
                    │   VAE Decode    │
                    │  (31 latent     │
                    │   → 121 frames) │
                    └────────┬────────┘
                             │
                             ▼
                    [Image Tensor]
                    Shape: (121, 704, 1280, 3)

                    ┌─────────────────┐
                    │  CreateVideo    │
                    │  (24 fps)       │
                    └────────┬────────┘
                             │
                             ▼
                    [MP4 Video File]
                    ~5 seconds @ 24fps
```

---

## Variable & Parameter Reference

### User-Adjustable Parameters

| Parameter | Location | Default | Range | Effect |
|-----------|----------|---------|-------|--------|
| **Prompt** | CLIPTextEncode (id:6) | Example text | Any descriptive text | Describes what to generate |
| **Negative Prompt** | CLIPTextEncode (id:7) | Empty | Any text | What to avoid |
| **Seed** | KSampler | Random | 0 - 2^64 | Reproducibility |
| **Steps** | KSampler | 20 | 15-30 | Quality vs speed |
| **CFG Scale** | KSampler | 5.0 | 3.0-8.0 | Prompt adherence |
| **Width** | Wan22ImageToVideoLatent | 1280 | 512-1920 | Output width |
| **Height** | Wan22ImageToVideoLatent | 704 | 320-1080 | Output height |
| **Num Frames** | Wan22ImageToVideoLatent | 121 | 33-241 | Video length |
| **FPS** | CreateVideo | 24 | 12-60 | Playback speed |

### Fixed/Advanced Parameters (Change with Caution)

| Parameter | Location | Value | Warning |
|-----------|----------|-------|---------|
| **Shift** | ModelSamplingSD3 | 8 | Model-specific, don't change |
| **Sampler** | KSampler | uni_pc | uni_pc/euler work best |
| **Scheduler** | KSampler | simple | Flow models use simple/normal |
| **Denoise** | KSampler | 1.0 | Only reduce for img2img |
| **Weight dtype** | UNETLoader | default | fp16 for quality |

---

## Prompt Engineering Guide

### Prompt Structure

WAN 2.2 responds well to structured prompts:

```
[Style/Aesthetic]. [Scene Description]. [Subject details]. [Actions/Motion]. 
[Camera movement]. [Atmosphere/Mood].
```

### Effective Prompt Elements

| Element | Example | Purpose |
|---------|---------|---------|
| **Style** | "Low contrast", "Cinematic", "Anime style" | Visual aesthetic |
| **Scene** | "In a retro 1970s subway station" | Environment |
| **Subject** | "A street musician plays guitar" | Main focus |
| **Motion** | "Commuters hurry by" | What moves |
| **Camera** | "Camera slowly moves right" | Viewpoint change |
| **Details** | "Old subway signs, mottled walls" | Texture/atmosphere |

### Prompt Tips

1. **Be specific about motion**: "slowly walks" vs "moves"
2. **Describe camera movement**: "tracking shot", "pan left", "zoom in"
3. **Include temporal cues**: "then", "while", "as"
4. **Avoid conflicting instructions**: Don't ask for both "still" and "moving"

### Example Prompts

**Cinematic Scene:**
```
Cinematic, 4K. A lone astronaut walks across a red Martian desert at sunset. 
Dust swirls around their boots. The camera slowly pulls back to reveal 
towering mountains in the distance. Golden hour lighting, lens flare.
```

**Action Sequence:**
```
Dynamic action shot. A motorcycle races through neon-lit Tokyo streets at night.
Rain reflects colorful signs on wet pavement. Camera tracks alongside the rider,
capturing motion blur and streaming lights. Cyberpunk aesthetic.
```

**Nature Documentary:**
```
Wildlife documentary style. A majestic eagle soars over snow-capped mountains.
Its wings catch the morning light as it circles slowly. Aerial perspective,
crisp details on feathers. Peaceful, contemplative mood.
```

---

## Training Data & Assumptions

### Inferred Training Data

Based on model behavior and Alibaba/Wan team publications:

| Data Type | Estimated Volume | Source Types |
|-----------|------------------|--------------|
| **Video clips** | ~14M clips | Stock footage, web videos |
| **Image-text pairs** | ~1B pairs | LAION, internal datasets |
| **High-quality subset** | ~2M curated | Professional cinematography |

### Training Assumptions

1. **Resolution**: Trained on 720p-1080p content, best results at native resolutions
2. **Duration**: 2-6 second clips, struggles with longer coherence
3. **Motion**: Trained on natural motion, may struggle with:
   - Extreme speed changes
   - Complex multi-object interactions
   - Physics-defying motion

4. **Style Strengths**:
   - ✅ Cinematic footage
   - ✅ Natural scenes
   - ✅ Human subjects
   - ✅ Animals and wildlife
   
5. **Style Weaknesses**:
   - ❌ Abstract art
   - ❌ Technical diagrams
   - ❌ Text rendering
   - ❌ Hands and fine details

---

## Tuning Guide

### What to Change (Safe)

| Goal | Parameter | Adjustment |
|------|-----------|------------|
| **Faster generation** | Steps | Reduce to 15 |
| **Higher quality** | Steps | Increase to 25-30 |
| **More creative** | CFG | Reduce to 3-4 |
| **Stricter prompt following** | CFG | Increase to 6-7 |
| **Longer video** | num_frames | Increase (33 frame increments) |
| **Different aspect ratio** | width/height | Keep multiples of 16 |

### Recommended Presets

**Fast Preview:**
```json
{
  "steps": 15,
  "cfg": 4.5,
  "num_frames": 49,
  "note": "~2 seconds, quick iteration"
}
```

**Balanced (Default):**
```json
{
  "steps": 20,
  "cfg": 5.0,
  "num_frames": 121,
  "note": "~5 seconds, good quality"
}
```

**High Quality:**
```json
{
  "steps": 28,
  "cfg": 5.5,
  "num_frames": 121,
  "note": "Best quality, slower"
}
```

### Resolution Guidelines

| Aspect Ratio | Dimensions | Use Case |
|--------------|------------|----------|
| 16:9 | 1280×704 | Standard widescreen (default) |
| 16:9 | 960×544 | Faster generation |
| 9:16 | 704×1280 | Vertical/mobile |
| 1:1 | 960×960 | Square format |
| 21:9 | 1280×544 | Ultra-wide cinematic |

> **Note**: Width and height must be divisible by 16.

### Frame Count Formula

```
frames = (seconds × 24) + 1
```

| Duration | Frames |
|----------|--------|
| 2 seconds | 49 |
| 3 seconds | 73 |
| 4 seconds | 97 |
| 5 seconds | 121 |

---

## Common Pitfalls & Best Practices

### Pitfalls to Avoid

| Pitfall | Why It Happens | Solution |
|---------|----------------|----------|
| **VRAM OOM** | 5B model + long video | Reduce num_frames or resolution |
| **Temporal flickering** | Too few steps | Increase steps to 20+ |
| **Motion blur** | Fast motion at low FPS | Match generation fps to playback |
| **Prompt ignored** | CFG too low | Increase CFG to 5-6 |
| **Oversaturated** | CFG too high | Reduce CFG below 6 |
| **First frame mismatch** | Poor I2V image | Use high-quality, clear first frame |
| **Artifacts at edges** | Resolution not divisible by 16 | Use standard resolutions |

### Best Practices

1. **Start with default settings** - they're optimized for the model
2. **Use I2V mode** when you need specific starting composition
3. **Keep prompts concise** - UMT5 handles ~150 words well
4. **Test at low resolution first** - scale up for final render
5. **Save seeds** of good generations for reproducibility
6. **Use negative prompts sparingly** - model is trained for positive guidance

### VRAM Requirements

| Resolution | Frames | Estimated VRAM |
|------------|--------|----------------|
| 960×544 | 49 | ~16GB |
| 1280×704 | 49 | ~20GB |
| 1280×704 | 121 | ~24GB |
| 1920×1080 | 121 | ~40GB+ |

### Memory Optimization Tips

1. Enable `--lowvram` or `--medvram` flags in ComfyUI
2. Use fp8 text encoder (already default)
3. Reduce batch size to 1
4. Close other GPU-intensive applications
5. Consider using model offloading nodes

---

## Why This Workflow?

### Comparison with Alternatives

| Workflow | Pros | Cons |
|----------|------|------|
| **WAN 2.2 5B Ti2V** | High quality, good motion, multilingual | High VRAM, slower |
| **WAN 2.1 1.3B** | Lower VRAM, faster | Lower quality |
| **AnimateDiff** | Works with any SD1.5 model | Limited motion, shorter clips |
| **SVD (Stable Video Diffusion)** | Good quality | I2V only, no text control |
| **CogVideoX** | Alternative architecture | Different training focus |

### Why Choose WAN 2.2 5B Ti2V?

1. **Quality**: State-of-the-art video generation at 5B scale
2. **Flexibility**: Supports both T2V and I2V modes
3. **Text Understanding**: UMT5-XXL provides excellent prompt comprehension
4. **Temporal Coherence**: 3D attention maintains consistent motion
5. **Multilingual**: Native support for 100+ languages
6. **Active Development**: Comfy-Org maintains official nodes

### Trade-offs

| Factor | WAN 2.2 5B | Alternative |
|--------|------------|-------------|
| **Quality** | ★★★★★ | ★★★☆☆ (smaller models) |
| **Speed** | ★★☆☆☆ | ★★★★☆ (AnimateDiff) |
| **VRAM** | 24GB+ | 8-12GB (smaller models) |
| **Control** | ★★★★☆ | ★★★★★ (ControlNet options) |

---

## Summary

The ComfyUI WAN 2.2 Ti2V workflow is a production-ready pipeline for generating high-quality short videos from text prompts and optional starting images.

### Quick Reference

| Aspect | Details |
|--------|---------|
| **Input** | Text prompt + optional starting image |
| **Processing** | UMT5 encodes text → DiT generates in latent space → VAE decodes to pixels |
| **Output** | 24fps video file (MP4) |
| **Key adjustables** | Prompt, seed, steps (15-30), CFG (3-8), resolution, frame count |
| **Keep fixed** | Model sampling shift, sampler type, scheduler |

### Getting Started Checklist

- [ ] Download all three model files to correct directories
- [ ] Load the workflow JSON in ComfyUI
- [ ] Write your prompt in the positive text encode node
- [ ] (Optional) Load a starting image for I2V mode
- [ ] Adjust resolution and frame count as needed
- [ ] Click Queue Prompt to generate

### Resources

- [ComfyUI Documentation](https://docs.comfy.org/)
- [WAN 2.2 Tutorial](https://docs.comfy.org/tutorials/video/wan/wan2_2)
- [Comfy-Org Workflow Templates](https://github.com/Comfy-Org/workflow_templates)
- [HuggingFace Model Repository](https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged)

---

*For best results, start with defaults, iterate on prompts, and scale resolution/frames as needed for your VRAM budget.*

