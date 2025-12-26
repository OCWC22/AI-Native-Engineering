## WAN 2.2 (5B) TI2V ComfyUI Workflow — Deep Dive

- **Workflow JSON**: [video_wan2_2_5B_ti2v.json](https://raw.githubusercontent.com/Comfy-Org/workflow_templates/refs/heads/main/templates/video_wan2_2_5B_ti2v.json)
- **Official tutorial (linked inside the workflow note)**: [ComfyUI WAN 2.2 tutorial](https://docs.comfy.org/tutorials/video/wan/wan2_2)

This document explains the WAN 2.2 “ti2v” ComfyUI template end-to-end, with a focus on how **text + image inputs** are processed, how prompts/values flow through the graph, and what you can safely tweak.

---

### High-level overview

This template is designed to generate a **video** conditioned on:

- **Text**: a positive prompt + a negative prompt
- **Image**: a **starting image** (used to initialize the video latent)

It does this by loading three model components (diffusion backbone, text encoder, VAE), turning text into **conditioning embeddings**, turning the start image into an initial **video latent**, running a single **diffusion sampling** loop over that latent, decoding frames back to images, then encoding/saving them as a video.

---

### Architecture diagram / ASCII flow (exact graph wiring)

Types shown are ComfyUI socket types (`MODEL`, `CLIP`, `VAE`, `CONDITIONING`, `LATENT`, `IMAGE`, `VIDEO`).

```
(37) UNETLoader
   MODEL
     │
     ▼
(48) ModelSamplingSD3
   MODEL
     │
     ▼
(3)  KSampler  ── LATENT ──► (8) VAEDecode ── IMAGE ──► (57) CreateVideo ── VIDEO ──► (58) SaveVideo
  ▲     ▲   ▲   ▲                                            ▲
  │     │   │   │                                            │
  │     │   │   └── LATENT ◄── (55) Wan22ImageToVideoLatent ◄─┘
  │     │   │                    ▲             ▲
  │     │   │                    │             │
  │     │   │                  VAE           IMAGE
  │     │   │                    │             │
  │     │   │                   (39)         (56)
  │     │   │                VAELoader     LoadImage
  │     │   │
  │     │   └── CONDITIONING ◄── (7) CLIPTextEncode (Negative)
  │     │
  │     └── CONDITIONING ◄── (6) CLIPTextEncode (Positive)
  │
  └── CLIP  ◄── (38) CLIPLoader
```

---

### Overall execution flow (input → output) + dependencies

ComfyUI executes nodes **on-demand**: when you queue the workflow, the final sink node (`SaveVideo`) requires a `VIDEO`, which forces upstream computation. A typical dependency-respecting execution order:

- **Model + encoder loads**: `UNETLoader` → `ModelSamplingSD3`; `CLIPLoader`; `VAELoader`
- **Inputs**: `LoadImage`
- **Conditioning**: `CLIPTextEncode` (positive) + `CLIPTextEncode` (negative)
- **Latent init**: `Wan22ImageToVideoLatent`
- **Sampling**: `KSampler`
- **Decode + encode**: `VAEDecode` → `CreateVideo` → `SaveVideo`

Because ComfyUI caches node outputs, changing just prompts typically re-runs **only** the text-encode + sampler + downstream nodes (loaders stay cached unless you change their settings/files).

---

### Node-by-node breakdown (purpose, inputs/outputs, key parameters)

This template contains **13 nodes**.

#### (37) `UNETLoader` — load the WAN 2.2 diffusion backbone

- **Role**: Loads the main generative model weights (the “diffusion model” / denoiser network).
- **Outputs**: `MODEL` → feeds `ModelSamplingSD3`.
- **Widget values**:
  - **Model file**: `wan2.2_ti2v_5B_fp16.safetensors`
  - **Device**: `default`
- **Model link**: [wan2.2_ti2v_5B_fp16.safetensors](https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/diffusion_models/wan2.2_ti2v_5B_fp16.safetensors)

#### (48) `ModelSamplingSD3` — configure the model’s sampling behavior

- **Role**: Wraps/configures the `MODEL` so the sampler uses the **correct noise/sigma schedule conventions** for this model family.
- **Inputs**: `MODEL` from `(37)`.
- **Outputs**: `MODEL` → feeds `KSampler.model`.
- **Widget values**: `[8]`
  - Practically: treat this as a **model-specific sampling parameter**. If you change it without knowing exactly what it does, you can get unstable motion, washed output, or total failure.

#### (38) `CLIPLoader` — load the text encoder (UMT5 XXL)

- **Role**: Loads the text encoder used to embed prompts into conditioning vectors.
- **Outputs**: `CLIP` → feeds both text-encode nodes.
- **Widget values**:
  - **Encoder file**: `umt5_xxl_fp8_e4m3fn_scaled.safetensors`
  - **Encoder type/profile**: `wan`
  - **Device**: `default`
- **Text encoder link**: [umt5_xxl_fp8_e4m3fn_scaled.safetensors](https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors)

#### (6) `CLIPTextEncode (Positive Prompt)` — turn positive text into conditioning

- **Role**: Tokenizes + encodes the **positive** prompt using the loaded text encoder.
- **Inputs**: `clip` (`CLIP`) from `(38)`
- **Outputs**: `CONDITIONING` → feeds `KSampler.positive`
- **Widget values**: the positive prompt string.

#### (7) `CLIPTextEncode (Negative Prompt)` — negative conditioning

- **Role**: Encodes the **negative** prompt to steer the sampler away from undesired artifacts.
- **Inputs**: `clip` (`CLIP`) from `(38)`
- **Outputs**: `CONDITIONING` → feeds `KSampler.negative`
- **Widget values**: a long negative prompt (artifacts, subtitles/watermarks, deformations, etc.).

#### (39) `VAELoader` — load the WAN 2.2 VAE

- **Role**: Loads the VAE used to move between pixel space (`IMAGE`) and latent space (`LATENT`).
- **Outputs**: `VAE` → feeds both `Wan22ImageToVideoLatent` and `VAEDecode`
- **Widget values**:
  - **VAE file**: `wan2.2_vae.safetensors`
- **VAE link**: [wan2.2_vae.safetensors](https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/vae/wan2.2_vae.safetensors)

#### (56) `LoadImage` — load the start frame

- **Role**: Loads an image from disk to serve as the start image for the video.
- **Outputs**:
  - `IMAGE` → feeds `Wan22ImageToVideoLatent.start_image`
  - `MASK` (unused in this template)
- **Widget values**:
  - **Filename**: `example.png`
  - **Mode**: `image`

#### (55) `Wan22ImageToVideoLatent` — build an initial *video latent* from the start image

- **Role**: Converts the start image into the latent structure expected by the WAN 2.2 TI2V model for video generation.
- **Inputs**:
  - `vae` (`VAE`) from `(39)`
  - `start_image` (`IMAGE`) from `(56)`
- **Outputs**: `LATENT` → feeds `KSampler.latent_image`
- **Widget values**: `[1280, 704, 121, 1]`, interpreted as:
  - **Width**: 1280
  - **Height**: 704
  - **Frames / length**: 121
  - **Batch**: 1
- **What it’s doing (conceptually)**:
  - Resizes/prepares the start image to the target resolution.
  - Uses the VAE to encode image content into latent space.
  - Produces a *time-shaped* latent tensor suitable for the video model (not just a single-frame latent).

#### (3) `KSampler` — the core diffusion denoising loop

- **Role**: Runs iterative denoising on the latent video using:
  - the diffusion `MODEL`
  - positive/negative `CONDITIONING`
  - the initial `LATENT` (seeded from the start image)
- **Inputs**:
  - `model` from `(48)`
  - `positive` from `(6)`
  - `negative` from `(7)`
  - `latent_image` from `(55)`
- **Output**: `LATENT` → to `VAEDecode`
- **Widget values**: `[seed, seed_mode, steps, cfg, sampler, scheduler, denoise]`
  - **Steps**: 20
  - **CFG**: 5
  - **Sampler**: `uni_pc`
  - **Scheduler**: `simple`
  - **Denoise**: 1

#### (8) `VAEDecode` — convert latent frames back to images

- **Role**: Decodes the sampled video latent back into per-frame images.
- **Inputs**:
  - `samples` (`LATENT`) from `(3)`
  - `vae` (`VAE`) from `(39)`
- **Output**: `IMAGE` → to `CreateVideo`

#### (57) `CreateVideo` — pack frames into a video stream

- **Role**: Converts the frame sequence (`IMAGE`) into a `VIDEO` object at a given fps, optionally muxing audio.
- **Inputs**:
  - `images` (`IMAGE`) from `(8)`
  - `audio` (`AUDIO`) is **unconnected** in this template
- **Outputs**: `VIDEO` → to `SaveVideo`
- **Widget values**: `[24]` → **FPS = 24**

#### (58) `SaveVideo` — write the output video file

- **Role**: Encodes/writes the `VIDEO` to disk.
- **Input**: `video` (`VIDEO`) from `(57)`
- **Widget values**: `video/ComfyUI`, `auto`, `auto`
  - **Output directory**: `video/ComfyUI`
  - **Format/codec**: auto-selected

#### (59) `MarkdownNote` — documentation only (not part of execution)

- **Role**: A note listing model links and where to place them under `ComfyUI/models/`.
- **Contains links**:
  - [ComfyUI WAN 2.2 tutorial](https://docs.comfy.org/tutorials/video/wan/wan2_2)
  - [wan2.2_ti2v_5B_fp16.safetensors](https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/diffusion_models/wan2.2_ti2v_5B_fp16.safetensors)
  - [wan2.2_vae.safetensors](https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/vae/wan2.2_vae.safetensors)
  - [umt5_xxl_fp8_e4m3fn_scaled.safetensors](https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors)

---

### Variables and prompt reference (what you can edit)

This workflow has **no explicit “variable nodes”** (like string/int primitives). Everything is controlled by node **widget values**.

- **Prompts**
  - **Positive prompt**: `(6) CLIPTextEncode` text box
  - **Negative prompt**: `(7) CLIPTextEncode` text box

- **Start image**
  - **Filename**: `(56) LoadImage` → `example.png`

- **Video shape**
  - **Width/Height/Frames/Batch**: `(55) Wan22ImageToVideoLatent` → `1280 × 704`, `121` frames, batch `1`
  - **FPS**: `(57) CreateVideo` → `24`

- **Sampling**
  - **Seed/randomize**: `(3) KSampler`
  - **Steps**: `(3) KSampler` → `20`
  - **CFG**: `(3) KSampler` → `5`
  - **Sampler/Scheduler**: `(3) KSampler` → `uni_pc` / `simple`
  - **Denoise**: `(3) KSampler` → `1`
  - **ModelSamplingSD3 param**: `(48)` → `8` (model-specific; treat as fixed unless you know why)

- **Output**
  - **Directory + encoding defaults**: `(58) SaveVideo` → `video/ComfyUI`, `auto`, `auto`

**Prompt templating**: none is built in. Your “template” is simply how you write the prompt.

A high-performing structure:

- **Subject + setting** → **action** → **camera motion** → **lighting/mood** → **style constraints**
- Negative prompt: **artifacts + undesired motion/content + “no text/subtitles/watermarks”**

---

### How text + image are combined (conditioning mechanics)

In this graph there are **two conditioning pathways** that meet inside the sampler:

#### 1) Text → `CONDITIONING` (guides every denoising step)

- `(38) CLIPLoader` loads **UMT5 XXL**
- `(6)` and `(7)` encode the prompts to produce `CONDITIONING`
- `(3) KSampler` uses positive vs negative conditioning for **classifier-free guidance (CFG)**

#### 2) Image → initial `LATENT` (anchors appearance/composition)

- `(56) LoadImage` provides the start frame
- `(55) Wan22ImageToVideoLatent` converts it into a **video-shaped latent** (width/height/frames)
- That latent is the starting point for `(3) KSampler`

Net effect: **text tells the model “what should happen”**, while the **start image biases “what it should look like at the beginning.”**

---

### Underlying WAN 2.2 architecture (what’s going on under the hood)

From what the workflow loads, WAN 2.2 here is a classic **latent diffusion video system** composed of:

- **Text encoder**: UMT5 XXL → produces token embeddings for prompts
- **Latent-space video diffusion model**: `wan2.2_ti2v_5B_fp16` → denoises a spatio-temporal latent conditioned on text
- **VAE**: `wan2.2_vae` → maps pixel frames ↔ latent frames

Why this architecture is chosen (practically):

- **Latent diffusion** is far cheaper than pixel-space diffusion (critical for video).
- A large text encoder supports long, detailed prompts.
- A dedicated WAN VAE matches the latent distribution the diffusion model expects.
- `ModelSamplingSD3` likely enforces the correct model-specific sampling/sigma behavior.

How it differs from “older/alternative ComfyUI video workflows”:

- Compared to “image model + motion module” graphs (AnimateDiff-style), WAN is closer to an **end-to-end video model** (fewer moving parts, often stronger temporal coherence, less modular).
- Compared to workflows with ControlNet/IP-Adapter, this template is **simpler** but offers fewer explicit controls.


#### WAN 2.2 system + model architecture (repo-level view)

This workflow is a **ComfyUI surface** over the underlying WAN 2.2 system, which (at a high level) is a diffusion-based video generator with shared text/VAE components and multiple task-specialized backbones.

Primary architecture reference: [DeepWiki: Wan-Video/Wan2.2](https://deepwiki.com/Wan-Video/Wan2.2)

##### High-level design goals

From the WAN 2.2 project description, the core goals are:

- **High-quality, controllable video synthesis** across multiple tasks (T2V, I2V, TI2V, S2V, Animate).
- **Scalable deployment**: from a single consumer GPU (notably TI2V-5B) to multi-GPU distributed inference (FSDP + sequence parallelism).
- **Unified components**: shared text encoder and VAE across variants, with task-specialized diffusion backbones.

(See [DeepWiki: Wan-Video/Wan2.2](https://deepwiki.com/Wan-Video/Wan2.2))

##### Model variants (where TI2V-5B fits)

WAN 2.2 is described as a family of checkpoints optimized for different tasks and hardware targets:

| Variant | Primary task | Architecture | Notes |
| --- | --- | --- | --- |
| **T2V-A14B** | Text → Video | **MoE** (2 experts; 27B total / 14B active) | Enterprise / multi-GPU focus |
| **I2V-A14B** | Image → Video | **MoE** (2 experts; 27B total / 14B active) | Enterprise / multi-GPU focus |
| **TI2V-5B** | Text+Image → Video | **Dense** 5B transformer | Consumer-GPU oriented (24GB class) |
| **S2V-14B** | Speech → Video | Dense + audio encoder | Lip-sync focus |
| **Animate-14B** | Animation/replacement | Dense + face/CLIP components | Character workflows |

(Source: [DeepWiki: Wan-Video/Wan2.2](https://deepwiki.com/Wan-Video/Wan2.2))

**Important implication for this ComfyUI template:** the `wan2.2_ti2v_5B_fp16.safetensors` checkpoint in this workflow corresponds to the **dense TI2V-5B** variant (not the MoE A14B variants). That’s why the graph stays simple: you don’t need MoE routing logic exposed at the workflow level.

##### End-to-end pipeline (how text + image are processed)

Below is a “systems view” of what the WAN 2.2 TI2V pipeline is doing conceptually. This matches the ComfyUI graph, but uses model-module names.

```
Text prompt(s)
  │
  ├─► Text encoder (T5/UMT5 XXL) ──► text embeddings / conditioning
  │
Start image
  │
  └─► VAE encoder (AutoencoderKLWan2) ──► image latent

Latent init (TI2V)
  └─► build spatio-temporal latent (resolution + length)

Diffusion sampling loop
  for t in timesteps:
    ├─► (optional) CFG combine: positive vs negative conditioning
    ├─► Wan diffusion backbone (dense DiT-style transformer for TI2V-5B)
    └─► update latent using scheduler/solver

VAE decode
  └─► latent frames → RGB frames

Video writer
  └─► RGB frames + fps → video file
```

Shared components are described as:

- **Text encoder**: `T5EncoderModel` using `google/umt5-xxl` (WAN uses a UMT5/T5-family text encoder). In ComfyUI WAN templates this is surfaced via `CLIPLoader`/`CLIPTextEncode`, but functionally it’s a T5-style encoder.
- **VAE**: `AutoencoderKLWan2` (WAN-specific VAE), surfaced as `VAELoader` + `VAEDecode`.

(See [DeepWiki: Wan-Video/Wan2.2](https://deepwiki.com/Wan-Video/Wan2.2))

##### High-compression VAE (why TI2V-5B is practical on 24GB GPUs)

A key TI2V-5B design point is **very aggressive latent compression** to reduce the spatio-temporal token count during diffusion.

DeepWiki summarizes the compression as:

- **Temporal compression**: **4×**
- **Spatial compression**:
  - “Standard” WAN VAE: **16×16** spatial compression (combined with 4× temporal = 1024× overall)
  - **TI2V-5B**: **32×32** spatial compression (combined with 4× temporal = 4096× overall)
- Plus an **additional 2×2 patchification** (effectively increasing efficiency for TI2V-5B)

(Source: [DeepWiki: Wan-Video/Wan2.2](https://deepwiki.com/Wan-Video/Wan2.2))

A helpful mental model for shape changes:

```
Pixel video:        [T, H, W, 3]
VAE latent (TI2V):  [T/4, H/32, W/32, C]
(tokens/patches):  sequence length shrinks further (implementation-specific)
Diffusion runs primarily over this compressed representation.
```

This is the *architectural* reason you can target “720P @ 24fps” workloads with TI2V-5B on consumer GPUs, while larger variants (A14B MoE) typically require multi-GPU.

##### MoE routing (only for T2V-A14B and I2V-A14B)

WAN 2.2’s Mixture-of-Experts design applies to the **A14B** T2V and I2V models:

- Two experts total (~13.5B each), **27B parameters** total
- **14B active per timestep** (constant inference cost)
- Routing is based on **SNR**: a “high-noise” expert is used earlier in denoising, and a “low-noise” expert later for refinement
- The switch threshold is described via `t_moe` (linked to an SNR threshold)

(Source: [DeepWiki: Wan-Video/Wan2.2](https://deepwiki.com/Wan-Video/Wan2.2))

ASCII intuition:

```
noise high (early timesteps)  ──► Expert A (structure / motion planning)
noise low  (late timesteps)   ──► Expert B (detail / refinement)
```

Again: **TI2V-5B is dense**, so you typically won’t think about `t_moe` when working with this specific ComfyUI workflow.

##### Mapping: ComfyUI nodes ↔ WAN 2.2 components

This mapping helps you reason about “what to change” in ComfyUI in terms of the underlying architecture:

| ComfyUI node | What it corresponds to | Primary output |
| --- | --- | --- |
| `UNETLoader` | WAN diffusion backbone weights (TI2V-5B) | `MODEL` |
| `ModelSamplingSD3` | Model-specific sampling configuration (sigma/noise conventions) | `MODEL` |
| `CLIPLoader` | WAN’s UMT5/T5 text encoder weights | `CLIP` |
| `CLIPTextEncode` | Prompt tokenization + text embedding | `CONDITIONING` |
| `VAELoader` | WAN VAE (`AutoencoderKLWan2`) | `VAE` |
| `Wan22ImageToVideoLatent` | VAE encode + latent initialization for TI2V | `LATENT` |
| `KSampler` | Diffusion denoising loop + CFG + scheduler/solver | `LATENT` |
| `VAEDecode` | VAE decode latent → frames | `IMAGE` |
| `CreateVideo` | Frame sequence → video container | `VIDEO` |
| `SaveVideo` | Encode/write video | file |

##### Key insights for modification (architecture-first)

- **Quality vs. speed is dominated by token count**: resolution × frames × (how much the VAE compresses). For TI2V-5B, the high-compression VAE is the enabler; your safest knobs are still **resolution, frames, steps**.
- **Text conditioning is shared across variants**: improvements in prompting generally transfer across tasks because WAN uses a large T5-family encoder.
- **If you need “more control”** (pose, depth, edges, identity locks), you’re not changing the WAN backbone—you’re adding **extra conditioning paths** in ComfyUI (LoRA/adapters/Control-like modules) and merging conditioning.
- **If you need “more raw quality”** beyond what TI2V-5B can deliver, you’re in “model variant selection” territory (MoE A14B models, multi-GPU inference), not just workflow tweaks.



##### Internal mechanics: diffusion backbone (dense DiT vs. MoE)

DeepWiki describes WAN’s diffusion backbone as a **DiT-style transformer** (Diffusion Transformer). Conceptually, it behaves like a denoiser function:

- **Inputs**:
  - A **noisy latent video** (in TI2V-5B, heavily VAE-compressed)
  - A **timestep / noise level** (what stage of denoising we’re at)
  - **Text conditioning embeddings** (from UMT5/T5)
- **Output**:
  - A **denoising direction** (often described as “predicted noise / residual”), which the sampler uses to update the latent

A useful mental diagram:

```
noisy latent tokens + time embedding
            │
            ▼
   transformer blocks (spatio-temporal attention)
            │
            ├─► cross-attend to text embeddings (conditioning)
            │
            ▼
   predicted denoising residual ("noise estimate")
            │
            ▼
   scheduler/solver step → next latent
```

Where this shows up in ComfyUI:

- `KSampler` is the **outer denoising loop** (scheduler + solver + CFG)
- `UNETLoader` provides the **denoiser weights** (called “UNet” in ComfyUI, but WAN is transformer-style internally)
- `ModelSamplingSD3` ensures the **model-specific sampling conventions** are applied correctly

(Architecture source: [DeepWiki: Wan-Video/Wan2.2](https://deepwiki.com/Wan-Video/Wan2.2))

##### Code structure (Wan2.2 repo) — where to modify what

If you’re modifying WAN 2.2 itself (not just the ComfyUI workflow), DeepWiki summarizes a code-level layout like:

- **Entry point**: `generate.py` parses CLI args and routes to the correct pipeline
- **Pipeline classes**:
  - `WanT2V`, `WanI2V`, `WanTI2V` (image/video generation pipelines)
  - `WanS2V` (speech-driven)
  - `WanAnimate` (animation/replacement)
- **Backbone**: `WanModel` implements the diffusion transformer (and MoE routing for the A14B variants)
- **Config**: `WAN_CONFIGS`, `SIZE_CONFIGS`, `MAX_AREA_CONFIGS` manage resolution/limits
- **Distributed inference**:
  - FSDP sharding utilities (`shard_model()`, `free_model()`) in `fsdp.py`
  - sequence parallel / distributed init (`init_distributed_group()`)

(Source: [DeepWiki: Wan-Video/Wan2.2](https://deepwiki.com/Wan-Video/Wan2.2))

##### System-level variables / parameters (repo / CLI) and what they control

These aren’t exposed directly in this ComfyUI template, but they explain **why some variants require multi-GPU** and how inference is scaled/optimized.

| Variable / flag | What it controls | Why it exists | How changing it affects behavior |
| --- | --- | --- | --- |
| `--task` (e.g. `ti2v-5B`) | Which WAN variant pipeline/checkpoint to run | Different tasks need different backbones | Changes capability + compute profile |
| `--offload_model` | Offload inactive model parts to CPU | Fit bigger models in limited VRAM | Slower, but can prevent OOM |
| `--convert_model_dtype` | Weight precision (FP16/BF16, etc.) | Reduce VRAM, speed up | Too aggressive can reduce quality/stability |
| `--t5_cpu` | Run T5 encoder on CPU | Save VRAM (text encoder is large) | Slower prompt encoding; frees VRAM |
| `--dit_fsdp`, `--t5_fsdp` | FSDP shard DiT / T5 across GPUs | Run models larger than a single GPU | Enables A14B MoE variants |
| `--ulysses_size N` | Sequence-parallel size | Parallelize attention/sequence compute | Enables higher resolution/longer sequences |
| `t_moe` (config) | MoE expert switching threshold (A14B only) | Use different experts for early vs late denoising | Can trade structure vs refinement behavior |

(Source: [DeepWiki: Wan-Video/Wan2.2](https://deepwiki.com/Wan-Video/Wan2.2))


---

### LoRAs / adapters / extra conditioning (what this workflow does and doesn’t do)

This template **does not load or apply any LoRA/adapters**. There are no `LoraLoader`, ControlNet, IP-Adapter, or conditioning-merge nodes.

If you add them, safe injection points (conceptually):

- **LoRA on the diffusion model**: inject **between** `(37) UNETLoader` and `(48) ModelSamplingSD3`.
- **LoRA for the text encoder**: inject on the `CLIP` path between `(38) CLIPLoader` and `(6)/(7) CLIPTextEncode`.
- **Adapters (ControlNet/IP-Adapter-like)**: produce extra `CONDITIONING`, then **merge** into positive conditioning before `KSampler.positive`.

---

### Training/data assumptions (what we know vs. what we can infer)

The workflow itself doesn’t encode training dataset metadata, so anything beyond model component choice is inference.

- **Known from the workflow**
  - Trained/packaged to use **UMT5 XXL** embeddings and a **WAN-specific VAE**.
  - A **TI2V** backbone suggests training included image-conditioned video generation (e.g., conditioning on a reference/start frame).

- **Reported training data improvements (Wan2.2 README summary)**
  - **+83.2% more training videos** (motion diversity)
  - **+65.6% more training images** (semantic coverage)
  - **Aesthetic curation** (labels for lighting, composition, contrast, color tone control)
  - **Cinematic-quality data** emphasis

  (Source: [DeepWiki: Wan-Video/Wan2.2](https://deepwiki.com/Wan-Video/Wan2.2))

- **Reasonable inferences**
  - The model likely learned camera/motion language from captioned video data.
  - Multilingual prompt tolerance depends on the training mix.

---

### Tuning guide (what to change, what to keep fixed)

#### Safe, high-impact knobs (recommended ranges)

- **Prompts (most important)**
  - Positive: be explicit about **subject**, **action**, **setting**, **camera motion**, **style**
  - Negative: keep it focused; too many constraints can suppress detail/motion

- **Start image**
  - Use a clean, high-quality image with the desired subject framing.
  - Align prompt with the image to reduce identity/style conflicts.

- **Steps** (`KSampler.steps`, default 20)
  - Try **15–30**

- **CFG** (`KSampler.cfg`, default 5)
  - Try **3–7**

- **Frames / length** (`Wan22ImageToVideoLatent.frames`, default 121)
  - Longer = more compute and more drift risk.
  - Many video systems prefer lengths of the form **8n+1** (121 = 8×15 + 1).

- **Resolution** (`Wan22ImageToVideoLatent.width/height`, default 1280×704)
  - Expect VRAM/time to scale sharply with pixels × frames.
  - Keep width/height **multiples of 64** (these defaults are).

- **FPS** (`CreateVideo`, default 24)
  - Changes playback speed only; doesn’t change number of generated frames.

- **Seed mode**
  - Use a fixed seed for reproducibility while tuning.

- **Denoise** (`KSampler.denoise`, default 1)
  - For stronger start-image adherence: **0.6–0.9**
  - For more creative divergence: **1.0**

#### Settings to keep fixed (unless you know exactly why)

- Keep the **model trio matched**: WAN diffusion model + WAN VAE + expected text encoder.
- Treat **`ModelSamplingSD3` value (8)** as part of the correct sampler config.
- Keep `CLIPLoader` profile as **`wan`**.

---

### Why this workflow was chosen (trade-offs)

This is the “minimum viable WAN 2.2 TI2V graph”:

- **Pros**
  - Robust, few moving parts.
  - Easy to tune: prompts + a handful of sampler/video knobs.
  - Bakes in correct model sampling wrapper.

- **Cons**
  - Less control than larger graphs (no LoRA slot, no ControlNet/IP-Adapter wiring).
  - Image influence is mainly through latent initialization rather than explicit adapters.

---

### Common pitfalls & best practices

- **OOM / VRAM blowups**
  - Reduce resolution, frames, steps; keep batch = 1.

- **Warped aspect ratio / stretched subjects**
  - Match the start image aspect ratio to the target; avoid aggressive resizing.

- **“Static video” (no motion)**
  - Add explicit motion/camera cues; lower CFG slightly; try another seed.

- **“Start image ignored”**
  - Lower denoise (0.7–0.9) and reduce prompt conflicts.

---

### Source references

- Architecture reference: [DeepWiki: Wan-Video/Wan2.2](https://deepwiki.com/Wan-Video/Wan2.2)
- Workflow template: [video_wan2_2_5B_ti2v.json](https://raw.githubusercontent.com/Comfy-Org/workflow_templates/refs/heads/main/templates/video_wan2_2_5B_ti2v.json)
- Tutorial: [ComfyUI WAN 2.2 tutorial](https://docs.comfy.org/tutorials/video/wan/wan2_2)
- Model files (as referenced by the workflow):
  - [wan2.2_ti2v_5B_fp16.safetensors](https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/diffusion_models/wan2.2_ti2v_5B_fp16.safetensors)
  - [wan2.2_vae.safetensors](https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/vae/wan2.2_vae.safetensors)
  - [umt5_xxl_fp8_e4m3fn_scaled.safetensors](https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors)
