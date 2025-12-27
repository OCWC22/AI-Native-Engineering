# Text + Image → Video as a General-Purpose Generative Medium

**Purpose**: Research + primitives exploration (not a feature proposal).  
**Product context**: AGI Inc is an AI-powered productivity concierge for knowledge workers (language-centric: chat, reasoning, automation, search, memory, tool orchestration).

---

## The Problem This Solves (For the CEO)

### The Current Pain: Understanding Transfer is Expensive

Right now, when someone needs to explain something complex:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    HOW UNDERSTANDING TRANSFER WORKS TODAY                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  EXAMPLE 1: Business Problem Chain                                          │
│  ──────────────────────────────────                                          │
│                                                                              │
│  Me: "Why did Q3 revenue miss?"                                             │
│  Claude: [500 words explaining 4 factors]                                   │
│  Me: "Wait, how did factor 2 cause factor 3?"                               │
│  Claude: [300 words of clarification]                                       │
│  Me: "Can you show me the timeline?"                                        │
│  Claude: [ASCII diagram that's hard to parse]                               │
│  Me: "I still don't see the causal chain..."                                │
│                                                                              │
│  ↓ 20 minutes later ↓                                                       │
│  Understanding achieved: ~70%                                               │
│  Transferable to others: NO (they'd need the same 20-min chat)             │
│                                                                              │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                              │
│  EXAMPLE 2: Technical Architecture                                          │
│  ─────────────────────────────────                                          │
│                                                                              │
│  Me: "Explain how matrix multiplication runs on a Blackwell GPU—            │
│       how work is partitioned across SMs, accelerated by tensor             │
│       cores, and how NVLink provides high-bandwidth data transfer           │
│       between GPUs—and why this matters for writing performant vLLM."       │
│                                                                              │
│  Claude: [1200 words, technically correct, dense]                           │
│  Me: "I need to re-read the SM partitioning part..."                        │
│  Me: "Wait, which step is the bottleneck?"                                  │
│  Me: "Can you draw a diagram?"                                              │
│  Claude: [Static diagram that doesn't show temporal flow]                   │
│                                                                              │
│  ↓ 45 minutes and 3 diagrams later ↓                                        │
│  Understanding achieved: ~85%                                               │
│  Transferable to others: BARELY (I'd need to re-explain it all)            │
│                                                                              │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                              │
│  EXAMPLE 3: Marketing Idea                                                  │
│  ─────────────────────────────                                              │
│                                                                              │
│  Me: "Help me explain our product to enterprise buyers"                     │
│  Claude: [Positioning document]                                             │
│  Me: "How do I show them the before/after?"                                │
│  Claude: [Comparison table]                                                 │
│  Me: "I need them to FEEL the difference, not just read it"               │
│                                                                              │
│  ↓ Still text ↓                                                             │
│  Impact: Medium (they'll skim it)                                          │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### What Video Changes

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      THE SAME EXAMPLES WITH VIDEO                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  EXAMPLE 1: Business Problem Chain                                          │
│  ──────────────────────────────────                                          │
│                                                                              │
│  Input:                                                                     │
│  • "Explain why Q3 revenue missed"                                          │
│  • [Attaches: revenue chart, sales data, 2 incident reports]               │
│                                                                              │
│  Output: 90-second video                                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  [0:00] Revenue target vs actual (animated chart)                   │   │
│  │  [0:15] "Here's what happened..." (timeline appears)               │   │
│  │  [0:25] Factor 1 → Factor 2 (arrow animates, shows causation)      │   │
│  │  [0:40] Factor 2 → Factor 3 (zooms into the mechanism)             │   │
│  │  [0:55] "The root cause was..." (highlights key node)              │   │
│  │  [1:15] "For Q4, this means..." (forward projection)               │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  Time to understand: 90 seconds                                             │
│  Transferable: YES (send the video to anyone)                              │
│                                                                              │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                              │
│  EXAMPLE 2: Technical Architecture                                          │
│  ─────────────────────────────────                                          │
│                                                                              │
│  Input:                                                                     │
│  • "Explain matrix multiplication on Blackwell GPUs for vLLM"              │
│  • [Attaches: architecture diagram, Nvidia whitepaper excerpt]             │
│                                                                              │
│  Output: 3-minute video                                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  [0:00] "Here's the full GPU. Let's zoom in..."                    │   │
│  │  [0:20] SMs light up as work is partitioned                        │   │
│  │  [0:45] Tensor cores animate matrix blocks moving through          │   │
│  │  [1:15] NVLink shows data flowing between GPUs                     │   │
│  │  [1:45] "Here's where the bottleneck happens..."                   │   │
│  │  [2:15] "In your vLLM code, this means..."                         │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  Time to understand: 3 minutes                                              │
│  Retention: HIGH (saw it happen, not just read about it)                   │
│  Transferable: YES (every new engineer watches this)                       │
│                                                                              │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                              │
│  EXAMPLE 3: Marketing Idea                                                  │
│  ─────────────────────────────                                              │
│                                                                              │
│  Input:                                                                     │
│  • "Show enterprise buyers the before/after of using AGI"                  │
│  • [Attaches: product screenshots, customer quote, workflow diagram]       │
│                                                                              │
│  Output: 60-second product video                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  [0:00] "This is how your team works today..." (painful flow)      │   │
│  │  [0:20] Friction points highlight in red                           │   │
│  │  [0:30] "With AGI..." (smooth flow animates)                       │   │
│  │  [0:45] Time saved appears as counter                              │   │
│  │  [0:55] "Get started in 5 minutes"                                 │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  Impact: HIGH (they felt the difference, not just read it)                 │
│  Reusable: YES (landing page, sales deck, email)                           │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Why This Matters: Humans Are Visual

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    THE FUNDAMENTAL INSIGHT                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  TEXT                              VIDEO                                     │
│  ────                              ─────                                     │
│  Sequential                        Parallel (see everything at once)       │
│  Abstract                          Concrete (see the thing happening)       │
│  Re-readable (but who does?)       Re-watchable (and people actually do)   │
│  Precise                           Intuitive                                │
│  Skimmable                         Guided attention                         │
│  Fast to produce                   Slower to produce                        │
│  Fast to consume (for experts)     Fast to consume (for everyone)          │
│                                                                              │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                              │
│  KEY POINT:                                                                 │
│                                                                              │
│  Text optimizes for the PRODUCER (fast to write).                          │
│  Video optimizes for the CONSUMER (fast to understand).                    │
│                                                                              │
│  For complex/hard-to-understand concepts:                                   │
│  • More people can understand visually                                     │
│  • Understanding is faster                                                  │
│  • Understanding is transferable                                           │
│  • Understanding sticks                                                     │
│                                                                              │
│  The trade-off is production time. The question is:                        │
│  CAN WE MAKE PRODUCTION FAST ENOUGH?                                       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Why Current AI Video Isn't Useful

**ChatGPT, Gemini, Runway, etc. can generate video. So why isn't this solved?**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│               WHY EXISTING AI VIDEO IS TOO GENERAL TO BE USEFUL              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  THE PROBLEM:                                                               │
│                                                                              │
│  Current tools optimize for:          Knowledge workers need:               │
│  ───────────────────────────          ──────────────────────                │
│  "Make impressive visuals"     vs     "Explain THIS specific thing"        │
│  Open-ended creativity         vs     Constrained, grounded outputs        │
│  Photorealistic renders        vs     Clear diagrams + motion              │
│  General-purpose prompts       vs     Structured intent                    │
│  One-shot generation           vs     Iterative editing                    │
│                                                                              │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                              │
│  ANALOGY TO CODE:                                                           │
│                                                                              │
│  Before Claude Code, LLMs could "write code."                              │
│  But they couldn't:                                                         │
│  • Work with YOUR files                                                     │
│  • Understand YOUR context                                                  │
│  • Iterate reliably                                                         │
│  • Produce production-ready outputs                                        │
│                                                                              │
│  Claude Code made LLMs useful for code by:                                  │
│  ✓ Constraining the scope (code, not "everything")                         │
│  ✓ Providing the right primitives (files, diffs, execution)               │
│  ✓ Being opinionated (about what good looks like)                          │
│  ✓ Enabling iteration (edit, run, see, repeat)                             │
│                                                                              │
│  The same opportunity exists for video.                                     │
│                                                                              │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                              │
│  THE CLAUDE CODE PATTERN:                                                   │
│                                                                              │
│  "General AI tool" → useless                                               │
│  "Scoped + constrained + modular harness" → useful                         │
│                                                                              │
│  For video, the constraint is:                                              │
│  NOT: "Generate any video"                                                  │
│  BUT: "Generate explanatory/demo videos grounded in your specific inputs"  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### The Strategic Timing Argument

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          WHY NOW                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  THE TREND:                                                                 │
│                                                                              │
│  Video generation inference is getting:                                     │
│  • FASTER (5 min → 30 sec within 18 months)                                │
│  • CHEAPER (10x cost reduction per generation)                              │
│  • EASIER (better APIs, smaller models, local inference)                   │
│                                                                              │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                              │
│  THE FUTURE:                                                                │
│                                                                              │
│  Generative UI is coming regardless.                                        │
│  The question is not IF but WHO builds the harness.                        │
│                                                                              │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                              │
│  THE OPPORTUNITY:                                                           │
│                                                                              │
│  Designing the harness RIGHT NOW positions AGI to be a leading lab         │
│  when the cost/latency curves cross the utility threshold.                  │
│                                                                              │
│  Claude Code wasn't built after LLMs got good at code.                      │
│  It was built while they were getting good—and it shaped the paradigm.     │
│                                                                              │
│  Same opportunity here.                                                     │
│                                                                              │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                              │
│  WHAT WE'RE PROPOSING:                                                      │
│                                                                              │
│  Not "build an AI video product."                                           │
│  But "design and prototype the harness primitives"—so when video           │
│  inference becomes fast/cheap enough, we're ready to ship.                 │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### The One-Line Pitch

> **Today, understanding transfer requires expensive back-and-forth or custom video production. We can build a harness that turns intent + grounded inputs into explanatory video—making understanding transferable, reusable, and instant.**

---

## Core Reframe

Don't think "AI video for learning."  
Think: **a foundational generative interface that turns intent + structure into visual motion, narrative, and explanation.**

Claude Code worked because it:
- Identified the right primitives (files, diffs, execution, feedback)
- Constrained the problem space (code, not "everything")
- Delivered immediate utility for real workflows
- Scaled from novice → expert without changing tools

This explores whether **text + image → video** can have an analogous "primitive" role for **visual + temporal + narrative work**.

## Goal

Explore whether text + image → generative video can become a useful, opinionated, general-purpose primitive for:
- Explanation
- Communication
- Persuasion
- Demonstration
- Sense-making

Focus areas:
- Where this medium provides real leverage
- Where it breaks down
- Which primitives must exist for it to be useful (not flashy)
- How it might integrate with / sit adjacent to AGI’s existing system

## Mental model

Reason from three angles:
- **Creator**: marketing, education, storytelling
- **Engineer**: systems thinking, primitives, constraints
- **Knowledge worker**: speed, clarity, reusability, intent-driven output

Assume users care about:
- Turning vague intent into concrete output
- Visual clarity over visual polish
- Reusability across contexts (learning, sales, internal, external)
- Control without micromanagement

---

## 1) Opportunity space (what this could be)

Text excels at **precision, scanability, reference**.  
Slides excel at **spatial chunking** and **presentational control**.  
Video is uniquely strong when the problem requires **causality over time** plus **attention control**.

### Where video can be a better abstraction (high leverage cases)
- **Temporal causality**: “this leads to that” (pipelines, incidents, org decisions, latency trade-offs)
- **Procedures + demos**: “do X, then Y” (product flows, onboarding, operational runbooks)
- **Progressive disclosure**: reveal complexity in layers without overwhelming
- **Guided persuasion**: pacing, emphasis, contrast (before/after, risk/reward) are inherently temporal
- **Sense-making from messy inputs**: turn a memo + a diagram + screenshots into a coherent narrative arc

### Where video adds friction (low leverage / wrong tool)
- **Reference work**: policies, specs, checklists, “what’s the API again?”
- **Fast iteration needs**: without strong edit primitives, video iteration is slower than text/slides
- **Auditability/trust**: “show the exact source for this claim” is harder in pure video
- **Non-linear consumption**: knowledge work often needs skimming, jumping, and search

**Reframe**: video isn’t “better content.” It’s a way to **compile intent into a time-based artifact** that reliably carries a message to an audience that won’t read the doc.

---

## 2) Core primitives (the most important section)

If this becomes “Claude Code for visual narrative,” the win will not be prettier renders.  
The win comes from a small set of **irreducible, editable, composable primitives**—a “Video IR” (intermediate representation) analogous to “files + diffs” in programming.

### Essential primitives (must exist)

#### A) Intent contract (what success means)
Make success explicit instead of buried in a prompt:
- Audience + purpose (explain / persuade / demo / align)
- Tone + voice constraints
- Allowed claims (and prohibited claims)
- Required disclaimers
- Target duration + depth level
- Outcome: “What should the viewer believe/know/do at the end?”

Why it matters: without an intent contract, the system optimizes for “looks plausible” instead of “moves the viewer to the desired understanding/action.”

#### B) Grounded inputs + provenance (what the video is allowed to use)
Inputs are first-class:
- Reference docs, diagrams, screenshots, charts, datasets, brand assets
- Rights/permissions metadata (logos, people, copyrighted imagery)
- Privacy + redaction constraints (especially for screenshots)

Provenance hooks:
- Every substantive claim maps to a source (or is explicitly marked as an **unsupported assertion**)

Why it matters: video is persuasive; ungrounded claims are higher-risk than in text.

#### C) Narrative structure (time as a typed object)
Represent temporal structure explicitly:
- Beat sheet / scene list
- Each beat has a rhetorical function (setup → problem → mechanism → trade-off → decision → next steps)
- Emphasis markers: “must remember” vs “optional color”

Why it matters: the system needs a stable scaffold to revise without rewriting everything.

#### D) Visual grammar (constrained semantics, not aesthetics)
Define a small set of visual “atoms” that carry meaning:
- Boxes/edges (systems), timelines, highlights/callouts
- Comparisons (A vs B), counters, state machines
- Step lists, UI cursor demos, chart reveals

Provide templates for common knowledge-work visuals:
- Architecture walkthrough
- Decision timeline / incident timeline
- Trade-off table (with emphasis)
- Lifecycle / funnel / flow

This is the likely “**code, not everything**” equivalent:
- **Explanatory motion graphics + grounded UI/diagram animation**
- Not open-ended “cinema” generation

Why it matters: open-ended style flexibility increases cost + reduces editability without increasing clarity for most workflows.

#### E) Temporal control (pacing + sync)
Explicit pacing primitives:
- Duration per beat
- Dwell time on key frames
- Transition types
- “Pause for comprehension”

Narration synchronization:
- On-screen changes aligned with spoken segments

Why it matters: mis-timed visuals undermine comprehension even if assets are good.

#### F) Edit operations (iteration without re-prompting)
Users need *meaning-level* edits:
- “Shorten scene 3”
- “Move the caveat earlier”
- “Make trade-offs section 2× longer”
- “Reduce jargon”
- “Swap this diagram with the updated one”
- “Replace the example with our company metrics”

Stability across revisions:
- Scene IDs + semantic anchors so small edits don’t cascade into full regeneration

Why it matters: professionals won’t iterate via repeated full rerolls; they need reliable deltas.

#### G) Determinism + diffability (professional requirement)
Treat output as an artifact with a stable representation:
- Versioned Video IR
- Reproducible renders (settings + seeds)
- Diffs at the IR level (“script changed, visuals unchanged”) rather than opaque re-generation

Why it matters: review, compliance, and collaboration require auditability.

#### H) Trust + QA gates (because video persuades)
Automated checks before final render:
- Unsupported claims / missing sources
- Prohibited topics, privacy leakage (screenshots), brand compliance
- Accessibility: captions, readability (font size/contrast)

Refuse/degrade gracefully:
- If grounding is missing, produce a **storyboard/animatic** with explicit TODOs instead of confident-looking fabrications

Why it matters: “looks real” + “wrong” is worse than “incomplete but honest.”

### Optional primitives (nice, not foundational)
- Cinematic camera language
- Character animation
- Complex 3D
- Photorealism

These often increase latency/cost and reduce editability without improving clarity for knowledge work.

### Actively harmful if overexposed
- Frame-level controls and keyframe micromanagement
- Dozens of style knobs
- Prompt-only iteration with no structured intermediate artifact

These push users into aesthetics and away from manipulating **meaning and structure**.

---

## 3) Example workflows (lightweight, illustrative)

### Architecture walkthrough from docs + one diagram
- Inputs: ADRs + a system diagram screenshot
- Output: 60–120s showing data flow, failure modes, and why key decisions were made
- Edit loop: “Add 10s on caching trade-offs; remove jargon; highlight SLO impact”

### Decision rationale timeline from a long memo
- Inputs: meeting notes + decision doc
- Output: timeline explaining “what happened and why” for people not in the room
- Trust behavior: every claim links back to the memo segment; disputes/uncertainty are labeled

### Product demo from screenshots (grounded UI animation)
- Inputs: 6 screenshots + release notes
- Output: guided flow with cursor/highlights + narration; suited for sales/onboarding
- Edit loop: “Swap industry example; add CTA; keep visuals”

### Sense-making explainer from a dataset + question
- Inputs: CSV + question (“why did latency spike last week?”)
- Output: animated chart progression with narrated hypotheses; conclusion vs uncertainty clearly separated

---

## 4) Possible product directions (with trade-offs)

### A) “Video as an artifact” inside AGI (fits current model)
- **Pros**: reuses AGI strengths (tool orchestration, memory, sourcing); natural artifact generation workflow
- **Cons**: adds a high-expectation surface; failures are obvious; risks bloating the core UX

### B) Adjacent “Studio” surface (separate mode)
- **Pros**: keeps AGI core focused; enables dedicated review/edit UI around the Video IR
- **Cons**: context fragmentation; risks becoming a side tool rather than a primitive

### C) Constrained “Explainer compiler” (narrow domain, high reliability)
- **Pros**: strongest “code, not everything” analog; can be opinionated about grammar, pacing, grounding, QA
- **Cons**: feels limiting early; creative storytelling won’t fit (by design)

Key cross-cutting trade-off:
- Optimize for **wow renders** vs **fast, trusted iteration**. Knowledge work disproportionately rewards the latter.

---

## 5) Clear reasons not to pursue this

- **Iteration latency kills adoption**: if edits aren’t materially faster than slides, this becomes novelty
- **Trust risk is higher by default**: video’s persuasiveness amplifies harm from hallucinations or misrepresented visuals
- **Mismatch with reference ergonomics**: many workflows need searchable artifacts; video is linear and harder to cite
- **Cost curve may not match value**: if compute + review time is high, only marketing teams justify it (too narrow)
- **“General-purpose” trap**: aiming for open-ended cinematic creativity early explodes primitives and collapses reliability
- **AGI brand dilution**: flashy-but-unreliable outputs distract from AGI’s core value (high-trust productivity orchestration)

---

## Working hypothesis (if pursued experimentally)

Start with a strict bet: **a grounded, editable Video IR for explanatory motion graphics and UI/diagram demos**.  
This is the smallest surface area where the system can be:
- Opinionated
- Fast to iterate
- Trustworthy
- Widely reusable across learning, marketing, and internal comms without becoming "a different product each time"

---

## Executive Summary: What to Tell Your CEO

### The Problem in One Sentence

**Understanding transfer is slow, expensive, and doesn't scale.**

When you explain something complex—a business problem, a technical architecture, a marketing concept—it takes 20-45 minutes of back-and-forth. That understanding dies with you. The next person who needs it has to repeat the entire process.

### The Opportunity in One Sentence

**Make understanding transferable by compiling intent + inputs into video.**

Instead of:
- Write doc → they skim it → they don't understand → you explain again

We get:
- Describe intent → attach inputs → generate video → they understand in 90 seconds → it works for everyone forever

### Why Video (And Why Now)

1. **Humans are visual.** Complex causality, temporal sequences, and spatial relationships are *hard* to convey in text. Video shows the thing happening.

2. **Current AI video is too general.** ChatGPT and Gemini can generate video, but they optimize for "impressive," not "useful." They lack:
   - Grounding to *your* specific inputs
   - Structured intent capture
   - Beat-level editing
   - Trustworthy sourcing

3. **The timing is right.** Video inference is getting faster/cheaper. Designing the harness now means we're ready when the curves cross.

### What We'd Build (If We Pursue This)

Not a general video generator. A **scoped, constrained harness** for:
- Explanatory videos (architecture walkthroughs, decision rationales)
- Product demos (grounded in screenshots and docs)
- Marketing narratives (before/after, value proposition)

Built on primitives that mirror Claude Code:
| Claude Code | Video Harness |
|-------------|---------------|
| Files | Grounded inputs (images, docs, diagrams) |
| Diffs | Beat-level editing |
| Execution | Preview renders |
| Feedback | Review + iteration |

### Why AGI Should Care

1. **Strategic positioning**: Generative UI is coming. First-mover on the harness shapes the paradigm.
2. **Product extension**: Video becomes another artifact type in AGI's knowledge work stack.
3. **Differentiation**: Competitors will add "video generation." We add "understanding transfer."

### The Ask

Not: "Build an AI video product."

But: "Fund a 2-4 week primitives exploration to validate whether the harness is tractable."

Outputs:
- Paper prototype of intent capture + beat structure
- User validation (do these primitives make sense?)
- Technical feasibility (can we get preview latency under 30s?)
- Go/no-go recommendation

---

## TL;DR For Busy Executives

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                               THE PITCH                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  PROBLEM:                                                                   │
│  Explaining complex things takes 20-45 minutes of back-and-forth.          │
│  That understanding isn't transferable. The next person starts over.       │
│                                                                              │
│  INSIGHT:                                                                   │
│  Humans understand complex things faster through video.                     │
│  Current AI video is too general to be useful.                             │
│  Claude Code showed: constrained + modular harness = useful.               │
│                                                                              │
│  OPPORTUNITY:                                                               │
│  Build a harness that turns intent + grounded inputs into                  │
│  explanatory video. Make understanding transferable and reusable.          │
│                                                                              │
│  TIMING:                                                                    │
│  Video inference is getting faster/cheaper.                                │
│  Designing the harness now positions AGI for when it crosses.              │
│                                                                              │
│  ASK:                                                                       │
│  2-4 week primitives exploration. Validate the harness is tractable.       │
│  Go/no-go recommendation at the end.                                        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

