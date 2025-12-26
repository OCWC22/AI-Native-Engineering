# Conductor: Architectural Deep Dive

## 1. What Problem Conductor Solves

### The Pain Before

When using LLMs for code generation, developers face a recurring failure mode:

1. **Context Loss**: LLMs forget previous decisions, leading to inconsistent implementations
2. **Scope Creep**: Without structure, conversations drift, and the LLM solves the wrong problem
3. **No Auditability**: Changes happen in a black box—no paper trail of *why* decisions were made
4. **Repeated Setup**: Every new feature requires re-explaining the tech stack, conventions, and goals
5. **Plan-Code Mismatch**: The LLM writes code without a verifiable plan, making review impossible

### Why Existing Tools Were Insufficient

| Tool/Approach | Gap |
|---------------|-----|
| **Raw LLM chat** | No memory, no structure, no accountability |
| **CLIs/SDKs** | Great for deterministic tasks, but don't manage intent or planning |
| **IDE copilots** | Line-level completion, not feature-level orchestration |
| **Ad-hoc scripts** | Brittle, no shared context, no iteration model |

**The core gap**: No tool enforced the discipline of *"specify before you build"* while also maintaining persistent, sharable project context.

---

## 2. What Conductor Is (and Is Not)

### What It Is

Conductor is a **prompt-based orchestration harness** that transforms an LLM into a disciplined software project manager. It's:

- **A protocol encoder**: It encodes a spec-driven development methodology directly into prompts
- **A state machine**: It uses markdown files as a persistent, human-readable state store
- **A context manager**: It ensures every LLM interaction has access to project-wide decisions (tech stack, guidelines, workflow)

### What It Is NOT

- **Not a framework or library**: There's no runtime code—it's pure prompts
- **Not an agent framework**: It doesn't manage multi-agent coordination or tool chaining
- **Not model-agnostic**: Currently tightly coupled to Gemini CLI's extension system
- **Not a code generator**: It orchestrates an LLM that generates code, but Conductor itself is just structure

**Mental Model**: Think of Conductor as a "constitution" for your LLM—it defines the rules of engagement, the phases of work, and where state lives.

---

## 3. High-Level Architecture

### Core Components

```
┌─────────────────────────────────────────────────────────────────┐
│                        Gemini CLI                               │
│  (Runtime: executes prompts, provides tool calls, manages chat) │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Conductor Extension                         │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐       │
│  │   Commands    │  │   Templates   │  │    Context    │       │
│  │  (TOML files) │  │  (MD files)   │  │  (GEMINI.md)  │       │
│  └───────────────┘  └───────────────┘  └───────────────┘       │
│         │                   │                   │               │
│         ▼                   ▼                   ▼               │
│  ┌─────────────────────────────────────────────────────┐       │
│  │              Project State (conductor/)              │       │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────────────────┐ │       │
│  │  │product.md│ │tech-stack│ │    tracks.md         │ │       │
│  │  │          │ │   .md    │ │  └─tracks/<id>/      │ │       │
│  │  │          │ │          │ │    ├─spec.md         │ │       │
│  │  │          │ │          │ │    ├─plan.md         │ │       │
│  │  │          │ │          │ │    └─metadata.json   │ │       │
│  │  └──────────┘ └──────────┘ └──────────────────────┘ │       │
│  └─────────────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────────┘
```

### Data, Context, and Control Flow

```
User invokes: /conductor:implement
        │
        ▼
┌──────────────────────────────────────────────────────────────┐
│  1. PROMPT INJECTION                                         │
│     - implement.toml prompt loaded                           │
│     - LLM receives ~4000 word "system directive"             │
└──────────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────────┐
│  2. CONTEXT LOADING (LLM-driven)                             │
│     - LLM reads: conductor/workflow.md                       │
│     - LLM reads: conductor/tracks/<id>/plan.md               │
│     - LLM reads: conductor/tracks/<id>/spec.md               │
└──────────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────────┐
│  3. TASK EXECUTION (LLM-driven)                              │
│     - LLM follows workflow.md procedures                     │
│     - LLM writes code via tool calls                         │
│     - LLM runs tests, commits, updates plan.md               │
└──────────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────────┐
│  4. STATE MUTATION                                           │
│     - plan.md: task marked [x] with commit SHA               │
│     - tracks.md: status updated to [~] or [x]                │
│     - Git: commits created, notes attached                   │
└──────────────────────────────────────────────────────────────┘
```

### Where Intelligence vs Plumbing Lives

| Layer | Intelligence | Plumbing |
|-------|-------------|----------|
| **Prompts (TOML)** | All of it—protocol logic, error handling, decision trees | None |
| **Gemini CLI** | None—pure executor | File I/O, shell commands, chat management |
| **Markdown State** | None—pure data | None—inert files |

**Key Insight**: All "smarts" are in the prompts. The architecture is "prompt-as-code."

---

## 4. Execution Model

### How Commands Are Scheduled

There is no scheduler. Each command is:
1. A single, monolithic prompt
2. Invoked manually by the user (`/conductor:implement`)
3. Executed sequentially in one conversation turn

**The LLM is the scheduler**: The prompt instructs the LLM to "loop through tasks" or "ask questions one by one." The loop logic is encoded in natural language.

### State Passing Between Steps

State is passed via **file system**:

```
Step 1 (setup): Writes conductor/product.md
                Writes conductor/setup_state.json ← Resume checkpoint
        │
        ▼
Step 2 (newTrack): Reads conductor/product.md
                   Writes conductor/tracks/<id>/spec.md
                   Writes conductor/tracks/<id>/plan.md
        │
        ▼
Step 3 (implement): Reads plan.md
                    Writes code files
                    Updates plan.md with [x] and SHA
                    Commits to Git
```

**Resume mechanism**: `setup_state.json` stores `last_successful_step`, allowing interrupted setups to continue.

### Coordination Model

```
┌────────────────────────────────────────────────────────────┐
│                    Single LLM Instance                      │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  "For each task in plan.md:"                         │  │
│  │    1. Mark [~] in progress                           │  │
│  │    2. Write failing tests                            │  │
│  │    3. Implement code                                 │  │
│  │    4. Run tests                                      │  │
│  │    5. Commit with message                            │  │
│  │    6. Attach git note                                │  │
│  │    7. Mark [x] with SHA                              │  │
│  │    8. Commit plan update                             │  │
│  └──────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────┘
```

This is **sequential, single-threaded, synchronous execution** encoded entirely in prose.

---

## 5. Prompting & Code Layer

### Prompt Structure

Each command TOML has:
```toml
description = "Human-readable command description"
prompt = """
## 1.0 SYSTEM DIRECTIVE
[Role definition, constraints]

## 1.1 SETUP CHECK
[Precondition validation]

## 2.0 MAIN PROTOCOL
[Step-by-step instructions with numbered sub-steps]

## 3.0 ERROR HANDLING
[Failure modes and recovery]
"""
```

**Key structural patterns**:
1. **Numbered sections** (1.0, 2.0, 3.0) for reference
2. **PROTOCOL headers** to signal discrete phases
3. **CRITICAL markers** for non-negotiable rules
4. **Explicit state checks** before proceeding
5. **Confirmation loops** requiring user approval

### Example: Task Execution Protocol (from workflow.md)

```markdown
### Standard Task Workflow

1. **Select Task:** Choose the next available task from `plan.md`
2. **Mark In Progress:** Change `[ ]` to `[~]`
3. **Write Failing Tests (Red Phase):**
   - Create test file
   - **CRITICAL:** Run tests and confirm failure
4. **Implement to Pass Tests (Green Phase):**
   - Write minimum code to pass
5. **Commit Code Changes:**
   - Stage changes
   - Propose commit message
6. **Attach Task Summary with Git Notes:**
   - Get commit hash
   - Draft note content
   - `git notes add -m "<content>" <hash>`
```

### Code-Prompt Interaction

There is **no code**. The "code" is the prompt. Abstractions are:

| Abstraction | Implementation |
|-------------|---------------|
| "Read a file" | LLM uses Gemini CLI's file read tool |
| "Write a file" | LLM uses Gemini CLI's file write tool |
| "Run a command" | LLM uses Gemini CLI's shell tool |
| "Loop through tasks" | Natural language instruction to iterate |
| "If-then-else" | Natural language conditional logic |

### Exposed vs Hidden Abstractions

**Exposed to users**:
- Markdown file structure (`tracks.md`, `plan.md`)
- Status markers (`[ ]`, `[~]`, `[x]`)
- Workflow customization (`workflow.md`)

**Hidden from users**:
- Prompt internals (the 4000-word protocols)
- LLM tool call mechanics
- State machine transitions

---

## 6. Strengths

### 1. Zero Runtime Dependencies
No code to install, no servers to run. Just prompts and markdown.

### 2. Human-Readable State
All state lives in markdown—diffable, reviewable, version-controlled.

```markdown
## [x] Phase 1: Project Setup [checkpoint: a1b2c3d]
- [x] Task: Initialize project structure (a1b2c3d)
- [x] Task: Configure build system (e4f5g6h)
```

### 3. Git-Native Auditability
Every task links to a commit. Git notes contain task summaries. History is reconstructable.

### 4. Team-Shareable Context
`product.md`, `tech-stack.md`, and `workflow.md` encode tribal knowledge:
```markdown
# Tech Stack
- **Language**: TypeScript 5.x
- **Framework**: Next.js 14 (App Router)
- **Database**: PostgreSQL with Drizzle ORM
```

New team members inherit context automatically.

### 5. Graceful Degradation
If the LLM makes mistakes, the user can:
- Edit markdown files directly
- Use `/conductor:revert` to undo work
- Resume from checkpoints

### 6. Methodology Enforcement
TDD, commit hygiene, and code coverage are baked into `workflow.md`—not optional.

---

## 7. Weaknesses & Limitations

### 1. Token Inefficiency
Every command re-reads context files. Large projects → high token consumption.

```
/conductor:implement reads:
- workflow.md (~3000 tokens)
- plan.md (~500-2000 tokens)
- spec.md (~500-1000 tokens)
- The prompt itself (~4000 tokens)
```

**Mitigation**: None built-in. Relies on user discipline to keep files concise.

### 2. Single-Model Bottleneck
No parallelism. Complex tracks with 20+ tasks run sequentially in one conversation.

### 3. Fragile Prompt Parsing
The LLM must correctly parse markdown state:
```markdown
## [~] Track: User Authentication
```
If formatting drifts, the LLM may fail to find tasks.

### 4. No Rollback Guarantee
`/conductor:revert` relies on LLM git analysis. Complex merge histories may confuse it.

### 5. Gemini CLI Lock-in
Commands use Gemini CLI's extension format. Porting to Claude or GPT requires rewriting the harness.

### 6. No Inter-Track Dependencies
Tracks are independent. No way to express "Track B depends on Track A completing."

### 7. Manual Verification Overhead
Phase completion requires user confirmation:
```
"Does this meet your expectations? Please confirm with yes..."
```
This blocks automation.

---

## 8. Future Work

### Missing Capabilities

| Capability | Current State | Needed |
|------------|---------------|--------|
| **Parallelism** | Sequential only | Parallel task execution |
| **Dependencies** | None | Track/task dependency graph |
| **Caching** | Re-reads everything | Context caching layer |
| **Multi-agent** | Single LLM | Specialist agents (reviewer, tester) |
| **Observability** | Git notes only | Structured logs, metrics |
| **Automation** | Manual triggers | CI/CD integration |

### Scaling Considerations

1. **Hierarchical Plans**: Break large tracks into sub-tracks with explicit dependencies
2. **Incremental Context**: Only load changed files since last checkpoint
3. **Checkpoint Compaction**: Summarize completed phases to reduce token load
4. **Event-Driven Execution**: Trigger `/conductor:implement` on git push or PR

---

## 9. How to Re-Implement This Simply

### Core Ideas to Extract

1. **Spec → Plan → Implement lifecycle** (the methodology)
2. **Markdown as state store** (human-readable, git-native)
3. **Prompt-as-protocol** (encode workflow in natural language)
4. **Checkpoint + resume** (fault tolerance via state file)
5. **Task ↔ Commit linking** (auditability)

### Minimal Implementation (Any LLM)

```
project/
├── .agent/
│   ├── context.md      # Product, tech stack, conventions
│   ├── workflow.md     # How tasks should be executed
│   └── state.json      # {"current_track": "...", "current_task": 3}
├── tracks/
│   └── feature-auth/
│       ├── spec.md     # What to build
│       └── plan.md     # Phased task list with [ ]/[x] markers
└── src/
    └── ...
```

**System prompt skeleton**:
```
You are a spec-driven development assistant.

BEFORE writing any code:
1. Read .agent/context.md for project constraints
2. Read .agent/workflow.md for execution rules
3. Read tracks/{current_track}/spec.md for requirements
4. Read tracks/{current_track}/plan.md for task list

EXECUTION RULES:
- Work on ONE task at a time
- Mark task [~] before starting
- Follow TDD: write failing test → implement → pass
- Commit after each task with message: "feat(scope): description"
- Mark task [x] with commit SHA after completion

NEVER skip steps. NEVER work on multiple tasks simultaneously.
```

### Adapting for Claude Code

Claude Code has built-in file and shell tools. Create a similar harness:

```python
# claude_conductor.py
import anthropic

SYSTEM_PROMPT = """
[Same protocol as above, adapted for Claude's tool format]
"""

def run_conductor(command: str, args: str):
    client = anthropic.Anthropic()
    
    # Load context files
    context = load_context_files()
    
    # Run conversation with tools
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=8192,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"/{command} {args}"}],
        tools=[
            {"name": "read_file", ...},
            {"name": "write_file", ...},
            {"name": "run_shell", ...},
        ]
    )
    
    # Handle tool calls in a loop
    while response.stop_reason == "tool_use":
        # Execute tools, continue conversation
        ...
```

### Adapting for a Custom Agent Harness

For maximum control, build a thin orchestrator:

```python
class ConductorAgent:
    def __init__(self, llm_client, workspace_path):
        self.llm = llm_client
        self.workspace = Path(workspace_path)
        self.state = self._load_state()
    
    def setup(self):
        """Interactive project setup."""
        prompt = self._load_prompt("setup")
        self._run_conversation(prompt)
    
    def new_track(self, description: str):
        """Create spec and plan for a new track."""
        prompt = self._load_prompt("new_track")
        self._run_conversation(prompt, {"description": description})
    
    def implement(self, track_id: str = None):
        """Execute tasks from a track's plan."""
        track = track_id or self._find_next_track()
        prompt = self._load_prompt("implement")
        context = self._load_track_context(track)
        self._run_conversation(prompt, context)
    
    def _run_conversation(self, system_prompt, context=None):
        """Main loop: send prompt, handle tool calls, update state."""
        messages = []
        while True:
            response = self.llm.chat(system_prompt, messages)
            if response.has_tool_calls:
                results = self._execute_tools(response.tool_calls)
                messages.append({"role": "tool", "content": results})
            else:
                break
        self._save_state()
```

### Key Design Decisions to Preserve

1. **Markdown state is sacred**: Don't use databases. The power is in human readability and git integration.

2. **One task at a time**: Resist the urge to parallelize. Sequential execution is easier to debug and audit.

3. **Explicit checkpoints**: Always save state after completing a logical unit. Assume crashes happen.

4. **User confirmation gates**: Don't auto-approve. The human-in-the-loop is a feature, not a bug.

5. **Git as audit log**: Link every task to a commit. Use git notes or structured commit messages.

---

## Summary

Conductor is a **prompt-encoded methodology** that transforms an LLM into a disciplined project manager. Its architecture is radically simple:

- **No runtime code**—just prompts and markdown
- **All state in files**—human-readable, git-native
- **Single LLM, sequential execution**—no coordination complexity

The design trades scalability for simplicity and auditability. It works because it constrains the LLM to a well-defined protocol rather than giving it open-ended freedom.

To adopt the core ideas elsewhere:
1. Encode your workflow in a system prompt
2. Use markdown files as your state store
3. Link tasks to commits for auditability
4. Require explicit confirmation at phase boundaries
5. Support resume via checkpoint files

The transferable insight: **structure your LLM interactions as protocols, not conversations**.

<chatName="Conductor architecture deep dive"/>