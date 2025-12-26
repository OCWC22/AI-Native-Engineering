# Building Long-Running LLM Agents: Memory Evolution & Context Engineering

> **Synthesized from**:  
> - MemEvolve: Meta-Evolution of Agent Memory Systems (Zhang et al., Dec 2025)  
> - Agent Skills for Context Engineering: A Design Rationale and Research Retrospective
>
> **Sources**: [arXiv:2512.18746](https://arxiv.org/abs/2512.18746) | [GitHub](https://github.com/bingreeky/MemEvolve)

---

## Executive Summary

This document synthesizes two complementary research streams into actionable guidance for building production-grade, long-running LLM agents:

1. **MemEvolve** — A meta-evolutionary framework that enables agents to evolve *how they learn*, not just *what they learn*. It provides the ESRM model (Encode-Store-Retrieve-Manage) for memory architecture.

2. **Context Engineering** — A paradigm shift from prompt engineering to holistic context management. The core insight: **context windows are attention budgets, not storage**.

**The Unified Thesis**: Agent failures are primarily *context failures*—either the wrong information is in context, critical information is buried, or the memory system cannot adapt to task requirements. Solving this requires both intelligent memory architecture (MemEvolve) and disciplined context engineering.

---

## Part 1: Foundational Mental Models

### The Two Paradigms

| Prompt Engineering | Context Engineering |
|-------------------|---------------------|
| Craft better instructions | Curate the complete inference state |
| Optimize phrasing | Optimize token allocation |
| Write clever prompts | Architect attention budgets |
| Single-turn focused | Multi-turn, long-horizon focused |
| Instruction quality | Signal-to-noise ratio |

**Key insight**: Prompts are one component of context. System prompts, tool definitions, message history, retrieved documents, and tool outputs all compete for finite attention capacity. **Optimize the complete context, not just the prompt.**

### Context Windows Are Attention Budgets

```
┌────────────────────────────────────────────────────────────────────┐
│                    CONTEXT WINDOW (e.g., 128K tokens)              │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│   ┌─────────────────────────────────────────────────────────────┐  │
│   │                    HIGH ATTENTION ZONE                      │  │
│   │              (Beginning of context)                         │  │
│   └─────────────────────────────────────────────────────────────┘  │
│                                                                    │
│   ┌─────────────────────────────────────────────────────────────┐  │
│   │                  ATTENTION DEAD ZONE                        │  │
│   │   "Lost in the Middle" - 10-40% accuracy drop               │  │
│   │   (Middle of long contexts)                                 │  │
│   └─────────────────────────────────────────────────────────────┘  │
│                                                                    │
│   ┌─────────────────────────────────────────────────────────────┐  │
│   │                    HIGH ATTENTION ZONE                      │  │
│   │                  (End of context)                           │  │
│   └─────────────────────────────────────────────────────────────┘  │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

**Research evidence** (Liu et al., TACL 2024): U-shaped attention curves cause 10-40% accuracy drops for information placed in the middle of long contexts. The RULER benchmark found that only 50% of models claiming 32K+ token windows maintain satisfactory performance at those lengths.

**Implication**: Larger context windows introduce more opportunities for failure, not fewer. **Curate ruthlessly. Load just-in-time. Preserve signal over volume.**

### The Three Levels of Agent Learning

From MemEvolve's analysis:

| Learner Type | Description | Memory Approach | Example |
|--------------|-------------|-----------------|---------|
| **Mediocre** | No persistent memory | Stateless - each task starts fresh | Basic ReAct agents |
| **Skillful** | Fixed memory architecture | Can learn content but not adapt structure | Voyager, ExpeL, SkillWeaver |
| **Adaptive** | Evolving memory architecture | Can modify how they encode, store, retrieve, and manage knowledge | MemEvolve-enabled agents |

**The gap**: Most production agents are "skillful learners" at best—they accumulate knowledge within a fixed structure but cannot adapt their learning approach to different task contexts. MemEvolve enables the jump to "adaptive learners."

### Non-Obvious Optimization Targets

| Common Metric | Better Metric | Why |
|--------------|---------------|-----|
| Tokens per request | **Tokens per task** | Re-fetching from poor compression costs more |
| Context utilization | **Signal-to-noise ratio** | 50% utilization with 90% signal beats 90% utilization with 50% signal |
| Compression ratio | **Functional quality** | 99% compression that loses file paths is worse than 95% that preserves them |
| Model capability | **Context curation** | Better model × poor context < worse model × excellent context |

---

## Part 2: Failed Hypotheses (What Doesn't Work)

### ❌ Hypothesis 1: Larger Context Windows Solve Memory Problems

**What was tested**: Give agents 100K+ token windows, let them accumulate context indefinitely.

**What failed**:
- RULER benchmark: 50% of models claiming 32K+ fail at 32K
- Chroma 2025 research: Even single irrelevant documents reduce performance; multiple compound
- Gemini 2.5 report: "Making effective use of 1M+ token context for agents presents a new research frontier"

**Failure mode**: **Context saturation**—models over-attend to context at the expense of training knowledge. Information placed in the middle becomes effectively invisible.

### ❌ Hypothesis 2: Aggressive Compression Minimizes Token Costs

**What was tested**: Squeeze context to minimum tokens per request.

**What failed**:
- Factory Research: OpenAI's 99.3% compression scored 0.35 quality points lower than 98.6% compression
- Agents forgot file paths, error messages, and approach decisions
- Re-fetching costs exceeded compression savings

**Failure mode**: **Tokens per task > tokens per request.** Aggressive compression creates re-fetch cycles that cost more overall.

### ❌ Hypothesis 3: Many Specialized Tools Enable Better Agent Behavior

**What was tested**: Build comprehensive tool libraries with specialized tools for each operation.

**What failed** (Vercel d0 agent case study):
- 17 specialized tools: 274.8s average execution, 80% success rate
- 2 primitive tools (bash + SQL): 77.4s average, **100% success rate**
- Overlapping tool descriptions caused selection confusion

**Failure mode**: **Consolidation principle violation**—if humans can't definitively choose which tool applies, neither can agents. Specialized tools constrain reasoning that models could handle with primitives.

### ❌ Hypothesis 4: Autonomous Agents Self-Correct via Prompt Engineering

**What was tested**: Add retry logic, self-correction prompts, "chain-of-verification" without external feedback.

**What failed**:
> "No prior work demonstrates successful self-correction with feedback from prompted LLMs, except for tasks exceptionally suited for self-correction." (MIT survey)

**Failure mode**: **Ungrounded correction**—without external verification (tool execution, retrieval, human feedback), self-correction amplifies errors rather than fixing them.

### ❌ Hypothesis 5: Supervisor Architectures Provide Reliable Coordination

**What was tested**: Central supervisor agent delegates to specialists and synthesizes results.

**What failed** (LangGraph benchmarks):
- 50% worse performance initially due to "telephone game" problem
- Supervisors paraphrase sub-agent responses incorrectly, losing fidelity
- Supervisor context becomes bottleneck as it accumulates worker outputs

**Failure mode**: **Translation errors compound**—every layer of synthesis introduces information loss.

**The fix**: Use `forward_message` tool for direct sub-agent → user communication, bypassing supervisor paraphrasing.

### ❌ Hypothesis 6: Vector RAG Provides Sufficient Memory

**What was tested**: Use semantic similarity search for agent memory.

**What failed**:
- Lost relationship structure between entities
- No temporal validity—couldn't distinguish current vs outdated facts
- DMR benchmark: Vector RAG accuracy ~60-70% vs Temporal Knowledge Graph 94.8%

**Failure mode**: **Structural amnesia**—semantic similarity retrieves isolated facts but loses relationships and temporal context needed for reasoning.

---

## Part 3: The ESRM Memory Architecture (What Works)

MemEvolve establishes a modular decomposition of any memory system into four fundamental components:

```
┌─────────────────────────────────────────────────────────────────┐
│                     AGENT MEMORY SYSTEM                          │
├──────────────┬──────────────┬──────────────┬───────────────────┤
│   ENCODE     │    STORE     │   RETRIEVE   │      MANAGE       │
│     (E)      │     (S)      │     (R)      │       (M)         │
├──────────────┼──────────────┼──────────────┼───────────────────┤
│ Raw          │ Integration  │ Context-     │ Offline           │
│ experience → │ into         │ aware        │ consolidation     │
│ structured   │ persistent   │ recall       │ & abstraction     │
│ representation│ storage     │              │                   │
└──────────────┴──────────────┴──────────────┴───────────────────┘
```

### 1. Encode (E) — Transform Raw Experience

**What it does**: Converts trajectories, observations, tool outputs, and feedback into structured representations.

**Context Engineering Integration**: Use **anchored iterative summarization** with explicit structure:

```markdown
## Session Intent
[What the user is trying to accomplish]

## Files Modified
- auth.controller.ts: Fixed JWT token generation
- config/redis.ts: Updated connection pooling

## Decisions Made
- Using Redis connection pool instead of per-request connections
- Retry logic with exponential backoff for transient failures

## Current State
- 14 tests passing, 2 failing

## Next Steps
1. Fix remaining test failures
2. Run full test suite
```

**Why structure matters**: Structure forces preservation—dedicated sections for files, decisions, and next steps prevent information drift across compression cycles.

**Implementation**:
```python
class StructuredEncoder:
    """Anchored iterative summarization encoder"""
    
    REQUIRED_SECTIONS = [
        "session_intent",
        "files_modified", 
        "decisions_made",
        "current_state",
        "next_steps"
    ]
    
    async def encode_trajectory(
        self, 
        trajectory: AgentTrajectory,
        existing_summary: Optional[str] = None
    ) -> MemoryRecord:
        """
        Encode with structure preservation.
        
        Key principles:
        - Multi-level: raw logs + summarized insights + abstract patterns
        - Agent-driven: LLM decides importance within structure
        - Anchored: new content merges into existing sections
        """
        if existing_summary:
            # Anchored update - merge into existing structure
            return await self._merge_into_anchor(
                anchor=existing_summary,
                new_content=trajectory
            )
        else:
            # Fresh encoding with required structure
            return await self._encode_fresh(trajectory)
    
    async def _merge_into_anchor(
        self, 
        anchor: str, 
        new_content: AgentTrajectory
    ) -> MemoryRecord:
        """Incremental merge prevents information drift"""
        prompt = f"""Update this session summary with new information.

EXISTING SUMMARY:
{anchor}

NEW ACTIVITY:
{self._format_trajectory(new_content)}

RULES:
1. Preserve all existing entries unless explicitly superseded
2. Add new files to Files Modified section
3. Update Current State to reflect latest status
4. Append new decisions with reasoning
5. Update Next Steps based on what was completed

Return the complete updated summary maintaining all sections."""
        
        return await self.llm.generate(prompt)
```

### 2. Store (S) — Integrate into Persistent Memory

**What it does**: Manages how encoded experiences are added to and organized in memory.

**Context Engineering Integration**: Use **typed storage with temporal validity**:

```python
class TypedMemoryStore:
    """Typed storage with temporal validity tracking"""
    
    def __init__(self):
        # Episodic: Specific experiences with full context
        self.episodic = EpisodicStore(
            retention_days=30,
            temporal_index=True
        )
        
        # Semantic: General knowledge and domain facts
        self.semantic = SemanticStore(
            entity_graph=True,  # Preserve relationships
            validity_periods=True  # Track when facts expire
        )
        
        # Procedural: Skills, tools, action patterns
        self.procedural = ProceduralStore(
            skill_library=True,
            success_tracking=True
        )
    
    async def store(
        self, 
        record: MemoryRecord, 
        store_type: str,
        validity_period: Optional[timedelta] = None
    ):
        """Store with temporal validity"""
        record.stored_at = datetime.utcnow()
        record.valid_until = (
            datetime.utcnow() + validity_period 
            if validity_period 
            else None  # Permanent
        )
        
        store = getattr(self, store_type)
        await store.add(record)
        
        # Deduplicate to prevent memory bloat
        await self._deduplicate_similar(store, record)
```

### 3. Retrieve (R) — Context-Aware Recall

**What it does**: Recalls relevant memory content based on current context and task.

**Context Engineering Integration**: Use **phase-aware retrieval with attention-budget limits**:

```python
class AttentionBudgetRetriever:
    """Retrieval that respects attention budget constraints"""
    
    # Placement priorities (high attention zones)
    PLACEMENT = {
        "critical": "start",      # System prompts, current task
        "reference": "end",       # Retrieved context
        "conversation": "middle"  # Historical (gets deprioritized)
    }
    
    async def retrieve(
        self, 
        query: str, 
        context: TaskContext,
        max_tokens: int = 4000  # Attention budget for retrieval
    ) -> RetrievalResult:
        """Phase-aware retrieval with token budgeting"""
        
        # Determine retrieval strategy based on task phase
        if context.phase == "planning":
            # High-level insights, strategic patterns
            results = await self._planning_retrieval(query)
        elif context.phase == "execution":
            # Specific tool examples, detailed procedures
            results = await self._execution_retrieval(query)
        elif context.phase == "debugging":
            # Error patterns, past failure recoveries
            results = await self._debugging_retrieval(query)
        else:
            results = await self._general_retrieval(query)
        
        # Apply token budget - prioritize by relevance
        budgeted = self._apply_token_budget(results, max_tokens)
        
        # Order for attention optimization
        return self._order_for_attention(budgeted)
    
    def _order_for_attention(self, results: List[MemoryRecord]) -> List[MemoryRecord]:
        """
        Order results to maximize attention.
        Most important at start and end (high attention zones).
        Less critical in middle (attention dead zone).
        """
        if len(results) <= 2:
            return results
        
        # Sort by importance
        by_importance = sorted(results, key=lambda r: r.importance, reverse=True)
        
        # Interleave: most important at boundaries, least important in middle
        ordered = []
        for i, record in enumerate(by_importance):
            if i % 2 == 0:
                ordered.insert(0, record)  # Add to start
            else:
                ordered.append(record)      # Add to end
        
        return ordered
```

### 4. Manage (M) — Offline Maintenance

**What it does**: Performs background operations like consolidation, abstraction, and cleanup.

**Context Engineering Integration**: Use **progressive disclosure for loading**:

```python
class ProgressiveMemoryManager:
    """
    Progressive disclosure: Load information only when needed.
    
    Startup: Load metadata only (~100 tokens per skill/memory block)
    Activation: Load full content on demand (<5000 tokens recommended)
    References: Load only when explicitly needed
    """
    
    def __init__(self):
        self.metadata_cache = {}  # Lightweight index
        self.content_cache = LRUCache(max_size=50)
        
    async def startup_load(self):
        """Load only metadata at startup"""
        for memory_block in self._scan_memory_blocks():
            self.metadata_cache[memory_block.id] = {
                "name": memory_block.name,
                "description": memory_block.description,  # For matching
                "path": memory_block.path,
                "size_tokens": memory_block.size_tokens,
                # Full content NOT loaded
            }
    
    async def activate(self, memory_block_id: str) -> MemoryContent:
        """Load full content on activation"""
        if memory_block_id in self.content_cache:
            return self.content_cache[memory_block_id]
        
        metadata = self.metadata_cache[memory_block_id]
        content = await self._load_full_content(metadata["path"])
        
        self.content_cache[memory_block_id] = content
        return content
    
    async def maintain(self):
        """Background maintenance operations"""
        # Consolidate similar memories
        await self._consolidate_duplicates()
        
        # Abstract patterns from specifics
        await self._extract_patterns()
        
        # Prune expired memories
        await self._prune_expired()
        
        # Evolve memory architecture if performance degrades
        if await self._performance_degraded():
            await self._trigger_architecture_evolution()
```

---

## Part 4: Context Degradation Patterns

Understanding how context degrades is essential for prevention:

### Pattern 1: Context Saturation

**Symptoms**: Agent loses effectiveness as context fills with tool outputs, message history, and retrieved documents.

**Mechanism**: Models over-attend to context at the expense of training knowledge. Information becomes diluted.

**Prevention**:
```python
class SaturationMonitor:
    """Detect and prevent context saturation"""
    
    THRESHOLDS = {
        "warning": 0.7,    # 70% context utilization
        "critical": 0.85,  # 85% context utilization
        "emergency": 0.95  # 95% context utilization
    }
    
    def check_saturation(self, context: Context) -> SaturationLevel:
        utilization = context.token_count / context.max_tokens
        
        if utilization >= self.THRESHOLDS["emergency"]:
            return SaturationLevel.EMERGENCY
        elif utilization >= self.THRESHOLDS["critical"]:
            return SaturationLevel.CRITICAL
        elif utilization >= self.THRESHOLDS["warning"]:
            return SaturationLevel.WARNING
        return SaturationLevel.HEALTHY
    
    async def handle_saturation(self, level: SaturationLevel, context: Context):
        if level == SaturationLevel.WARNING:
            # Start proactive compression
            await self._compress_low_priority_content(context)
        elif level == SaturationLevel.CRITICAL:
            # Aggressive compression + summarization
            await self._emergency_compression(context)
        elif level == SaturationLevel.EMERGENCY:
            # Checkpoint and reset
            await self._checkpoint_and_reset(context)
```

### Pattern 2: Context Poisoning

**Symptoms**: Irrelevant or incorrect information in context leads to degraded outputs.

**Mechanism**: Even single irrelevant documents reduce performance; multiple compound the effect.

**Prevention**:
```python
class ContextValidator:
    """Validate context quality before use"""
    
    async def validate_retrieval(
        self, 
        query: str, 
        retrieved: List[Document]
    ) -> List[Document]:
        """Filter out potentially poisoning content"""
        validated = []
        
        for doc in retrieved:
            relevance = await self._compute_relevance(query, doc)
            freshness = self._compute_freshness(doc)
            source_trust = self._get_source_trust(doc)
            
            # Multi-signal validation
            if relevance > 0.6 and freshness > 0.5 and source_trust > 0.7:
                validated.append(doc)
            else:
                self._log_filtered(doc, relevance, freshness, source_trust)
        
        return validated
```

### Pattern 3: Lost in the Middle

**Symptoms**: Information in the middle of long contexts is ignored or forgotten.

**Mechanism**: U-shaped attention curves—models attend strongly to beginning and end, weakly to middle.

**Prevention**:
```python
class MiddlePreservation:
    """Prevent information loss in middle context"""
    
    def arrange_for_attention(self, content_blocks: List[ContentBlock]) -> str:
        """
        Arrange content to maximize attention to important information.
        
        Strategy:
        1. Most critical at very start (system prompt zone)
        2. Second most critical at end (recent context zone)
        3. Reference/background in middle (may be partially ignored)
        """
        by_priority = sorted(
            content_blocks, 
            key=lambda b: b.priority, 
            reverse=True
        )
        
        arranged = []
        for i, block in enumerate(by_priority):
            if i == 0:
                arranged.insert(0, block)  # Most important → start
            elif i % 2 == 1:
                arranged.append(block)      # Odd priority → end
            else:
                # Even priority → middle (accept some attention loss)
                mid = len(arranged) // 2
                arranged.insert(mid, block)
        
        return self._format_blocks(arranged)
```

### Pattern 4: Re-fetch Cycles

**Symptoms**: Agents repeatedly re-read files, re-explore approaches, wasting tokens.

**Mechanism**: Poor compression loses critical details (file paths, decisions), forcing re-discovery.

**Prevention**: Use anchored iterative summarization with explicit preservation of:
- File paths and line numbers
- Decisions made and their rationale
- Error messages and their resolutions
- Approach attempts and their outcomes

---

## Part 5: Tool Design Principles

### The Consolidation Principle

> **"If humans can't definitively choose which tool applies, neither can agents."**

**The test**: Given two tools with overlapping descriptions, can a human immediately and unambiguously know which to use? If not, consolidate them.

**Evidence** (Vercel d0 case study):
- 17 specialized tools: 274.8s, 80% success
- 2 primitive tools: 77.4s, **100% success**

```python
# ❌ BAD: Overlapping specialized tools
tools = [
    "search_code_by_function_name",
    "search_code_by_class_name", 
    "search_code_by_variable",
    "search_code_by_import",
    "search_code_semantic",
    "grep_codebase",
    "find_files",
    # ... 10 more search-like tools
]

# ✅ GOOD: Consolidated primitives
tools = [
    "grep",   # Exact text search
    "search"  # Semantic search
]
```

### Tool Description Engineering

Tool descriptions function as prompt engineering. They should answer:
1. **What** does it do?
2. **When** should it be used?
3. **What** inputs does it expect (with examples)?
4. **What** outputs does it produce?
5. **What** errors can occur and how to handle them?

```python
def get_customer(customer_id: str, format: str = "concise"):
    """
    Retrieve customer information by ID.
    
    USE WHEN:
    - User asks about specific customer details
    - Need customer context for decision-making
    
    INPUTS:
        customer_id: Format "CUST-######" (e.g., "CUST-000001")
        format: "concise" for key fields, "detailed" for complete record
    
    OUTPUTS:
        Customer object with requested fields
        - concise: id, name, email, status
        - detailed: full record including history
    
    ERRORS:
        NOT_FOUND: Customer ID doesn't exist
            → Verify ID format, check for typos
        INVALID_FORMAT: ID doesn't match CUST-###### pattern
            → Correct the format before retrying
    """
```

### Error Design for Recovery

Errors should tell agents **how to fix them**:

```python
# ❌ BAD: Opaque error
{"error": "Invalid input"}

# ✅ GOOD: Recovery-oriented error
{
    "error": {
        "code": "INVALID_CUSTOMER_ID",
        "message": "Customer ID 'CUST-123' does not match required format",
        "expected_format": {
            "pattern": "CUST-######",
            "example": "CUST-000001"
        },
        "resolution": "Provide a customer ID matching pattern CUST-######",
        "retryable": True
    }
}
```

### Response Format Optimization

Provide configurable verbosity to reduce token consumption:

```python
class ToolResponseFormatter:
    """Format responses based on context needs"""
    
    async def format(
        self, 
        result: Any, 
        verbosity: str = "concise"
    ) -> Dict:
        """
        verbosity levels:
        - "minimal": Just the answer (saves 80% tokens)
        - "concise": Answer + key context (saves 50% tokens)  
        - "detailed": Full response with metadata
        """
        if verbosity == "minimal":
            return {"result": self._extract_core(result)}
        elif verbosity == "concise":
            return {
                "result": self._extract_core(result),
                "context": self._key_context(result)
            }
        else:
            return {"result": result, "metadata": self._full_metadata(result)}
```

---

## Part 6: Multi-Agent Coordination

### The Telephone Game Problem

**Problem**: Supervisor agents paraphrase sub-agent responses, introducing translation errors that compound with each layer.

**Solution**: Use `forward_message` for direct communication:

```python
class MultiAgentCoordinator:
    """Avoid telephone game through direct forwarding"""
    
    async def coordinate(self, task: Task) -> Result:
        # Decompose task
        subtasks = await self.planner.decompose(task)
        
        results = {}
        for subtask in subtasks:
            # Delegate to specialist
            agent = self._select_agent(subtask)
            result = await agent.execute(subtask)
            
            # CRITICAL: Forward directly, don't paraphrase
            if result.requires_user_visibility:
                await self.forward_message(
                    from_agent=agent.id,
                    to="user",
                    message=result.raw_output  # Direct forwarding
                )
            
            results[subtask.id] = result
        
        # Synthesize only what requires synthesis
        return await self._synthesize_final(results)
```

### Context Isolation Pattern

**Problem**: Single-agent with massive context leads to confusion and degradation.

**Solution**: Partition context across specialized sub-agents:

```
┌─────────────────────────────────────────────────────────────────┐
│                        COORDINATOR                               │
│  (Minimal context: task routing, completion tracking)           │
└──────────────────────────┬──────────────────────────────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        ▼                  ▼                  ▼
┌───────────────┐  ┌───────────────┐  ┌───────────────┐
│   PLANNER     │  │   CODER       │  │   REVIEWER    │
│ ┌───────────┐ │  │ ┌───────────┐ │  │ ┌───────────┐ │
│ │Isolated   │ │  │ │Isolated   │ │  │ │Isolated   │ │
│ │Context:   │ │  │ │Context:   │ │  │ │Context:   │ │
│ │- Goals    │ │  │ │- Code     │ │  │ │- Diff     │ │
│ │- Approach │ │  │ │- Errors   │ │  │ │- Tests    │ │
│ └───────────┘ │  │ └───────────┘ │  │ └───────────┘ │
└───────────────┘  └───────────────┘  └───────────────┘
```

**Benefits**:
- Each sub-agent operates in clean context focused on its subtask
- Prevents KV-cache penalties from context confusion
- Enables parallel execution of independent subtasks

### Shared Memory Pool Implementation

```python
class IsolatedContextMultiAgent:
    """Multi-agent system with context isolation"""
    
    def __init__(self):
        # Shared memory for cross-agent learning
        self.shared_memory = SharedMemoryPool()
        
        # Isolated agents
        self.agents = {
            "planner": Agent(
                role="planner",
                context_limit=8000,  # Small, focused
                memory_access=["semantic", "episodic"]
            ),
            "coder": Agent(
                role="coder",
                context_limit=16000,  # More for code
                memory_access=["procedural", "episodic"]
            ),
            "reviewer": Agent(
                role="reviewer",
                context_limit=8000,
                memory_access=["semantic", "procedural"]
            )
        }
    
    async def execute(self, task: Task) -> Result:
        # Phase 1: Planning (isolated context)
        plan = await self.agents["planner"].execute(
            task=task,
            context=await self._build_planner_context(task)
        )
        
        # Phase 2: Coding (isolated context)
        code = await self.agents["coder"].execute(
            task=plan.subtasks,
            context=await self._build_coder_context(plan)
        )
        
        # Phase 3: Review (isolated context)
        review = await self.agents["reviewer"].execute(
            task=code,
            context=await self._build_reviewer_context(code)
        )
        
        # Learn from interaction
        await self._update_shared_memory(task, plan, code, review)
        
        return review.result
```

---

## Part 7: Evaluation That Actually Works

### Probe-Based Evaluation (Not ROUGE)

**Problem**: ROUGE, embedding similarity, and generic quality metrics give false confidence. They measure lexical overlap, not functional capability.

**Solution**: Probe-based evaluation asks questions that require remembering specifics:

```python
class ProbeBasedEvaluator:
    """Evaluate by testing functional capability"""
    
    async def evaluate_compression(
        self, 
        original: str, 
        compressed: str
    ) -> CompressionQuality:
        """
        Test if agent can still work after compression.
        
        Generate probes from original, test against compressed.
        """
        # Generate probes from original content
        probes = await self._generate_probes(original)
        
        # Test each probe against compressed
        scores = []
        for probe in probes:
            answer = await self._answer_from_context(
                question=probe.question,
                context=compressed
            )
            
            score = self._evaluate_answer(
                answer=answer,
                expected=probe.expected_answer,
                tolerance=probe.tolerance
            )
            scores.append(score)
        
        return CompressionQuality(
            functional_accuracy=sum(scores) / len(scores),
            probes_passed=sum(1 for s in scores if s > 0.8),
            total_probes=len(probes),
            critical_failures=[
                p for p, s in zip(probes, scores) 
                if p.is_critical and s < 0.5
            ]
        )
    
    async def _generate_probes(self, content: str) -> List[Probe]:
        """Generate probes for critical information"""
        probes = []
        
        # File path probes
        for file_path in self._extract_file_paths(content):
            probes.append(Probe(
                question=f"What is the path to the file that handles {file_path.purpose}?",
                expected_answer=file_path.path,
                is_critical=True
            ))
        
        # Decision probes
        for decision in self._extract_decisions(content):
            probes.append(Probe(
                question=f"What approach was chosen for {decision.topic}?",
                expected_answer=decision.choice,
                is_critical=True
            ))
        
        # Error resolution probes
        for error in self._extract_errors(content):
            probes.append(Probe(
                question=f"How was the '{error.type}' error resolved?",
                expected_answer=error.resolution,
                is_critical=True
            ))
        
        return probes
```

### LLM-as-Judge with Bias Mitigation

```python
class BiasAwareLLMJudge:
    """LLM evaluation with position bias mitigation"""
    
    async def compare_pairwise(
        self, 
        response_a: str, 
        response_b: str,
        criteria: List[str]
    ) -> ComparisonResult:
        """
        Compare responses with position bias mitigation.
        
        Method: Swap positions and average results.
        """
        # Evaluation with A first
        result_a_first = await self._evaluate_pair(
            first=response_a,
            second=response_b,
            criteria=criteria
        )
        
        # Evaluation with B first (swapped)
        result_b_first = await self._evaluate_pair(
            first=response_b,
            second=response_a,
            criteria=criteria
        )
        
        # Average to mitigate position bias
        return ComparisonResult(
            winner=self._reconcile_winners(result_a_first, result_b_first),
            confidence=self._compute_agreement(result_a_first, result_b_first),
            criteria_scores={
                c: (result_a_first.scores[c] + result_b_first.scores[c]) / 2
                for c in criteria
            }
        )
```

---

## Part 8: Dual-Loop Architecture Evolution

The core MemEvolve pattern for long-running agents:

### Inner Loop: Experience Learning

```python
class InnerLoop:
    """Learn content within current architecture"""
    
    def __init__(self, memory: AgentMemory):
        self.memory = memory
        self.trajectory_buffer = []
        self.performance_metrics = []
    
    async def learn(
        self, 
        task: Task, 
        result: TaskResult, 
        trajectory: Trajectory
    ):
        """Update memory with new experience"""
        # Encode experience
        encoded = await self.memory.encoder.encode(trajectory)
        
        # Store in appropriate memory type
        if result.success:
            await self.memory.store(encoded, "procedural")  # Skill
        else:
            await self.memory.store(encoded, "episodic")    # Experience
        
        # Track for outer loop
        self.trajectory_buffer.append(trajectory)
        self.performance_metrics.append({
            "task": task.id,
            "success": result.success,
            "tokens": result.tokens_used,
            "time": result.duration
        })
```

### Outer Loop: Architecture Evolution

```python
class OuterLoop:
    """Evolve the memory architecture itself"""
    
    def __init__(self, memory: AgentMemory):
        self.memory = memory
        self.evolution_history = []
    
    async def evolve(self, feedback: PerformanceFeedback):
        """Diagnose-and-Design architecture evolution"""
        
        # Step 1: Diagnose bottlenecks
        diagnosis = await self._diagnose(feedback)
        
        if not diagnosis.has_actionable_insights:
            return EvolutionResult.NO_ACTION
        
        # Step 2: Design improvements
        candidates = await self._generate_architecture_candidates(diagnosis)
        
        # Step 3: Tournament selection
        winner = await self._tournament_select(candidates, feedback.tasks)
        
        # Step 4: Apply if improvement
        if winner.performance > self.memory.current_performance * 1.05:
            await self._apply_architecture(winner)
            return EvolutionResult.EVOLVED
        
        return EvolutionResult.RETAINED
    
    async def _diagnose(self, feedback: PerformanceFeedback) -> Diagnosis:
        """Identify memory system bottlenecks"""
        return Diagnosis(
            encoding_issues=self._analyze_encoding(feedback),
            storage_issues=self._analyze_storage(feedback),
            retrieval_issues=self._analyze_retrieval(feedback),
            management_issues=self._analyze_management(feedback),
            recommendations=await self._generate_recommendations(feedback)
        )
```

### Dual-Loop Integration

```python
class DualLoopAgent:
    """
    Agent with dual-loop learning:
    - Inner: Learn content (every task)
    - Outer: Evolve architecture (periodically)
    """
    
    def __init__(self, config: AgentConfig):
        self.memory = EvolvableMemory(config.memory)
        self.inner_loop = InnerLoop(self.memory)
        self.outer_loop = OuterLoop(self.memory)
        
        self.inner_iterations = 0
        self.evolution_threshold = config.evolution_threshold  # e.g., 100
        
    async def run(self, task: Task) -> TaskResult:
        """Execute with dual-loop learning"""
        
        # Build context with attention budget
        context = await self._build_context(task)
        
        # Execute task
        result = await self._execute(task, context)
        
        # Inner loop: Always learn
        await self.inner_loop.learn(task, result, self.get_trajectory())
        self.inner_iterations += 1
        
        # Outer loop: Periodically evolve
        if self.inner_iterations >= self.evolution_threshold:
            feedback = self.inner_loop.get_performance_summary()
            await self.outer_loop.evolve(feedback)
            self.inner_iterations = 0
        
        return result
    
    async def _build_context(self, task: Task) -> Context:
        """Build context respecting attention budget"""
        context = Context(max_tokens=self.config.context_limit)
        
        # Priority 1: Current task (start of context)
        context.add_at_start(task.to_prompt())
        
        # Priority 2: Relevant memory (end of context)
        memories = await self.memory.retrieve(
            query=task.description,
            max_tokens=context.remaining * 0.3  # 30% for memory
        )
        context.add_at_end(self._format_memories(memories))
        
        # Priority 3: Tool definitions (after task)
        tools = await self._select_tools(task)
        context.add_after_start(self._format_tools(tools))
        
        return context
```

---

## Part 9: Failure Handling Strategies

### Diagnose-and-Design Pattern

From MemEvolve, applied to runtime failures:

```python
class DiagnoseAndDesignHandler:
    """Runtime failure handling using MemEvolve pattern"""
    
    def __init__(self, memory: AgentMemory):
        self.memory = memory
        self.failure_patterns = FailurePatternLibrary()
    
    async def handle(self, failure: Failure) -> RecoveryResult:
        """1. Diagnose, 2. Retrieve similar cases, 3. Design recovery"""
        
        # Step 1: Diagnose
        diagnosis = Diagnosis(
            failure_type=self._categorize(failure),
            context=self._extract_context(failure),
            severity=self._assess_severity(failure),
            is_retryable=self._check_retryable(failure)
        )
        
        # Step 2: Retrieve similar past failures
        similar_cases = await self.memory.retrieve(
            query=f"failure: {diagnosis.failure_type}",
            memory_type="episodic",
            filters={"type": "failure_recovery"},
            k=5
        )
        
        # Step 3: Design recovery
        if similar_cases and similar_cases[0].recovery_success:
            recovery = self._adapt_past_strategy(
                similar_cases[0].recovery_strategy,
                diagnosis
            )
        else:
            recovery = await self._generate_recovery(diagnosis)
        
        # Execute and learn
        result = await self._execute_recovery(recovery)
        await self._learn_from_outcome(failure, diagnosis, recovery, result)
        
        return result
```

### Graceful Degradation Chain

```python
class GracefulDegradation:
    """Progressive fallback for agent failures"""
    
    FALLBACK_CHAIN = [
        ("retry_same", "Retry with same approach"),
        ("retry_simplified", "Retry with simplified approach"),
        ("switch_tool", "Switch to alternative tool"),
        ("reduce_scope", "Reduce scope and retry"),
        ("checkpoint_escalate", "Checkpoint and escalate to human"),
        ("graceful_terminate", "Terminate with partial results")
    ]
    
    async def handle(
        self, 
        task: Task, 
        failure: Failure,
        attempt: int = 0
    ) -> TaskResult:
        """Try progressively simpler approaches"""
        
        if attempt >= len(self.FALLBACK_CHAIN):
            return TaskResult.failed(
                reason="All fallback strategies exhausted",
                partial_result=self._get_partial_result()
            )
        
        strategy_name, description = self.FALLBACK_CHAIN[attempt]
        strategy = getattr(self, strategy_name)
        
        try:
            logger.info(f"Attempting recovery: {description}")
            result = await strategy(task, failure)
            
            # Learn from successful recovery
            await self._record_success(strategy_name, task, failure)
            return result
            
        except Exception as new_failure:
            logger.warning(f"Recovery failed: {strategy_name}")
            await self._record_failure(strategy_name, new_failure)
            return await self.handle(task, new_failure, attempt + 1)
```

---

## Part 10: Production Implementation Checklist

### Memory System Checklist

```
□ ESRM Components
  □ Encoder with anchored iterative summarization
  □ Typed storage (episodic/semantic/procedural)
  □ Phase-aware retriever with attention budget
  □ Progressive disclosure manager

□ Context Engineering
  □ Token budget monitoring (warning at 70%, critical at 85%)
  □ Compression with structured preservation
  □ Attention-aware content ordering
  □ Probe-based compression validation

□ Evolution Capability
  □ Inner loop for experience learning
  □ Outer loop for architecture evolution
  □ Tournament selection for improvements
  □ Rollback capability
```

### Tool Design Checklist

```
□ Consolidation Principle
  □ No overlapping tool descriptions
  □ Human can unambiguously select tool
  □ Primitives over specialists when possible

□ Description Engineering
  □ Clear when-to-use conditions
  □ Input examples with format
  □ Output structure documented
  □ Error codes with resolution steps

□ Response Optimization
  □ Configurable verbosity levels
  □ Default to concise format
  □ Metadata only when needed
```

### Multi-Agent Checklist

```
□ Context Isolation
  □ Agents have focused, minimal context
  □ No cross-contamination between subtasks
  □ Coordinator has routing-only context

□ Communication
  □ Direct forwarding for user-visible results
  □ No supervisor paraphrasing of sub-agent output
  □ Explicit handoff protocols

□ Shared Learning
  □ Shared memory pool for cross-agent learning
  □ Federated updates for experience propagation
  □ Architecture evolution applied to all agents
```

### Evaluation Checklist

```
□ Probe-Based Testing
  □ Generate probes from critical content
  □ Test file path preservation
  □ Test decision preservation
  □ Test error resolution preservation

□ Bias Mitigation
  □ Position swapping for pairwise comparison
  □ Multi-evaluator averaging
  □ Explicit criteria rubrics

□ Multi-Objective
  □ Track performance (accuracy)
  □ Track cost (tokens used)
  □ Track latency (time taken)
  □ Pareto-optimal selection
```

---

## Summary: Key Principles

### 1. Context Quality > Model Capability

Better model × poor context < worse model × excellent context. Curate ruthlessly.

### 2. Tokens Per Task > Tokens Per Request

Aggressive compression creates re-fetch cycles. Preserve signal for functional quality.

### 3. Consolidation Over Specialization

Fewer unambiguous tools beat many overlapping specialists. Let models reason.

### 4. External Verification Is Mandatory

Self-correction without external feedback amplifies errors. Use tools, RAG, humans.

### 5. Isolation Over Accumulation

Partition context across sub-agents. Single-agent massive context degrades.

### 6. Structure Forces Preservation

Anchored sections (files, decisions, next steps) prevent information drift.

### 7. Evolution Over Static Design

Memory architectures that can change beat fixed architectures across tasks.

### 8. Probe-Based Evaluation

Measure functional capability, not lexical similarity. Can the agent still work?

---

## References

- **MemEvolve Paper**: Zhang et al. "MemEvolve: Meta-Evolution of Agent Memory Systems" (arXiv:2512.18746)
- **MemEvolve Code**: https://github.com/bingreeky/MemEvolve
- **Lost in the Middle**: Liu et al., TACL 2024
- **RULER Benchmark**: Context window validation research
- **Factory Research**: Anchored iterative summarization (Dec 2025)
- **Vercel d0 Agent**: Tool consolidation case study
- **MT-Bench**: LLM-as-judge methodology (Zheng et al., 2023)
- **Baseline Systems**: Voyager, ExpeL, SkillWeaver, DILU, Generative Agents, Agent-KB, Mobile-Agent-E, MEMP, Dynamic Cheatsheet, Agent Workflow Memory, Evolver

