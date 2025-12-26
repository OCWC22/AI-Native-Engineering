# MemEvolve Paper Synthesis: Actionable Guidance for Long-Running LLM Agents

> **Paper**: MemEvolve: Meta-Evolution of Agent Memory Systems  
> **Authors**: OPPO AI Agent Team, LV-NUS Lab (Zhang et al., December 2025)  
> **Source**: [arXiv:2512.18746](https://arxiv.org/abs/2512.18746) | [GitHub](https://github.com/bingreeky/MemEvolve)

---

## Executive Summary

MemEvolve introduces a **meta-evolutionary framework** that enables LLM agents to not only learn from experience, but to **evolve how they learn**. The key insight: while current agent memory systems can accumulate knowledge within a fixed structure, they cannot adapt their fundamental memory architecture to different task contexts. MemEvolve addresses this by jointly evolving both the *content* of agent memory and the *architecture* of the memory system itself.

**Key Results**:
- 3.54% to 17.06% performance improvements across diverse agent frameworks
- Strong cross-task, cross-LLM, and cross-framework generalization
- No significant increase in computational overhead

---

## Part 1: Core Concepts for Agent Builders

### The Three Levels of Agent Learning

The paper distinguishes three tiers of learning capability:

| Learner Type | Description | Memory Approach | Example |
|--------------|-------------|-----------------|---------|
| **Mediocre** | No persistent memory | Stateless - each task starts fresh | Basic ReAct agents |
| **Skillful** | Fixed memory architecture | Can learn content but not adapt structure | Voyager, ExpeL, SkillWeaver |
| **Adaptive** | Evolving memory architecture | Can modify how they encode, store, retrieve, and manage knowledge | MemEvolve-enabled agents |

**Implication for builders**: Most production agents today are "skillful learners" at best. To build truly robust long-running agents, you need to design for **architectural adaptability**, not just content accumulation.

### The Four Pillars of Agent Memory (ESRM Model)

MemEvolve establishes a modular decomposition of any memory system into four fundamental components:

```
┌─────────────────────────────────────────────────────────────────┐
│                     AGENT MEMORY SYSTEM                          │
├──────────────┬──────────────┬──────────────┬───────────────────┤
│   ENCODE     │    STORE     │   RETRIEVE   │      MANAGE       │
│     (E)      │     (U)      │     (R)      │       (G)         │
├──────────────┼──────────────┼──────────────┼───────────────────┤
│ Raw          │ Integration  │ Context-     │ Offline           │
│ experience → │ into         │ aware        │ consolidation     │
│ structured   │ persistent   │ recall       │ & abstraction     │
│ representation│ storage     │              │                   │
└──────────────┴──────────────┴──────────────┴───────────────────┘
```

#### 1. Encode (E) — Transform Raw Experience

**What it does**: Converts trajectories, observations, tool outputs, and feedback into structured representations.

**Implementation patterns**:
```python
class MemoryEncoder:
    """Strategies for encoding agent experiences"""
    
    def encode_trajectory(self, trajectory: AgentTrajectory) -> MemoryRecord:
        """Options:
        - Raw storage: Store complete trajectory as-is
        - Distillation: Extract key insights/patterns
        - Multi-level: Create hierarchical representations
          (action-level → task-level → domain-level)
        - Agent-driven: Let LLM decide what to remember
        """
        pass
    
    def encode_tool_result(self, tool_call: ToolCall, result: Any) -> MemoryRecord:
        """Encode tool usage patterns for reuse"""
        pass
```

**Design recommendations**:
- Use **multi-level encoding** for long-running agents (raw logs + summarized insights + abstract patterns)
- Implement **agent-driven encoding** where the LLM decides importance
- Consider **temporal decay** - recent experiences encoded in more detail

#### 2. Store (U) — Integrate into Persistent Memory

**What it does**: Manages how encoded experiences are added to and organized in memory.

**Implementation patterns**:
```python
class MemoryStore:
    """Strategies for storing encoded experiences"""
    
    def __init__(self):
        self.episodic = []      # Specific experiences
        self.semantic = {}       # General knowledge
        self.procedural = {}     # Skills/tools
        
    def store(self, record: MemoryRecord, store_type: str):
        """Options:
        - Append-only: Simple log-style storage
        - Indexed: Organize by task type, domain, tool
        - Hierarchical: Tree structure (domain → task → action)
        - Hybrid: Multiple storage backends for different purposes
        """
        pass
```

**Design recommendations**:
- Use **typed storage** (episodic for experiences, semantic for facts, procedural for skills)
- Implement **deduplication** to prevent memory bloat
- Add **versioning** for memory evolution tracking

#### 3. Retrieve (R) — Context-Aware Recall

**What it does**: Recalls relevant memory content based on current context and task.

**Implementation patterns**:
```python
class MemoryRetriever:
    """Strategies for retrieving relevant memories"""
    
    def retrieve(self, query: str, context: TaskContext) -> List[MemoryRecord]:
        """Options:
        - Keyword matching: Fast but shallow
        - Embedding similarity: Semantic but slow
        - Hybrid: Combine multiple strategies
        - Adaptive: Different strategies for different phases
          (planning → high-level insights, execution → specific tool examples)
        """
        pass
```

**Design recommendations**:
- Use **phase-aware retrieval** (different strategies for planning vs execution)
- Implement **relevance scoring** with multiple signals (semantic similarity + recency + success rate)
- Add **retrieval limits** to prevent context window overflow

#### 4. Manage (G) — Offline Maintenance

**What it does**: Performs background operations like consolidation, abstraction, and cleanup.

**Implementation patterns**:
```python
class MemoryManager:
    """Strategies for memory maintenance"""
    
    async def maintain(self, memory: MemoryStore):
        """Options:
        - Periodic consolidation: Merge similar experiences
        - Abstraction: Extract general patterns from specifics
        - Pruning: Remove outdated or low-value memories
        - Indexing: Rebuild search indices
        - Evolution: Upgrade memory representations
        """
        pass
```

**Design recommendations**:
- Run maintenance **asynchronously** during idle periods
- Implement **graduated retention** (compress older memories)
- Add **memory health metrics** (size, retrieval latency, hit rate)

---

## Part 2: Multi-Agent System Design

### Coordination Patterns

MemEvolve's insights apply directly to multi-agent coordination challenges:

#### Pattern 1: Shared Memory Pool

```
┌─────────────────────────────────────────────────────────┐
│                    SHARED MEMORY POOL                   │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐   │
│  │Episodic │  │Semantic │  │Procedural│ │Skills   │   │
│  │Memory   │  │Memory   │  │Memory    │ │Library  │   │
│  └────┬────┘  └────┬────┘  └────┬─────┘ └────┬────┘   │
│       │            │            │             │        │
└───────┼────────────┼────────────┼─────────────┼────────┘
        │            │            │             │
   ┌────▼────┐  ┌────▼────┐  ┌────▼────┐  ┌────▼────┐
   │Agent A  │  │Agent B  │  │Agent C  │  │Agent D  │
   │(Planner)│  │(Coder)  │  │(Reviewer)│ │(Deployer)│
   └─────────┘  └─────────┘  └─────────┘  └─────────┘
```

**Implementation**:
```python
class SharedMemoryPool:
    """Centralized memory accessible by all agents"""
    
    def __init__(self):
        self.encoder = SharedEncoder()
        self.store = DistributedStore()
        self.retriever = ContextAwareRetriever()
        self.manager = AsyncMemoryManager()
        self._locks = {}  # Prevent race conditions
    
    async def record_experience(self, agent_id: str, experience: Experience):
        """Thread-safe experience recording"""
        async with self._locks.get(agent_id, asyncio.Lock()):
            encoded = self.encoder.encode(experience, agent_id)
            await self.store.add(encoded)
    
    async def get_relevant_context(self, agent_id: str, task: Task) -> Context:
        """Retrieve context customized for specific agent role"""
        memories = await self.retriever.retrieve(
            query=task.description,
            filters={"relevant_to_role": agent_id}
        )
        return self._format_for_agent(memories, agent_id)
```

#### Pattern 2: Federated Memory with Cross-Agent Learning

```
┌──────────────────────────────────────────────────────────────┐
│                    META-COORDINATOR                          │
│     (Aggregates insights, evolves shared patterns)           │
└──────────────────────────┬───────────────────────────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        ▼                  ▼                  ▼
┌───────────────┐  ┌───────────────┐  ┌───────────────┐
│   Agent A     │  │   Agent B     │  │   Agent C     │
│ ┌───────────┐ │  │ ┌───────────┐ │  │ ┌───────────┐ │
│ │Local Mem  │ │  │ │Local Mem  │ │  │ │Local Mem  │ │
│ └───────────┘ │  │ └───────────┘ │  │ └───────────┘ │
└───────────────┘  └───────────────┘  └───────────────┘
```

**Key insight from MemEvolve**: Agents can learn from each other's memory architectures, not just content. The meta-coordinator can:
1. Identify which agent's memory strategy works best for which task types
2. Cross-pollinate successful patterns
3. Evolve shared memory primitives

### State Management for Long-Running Agents

#### Hierarchical State Architecture

```python
class AgentStateManager:
    """Three-tier state management for long-running agents"""
    
    def __init__(self):
        # Tier 1: Working Memory (immediate context)
        self.working_memory = WorkingMemory(max_tokens=8000)
        
        # Tier 2: Session Memory (current task chain)
        self.session_memory = SessionMemory(ttl_hours=24)
        
        # Tier 3: Long-term Memory (persistent across sessions)
        self.long_term_memory = LongTermMemory(
            encoder=AdaptiveEncoder(),
            store=HybridStore(),
            retriever=HierarchicalRetriever(),
            manager=EvolvingManager()
        )
    
    async def get_context(self, task: Task) -> str:
        """Build context from all memory tiers"""
        context_parts = []
        
        # Always include working memory
        context_parts.append(self.working_memory.get_recent(k=10))
        
        # Add relevant session context
        session_context = self.session_memory.get_related(task)
        if session_context:
            context_parts.append(session_context)
        
        # Selectively retrieve from long-term memory
        if task.requires_domain_knowledge:
            lt_context = await self.long_term_memory.retrieve(
                query=task.description,
                max_items=5
            )
            context_parts.append(lt_context)
        
        return self._format_context(context_parts)
    
    async def update_state(self, event: AgentEvent):
        """Update appropriate memory tier based on event type"""
        # Always update working memory
        self.working_memory.add(event)
        
        # Update session memory for significant events
        if event.is_significant:
            self.session_memory.add(event)
        
        # Asynchronously update long-term memory
        if event.is_learnable:
            asyncio.create_task(
                self.long_term_memory.learn(event)
            )
```

#### Checkpointing and Recovery

For long-running agents, implement robust state persistence:

```python
class AgentCheckpointManager:
    """Checkpoint and recovery for agent state"""
    
    def __init__(self, storage_backend: str = "sqlite"):
        self.storage = self._init_storage(storage_backend)
        self.checkpoint_interval = 100  # steps
        
    async def checkpoint(self, agent_state: AgentState):
        """Save agent state to persistent storage"""
        checkpoint = {
            "timestamp": datetime.utcnow().isoformat(),
            "step": agent_state.step_count,
            "working_memory": agent_state.working_memory.serialize(),
            "session_memory": agent_state.session_memory.serialize(),
            "task_queue": agent_state.task_queue.serialize(),
            "memory_architecture": agent_state.memory.get_architecture(),
        }
        
        await self.storage.save(
            key=f"checkpoint_{agent_state.agent_id}",
            value=checkpoint
        )
        
        # Keep history for rollback
        await self.storage.save(
            key=f"checkpoint_{agent_state.agent_id}_{agent_state.step_count}",
            value=checkpoint
        )
    
    async def restore(self, agent_id: str, step: Optional[int] = None) -> AgentState:
        """Restore agent state from checkpoint"""
        key = f"checkpoint_{agent_id}"
        if step:
            key = f"{key}_{step}"
        
        checkpoint = await self.storage.load(key)
        return AgentState.deserialize(checkpoint)
```

---

## Part 3: Failure Handling Strategies

### The Diagnose-and-Design Pattern

MemEvolve's meta-evolution uses a "Diagnose-and-Design" process that's directly applicable to runtime failure handling:

```python
class AdaptiveFailureHandler:
    """MemEvolve-inspired failure handling"""
    
    def __init__(self, agent: Agent):
        self.agent = agent
        self.failure_memory = FailureMemory()
        self.recovery_strategies = StrategyLibrary()
    
    async def handle_failure(self, failure: Failure) -> RecoveryAction:
        """1. Diagnose, 2. Retrieve similar cases, 3. Design recovery"""
        
        # Step 1: Diagnose - Identify failure type and root cause
        diagnosis = await self._diagnose(failure)
        
        # Step 2: Retrieve - Find similar past failures
        similar_cases = await self.failure_memory.find_similar(
            failure_type=diagnosis.type,
            context=diagnosis.context,
            k=5
        )
        
        # Step 3: Design - Generate recovery strategy
        if similar_cases:
            # Use past successful recovery strategies
            recovery = self._adapt_past_strategy(similar_cases, diagnosis)
        else:
            # Generate new recovery strategy
            recovery = await self._generate_recovery(diagnosis)
        
        # Step 4: Execute and learn
        result = await self._execute_recovery(recovery)
        await self._learn_from_outcome(failure, diagnosis, recovery, result)
        
        return result
    
    async def _diagnose(self, failure: Failure) -> Diagnosis:
        """Categorize and analyze failure"""
        return Diagnosis(
            type=self._categorize_failure(failure),
            context=self._extract_context(failure),
            bottleneck=self._identify_bottleneck(failure),
            severity=self._assess_severity(failure)
        )
    
    async def _learn_from_outcome(
        self, 
        failure: Failure, 
        diagnosis: Diagnosis,
        recovery: RecoveryAction,
        result: RecoveryResult
    ):
        """Update failure memory with outcome"""
        record = FailureRecord(
            failure=failure,
            diagnosis=diagnosis,
            recovery=recovery,
            success=result.success,
            cost=result.cost
        )
        
        await self.failure_memory.store(record)
        
        # If recovery strategy was successful, add to strategy library
        if result.success:
            await self.recovery_strategies.add(
                failure_type=diagnosis.type,
                strategy=recovery,
                effectiveness=result.effectiveness
            )
```

### Failure Categories and Recovery Strategies

Based on MemEvolve's multi-objective optimization, handle failures across three dimensions:

| Failure Type | Detection | Recovery Strategy |
|--------------|-----------|-------------------|
| **Performance** | Task not completed, incorrect output | Retry with different approach, escalate to human |
| **Cost** | Token budget exceeded | Reduce context, switch to smaller model, batch operations |
| **Latency** | Timeout, slow responses | Implement caching, parallelize steps, precompute |

```python
class MultiObjectiveFailureHandler:
    """Handle failures considering performance, cost, and latency"""
    
    async def select_recovery(
        self, 
        failure: Failure, 
        candidates: List[RecoveryStrategy]
    ) -> RecoveryStrategy:
        """Pareto-optimal recovery selection"""
        
        # Score each candidate on three objectives
        scored = []
        for strategy in candidates:
            score = ParetoScore(
                performance=await self._estimate_success_prob(strategy),
                cost=await self._estimate_cost(strategy),
                latency=await self._estimate_latency(strategy)
            )
            scored.append((strategy, score))
        
        # Return Pareto-optimal choice based on current constraints
        return self._pareto_select(
            scored,
            constraints=self._get_current_constraints()
        )
```

### Graceful Degradation Patterns

```python
class GracefulDegradationHandler:
    """Progressive fallback for agent failures"""
    
    FALLBACK_CHAIN = [
        "retry_with_same_approach",
        "retry_with_simplified_approach",
        "switch_to_alternative_tool",
        "reduce_scope_and_retry",
        "checkpoint_and_escalate",
        "graceful_termination"
    ]
    
    async def handle_with_degradation(
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
        
        strategy = self.FALLBACK_CHAIN[attempt]
        
        try:
            result = await getattr(self, strategy)(task, failure)
            
            # Learn from successful recovery
            await self._record_recovery_success(strategy, task, failure)
            
            return result
            
        except Exception as new_failure:
            # Log and try next fallback
            await self._record_failure(strategy, new_failure)
            return await self.handle_with_degradation(
                task, new_failure, attempt + 1
            )
```

---

## Part 4: Tool Calling Integration

### Memory-Aware Tool Selection

MemEvolve's retrieval mechanisms enhance tool calling:

```python
class MemoryAwareToolSelector:
    """Select tools based on memory of past tool usage"""
    
    def __init__(self, tool_registry: ToolRegistry, memory: AgentMemory):
        self.tools = tool_registry
        self.memory = memory
        self.usage_stats = ToolUsageStats()
    
    async def select_tools(
        self, 
        task: Task, 
        max_tools: int = 5
    ) -> List[Tool]:
        """Select most relevant tools for task"""
        
        # Get task embedding
        task_embedding = await self._embed_task(task)
        
        # Retrieve past similar tasks and their tool usage
        similar_tasks = await self.memory.retrieve(
            query=task.description,
            memory_type="procedural",  # Tool/skill memory
            k=10
        )
        
        # Score tools based on:
        # 1. Semantic relevance to task
        # 2. Success rate in similar past tasks
        # 3. Efficiency (cost/latency)
        tool_scores = {}
        for tool in self.tools.all():
            score = ToolScore(
                relevance=self._compute_relevance(tool, task_embedding),
                success_rate=self._get_success_rate(tool, similar_tasks),
                efficiency=self._get_efficiency(tool)
            )
            tool_scores[tool.name] = score.combined()
        
        # Return top-k tools
        sorted_tools = sorted(
            tool_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        return [self.tools.get(name) for name, _ in sorted_tools[:max_tools]]
```

### Tool Execution with Learning

```python
class LearningToolExecutor:
    """Execute tools and learn from outcomes"""
    
    def __init__(self, memory: AgentMemory):
        self.memory = memory
        self.execution_history = []
    
    async def execute(
        self, 
        tool: Tool, 
        args: Dict[str, Any],
        context: TaskContext
    ) -> ToolResult:
        """Execute tool and record experience"""
        
        start_time = time.time()
        try:
            result = await tool.run(**args)
            success = True
            error = None
        except Exception as e:
            result = None
            success = False
            error = str(e)
        
        execution_time = time.time() - start_time
        
        # Record experience
        experience = ToolExperience(
            tool_name=tool.name,
            args=args,
            context=context.summary(),
            result=result,
            success=success,
            error=error,
            execution_time=execution_time,
            timestamp=datetime.utcnow()
        )
        
        # Store in procedural memory
        await self.memory.store(
            record=self._encode_experience(experience),
            memory_type="procedural"
        )
        
        # Update tool statistics
        await self._update_tool_stats(tool.name, experience)
        
        return ToolResult(
            output=result,
            success=success,
            error=error,
            metadata={"execution_time": execution_time}
        )
    
    async def _encode_experience(self, exp: ToolExperience) -> MemoryRecord:
        """Encode tool experience for memory storage"""
        return MemoryRecord(
            content={
                "tool": exp.tool_name,
                "pattern": self._extract_pattern(exp),
                "outcome": "success" if exp.success else "failure",
                "lesson": await self._extract_lesson(exp)
            },
            embeddings=await self._compute_embeddings(exp),
            metadata={
                "type": "tool_usage",
                "timestamp": exp.timestamp,
                "execution_time": exp.execution_time
            }
        )
```

### Tool Skill Synthesis (Voyager/SkillWeaver Pattern)

From the baseline systems in EvolveLab:

```python
class ToolSkillSynthesizer:
    """Synthesize reusable tool patterns from experiences"""
    
    def __init__(self, memory: AgentMemory, llm: LLM):
        self.memory = memory
        self.llm = llm
        self.skill_library = SkillLibrary()
    
    async def synthesize_skills(self):
        """Periodically extract reusable patterns from tool usage"""
        
        # Get recent successful tool executions
        recent_successes = await self.memory.retrieve(
            query="successful tool executions",
            memory_type="procedural",
            filters={"outcome": "success"},
            k=100
        )
        
        # Cluster similar usage patterns
        clusters = self._cluster_by_pattern(recent_successes)
        
        for cluster in clusters:
            if len(cluster) >= 3:  # Minimum occurrences for pattern
                # Synthesize skill from cluster
                skill = await self._synthesize_skill(cluster)
                
                if skill and await self._validate_skill(skill):
                    await self.skill_library.add(skill)
    
    async def _synthesize_skill(self, examples: List[ToolExperience]) -> Skill:
        """Use LLM to extract reusable skill from examples"""
        
        prompt = f"""Analyze these tool usage examples and extract a reusable pattern:

Examples:
{self._format_examples(examples)}

Extract:
1. Pattern name
2. When to apply this pattern
3. Parameterized tool call template
4. Pre-conditions and post-conditions
"""
        
        response = await self.llm.generate(prompt)
        return Skill.parse(response)
```

---

## Part 5: System Design Principles

### Principle 1: Design for Architectural Evolution

**From MemEvolve**: Memory architectures should not be static. Design systems that can modify how they learn, not just what they learn.

**Implementation**:
```python
class EvolvableMemorySystem:
    """Memory system with swappable components"""
    
    def __init__(self, config: MemoryConfig):
        # Components are interfaces, not concrete implementations
        self.encoder: Encoder = self._load_encoder(config.encoder_type)
        self.store: Store = self._load_store(config.store_type)
        self.retriever: Retriever = self._load_retriever(config.retriever_type)
        self.manager: Manager = self._load_manager(config.manager_type)
        
        # Evolution tracking
        self.architecture_version = config.version
        self.performance_history = []
    
    def evolve_component(
        self, 
        component: str, 
        new_type: str,
        reason: str
    ):
        """Hot-swap a memory component"""
        old_component = getattr(self, component)
        new_component = self._load_component(component, new_type)
        
        # Migrate state if possible
        if hasattr(old_component, 'export_state'):
            state = old_component.export_state()
            new_component.import_state(state)
        
        setattr(self, component, new_component)
        
        # Log evolution
        self._log_evolution(component, old_component, new_component, reason)
```

### Principle 2: Multi-Objective Optimization

**From MemEvolve**: Evaluate agent systems on performance, cost, AND latency simultaneously.

```python
class MultiObjectiveEvaluator:
    """Evaluate agent performance across multiple dimensions"""
    
    OBJECTIVES = {
        "performance": {"weight": 0.5, "direction": "maximize"},
        "api_cost": {"weight": 0.3, "direction": "minimize"},
        "latency": {"weight": 0.2, "direction": "minimize"}
    }
    
    def evaluate(self, agent: Agent, task_batch: List[Task]) -> EvaluationResult:
        """Comprehensive multi-objective evaluation"""
        
        results = []
        for task in task_batch:
            result = agent.run(task)
            results.append(TaskResult(
                success=result.success,
                tokens_used=result.token_count,
                time_taken=result.duration,
                output_quality=self._assess_quality(result)
            ))
        
        return EvaluationResult(
            performance=self._compute_performance(results),
            cost=self._compute_cost(results),
            latency=self._compute_latency(results),
            pareto_score=self._compute_pareto_score(results)
        )
    
    def _compute_pareto_score(self, results: List[TaskResult]) -> float:
        """Compute Pareto-optimal score considering all objectives"""
        scores = {}
        for obj_name, obj_config in self.OBJECTIVES.items():
            raw_score = getattr(self, f"_compute_{obj_name}")(results)
            if obj_config["direction"] == "minimize":
                raw_score = -raw_score
            scores[obj_name] = raw_score * obj_config["weight"]
        
        return sum(scores.values())
```

### Principle 3: Hierarchical Memory with Typed Storage

```python
class ProductionMemoryArchitecture:
    """Production-ready hierarchical memory system"""
    
    def __init__(self):
        # Episodic: Specific experiences with full context
        self.episodic = EpisodicMemory(
            storage=TimestampedVectorStore(),
            retention_days=30,
            max_entries=10000
        )
        
        # Semantic: Factual knowledge and domain information
        self.semantic = SemanticMemory(
            storage=KnowledgeGraph(),
            auto_consolidate=True
        )
        
        # Procedural: Skills, tools, and action patterns
        self.procedural = ProceduralMemory(
            storage=SkillLibrary(),
            skill_synthesizer=AutoSkillSynthesizer()
        )
        
        # Working: Current task context
        self.working = WorkingMemory(
            max_tokens=8000,
            eviction_policy="importance_weighted_lru"
        )
    
    async def contextual_retrieve(
        self, 
        query: str, 
        task_phase: str
    ) -> RetrievalResult:
        """Phase-aware retrieval from appropriate memory types"""
        
        if task_phase == "planning":
            # Focus on semantic and high-level episodic
            return await self._planning_retrieval(query)
        elif task_phase == "execution":
            # Focus on procedural and detailed episodic
            return await self._execution_retrieval(query)
        elif task_phase == "reflection":
            # Broad retrieval for learning
            return await self._reflection_retrieval(query)
```

### Principle 4: Continuous Learning with Stability

```python
class StableContinuousLearner:
    """Learn continuously while maintaining stability"""
    
    def __init__(self, memory: AgentMemory):
        self.memory = memory
        self.performance_baseline = None
        self.evolution_candidates = []
        
    async def learn_from_experience(self, experience: Experience):
        """Safe continuous learning with rollback capability"""
        
        # Create checkpoint before learning
        checkpoint = await self.memory.checkpoint()
        
        try:
            # Apply learning update
            await self.memory.integrate(experience)
            
            # Validate learning didn't degrade performance
            if self.performance_baseline:
                current_performance = await self._quick_eval()
                
                if current_performance < self.performance_baseline * 0.95:
                    # Rollback if significant degradation
                    await self.memory.restore(checkpoint)
                    logger.warning(
                        f"Learning rollback: {current_performance} < "
                        f"{self.performance_baseline * 0.95}"
                    )
                    return LearningResult.ROLLBACK
            
            return LearningResult.SUCCESS
            
        except Exception as e:
            await self.memory.restore(checkpoint)
            logger.error(f"Learning error, rolled back: {e}")
            return LearningResult.ERROR
```

---

## Part 6: Architectural Patterns

### Pattern 1: Tournament-Style Evolution

From MemEvolve's automatic multi-round evolution:

```python
class TournamentEvolution:
    """Evolve agent components through tournament selection"""
    
    async def evolve(
        self,
        base_agent: Agent,
        task_batch: List[Task],
        num_rounds: int = 3,
        candidates_per_round: int = 3
    ) -> Agent:
        """Tournament-style evolution"""
        
        current_best = base_agent
        
        for round_num in range(num_rounds):
            # Generate candidate variations
            candidates = [current_best]
            for _ in range(candidates_per_round):
                variation = await self._generate_variation(current_best)
                candidates.append(variation)
            
            # Evaluate all candidates
            scores = {}
            for candidate in candidates:
                scores[candidate.id] = await self._evaluate(
                    candidate, 
                    task_batch[:20]  # Initial evaluation
                )
            
            # Top performers go to finals
            finalists = sorted(
                candidates, 
                key=lambda c: scores[c.id],
                reverse=True
            )[:2]
            
            # Final evaluation with more tasks
            final_scores = {}
            for finalist in finalists:
                final_scores[finalist.id] = await self._evaluate(
                    finalist,
                    task_batch  # Full evaluation
                )
            
            # Select winner
            current_best = max(finalists, key=lambda f: final_scores[f.id])
        
        return current_best
```

### Pattern 2: Dual-Loop Learning

The core MemEvolve pattern:

```python
class DualLoopLearningSystem:
    """
    Inner loop: Learn content within fixed architecture
    Outer loop: Evolve the architecture itself
    """
    
    def __init__(self, agent: Agent):
        self.agent = agent
        self.inner_loop = ExperienceLearner(agent.memory)
        self.outer_loop = ArchitectureEvolver(agent.memory)
        
        self.inner_iterations = 0
        self.outer_iterations = 0
        self.evolution_threshold = 100  # Inner iterations before outer
        
    async def run(self, task: Task) -> TaskResult:
        """Execute task with dual-loop learning"""
        
        # Run task
        result = await self.agent.execute(task)
        
        # Inner loop: Learn from experience
        await self.inner_loop.learn(
            task=task,
            result=result,
            trajectory=self.agent.get_trajectory()
        )
        self.inner_iterations += 1
        
        # Outer loop: Periodically evolve architecture
        if self.inner_iterations >= self.evolution_threshold:
            await self._run_outer_loop()
            self.inner_iterations = 0
            self.outer_iterations += 1
        
        return result
    
    async def _run_outer_loop(self):
        """Evolve memory architecture based on accumulated feedback"""
        
        # Collect performance feedback
        feedback = await self.inner_loop.get_performance_summary()
        
        # Diagnose bottlenecks
        diagnosis = await self.outer_loop.diagnose(feedback)
        
        # Generate architectural improvements
        if diagnosis.has_actionable_insights:
            new_architecture = await self.outer_loop.design(diagnosis)
            
            # Validate and apply
            if await self.outer_loop.validate(new_architecture):
                await self.outer_loop.apply(new_architecture)
```

### Pattern 3: Federated Learning Across Agents

```python
class FederatedAgentLearning:
    """Share learning across multiple agent instances"""
    
    def __init__(self, agents: List[Agent], coordinator: Coordinator):
        self.agents = agents
        self.coordinator = coordinator
        self.global_memory = GlobalMemoryPool()
        
    async def federated_learning_round(self):
        """One round of federated learning"""
        
        # Collect local updates from all agents
        local_updates = []
        for agent in self.agents:
            update = await agent.memory.export_recent_learnings()
            local_updates.append(update)
        
        # Aggregate at coordinator
        aggregated = await self.coordinator.aggregate(local_updates)
        
        # Distribute back to agents
        for agent in self.agents:
            await agent.memory.import_global_learnings(aggregated)
        
        # Optionally evolve shared memory architecture
        if self._should_evolve():
            best_architecture = await self._identify_best_architecture()
            await self._propagate_architecture(best_architecture)
```

---

## Part 7: Implementation Recommendations

### Quick Start: Minimal Viable Memory System

```python
class MinimalAgentMemory:
    """Start here, then evolve"""
    
    def __init__(self):
        self.experiences = []  # Simple list storage
        self.embeddings_cache = {}
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
    
    def store(self, experience: Dict):
        """Store experience with embedding"""
        embedding = self.embedder.encode(str(experience))
        self.experiences.append({
            "data": experience,
            "embedding": embedding,
            "timestamp": datetime.utcnow()
        })
    
    def retrieve(self, query: str, k: int = 5) -> List[Dict]:
        """Simple similarity-based retrieval"""
        query_embedding = self.embedder.encode(query)
        
        scored = []
        for exp in self.experiences:
            similarity = cosine_similarity(
                query_embedding, 
                exp["embedding"]
            )
            scored.append((exp["data"], similarity))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        return [x[0] for x in scored[:k]]
```

### Production: Full-Featured Implementation

```python
class ProductionAgentMemory:
    """Production-ready memory system with all MemEvolve principles"""
    
    def __init__(self, config: MemoryConfig):
        # ESRM Components
        self.encoder = self._init_encoder(config)
        self.store = self._init_store(config)
        self.retriever = self._init_retriever(config)
        self.manager = self._init_manager(config)
        
        # Multi-objective tracking
        self.metrics = MemoryMetrics()
        
        # Evolution capability
        self.evolver = ArchitectureEvolver(self)
        
        # Checkpointing
        self.checkpointer = CheckpointManager(config.checkpoint_path)
        
    async def integrate_experience(self, experience: Experience):
        """Full experience integration pipeline"""
        # Encode
        encoded = await self.encoder.encode(experience)
        
        # Store
        await self.store.add(encoded)
        
        # Update metrics
        self.metrics.record_integration()
        
        # Trigger background management if needed
        if self.metrics.needs_maintenance:
            asyncio.create_task(self.manager.maintain())
    
    async def get_context(
        self, 
        query: str, 
        context: TaskContext
    ) -> RetrievalResult:
        """Context-aware retrieval"""
        result = await self.retriever.retrieve(
            query=query,
            context=context,
            max_items=self._dynamic_limit(context)
        )
        
        self.metrics.record_retrieval(result)
        return result
    
    def _dynamic_limit(self, context: TaskContext) -> int:
        """Adjust retrieval limit based on context"""
        base_limit = 10
        
        # Reduce limit if context window is constrained
        if context.remaining_tokens < 4000:
            return min(base_limit, 3)
        
        # Increase limit for complex tasks
        if context.task_complexity == "high":
            return base_limit * 2
        
        return base_limit
```

### Testing Memory Systems

```python
class MemorySystemTester:
    """Test suite for agent memory systems"""
    
    async def test_encode_store_retrieve_cycle(self, memory: AgentMemory):
        """Basic ESRM cycle test"""
        experience = self._create_test_experience()
        
        # Store
        await memory.integrate_experience(experience)
        
        # Retrieve
        results = await memory.get_context(
            query=experience.summary,
            context=TaskContext.default()
        )
        
        assert len(results) > 0
        assert self._experience_in_results(experience, results)
    
    async def test_retrieval_relevance(self, memory: AgentMemory):
        """Test that retrieval returns relevant results"""
        # Store diverse experiences
        experiences = self._create_diverse_experiences(n=100)
        for exp in experiences:
            await memory.integrate_experience(exp)
        
        # Query for specific type
        query = "database connection errors"
        results = await memory.get_context(query, TaskContext.default())
        
        # Verify relevance
        relevance_scores = [
            self._compute_relevance(query, r) for r in results
        ]
        assert all(score > 0.5 for score in relevance_scores)
    
    async def test_memory_evolution(self, memory: EvolvableMemory):
        """Test architecture evolution capability"""
        original_architecture = memory.get_architecture()
        
        # Trigger evolution
        feedback = self._create_feedback_indicating_retrieval_issues()
        await memory.evolver.evolve_based_on_feedback(feedback)
        
        new_architecture = memory.get_architecture()
        assert new_architecture != original_architecture
```

---

## Summary: Key Takeaways for Agent Builders

### 1. Memory Architecture Matters

Don't treat memory as an afterthought. The ESRM (Encode-Store-Retrieve-Manage) framework provides a principled way to design agent memory:
- **Encode**: Multi-level, agent-driven encoding
- **Store**: Typed storage (episodic/semantic/procedural)
- **Retrieve**: Phase-aware, context-sensitive retrieval
- **Manage**: Async consolidation and evolution

### 2. Design for Evolution

Build memory systems that can change:
- Use interfaces, not concrete implementations
- Track performance metrics across objectives
- Implement component hot-swapping
- Use tournament-style selection for improvements

### 3. Multi-Agent Coordination Through Shared Memory

For multi-agent systems:
- Use shared memory pools with role-based retrieval
- Implement federated learning for cross-agent improvement
- Design for eventual consistency, not perfect synchronization

### 4. Failure Handling = Learning Opportunity

Apply the Diagnose-and-Design pattern:
- Categorize failures systematically
- Store recovery strategies in memory
- Learn from both successes and failures
- Implement graceful degradation chains

### 5. Tool Calling Benefits from Memory

Memory-aware tool usage:
- Select tools based on past success rates
- Learn tool patterns and synthesize reusable skills
- Update tool statistics from every execution

### 6. Balance Performance, Cost, and Latency

Always evaluate across multiple objectives:
- Use Pareto-optimal selection for trade-offs
- Implement budget-aware retrieval limits
- Track and optimize all three dimensions

---

## References

- **Paper**: Zhang et al. "MemEvolve: Meta-Evolution of Agent Memory Systems" (arXiv:2512.18746)
- **Code**: https://github.com/bingreeky/MemEvolve
- **Baseline Systems**: Voyager, ExpeL, SkillWeaver, DILU, Generative Agents, Agent-KB, Mobile-Agent-E, MEMP, Dynamic Cheatsheet, Agent Workflow Memory, Evolver

