# Graph Report - go-wgpu-mat  (2026-08-02)

## Corpus Check

- 47 files · ~34,381 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary

- 629 nodes · 1734 edges · 29 communities (25 shown, 4 thin omitted)
- Extraction: 72% EXTRACTED · 28% INFERRED · 0% AMBIGUOUS · INFERRED: 488 edges (avg confidence: 0.8)
- Token cost: 0 input · 0 output

## Graph Freshness

- Built from commit: `313ac51b`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)

- Matrix
- mat_unit_test.go
- NewContext
- newPipelineCache
- matMulWGPUDeps
- newMockMatrix
- Context
- TestP4MetalKernels
- Decisions
- robustness_test.go
- go-wgpu-mat
- mat_bench_test.go
- P4 Device-resident Kernel Plan
- .agents/README.md
- Findings
- Submission Lifetime 詳細計画(SL-002 成果物)
- Current Status
- readBuffer
- Repo Agent Notes
- Work Log
- Commands
- Tasks
- add
- contextAdapterOptions
- bench-isolated.sh
- github.com/KEINOS/go-wgpu-mat
- ContextMode
- Shape

## God Nodes (most connected - your core abstractions)

1. `Matrix` - 100 edges
2. `NewContext()` - 67 edges
3. `NewMatrix()` - 58 edges
4. `Context` - 53 edges
5. `newMockMatrix()` - 48 edges
6. `sentinelError()` - 42 edges
7. `matMulWGPUDeps` - 42 edges
8. `wrapError()` - 36 edges
9. `serializeGPUTest()` - 27 edges
10. `shareMockContext()` - 23 edges

## Surprising Connections (you probably didn't know these)

- `Add()` --calls--> `defaultAddDeps()`  [INFERRED]
  mat/ops.go → mat/add.go
- `add()` --calls--> `wrapError()`  [INFERRED]
  mat/add.go → mat/errors.go
- `add()` --calls--> `runBinaryBroadcast()`  [INFERRED]
  mat/add.go → mat/ops.go
- `Add()` --calls--> `add()`  [INFERRED]
  mat/ops.go → mat/add.go
- `TestAddDispatchError()` --calls--> `add()`  [INFERRED]
  mat/ops_test.go → mat/add.go

## Import Cycles

- None detected.

## Communities (29 total, 4 thin omitted)

### Community 0 - "Matrix"

Cohesion: 0.07
Nodes (83): BindGroupEntry, createAddBindGroup(), dispatchAdd(), sentinelError(), wrapError(), Uint32, ceilDiv(), createMatMulUniform() (+75 more)

### Community 1 - "mat_unit_test.go"

Cohesion: 0.13
Nodes (30): defaultMatrixDeps(), T, TestContextReleaseIsIdempotent(), TestContextReleaseWithNilFields(), TestMatrixReadBackendError(), TestMatrixReadSuccessConvertsFromBytes(), TestMatrixReadUninitialized(), TestMatrixReadWriteRejectReleasedContext() (+22 more)

### Community 2 - "NewContext"

Cohesion: 0.12
Nodes (53): F, NewContext(), Example(), ExampleAdd(), ExampleMatMul(), ExampleNewMatrix(), ExampleReduceMax(), ExampleReduceSum() (+45 more)

### Community 3 - "newPipelineCache"

Cohesion: 0.08
Nodes (30): classifiedError, ComputePipeline, newError(), DefaultReleaseComputePipeline(), ComputePipeline, New(), ComputePipeline, T (+22 more)

### Community 4 - "matMulWGPUDeps"

Cohesion: 0.06
Nodes (68): BindGroupDescriptor, BindGroupLayoutDescriptor, BindGroupLayoutEntry, BufferBindingType, CommandEncoderDescriptor, ComputePassDescriptor, ComputePassEncoder, ComputePipelineDescriptor (+60 more)

### Community 5 - "newMockMatrix"

Cohesion: 0.14
Nodes (38): mockMatrixIO, decodeFloat32(), encodeFloat32(), T, newMockMatrix(), shareMockContext(), TestAddDispatchError(), TestAddDispatchesWithoutHostIO() (+30 more)

### Community 6 - "Context"

Cohesion: 0.11
Nodes (18): Adapter, AdapterInfo, DeviceDescriptor, Instance, InstanceDescriptor, Limits, Context, defaultContextDeps() (+10 more)

### Community 7 - "TestP4MetalKernels"

Cohesion: 0.13
Nodes (41): BenchmarkP4DeviceResidentChain(), Add(), BroadcastTo(), Mul(), ReduceSumTo(), ReshapeTo(), Scale(), Transp() (+33 more)

### Community 8 - "Decisions"

Cohesion: 0.17
Nodes (11): D-001 Repo-local handover, D-002 Completed P4 plan location, D-003 No readback workaround, D-004 Cause labeling, D-005 Remote authority, D-006 Fix by dependency pin bump, D-007 Fail-fast newest upstream during investigation, D-008 Upstream contribution boundary (+3 more)

### Community 9 - "robustness_test.go"

Cohesion: 0.16
Nodes (27): sentinelWrapError(), newTestContextDeps(), assertUnaryContract(), T, TestAddReturnsKernelValidationError(), TestContextDiagnosticAPI(), TestContextModeString(), TestMatMulCPUReadErrors() (+19 more)

### Community 10 - "go-wgpu-mat"

Cohesion: 0.14
Nodes (13): API, Concurrency, Data layout, Development, go-wgpu-mat, How it works, Installation, License (+5 more)

### Community 11 - "mat_bench_test.go"

Cohesion: 0.37
Nodes (12): B, binaryOperation, BenchmarkAdd256x256(), benchmarkBinaryOperation(), benchmarkData(), BenchmarkMatMul64x64(), benchmarkMatMulOperation(), BenchmarkMul256x256() (+4 more)

### Community 12 - "P4 Device-resident Kernel Plan"

Cohesion: 0.11
Nodes (18): Acceptance Gates, Baseline, Broadcasting, Close-out, Goal, Gradient shape primitives, Internal Design, Non-goals (+10 more)

### Community 14 - "Findings"

Cohesion: 0.18
Nodes (11): Confirmed facts(現行状態), Evidence rules, Findings, Historical baseline(v0.30.22、RED-before evidence), Isolation ladder results(SL-004続、2026-07-31、Kimi Code CLI), Module contract facts(SL-002、2026-07-31、Kimi Code CLI), Questions to resolve, Reproduction and isolation evidence(SL-003/004、2026-07-31、Kimi Code CLI) (+3 more)

### Community 15 - "Submission Lifetime 詳細計画(SL-002 成果物)"

Cohesion: 0.15
Nodes (13): Acceptance gates, Non-goals, Phase 1: SL-003 RED regression test(production変更なし), Phase 2: SL-004 切り分けladder(安い順), Phase 3: SL-005 修正(証拠依存), Phase 4: SL-006 Context.Release drain, Phase 5: SL-007/SL-008 検証, Rollback (+5 more)

### Community 16 - "Current Status"

Cohesion: 0.29
Nodes (7): Active handover, Authority, Current phase, Current Status, Next handover point, Repository, Validation state

### Community 17 - "readBuffer"

Cohesion: 0.14
Nodes (17): MappedRange, defaultReadBufferDeps(), Buffer, BufferDescriptor, CommandBuffer, CommandEncoder, matrixBufferSize(), readBuffer() (+9 more)

### Community 18 - "Repo Agent Notes"

Cohesion: 0.33
Nodes (6): Index, Repo Agent Notes, Session close-out, Session start, 情報の優先順位, 読む順序

### Community 19 - "Work Log"

Cohesion: 0.12
Nodes (15): 2026-07-31 fail-fastでv0.30.30を採用(Kimi Code CLI), 2026-07-31 Handover initialization, 2026-07-31 independent review(Codex), 2026-07-31 review close-out(Codex), 2026-07-31 review remediation(Kimi Code CLI), 2026-07-31 SL-001/SL-002(Kimi Code CLI), 2026-07-31 SL-003〜SL-008(Kimi Code CLI), 2026-07-31 SL-009 reviewとcommit(Kimi Code CLI) (+7 more)

### Community 20 - "Commands"

Cohesion: 0.50
Nodes (4): Commands, Existing repository gates, Index, Resume

### Community 21 - "Tasks"

Cohesion: 0.50
Nodes (4): Acceptance outline, Handover setup, Next implementation candidate: submission lifetime, Tasks

### Community 22 - "add"

Cohesion: 0.31
Nodes (11): add(), defaultAddDeps(), isCPUAdapter(), T, stubContext(), TestAdd_cpuFallback(), TestIsCPUAdapter_realAdapter(), TestIsCPUAdapter_stubContext() (+3 more)

### Community 23 - "contextAdapterOptions"

Cohesion: 0.36
Nodes (6): contextAdapterOptions(), T, TestUseCPU(), TestUseGPU(), UseCPU(), UseGPU()

### Community 27 - "ContextMode"

Cohesion: 0.33
Nodes (5): resolveContextMode(), ContextMode, TestNewContext_internalInvalidMode(), TestNewContext_invalidModes(), TestResolveContextMode()

## Knowledge Gaps

- **91 isolated node(s):** `github.com/KEINOS/go-wgpu-mat`, `bench-isolated.sh script`, `読む順序`, `情報の優先順位`, `Index` (+86 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **4 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions

_Questions this graph is uniquely positioned to answer:_

- **Why does `Matrix` connect `Matrix` to `NewContext`, `matMulWGPUDeps`, `newMockMatrix`, `Context`, `TestP4MetalKernels`, `robustness_test.go`, `readBuffer`, `add`?**
  _High betweenness centrality (0.185) - this node is a cross-community bridge._
- **Why does `Context` connect `Context` to `Matrix`, `NewContext`, `newPipelineCache`, `matMulWGPUDeps`, `TestP4MetalKernels`, `readBuffer`, `add`, `contextAdapterOptions`, `ContextMode`?**
  _High betweenness centrality (0.140) - this node is a cross-community bridge._
- **Why does `NewContext()` connect `NewContext` to `Matrix`, `newPipelineCache`, `Context`, `TestP4MetalKernels`, `robustness_test.go`, `mat_bench_test.go`, `contextAdapterOptions`, `ContextMode`?**
  _High betweenness centrality (0.115) - this node is a cross-community bridge._
- **Are the 59 inferred relationships involving `NewContext()` (e.g. with `sentinelError()` and `sentinelWrapError()`) actually correct?**
  _`NewContext()` has 59 INFERRED edges - model-reasoned connections that need verification._
- **Are the 50 inferred relationships involving `NewMatrix()` (e.g. with `benchmarkBinaryOperation()` and `benchmarkMatMulOperation()`) actually correct?**
  _`NewMatrix()` has 50 INFERRED edges - model-reasoned connections that need verification._
- **Are the 18 inferred relationships involving `newMockMatrix()` (e.g. with `TestAddKernelLimits()` and `TestAdd_cpuFallback()`) actually correct?**
  _`newMockMatrix()` has 18 INFERRED edges - model-reasoned connections that need verification._
- **What connects `github.com/KEINOS/go-wgpu-mat`, `bench-isolated.sh script`, `読む順序` to the rest of the system?**
  _91 weakly-connected nodes found - possible documentation gaps or missing edges._
