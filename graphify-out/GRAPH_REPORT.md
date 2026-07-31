# Graph Report - go-wgpu-mat  (2026-07-31)

## Corpus Check

- 45 files · ~31,942 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary

- 591 nodes · 1639 edges · 25 communities (22 shown, 3 thin omitted)
- Extraction: 72% EXTRACTED · 28% INFERRED · 0% AMBIGUOUS · INFERRED: 465 edges (avg confidence: 0.8)
- Token cost: 0 input · 0 output

## Graph Freshness

- Built from commit: `ba3ea83b`
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
- 実行計画
- Repo Agent Notes
- Work Log
- Commands
- Tasks
- bench-isolated.sh
- github.com/KEINOS/go-wgpu-mat

## God Nodes (most connected - your core abstractions)

1. `Matrix` - 94 edges
2. `NewContext()` - 64 edges
3. `NewMatrix()` - 56 edges
4. `Context` - 51 edges
5. `newMockMatrix()` - 48 edges
6. `sentinelError()` - 42 edges
7. `matMulWGPUDeps` - 42 edges
8. `wrapError()` - 36 edges
9. `serializeGPUTest()` - 27 edges
10. `shareMockContext()` - 23 edges

## Surprising Connections (you probably didn't know these)

- `TestAdd_cpuFallback()` --calls--> `defaultAddDeps()`  [INFERRED]
  mat/add_stub_test.go → mat/add.go
- `Add()` --calls--> `defaultAddDeps()`  [INFERRED]
  mat/ops.go → mat/add.go
- `add()` --calls--> `wrapError()`  [INFERRED]
  mat/add.go → mat/errors.go
- `add()` --calls--> `runBinaryBroadcast()`  [INFERRED]
  mat/add.go → mat/ops.go
- `TestAdd_cpuFallback()` --calls--> `add()`  [INFERRED]
  mat/add_stub_test.go → mat/add.go

## Import Cycles

- None detected.

## Communities (25 total, 3 thin omitted)

### Community 0 - "Matrix"

Cohesion: 0.08
Nodes (70): validateAdd(), classifiedError, sentinelError(), wrapError(), Uint32, matMul(), matMulCPU(), validateMatMul() (+62 more)

### Community 1 - "mat_unit_test.go"

Cohesion: 0.06
Nodes (52): MappedRange, defaultMatrixDeps(), defaultReadBufferDeps(), Buffer, BufferDescriptor, CommandBuffer, CommandEncoder, matrixBufferSize() (+44 more)

### Community 2 - "NewContext"

Cohesion: 0.12
Nodes (53): F, NewContext(), Example(), ExampleAdd(), ExampleMatMul(), ExampleNewMatrix(), ExampleReduceMax(), ExampleReduceSum() (+45 more)

### Community 3 - "newPipelineCache"

Cohesion: 0.07
Nodes (35): isCPUAdapter(), T, stubContext(), TestAdd_cpuFallback(), TestIsCPUAdapter_realAdapter(), TestIsCPUAdapter_stubContext(), ComputePipeline, newError() (+27 more)

### Community 4 - "matMulWGPUDeps"

Cohesion: 0.05
Nodes (84): BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor, BindGroupLayoutEntry, BufferBindingType, CommandEncoderDescriptor, ComputePassDescriptor, ComputePassEncoder (+76 more)

### Community 5 - "newMockMatrix"

Cohesion: 0.13
Nodes (42): add(), mockMatrixIO, decodeFloat32(), encodeFloat32(), T, newMockMatrix(), shareMockContext(), TestAddDispatchError() (+34 more)

### Community 6 - "Context"

Cohesion: 0.08
Nodes (26): Adapter, AdapterInfo, DeviceDescriptor, Instance, InstanceDescriptor, Limits, Context, contextAdapterOptions() (+18 more)

### Community 7 - "TestP4MetalKernels"

Cohesion: 0.23
Nodes (27): BenchmarkP4DeviceResidentChain(), Add(), BroadcastTo(), Mul(), ReduceSumTo(), ReshapeTo(), Scale(), Transp() (+19 more)

### Community 8 - "Decisions"

Cohesion: 0.25
Nodes (7): D-001 Repo-local handover, D-002 Completed P4 plan location, D-003 No readback workaround, D-004 Cause labeling, D-005 Remote authority, D-006 Fix by dependency pin bump, Decisions

### Community 9 - "robustness_test.go"

Cohesion: 0.17
Nodes (25): sentinelWrapError(), newTestContextDeps(), assertUnaryContract(), T, TestContextDiagnosticAPI(), TestContextModeString(), TestMatMulCPUReadErrors(), TestMatrixDiagnosticAPI() (+17 more)

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

Cohesion: 0.25
Nodes (8): Confirmed facts, Evidence rules, Findings, Module contract facts(SL-002、2026-07-31、Kimi Code CLI), Questions to resolve, Reproduction and isolation evidence(SL-003/004、2026-07-31、Kimi Code CLI), SL-001 inventory(2026-07-31、Kimi Code CLI), Working hypothesis

### Community 15 - "Submission Lifetime 詳細計画(SL-002 成果物)"

Cohesion: 0.29
Nodes (7): Acceptance gates, Non-goals, Rollback, Submission Lifetime 詳細計画(SL-002 成果物), Test selectors, 仮説の状態, 前提: 確認済みcontract事実

### Community 16 - "Current Status"

Cohesion: 0.29
Nodes (7): Active handover, Authority, Current phase, Current Status, Next handover point, Repository, Validation state

### Community 17 - "実行計画"

Cohesion: 0.33
Nodes (6): Phase 1: SL-003 RED regression test(production変更なし), Phase 2: SL-004 切り分けladder(安い順), Phase 3: SL-005 修正(証拠依存), Phase 4: SL-006 Context.Release drain, Phase 5: SL-007/SL-008 検証, 実行計画

### Community 18 - "Repo Agent Notes"

Cohesion: 0.33
Nodes (6): Index, Repo Agent Notes, Session close-out, Session start, 情報の優先順位, 読む順序

### Community 19 - "Work Log"

Cohesion: 0.33
Nodes (5): 2026-07-31 Handover initialization, 2026-07-31 SL-001/SL-002(Kimi Code CLI), 2026-07-31 SL-003〜SL-008(Kimi Code CLI), 2026-07-31 SL-009 reviewとcommit(Kimi Code CLI), Work Log

### Community 20 - "Commands"

Cohesion: 0.50
Nodes (4): Commands, Existing repository gates, Index, Resume

### Community 21 - "Tasks"

Cohesion: 0.50
Nodes (4): Acceptance outline, Handover setup, Next implementation candidate: submission lifetime, Tasks

## Knowledge Gaps

- **74 isolated node(s):** `github.com/KEINOS/go-wgpu-mat`, `bench-isolated.sh script`, `読む順序`, `情報の優先順位`, `Index` (+69 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **3 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions

_Questions this graph is uniquely positioned to answer:_

- **Why does `Matrix` connect `Matrix` to `mat_unit_test.go`, `NewContext`, `matMulWGPUDeps`, `newMockMatrix`, `Context`, `TestP4MetalKernels`, `robustness_test.go`?**
  _High betweenness centrality (0.185) - this node is a cross-community bridge._
- **Why does `Context` connect `Context` to `Matrix`, `mat_unit_test.go`, `NewContext`, `newPipelineCache`, `matMulWGPUDeps`, `TestP4MetalKernels`?**
  _High betweenness centrality (0.147) - this node is a cross-community bridge._
- **Why does `NewContext()` connect `NewContext` to `Matrix`, `mat_unit_test.go`, `newPipelineCache`, `Context`, `TestP4MetalKernels`, `robustness_test.go`, `mat_bench_test.go`?**
  _High betweenness centrality (0.114) - this node is a cross-community bridge._
- **Are the 56 inferred relationships involving `NewContext()` (e.g. with `sentinelError()` and `sentinelWrapError()`) actually correct?**
  _`NewContext()` has 56 INFERRED edges - model-reasoned connections that need verification._
- **Are the 48 inferred relationships involving `NewMatrix()` (e.g. with `benchmarkBinaryOperation()` and `benchmarkMatMulOperation()`) actually correct?**
  _`NewMatrix()` has 48 INFERRED edges - model-reasoned connections that need verification._
- **Are the 18 inferred relationships involving `newMockMatrix()` (e.g. with `TestAddKernelLimits()` and `TestAdd_cpuFallback()`) actually correct?**
  _`newMockMatrix()` has 18 INFERRED edges - model-reasoned connections that need verification._
- **What connects `github.com/KEINOS/go-wgpu-mat`, `bench-isolated.sh script`, `読む順序` to the rest of the system?**
  _74 weakly-connected nodes found - possible documentation gaps or missing edges._
