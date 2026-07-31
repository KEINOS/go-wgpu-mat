# Graph Report - go-wgpu-mat  (2026-07-31)

## Corpus Check

- 43 files · ~30,005 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary

- 558 nodes · 1572 edges · 27 communities (22 shown, 5 thin omitted)
- Extraction: 71% EXTRACTED · 29% INFERRED · 0% AMBIGUOUS · INFERRED: 450 edges (avg confidence: 0.8)
- Token cost: 0 input · 0 output

## Graph Freshness

- Built from commit: `99490882`
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
- ops.go
- successfulMatMulWGPUDeps
- robustness_test.go
- go-wgpu-mat
- mat_bench_test.go
- P4 Device-resident Kernel Plan
- .agents/README.md
- Decisions
- Current Status
- Findings
- Public Contract
- Phases
- Tasks
- Commands
- Repo Agent Notes
- worklog.md
- classifiedError
- bench-isolated.sh
- github.com/KEINOS/go-wgpu-mat

## God Nodes (most connected - your core abstractions)

1. `Matrix` - 93 edges
2. `NewContext()` - 61 edges
3. `NewMatrix()` - 55 edges
4. `Context` - 50 edges
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

## Communities (27 total, 5 thin omitted)

### Community 0 - "Matrix"

Cohesion: 0.08
Nodes (72): createAddBindGroup(), createAddBindGroupLayout(), createAddPipeline(), defaultAddDeps(), dispatchAdd(), dispatchAddWithDeps(), encodeAndSubmitAdd(), BindGroup (+64 more)

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

Cohesion: 0.08
Nodes (44): BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor, BindGroupLayoutEntry, BufferBindingType, CommandEncoderDescriptor, ComputePassDescriptor, ComputePassEncoder (+36 more)

### Community 5 - "newMockMatrix"

Cohesion: 0.13
Nodes (43): add(), mockMatrixIO, runBinaryElementwise(), decodeFloat32(), encodeFloat32(), T, newMockMatrix(), shareMockContext() (+35 more)

### Community 6 - "Context"

Cohesion: 0.08
Nodes (26): Adapter, AdapterInfo, DeviceDescriptor, Instance, InstanceDescriptor, Limits, Context, contextAdapterOptions() (+18 more)

### Community 7 - "ops.go"

Cohesion: 0.17
Nodes (31): BenchmarkP4DeviceResidentChain(), Add(), applyFiniteSoftmaxRow(), applyRMSNormRow(), applySoftmaxRow(), applySpecialSoftmaxRow(), broadcastDimension(), BroadcastTo() (+23 more)

### Community 8 - "successfulMatMulWGPUDeps"

Cohesion: 0.16
Nodes (24): T, TestAddKernelLimits(), TestCreateAddPipelineErrors(), TestDispatchAddWGPUStages(), TestEncodeAndSubmitAddErrors(), T, matMulTestMatrices(), successfulMatMulWGPUDeps() (+16 more)

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

Cohesion: 0.25
Nodes (8): Acceptance Gates, Baseline, Close-out, Goal, Internal Design, Non-goals, P4 Device-resident Kernel Plan, Plan Review

### Community 14 - "Decisions"

Cohesion: 0.29
Nodes (6): D-001 Repo-local handover, D-002 Completed P4 plan location, D-003 No readback workaround, D-004 Cause labeling, D-005 Remote authority, Decisions

### Community 15 - "Current Status"

Cohesion: 0.33
Nodes (6): Authority, Current phase, Current Status, Next handover point, Repository, Validation state

### Community 16 - "Findings"

Cohesion: 0.40
Nodes (5): Confirmed facts, Evidence rules, Findings, Questions to resolve, Working hypothesis

### Community 17 - "Public Contract"

Cohesion: 0.40
Nodes (5): Broadcasting, Gradient shape primitives, Public Contract, Statistics, Unary kernels

### Community 18 - "Phases"

Cohesion: 0.40
Nodes (5): P4.0 Plan and review, P4.1 RED contract tests, P4.2 GREEN kernels and statistics, P4.3 Documentation and validation, Phases

### Community 19 - "Tasks"

Cohesion: 0.40
Nodes (4): Acceptance outline, Handover setup, Next implementation candidate: submission lifetime, Tasks

### Community 20 - "Commands"

Cohesion: 0.50
Nodes (4): Commands, Existing repository gates, Index, Resume

### Community 21 - "Repo Agent Notes"

Cohesion: 0.50
Nodes (4): Index, Repo Agent Notes, 情報の優先順位, 読む順序

## Knowledge Gaps

- **53 isolated node(s):** `github.com/KEINOS/go-wgpu-mat`, `bench-isolated.sh script`, `読む順序`, `情報の優先順位`, `Index` (+48 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **5 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions

_Questions this graph is uniquely positioned to answer:_

- **Why does `Matrix` connect `Matrix` to `mat_unit_test.go`, `NewContext`, `matMulWGPUDeps`, `newMockMatrix`, `Context`, `ops.go`, `successfulMatMulWGPUDeps`, `robustness_test.go`?**
  _High betweenness centrality (0.201) - this node is a cross-community bridge._
- **Why does `Context` connect `Context` to `Matrix`, `mat_unit_test.go`, `NewContext`, `newPipelineCache`, `matMulWGPUDeps`, `ops.go`?**
  _High betweenness centrality (0.160) - this node is a cross-community bridge._
- **Why does `NewContext()` connect `NewContext` to `Matrix`, `mat_unit_test.go`, `newPipelineCache`, `Context`, `ops.go`, `robustness_test.go`, `mat_bench_test.go`?**
  _High betweenness centrality (0.117) - this node is a cross-community bridge._
- **Are the 53 inferred relationships involving `NewContext()` (e.g. with `sentinelError()` and `sentinelWrapError()`) actually correct?**
  _`NewContext()` has 53 INFERRED edges - model-reasoned connections that need verification._
- **Are the 47 inferred relationships involving `NewMatrix()` (e.g. with `benchmarkBinaryOperation()` and `benchmarkMatMulOperation()`) actually correct?**
  _`NewMatrix()` has 47 INFERRED edges - model-reasoned connections that need verification._
- **Are the 18 inferred relationships involving `newMockMatrix()` (e.g. with `TestAddKernelLimits()` and `TestAdd_cpuFallback()`) actually correct?**
  _`newMockMatrix()` has 18 INFERRED edges - model-reasoned connections that need verification._
- **What connects `github.com/KEINOS/go-wgpu-mat`, `bench-isolated.sh script`, `読む順序` to the rest of the system?**
  _53 weakly-connected nodes found - possible documentation gaps or missing edges._
