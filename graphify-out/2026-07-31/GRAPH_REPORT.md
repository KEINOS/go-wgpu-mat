# Graph Report - go-wgpu-mat  (2026-07-31)

## Corpus Check

- 43 files · ~30,221 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary

- 561 nodes · 1575 edges · 17 communities (15 shown, 2 thin omitted)
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
- TestP4MetalKernels
- successfulMatMulWGPUDeps
- robustness_test.go
- go-wgpu-mat
- mat_bench_test.go
- P4 Device-resident Kernel Plan
- .agents/README.md
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

- `Add()` --calls--> `defaultAddDeps()`  [INFERRED]
  mat/ops.go → mat/add.go
- `add()` --calls--> `wrapError()`  [INFERRED]
  mat/add.go → mat/errors.go
- `add()` --calls--> `runBinaryBroadcast()`  [INFERRED]
  mat/add.go → mat/ops.go
- `TestAdd_cpuFallback()` --calls--> `add()`  [INFERRED]
  mat/add_stub_test.go → mat/add.go
- `Add()` --calls--> `add()`  [INFERRED]
  mat/ops.go → mat/add.go

## Import Cycles

- None detected.

## Communities (17 total, 2 thin omitted)

### Community 0 - "Matrix"

Cohesion: 0.07
Nodes (71): validateAdd(), validateAddKernelContract(), addDeps, classifiedError, sentinelError(), wrapError(), Uint32, ceilDiv() (+63 more)

### Community 1 - "mat_unit_test.go"

Cohesion: 0.06
Nodes (52): MappedRange, defaultMatrixDeps(), defaultReadBufferDeps(), Buffer, BufferDescriptor, CommandBuffer, CommandEncoder, matrixBufferSize() (+44 more)

### Community 2 - "NewContext"

Cohesion: 0.12
Nodes (53): F, NewContext(), Example(), ExampleAdd(), ExampleMatMul(), ExampleNewMatrix(), ExampleReduceMax(), ExampleReduceSum() (+45 more)

### Community 3 - "newPipelineCache"

Cohesion: 0.07
Nodes (36): defaultAddDeps(), isCPUAdapter(), T, stubContext(), TestAdd_cpuFallback(), TestIsCPUAdapter_realAdapter(), TestIsCPUAdapter_stubContext(), ComputePipeline (+28 more)

### Community 4 - "matMulWGPUDeps"

Cohesion: 0.07
Nodes (57): BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor, BindGroupLayoutEntry, BufferBindingType, CommandEncoderDescriptor, ComputePassDescriptor, ComputePassEncoder (+49 more)

### Community 5 - "newMockMatrix"

Cohesion: 0.13
Nodes (43): add(), mockMatrixIO, runBinaryElementwise(), decodeFloat32(), encodeFloat32(), T, newMockMatrix(), shareMockContext() (+35 more)

### Community 6 - "Context"

Cohesion: 0.08
Nodes (26): Adapter, AdapterInfo, DeviceDescriptor, Instance, InstanceDescriptor, Limits, Context, contextAdapterOptions() (+18 more)

### Community 7 - "TestP4MetalKernels"

Cohesion: 0.32
Nodes (19): BenchmarkP4DeviceResidentChain(), Add(), BroadcastTo(), Mul(), ReduceSumTo(), ReshapeTo(), Scale(), Transp() (+11 more)

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

Cohesion: 0.11
Nodes (18): Acceptance Gates, Baseline, Broadcasting, Close-out, Goal, Gradient shape primitives, Internal Design, Non-goals (+10 more)

### Community 13 - ".agents/README.md"

Cohesion: 0.05
Nodes (35): Commands, Existing repository gates, Index, Resume, D-001 Repo-local handover, D-002 Completed P4 plan location, D-003 No readback workaround, D-004 Cause labeling (+27 more)

## Knowledge Gaps

- **56 isolated node(s):** `github.com/KEINOS/go-wgpu-mat`, `bench-isolated.sh script`, `読む順序`, `情報の優先順位`, `Index` (+51 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **2 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions

_Questions this graph is uniquely positioned to answer:_

- **Why does `Matrix` connect `Matrix` to `mat_unit_test.go`, `NewContext`, `matMulWGPUDeps`, `newMockMatrix`, `Context`, `TestP4MetalKernels`, `successfulMatMulWGPUDeps`, `robustness_test.go`?**
  _High betweenness centrality (0.199) - this node is a cross-community bridge._
- **Why does `Context` connect `Context` to `Matrix`, `mat_unit_test.go`, `NewContext`, `newPipelineCache`, `matMulWGPUDeps`, `TestP4MetalKernels`?**
  _High betweenness centrality (0.158) - this node is a cross-community bridge._
- **Why does `NewContext()` connect `NewContext` to `Matrix`, `mat_unit_test.go`, `newPipelineCache`, `Context`, `TestP4MetalKernels`, `robustness_test.go`, `mat_bench_test.go`?**
  _High betweenness centrality (0.116) - this node is a cross-community bridge._
- **Are the 53 inferred relationships involving `NewContext()` (e.g. with `sentinelError()` and `sentinelWrapError()`) actually correct?**
  _`NewContext()` has 53 INFERRED edges - model-reasoned connections that need verification._
- **Are the 47 inferred relationships involving `NewMatrix()` (e.g. with `benchmarkBinaryOperation()` and `benchmarkMatMulOperation()`) actually correct?**
  _`NewMatrix()` has 47 INFERRED edges - model-reasoned connections that need verification._
- **Are the 18 inferred relationships involving `newMockMatrix()` (e.g. with `TestAddKernelLimits()` and `TestAdd_cpuFallback()`) actually correct?**
  _`newMockMatrix()` has 18 INFERRED edges - model-reasoned connections that need verification._
- **What connects `github.com/KEINOS/go-wgpu-mat`, `bench-isolated.sh script`, `読む順序` to the rest of the system?**
  _56 weakly-connected nodes found - possible documentation gaps or missing edges._
