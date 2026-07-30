# P4 Device-resident Kernel Plan

## Goal

Provide the minimum 2D `float32` device-resident operations and observable
transfer statistics required by `go-nn` tensor autograd, without adding
autograd or tensor ownership to this repository.

This phase ends after the kernels and their Metal validation are complete.
`go-nn` integration, its submodule update, tags, and pushes are separate
Maintainer-controlled steps.

## Baseline

- Baseline commit: `99a7369`.
- `MatMul` and same-shape `Add` already use WGSL on a hardware adapter.
- `Scale`, `Transp`, and `ReduceSum` use host read/compute/write compatibility
  paths.
- There is no elementwise multiply operation, general 2D broadcast operation,
  broadcast-gradient reduction, device-resident reshape copy, or public
  host-transfer measurement.
- `CGO_ENABLED=0 make test` packages pass with 100% statement coverage.
- Before P4 changes, the full CGO-enabled race suite can fail inside upstream
  Metal FFI while draining a queue after compatibility-path readback. Targeted
  non-race Metal `MatMul` and `Add` tests pass serially. P4 therefore reports
  the full race result and a separate mandatory serialized Metal kernel gate.

## Public Contract

### Broadcasting

`Add` and the new `Mul` accept 2D NumPy-style broadcasting. For each axis,
input dimensions must either equal the output dimension or be `1`. The output
shape must be the axis-wise maximum. Existing same-shape calls remain source
and behavior compatible.

The software adapter executes an equivalent pure-Go implementation. A hardware
adapter dispatches WGSL and performs no implicit `Matrix.Read` or
`Matrix.Write`.

### Unary kernels

`Scale` keeps its existing same-shape contract and becomes a hardware WGSL
kernel. `Transp` keeps its existing transposed-output contract and becomes a
hardware WGSL kernel.

### Gradient shape primitives

`ReduceSumTo(input, out)` reduces a 2D input into an output whose axes are
either equal to the corresponding input axis or `1`. At least one axis must be
reduced unless the shapes are equal. Equal shapes are permitted and copy the
input, which keeps executor code uniform.

`BroadcastTo(input, out)` expands an input whose axes are either equal to the
corresponding output axis or `1`. Equal shapes are permitted and copy the
input.

`ReshapeTo(input, out)` copies device data in row-major order between matrices
with equal element counts. It permits shape changes such as `2x3` to `3x2`
without a host transfer. The matrices keep distinct ownership and buffers;
P4 does not introduce aliasing views.

Existing `ReduceSum(input, out)` remains row-wise (`rows x cols` to
`rows x 1`) and delegates to the same implementation. `ReduceMax`, `Softmax`,
and `RMSNorm` remain compatibility-path operations in P4 because tensor
autograd does not require them yet.

All operations preserve the existing same-context, initialized-resource,
non-aliased-output, dimension, device-limit, and sentinel-error contracts.
Unsupported shapes return `ErrDimensionMismatch`; hardware kernel limits return
`ErrKernelLimit` or `ErrDeviceLimit`. There is no implicit hardware-to-CPU
fallback after a hardware adapter has been selected.

### Statistics

`Context.Stats()` returns an immutable snapshot with cumulative counters:

- `HostReads`
- `HostWrites`
- `CommandSubmissions`
- `BufferAllocations`
- `LiveBuffers`
- `PeakLiveBuffers`

Counters are concurrency-safe. Tests compare two snapshots instead of resetting
global state. `HostReads` and `HostWrites` count completed matrix-payload
transfers initiated by public `Matrix.Read` and `Matrix.Write`; internal uniform
metadata uploads are deliberately excluded. Submission and allocation counters
increment only after successful backend operations. Live-buffer accounting
includes matrix, staging, and temporary uniform buffers created by this package
and decrements exactly once on release. A temporary uniform allocation therefore
changes buffer counters but not `HostWrites`.

The statistics are diagnostic, not a synchronization primitive. They do not
change the existing rule that callers must not release a context or matrix
concurrently with operations that use it.

## Internal Design

1. Add small operation-specific dispatch seams following the existing `Add`
   and `MatMul` dependency-injection pattern.
2. Reuse common compute pipeline, bind-group, encoder, submission, and release
   helpers; keep WGSL and uniform layout next to each operation contract.
3. Use an operation uniform for source/output dimensions and scalar values.
   Dispatch elementwise and broadcast operations over output elements.
4. Implement `ReduceSumTo` with one invocation per output element. Each
   invocation loops only over axes reduced to `1`; this favors correctness and
   deterministic accumulation over premature reduction optimization.
5. Implement `ReshapeTo` as a device-to-device copy command or an equivalent
   elementwise copy kernel. It must not map either matrix to the host.
6. Keep software-adapter paths pure Go and shape-identical to the hardware
   result. Do not call public `Read`/`Write` from hardware paths.
7. Add private context accounting helpers around every package-owned buffer and
   queue submission. Injected unit-test dependencies must be able to verify
   success, failure, and exact-once release behavior without hardware.
8. Discard unfinished command encoders on every failure before `Finish`, and
   release every non-nil partial resource returned together with an error.

## Phases

### P4.0 Plan and review

- Inventory existing kernels, compatibility paths, resource lifecycle, limits,
  and validation behavior.
- Record baseline software coverage and the pre-existing Metal race failure.
- Obtain read-only review with no unresolved blocking or major finding.
- Commit this reviewed plan before production code changes.

### P4.1 RED contract tests

- Add table-driven validation and CPU-reference tests for all broadcast shapes:
  same shape, row, column, scalar, and invalid dimensions.
- Add no-host-I/O dispatch tests for `Mul`, `Scale`, `Transp`, `ReduceSumTo`,
  `BroadcastTo`, and `ReshapeTo`.
- Add injected failure tests for every resource-creation and submission stage,
  including `(nil, nil)` and `(non-nil, error)` backend returns.
- Assert exact-once release/accounting for every partial resource and
  `CommandEncoder.DiscardEncoding()` on failures before `Finish`.
- Add statistics tests for success/failure counting, snapshots, peak/live
  accounting, idempotent release, concurrent independent contexts, and the
  exclusion of internal uniform metadata from `HostWrites`.
- Run the focused tests and record that they fail because production symbols or
  dispatch behavior are absent.

### P4.2 GREEN kernels and statistics

- Implement the minimum WGSL kernels and software references.
- Implement context statistics and exact resource accounting.
- Make focused tests pass, then refactor shared helpers without broadening API
  scope.
- Preserve 100% statement coverage for production packages in software mode.

### P4.3 Documentation and validation

- Update package GoDoc, README API/behavior tables, examples, and benchmarks.
- Run an independent read-only implementation review and resolve verified
  blocking findings.
- Run build, lint, software tests, race tests, fuzz smoke, and benchmarks.
- Run serialized `UseGPU` Metal tests for every P4 kernel and assert zero host
  transfers between uploaded inputs and explicit final readback.
- Report the pre-existing full-suite Metal FFI failure separately if it remains;
  it must not be hidden by skipping the mandatory targeted Metal gate.

## Acceptance Gates

The P4 implementation is complete only when all of the following hold:

- Existing API and sentinel behavior remain compatible.
- Hardware `MatMul`, `Add`, `Mul`, `Scale`, `Transp`, `ReduceSumTo`, and
  `BroadcastTo` execute without implicit host reads or writes.
- `ReshapeTo` performs an equal-element-count device copy without host I/O.
- Software and Metal results match a pure-Go reference for same-shape and all
  supported 2D broadcast/reduction forms within a documented `float32`
  tolerance.
- A representative device-resident operation chain has zero intermediate host
  reads/writes and one explicit final read.
- Statistics correctly report host transfers, submissions, allocations, live
  buffers, and peak live buffers under success and injected failures.
- Production statement coverage remains 100% in the software-test profile.
- `go build ./...`, both CGO build modes, `go test` software gates,
  `golangci-lint`, Markdown lint, examples, bounded fuzz smoke, and the targeted
  serialized Metal gate pass.
- Final read-only review has no unresolved blocking finding.
- No `go-nn` checkout, submodule pointer, remote branch, tag, or upstream state
  is changed.

The mandatory P4 selectors are:

```sh
CGO_ENABLED=1 go test -race -cover -count=1 \
  -run '^TestP4Software' ./mat
GO_WGPU_MAT_GPU=1 CGO_ENABLED=1 go test -count=1 -parallel=1 \
  -run '^TestP4Metal' ./mat
```

`TestP4Software*` creates only `UseCPU` contexts or injected test doubles and
must not request a hardware adapter. `TestP4Metal*` requires `UseGPU`; when
`GO_WGPU_MAT_GPU=1` is present, adapter unavailability is a failure rather than
a skip. The full `make test` result is reported separately so an upstream Metal
FFI crash cannot be mistaken for either selector's result.

## Plan Review

- Review round 1 found one blocking and three major omissions: device-resident
  reshape, payload-versus-uniform transfer semantics, partial resource cleanup,
  and deterministic software/Metal selectors.
- All four findings were accepted and incorporated above.
- Final read-only rereview result: `AGREED: no blocking findings.`

## Non-goals

- `go-nn` integration or autograd implementation.
- N-dimensional tensors or batched matrix ownership in this repository.
- Kernelizing `ReduceMax`, `Softmax`, or `RMSNorm`.
- Mixed precision, shader micro-optimization, command fusion, or buffer pooling.
- Pushes, tags, or releases.

## Close-out

- Both `CGO_ENABLED=0` and `CGO_ENABLED=1` builds pass.
- Full `make test` passes with race detection and 100% statement coverage in
  both CGO modes. The baseline Metal FFI failure did not reproduce after P4.
- The mandatory software race selector and serialized Metal selector pass.
- Go lint and Markdown lint report zero issues.
- Both fuzz targets pass their 10-second smoke runs.
- `BenchmarkMul256x256` and `BenchmarkP4DeviceResidentChain` execute
  successfully on local Metal.
- Final Hermes review result: `AGREED: no blocking findings.` An additional
  medium observation about `source_index` was rejected after verification:
  its `cols` argument is the source stride, and the reviewer's counterexamples
  were invalid broadcast shapes. Valid row and column broadcasts are covered
  by the serialized Metal test. Earlier unavailable reviewer attempts changed
  no files.
- No `go-nn` checkout, submodule pointer, remote branch, tag, or upstream state
  was changed.
