//nolint:lll,paralleltest,tparallel,wsl_v5 // Shared injected dependencies make sequential subtests explicit.
package mat

import (
	"io"
	"math"
	"testing"

	"github.com/gogpu/wgpu"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func markHardwareMock(first *Matrix, matrices ...*Matrix) {
	first.ctx.device = new(wgpu.Device)
	first.ctx.infoSet = true
	first.ctx.isCPU = false
	shareMockContext(first, matrices...)
}

func TestTensorOperationsUseInjectedHardwareDispatch(t *testing.T) { //nolint:funlen // Operation contracts are clearest together.
	t.Parallel()

	tests := []struct {
		name      string
		operation tensorOperation
		newArgs   func() (*Matrix, *Matrix, float32)
		call      func(*Matrix, *Matrix, float32, tensorOpExecutionDeps) error
	}{
		{
			name: "mul", operation: tensorOpMul,
			newArgs: func() (*Matrix, *Matrix, float32) {
				left, _ := newMockMatrix(2, 2, make([]float32, 4))
				out, _ := newMockMatrix(2, 2, make([]float32, 4))

				return left, out, 0
			},
			call: func(input, out *Matrix, _ float32, deps tensorOpExecutionDeps) error {
				right, _ := newMockMatrix(1, 2, make([]float32, 2))
				shareMockContext(input, right, out)

				return mulWithDeps(input, right, out, deps)
			},
		},
		{
			name: "scale", operation: tensorOpScale,
			newArgs: sameShapeTensorOpArgs,
			call: func(input, out *Matrix, scalar float32, deps tensorOpExecutionDeps) error {
				return scaleWithDeps(input, scalar, out, deps)
			},
		},
		{
			name: "transpose", operation: tensorOpTranspose,
			newArgs: func() (*Matrix, *Matrix, float32) {
				input, _ := newMockMatrix(2, 3, make([]float32, 6))
				out, _ := newMockMatrix(3, 2, make([]float32, 6))

				return input, out, 0
			},
			call: func(input, out *Matrix, _ float32, deps tensorOpExecutionDeps) error {
				return transpWithDeps(input, out, deps)
			},
		},
		{
			name: "reduce", operation: tensorOpReduceSumTo,
			newArgs: func() (*Matrix, *Matrix, float32) {
				input, _ := newMockMatrix(2, 3, make([]float32, 6))
				out, _ := newMockMatrix(1, 3, make([]float32, 3))

				return input, out, 0
			},
			call: func(input, out *Matrix, _ float32, deps tensorOpExecutionDeps) error {
				return reduceSumToWithDeps(input, out, deps)
			},
		},
		{
			name: "broadcast", operation: tensorOpBroadcastTo,
			newArgs: func() (*Matrix, *Matrix, float32) {
				input, _ := newMockMatrix(1, 3, make([]float32, 3))
				out, _ := newMockMatrix(2, 3, make([]float32, 6))

				return input, out, 0
			},
			call: func(input, out *Matrix, _ float32, deps tensorOpExecutionDeps) error {
				return broadcastToWithDeps(input, out, deps)
			},
		},
		{
			name: "reshape", operation: tensorOpReshapeTo,
			newArgs: func() (*Matrix, *Matrix, float32) {
				input, _ := newMockMatrix(2, 3, make([]float32, 6))
				out, _ := newMockMatrix(3, 2, make([]float32, 6))

				return input, out, 0
			},
			call: func(input, out *Matrix, _ float32, deps tensorOpExecutionDeps) error {
				return reshapeToWithDeps(input, out, deps)
			},
		},
	}

	for _, testCase := range tests {
		t.Run(testCase.name, func(t *testing.T) {
			t.Parallel()

			input, out, scalar := testCase.newArgs()
			markHardwareMock(input, out)
			dispatched := false
			deps := tensorOpExecutionDeps{
				dispatch: func(
					operation tensorOperation,
					_, _, _ *Matrix,
					gotScalar float32,
				) error {
					dispatched = true
					assert.Equal(t, testCase.operation, operation)
					if testCase.operation == tensorOpScale {
						assert.InDelta(t, scalar, gotScalar, 0)
					}

					return io.EOF
				},
			}

			err := testCase.call(input, out, scalar, deps)
			require.ErrorIs(t, err, io.EOF)
			assert.True(t, dispatched)
		})
	}
}

func sameShapeTensorOpArgs() (*Matrix, *Matrix, float32) {
	input, _ := newMockMatrix(2, 2, make([]float32, 4))
	out, _ := newMockMatrix(2, 2, make([]float32, 4))

	return input, out, 2
}

func TestAddBroadcastUsesInjectedDispatch(t *testing.T) {
	t.Parallel()

	left, _ := newMockMatrix(2, 3, make([]float32, 6))
	right, _ := newMockMatrix(1, 3, make([]float32, 3))
	out, _ := newMockMatrix(2, 3, make([]float32, 6))
	markHardwareMock(left, right, out)

	err := add(left, right, out, addDeps{
		dispatch: nil,
		dispatchBroadcast: func(
			operation tensorOperation,
			_, _, _ *Matrix,
			_ float32,
		) error {
			assert.Equal(t, tensorOpAdd, operation)

			return io.EOF
		},
	})
	require.ErrorIs(t, err, io.EOF)
}

func TestTensorOperationCPUTransferErrors(t *testing.T) {
	t.Parallel()

	t.Run("mul right read", func(t *testing.T) {
		left, _ := newMockMatrix(1, 1, []float32{1})
		right, rightIO := newMockMatrix(1, 1, []float32{2})
		out, _ := newMockMatrix(1, 1, []float32{0})
		shareMockContext(left, right, out)
		rightIO.readErr = io.EOF
		require.ErrorContains(t, mul(left, right, out), "failed to read right")
	})

	t.Run("mul write", func(t *testing.T) {
		left, _ := newMockMatrix(1, 1, []float32{1})
		right, _ := newMockMatrix(1, 1, []float32{2})
		out, outIO := newMockMatrix(1, 1, []float32{0})
		shareMockContext(left, right, out)
		outIO.writeErr = io.EOF
		require.ErrorContains(t, mul(left, right, out), "failed to write out")
	})

	tests := []struct {
		name string
		call func(*Matrix, *Matrix) error
		rows int
		cols int
		outR int
		outC int
	}{
		{name: "reduce", call: reduceSumTo, rows: 2, cols: 2, outR: 1, outC: 2},
		{name: "broadcast", call: broadcastTo, rows: 1, cols: 2, outR: 2, outC: 2},
		{name: "reshape", call: reshapeTo, rows: 1, cols: 4, outR: 2, outC: 2},
	}

	for _, testCase := range tests {
		t.Run(testCase.name+" read", func(t *testing.T) {
			input, inputIO := newMockMatrix(testCase.rows, testCase.cols, make([]float32, testCase.rows*testCase.cols))
			out, _ := newMockMatrix(testCase.outR, testCase.outC, make([]float32, testCase.outR*testCase.outC))
			shareMockContext(input, out)
			inputIO.readErr = io.EOF
			require.ErrorContains(t, testCase.call(input, out), "failed to read input")
		})

		t.Run(testCase.name+" write", func(t *testing.T) {
			input, _ := newMockMatrix(testCase.rows, testCase.cols, make([]float32, testCase.rows*testCase.cols))
			out, outIO := newMockMatrix(testCase.outR, testCase.outC, make([]float32, testCase.outR*testCase.outC))
			shareMockContext(input, out)
			outIO.writeErr = io.EOF
			require.ErrorContains(t, testCase.call(input, out), "failed to write out")
		})
	}
}

func TestDispatchTensorOperationSuccessAndStats(t *testing.T) {
	t.Parallel()

	left, right, out := matMulTestMatrices()
	markHardwareMock(left, right, out)
	deps := successfulMatMulWGPUDeps()

	err := dispatchTensorOperationWithDeps(tensorOpMul, left, right, out, 0, deps)
	require.NoError(t, err)
	assert.Equal(t, Stats{
		HostReadCount:           0,
		HostReadBytes:           0,
		HostWriteCount:          0,
		HostWriteBytes:          0,
		ComputeSubmissionCount:  1,
		ReadbackSubmissionCount: 0,
		MatrixAllocationCount:   0,
		MatrixReleaseCount:      0,
		LiveMatrixBytes:         0,
		PeakLiveMatrixBytes:     0,
	}, left.ctx.Stats())
}

func TestTensorOperationKernelLimits(t *testing.T) {
	t.Parallel()

	out, _ := newMockMatrix(0, 1, nil)
	require.ErrorIs(t, validateTensorOpKernelContract(out), ErrKernelLimit)

	out.rows = math.MaxUint32
	out.cols = 2
	require.ErrorIs(t, validateTensorOpKernelContract(out), ErrKernelLimit)

	out.rows = 257
	out.cols = 1
	out.ctx.limits.MaxComputeWorkgroupsPerDimension = 1
	require.ErrorIs(t, validateTensorOpKernelContract(out), ErrDeviceLimit)
}

func TestContextStatsNil(t *testing.T) {
	t.Parallel()

	var ctx *Context
	assert.Equal(t, Stats{
		HostReadCount:           0,
		HostReadBytes:           0,
		HostWriteCount:          0,
		HostWriteBytes:          0,
		ComputeSubmissionCount:  0,
		ReadbackSubmissionCount: 0,
		MatrixAllocationCount:   0,
		MatrixReleaseCount:      0,
		LiveMatrixBytes:         0,
		PeakLiveMatrixBytes:     0,
	}, ctx.Stats())
	ctx.recordMatrixAllocation(4)
	ctx.recordMatrixRelease(4)
	ctx.recordHostRead(4)
	ctx.recordHostWrite(4)
	ctx.recordComputeSubmission()
	ctx.recordReadbackSubmission()
}
