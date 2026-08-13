package mat

import (
	"testing"

	"github.com/stretchr/testify/require"
)

// TestDefaultDeviceDispatchWithSoftwareAdapter exercises the real WGPU
// dependency wiring without requiring a hardware adapter. The context still
// owns a software device; only the package's host-compatibility selector is
// disabled so the operations take their device dispatch paths.
// Numeric device results are checked by the Metal parity tests; the software
// adapter is used here only to verify that every default dispatch is accepted.
func TestDefaultDeviceDispatchWithSoftwareAdapter(t *testing.T) {
	t.Parallel()

	ctx, err := NewContext(UseCPU)
	require.NoError(t, err)
	t.Cleanup(ctx.Release)

	ctx.isCPU = false
	require.False(t, useHostCompatibility(ctx))

	testDefaultAddAndMatMul(t, ctx)
	testDefaultTensorDispatch(t, ctx)
	testDefaultOptimizerDispatch(t, ctx)
}

func testDefaultAddAndMatMul(t *testing.T, ctx *Context) {
	t.Helper()

	left := newDeviceTestMatrix(t, ctx, 1, 2, []float32{2, 3})
	right := newDeviceTestMatrix(t, ctx, 1, 2, []float32{4, 5})
	out := newDeviceTestMatrix(t, ctx, 1, 2, []float32{0, 0})
	requireDeviceDispatch(t, ctx, func() error { return Add(left, right, out) })

	matMulRight := newDeviceTestMatrix(t, ctx, 2, 1, []float32{4, 5})
	matMulOut := newDeviceTestMatrix(t, ctx, 1, 1, []float32{0})
	requireDeviceDispatch(t, ctx, func() error { return MatMul(left, matMulRight, matMulOut) })
}

func testDefaultTensorDispatch(t *testing.T, ctx *Context) {
	t.Helper()

	input := newDeviceTestMatrix(t, ctx, 1, 2, []float32{2, 3})
	out := newDeviceTestMatrix(t, ctx, 1, 2, []float32{0, 0})
	requireDeviceDispatch(t, ctx, func() error { return Scale(input, 2, out) })

	state := RandomState{Seed: 0, StreamID: 0, Counter: 0}

	requireDeviceDispatch(t, ctx, func() error { return Dropout(input, 0, state, out) })
}

func testDefaultOptimizerDispatch(t *testing.T, ctx *Context) {
	t.Helper()

	moment := newDeviceTestMatrix(t, ctx, 1, 2, []float32{0, 0})
	gradient := newDeviceTestMatrix(t, ctx, 1, 2, []float32{0.1, -0.2})
	first := newDeviceTestMatrix(t, ctx, 1, 2, []float32{0, 0})
	second := newDeviceTestMatrix(t, ctx, 1, 2, []float32{0, 0})
	delta := newDeviceTestMatrix(t, ctx, 1, 2, []float32{0, 0})
	flag := newDeviceTestMatrix(t, ctx, 1, 1, []float32{1})
	selected := newDeviceTestMatrix(t, ctx, 1, 2, []float32{0, 0})

	requireDeviceDispatch(t, ctx, func() error {
		return AdamFirstMoment(moment, gradient, 0.9, first)
	})
	requireDeviceDispatch(t, ctx, func() error {
		return AdamSecondMoment(moment, gradient, 0.999, second)
	})
	requireDeviceDispatch(t, ctx, func() error {
		return AdamDelta(first, second, 0.1, 1, delta)
	})
	requireDeviceDispatch(t, ctx, func() error { return AllFiniteAccumulate(delta, flag) })
	requireDeviceDispatch(t, ctx, func() error {
		return SelectFinite(first, second, flag, selected)
	})
}

func requireDeviceDispatch(t *testing.T, ctx *Context, operation func() error) {
	t.Helper()

	before := ctx.Stats().ComputeSubmissionCount

	require.NoError(t, operation())
	require.Greater(t, ctx.Stats().ComputeSubmissionCount, before)
}

func newDeviceTestMatrix(
	t *testing.T,
	ctx *Context,
	rows, cols int,
	values []float32,
) *Matrix {
	t.Helper()

	matrix, err := NewMatrix(ctx, rows, cols)
	require.NoError(t, err)
	t.Cleanup(matrix.Release)
	require.NoError(t, matrix.Write(values))

	return matrix
}
