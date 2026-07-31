//nolint:paralleltest // WGPU backend selection is process-global and Metal tests are serialized.
package mat_test

import (
	"os"
	"testing"

	"github.com/KEINOS/go-wgpu-mat/mat"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func newP4Matrix(t *testing.T, ctx *mat.Context, rows, cols int, data []float32) *mat.Matrix {
	t.Helper()

	matrix, err := mat.NewMatrix(ctx, rows, cols)
	require.NoError(t, err)
	t.Cleanup(matrix.Release)

	if data != nil {
		require.NoError(t, matrix.Write(data))
	}

	return matrix
}

func readP4Matrix(t *testing.T, matrix *mat.Matrix) []float32 {
	t.Helper()

	data, err := matrix.Read()
	require.NoError(t, err)

	return data
}

func assertP4DeviceResidentStats(t *testing.T, start, computed mat.Stats) {
	t.Helper()

	assert.Equal(t, uint64(0), computed.HostReadCount-start.HostReadCount)
	assert.Equal(t, uint64(0), computed.HostReadBytes-start.HostReadBytes)
	assert.Equal(t, uint64(0), computed.HostWriteCount-start.HostWriteCount)
	assert.Equal(t, uint64(0), computed.HostWriteBytes-start.HostWriteBytes)
	assert.Equal(t, uint64(9), computed.ComputeSubmissionCount-start.ComputeSubmissionCount)
	assert.Equal(t, uint64(0), computed.ReadbackSubmissionCount-start.ReadbackSubmissionCount)
	assert.Equal(t, uint64(0), computed.MatrixAllocationCount-start.MatrixAllocationCount)
	assert.Equal(t, uint64(0), computed.MatrixReleaseCount-start.MatrixReleaseCount)
	assert.Equal(t, start.LiveMatrixBytes, computed.LiveMatrixBytes)
}

func assertP4FinalReadbackStats(t *testing.T, start, finished mat.Stats) {
	t.Helper()

	assert.Equal(t, uint64(1), finished.HostReadCount-start.HostReadCount)
	assert.Equal(t, uint64(16), finished.HostReadBytes-start.HostReadBytes)
	assert.Equal(t, uint64(0), finished.HostWriteCount-start.HostWriteCount)
	assert.Equal(t, uint64(0), finished.HostWriteBytes-start.HostWriteBytes)
	assert.Equal(t, uint64(9), finished.ComputeSubmissionCount-start.ComputeSubmissionCount)
	assert.Equal(t, uint64(1), finished.ReadbackSubmissionCount-start.ReadbackSubmissionCount)
	assert.Equal(t, uint64(0), finished.MatrixAllocationCount-start.MatrixAllocationCount)
	assert.Equal(t, uint64(0), finished.MatrixReleaseCount-start.MatrixReleaseCount)
}

func assertP4ReleasedStats(t *testing.T, released mat.Stats) {
	t.Helper()

	assert.Equal(t, released.MatrixAllocationCount, released.MatrixReleaseCount)
	assert.Equal(t, uint64(0), released.LiveMatrixBytes)
	assert.GreaterOrEqual(t, released.PeakLiveMatrixBytes, uint64(236))
}

func TestP4SoftwareKernels(t *testing.T) {
	ctx, err := mat.NewContext(mat.UseCPU)
	require.NoError(t, err)
	t.Cleanup(ctx.Release)

	input := newP4Matrix(t, ctx, 2, 3, []float32{1, 2, 3, 4, 5, 6})
	row := newP4Matrix(t, ctx, 1, 3, []float32{10, 20, 30})
	column := newP4Matrix(t, ctx, 2, 1, []float32{2, 3})

	added := newP4Matrix(t, ctx, 2, 3, nil)
	require.NoError(t, mat.Add(input, row, added))
	assert.Equal(t, []float32{11, 22, 33, 14, 25, 36}, readP4Matrix(t, added))

	multiplied := newP4Matrix(t, ctx, 2, 3, nil)
	require.NoError(t, mat.Mul(added, column, multiplied))
	assert.Equal(t, []float32{22, 44, 66, 42, 75, 108}, readP4Matrix(t, multiplied))

	scaled := newP4Matrix(t, ctx, 2, 3, nil)
	require.NoError(t, mat.Scale(multiplied, 0.5, scaled))
	assert.Equal(t, []float32{11, 22, 33, 21, 37.5, 54}, readP4Matrix(t, scaled))

	transposed := newP4Matrix(t, ctx, 3, 2, nil)
	require.NoError(t, mat.Transp(scaled, transposed))
	assert.Equal(t, []float32{11, 21, 22, 37.5, 33, 54}, readP4Matrix(t, transposed))

	reduced := newP4Matrix(t, ctx, 1, 2, nil)
	require.NoError(t, mat.ReduceSumTo(transposed, reduced))
	assert.Equal(t, []float32{66, 112.5}, readP4Matrix(t, reduced))

	broadcast := newP4Matrix(t, ctx, 3, 2, nil)
	require.NoError(t, mat.BroadcastTo(reduced, broadcast))
	assert.Equal(t, []float32{66, 112.5, 66, 112.5, 66, 112.5}, readP4Matrix(t, broadcast))

	reshaped := newP4Matrix(t, ctx, 2, 3, nil)
	require.NoError(t, mat.ReshapeTo(broadcast, reshaped))
	assert.Equal(t, []float32{66, 112.5, 66, 112.5, 66, 112.5}, readP4Matrix(t, reshaped))

	rowSums := newP4Matrix(t, ctx, 2, 1, nil)
	require.NoError(t, mat.ReduceSum(input, rowSums))
	assert.Equal(t, []float32{6, 15}, readP4Matrix(t, rowSums))
}

func TestP4SoftwareShapeErrors(t *testing.T) {
	ctx, err := mat.NewContext(mat.UseCPU)
	require.NoError(t, err)
	t.Cleanup(ctx.Release)

	input := newP4Matrix(t, ctx, 2, 3, nil)
	wrong := newP4Matrix(t, ctx, 2, 2, nil)
	out := newP4Matrix(t, ctx, 2, 3, nil)

	require.ErrorIs(t, mat.Add(input, wrong, out), mat.ErrDimensionMismatch)
	require.ErrorIs(t, mat.Mul(input, wrong, out), mat.ErrDimensionMismatch)
	require.ErrorIs(t, mat.ReduceSumTo(input, wrong), mat.ErrDimensionMismatch)
	require.ErrorIs(t, mat.BroadcastTo(wrong, out), mat.ErrDimensionMismatch)
	require.ErrorIs(t, mat.ReshapeTo(input, wrong), mat.ErrDimensionMismatch)
}

func TestP4SoftwareStatsSnapshot(t *testing.T) {
	ctx, err := mat.NewContext(mat.UseCPU)
	require.NoError(t, err)
	t.Cleanup(ctx.Release)

	before := ctx.Stats()
	input := newP4Matrix(t, ctx, 1, 2, []float32{1, 2})
	out := newP4Matrix(t, ctx, 1, 2, nil)
	require.NoError(t, mat.Scale(input, 2, out))
	_ = readP4Matrix(t, out)
	after := ctx.Stats()

	assert.Equal(t, uint64(2), after.MatrixAllocationCount-before.MatrixAllocationCount)
	assert.Equal(t, uint64(0), after.MatrixReleaseCount-before.MatrixReleaseCount)
	assert.Equal(t, uint64(16), after.LiveMatrixBytes-before.LiveMatrixBytes)
	assert.GreaterOrEqual(t, after.PeakLiveMatrixBytes, after.LiveMatrixBytes)
	assert.Equal(t, uint64(2), after.HostReadCount-before.HostReadCount)
	assert.Equal(t, uint64(16), after.HostReadBytes-before.HostReadBytes)
	assert.Equal(t, uint64(2), after.HostWriteCount-before.HostWriteCount)
	assert.Equal(t, uint64(16), after.HostWriteBytes-before.HostWriteBytes)
	assert.Equal(t, uint64(0), after.ComputeSubmissionCount-before.ComputeSubmissionCount)
	assert.Equal(t, uint64(2), after.ReadbackSubmissionCount-before.ReadbackSubmissionCount)
}

func TestP4MetalKernels(t *testing.T) {
	if os.Getenv("GO_WGPU_MAT_GPU") != "1" {
		t.Skip("set GO_WGPU_MAT_GPU=1 to require the local Metal gate")
	}

	ctx, err := mat.NewContext(mat.UseGPU)
	require.NoError(t, err, "P4 Metal gate requires a hardware adapter")
	t.Cleanup(ctx.Release)

	input := newP4Matrix(t, ctx, 2, 3, []float32{1, 2, 3, 4, 5, 6})
	row := newP4Matrix(t, ctx, 1, 3, []float32{10, 20, 30})
	column := newP4Matrix(t, ctx, 2, 1, []float32{2, 3})
	right := newP4Matrix(t, ctx, 3, 2, []float32{1, 0, 0, 1, 1, 1})
	added := newP4Matrix(t, ctx, 2, 3, nil)
	multiplied := newP4Matrix(t, ctx, 2, 3, nil)
	scaled := newP4Matrix(t, ctx, 2, 3, nil)
	transposed := newP4Matrix(t, ctx, 3, 2, nil)
	reduced := newP4Matrix(t, ctx, 1, 2, nil)
	broadcast := newP4Matrix(t, ctx, 3, 2, nil)
	reshaped := newP4Matrix(t, ctx, 2, 3, nil)
	product := newP4Matrix(t, ctx, 2, 2, nil)

	start := ctx.Stats()

	require.NoError(t, mat.Add(input, input, added))
	require.NoError(t, mat.Add(input, row, added))
	require.NoError(t, mat.Mul(added, column, multiplied))
	require.NoError(t, mat.Scale(multiplied, 0.5, scaled))
	require.NoError(t, mat.Transp(scaled, transposed))
	require.NoError(t, mat.ReduceSumTo(transposed, reduced))
	require.NoError(t, mat.BroadcastTo(reduced, broadcast))
	require.NoError(t, mat.ReshapeTo(broadcast, reshaped))
	require.NoError(t, mat.MatMul(reshaped, right, product))

	computed := ctx.Stats()

	assertP4DeviceResidentStats(t, start, computed)

	result := readP4Matrix(t, product)
	assert.InDeltaSlice(t, []float32{132, 178.5, 225, 178.5}, result, 1e-4)

	finished := ctx.Stats()
	assertP4FinalReadbackStats(t, start, finished)

	for _, matrix := range []*mat.Matrix{
		input, row, column, right, added, multiplied,
		scaled, transposed, reduced, broadcast, reshaped, product,
	} {
		matrix.Release()
	}

	assertP4ReleasedStats(t, ctx.Stats())
}
