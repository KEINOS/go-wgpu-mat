//nolint:paralleltest // WGPU backend selection is process-global and Metal tests are serialized.
package mat_test

import (
	"os"
	"strconv"
	"testing"

	"github.com/KEINOS/go-wgpu-mat/mat"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// Submission-lifetime regression shape: repeated rounds of dependent compute
// operations without intermediate host readback, mimicking the downstream
// go-nn device-resident backward. Any corruption in a round persists in the
// accumulator, so a single final read detects a failure in any round.

const (
	slRoundsEnv     = "GO_WGPU_MAT_SL_ROUNDS"
	slDefaultRounds = 256

	slRows   = 8
	slShared = 4
	slCols   = 8
)

func slRoundCount(t *testing.T) uint64 {
	t.Helper()

	raw := os.Getenv(slRoundsEnv)
	if raw == "" {
		return slDefaultRounds
	}

	rounds, err := strconv.ParseUint(raw, 10, 64)
	require.NoError(t, err, "%s must be an unsigned integer", slRoundsEnv)
	require.Positive(t, rounds, "%s must be positive", slRoundsEnv)

	return rounds
}

// slReferenceAccumulate computes one round of the chain in pure Go:
// t = 0.5 * (left x right) * mask, then acc += rowwise-broadcast(rowSum(t)).
func slReferenceAccumulate(acc, left, right, mask []float32) {
	product := make([]float32, slRows*slCols)

	for row := range slRows {
		for col := range slCols {
			var sum float32
			for shared := range slShared {
				sum += left[row*slShared+shared] * right[shared*slCols+col]
			}

			product[row*slCols+col] = sum * mask[row*slCols+col] * 0.5
		}
	}

	for row := range slRows {
		var rowSum float32
		for col := range slCols {
			rowSum += product[row*slCols+col]
		}

		for col := range slCols {
			acc[row*slCols+col] += rowSum
		}
	}
}

// runSLChainedRound submits one round of dependent operations with fresh
// intermediate matrices and no intermediate host transfer. Intermediates are
// allocated directly so each is released exactly once per round, including
// FailNow paths (deferred releases run on runtime.Goexit).
func runSLChainedRound(
	t *testing.T,
	ctx *mat.Context,
	left, right, mask, accIn, accOut *mat.Matrix,
) {
	t.Helper()

	product, err := mat.NewMatrix(ctx, slRows, slCols)
	require.NoError(t, err)

	defer product.Release()

	masked, err := mat.NewMatrix(ctx, slRows, slCols)
	require.NoError(t, err)

	defer masked.Release()

	scaled, err := mat.NewMatrix(ctx, slRows, slCols)
	require.NoError(t, err)

	defer scaled.Release()

	reduced, err := mat.NewMatrix(ctx, slRows, 1)
	require.NoError(t, err)

	defer reduced.Release()

	broadcast, err := mat.NewMatrix(ctx, slRows, slCols)
	require.NoError(t, err)

	defer broadcast.Release()

	require.NoError(t, mat.MatMul(left, right, product))
	require.NoError(t, mat.Mul(product, mask, masked))
	require.NoError(t, mat.Scale(masked, 0.5, scaled))
	require.NoError(t, mat.ReduceSumTo(scaled, reduced))
	require.NoError(t, mat.BroadcastTo(reduced, broadcast))
	require.NoError(t, mat.Add(accIn, broadcast, accOut))
}

func slInputData() ([]float32, []float32, []float32, []float32) {
	left := make([]float32, slRows*slShared)
	right := make([]float32, slShared*slCols)
	mask := make([]float32, slRows*slCols)
	zeros := make([]float32, slRows*slCols)

	for i := range left {
		left[i] = float32(i%3) - 1
	}

	for i := range right {
		right[i] = float32(i%5) - 2
	}

	for i := range mask {
		mask[i] = float32(i % 2)
	}

	return left, right, mask, zeros
}

func TestSLSoftwareChainedCompute(t *testing.T) {
	ctx, err := mat.NewContext(mat.UseCPU)
	require.NoError(t, err)
	t.Cleanup(ctx.Release)

	leftData, rightData, maskData, zeros := slInputData()
	left := newP4Matrix(t, ctx, slRows, slShared, leftData)
	right := newP4Matrix(t, ctx, slShared, slCols, rightData)
	mask := newP4Matrix(t, ctx, slRows, slCols, maskData)
	accIn := newP4Matrix(t, ctx, slRows, slCols, zeros)
	accOut := newP4Matrix(t, ctx, slRows, slCols, nil)

	expected := make([]float32, slRows*slCols)

	rounds := slRoundCount(t)
	for range rounds {
		runSLChainedRound(t, ctx, left, right, mask, accIn, accOut)
		slReferenceAccumulate(expected, leftData, rightData, maskData)

		accIn, accOut = accOut, accIn
	}

	assert.InDeltaSlice(t, expected, readP4Matrix(t, accIn), 1e-4)
}

func TestSLMetalChainedCompute(t *testing.T) {
	if os.Getenv("GO_WGPU_MAT_GPU") != "1" {
		t.Skip("set GO_WGPU_MAT_GPU=1 to require the local Metal gate")
	}

	ctx, err := mat.NewContext(mat.UseGPU)
	require.NoError(t, err, "SL Metal gate requires a hardware adapter")
	t.Cleanup(ctx.Release)

	leftData, rightData, maskData, zeros := slInputData()
	left := newP4Matrix(t, ctx, slRows, slShared, leftData)
	right := newP4Matrix(t, ctx, slShared, slCols, rightData)
	mask := newP4Matrix(t, ctx, slRows, slCols, maskData)
	accIn := newP4Matrix(t, ctx, slRows, slCols, zeros)
	accOut := newP4Matrix(t, ctx, slRows, slCols, nil)

	expected := make([]float32, slRows*slCols)

	rounds := slRoundCount(t)
	start := ctx.Stats()

	for range rounds {
		runSLChainedRound(t, ctx, left, right, mask, accIn, accOut)
		slReferenceAccumulate(expected, leftData, rightData, maskData)

		accIn, accOut = accOut, accIn
	}

	computed := ctx.Stats()

	assert.Zero(t, computed.HostReadCount-start.HostReadCount)
	assert.Zero(t, computed.HostReadBytes-start.HostReadBytes)
	assert.Zero(t, computed.HostWriteCount-start.HostWriteCount)
	assert.Zero(t, computed.HostWriteBytes-start.HostWriteBytes)
	assert.Equal(t, 6*rounds,
		computed.ComputeSubmissionCount-start.ComputeSubmissionCount)
	assert.Zero(t, computed.ReadbackSubmissionCount-start.ReadbackSubmissionCount)

	assert.InDeltaSlice(t, expected, readP4Matrix(t, accIn), 1e-4)

	finished := ctx.Stats()
	assert.Equal(t, uint64(1), finished.HostReadCount-start.HostReadCount)
	assert.Equal(t, uint64(1), finished.ReadbackSubmissionCount-start.ReadbackSubmissionCount)
}

// TestSLMetalReleaseWithInflightWork releases the Context while submissions
// may still be in flight: no final Read synchronizes the queue. Release must
// drain outstanding work without crashing, and stay idempotent.
func TestSLMetalReleaseWithInflightWork(t *testing.T) {
	if os.Getenv("GO_WGPU_MAT_GPU") != "1" {
		t.Skip("set GO_WGPU_MAT_GPU=1 to require the local Metal gate")
	}

	ctx, err := mat.NewContext(mat.UseGPU)
	require.NoError(t, err, "SL Metal gate requires a hardware adapter")
	t.Cleanup(ctx.Release)

	leftData, rightData, maskData, zeros := slInputData()
	left := newP4Matrix(t, ctx, slRows, slShared, leftData)
	right := newP4Matrix(t, ctx, slShared, slCols, rightData)
	mask := newP4Matrix(t, ctx, slRows, slCols, maskData)
	accIn := newP4Matrix(t, ctx, slRows, slCols, zeros)
	accOut := newP4Matrix(t, ctx, slRows, slCols, nil)

	for range slRoundCount(t) {
		runSLChainedRound(t, ctx, left, right, mask, accIn, accOut)
		accIn, accOut = accOut, accIn
	}

	for _, matrix := range []*mat.Matrix{left, right, mask, accIn, accOut} {
		matrix.Release()
		matrix.Release()
		assert.True(t, matrix.Released())
	}

	ctx.Release()
	assert.True(t, ctx.Released())

	ctx.Release()
	assert.True(t, ctx.Released())
}
