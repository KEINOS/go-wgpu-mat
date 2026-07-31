//nolint:paralleltest // WGPU backend selection is process-global and Metal tests are serialized.
package mat_test

import (
	"os"
	"testing"

	"github.com/KEINOS/go-wgpu-mat/mat"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// Isolation repro for the go-nn repeated-backward corruption that persists on
// go-wgpu-mat v0.0.3 (gogpu/wgpu v0.30.29) and on v0.30.30, while
// TestSLMetalChainedCompute stays GREEN. These tests document an UNFIXED
// upstream bug: the reuse/read variants are EXPECTED TO FAIL while it stands,
// so they are quarantined under the TestRepro prefix and excluded from the
// default ^TestSLMetal gate. Run them explicitly with:
//
//	GO_WGPU_MAT_GPU=1 go test -count=10 -parallel=1 -run '^TestReproMetal' ./mat
//
// It replicates the exact op sequence of go-nn's TensorNode backward commit
// phase:
//
//	seed      = BroadcastTo(one, 1x1); dProduct = BroadcastTo(seed, 2x2)
//	rightT    = Transp(right);         deltaL   = MatMul(dProduct, rightT)
//	leftT     = Transp(left);          deltaR   = MatMul(leftT, dProduct)
//	committed = Add(grad, delta)       // per leaf, then old grads close
//
// and, like go-nn's matrix pool, closed matrices return to a pool that hands
// them out again as new outputs. An intermediate host read of the gradients
// sits between rounds, matching go-nn's contract test shape. The pooled
// variant reuses buffers; the fresh variant allocates and releases for real,
// matching go-nn's pool-disabled experiment.

type slGradPool struct {
	ctx        *mat.Context
	reuseTemps bool
	reuseGrads bool
	all        []*mat.Matrix
	pooled     map[[2]int][]*mat.Matrix
}

func newSLGradPool(ctx *mat.Context, reuseTemps, reuseGrads bool) *slGradPool {
	return &slGradPool{
		ctx:        ctx,
		reuseTemps: reuseTemps,
		reuseGrads: reuseGrads,
		all:        make([]*mat.Matrix, 0),
		pooled:     make(map[[2]int][]*mat.Matrix),
	}
}

func (pool *slGradPool) acquire(t *testing.T, rows, cols int) *mat.Matrix {
	t.Helper()

	key := [2]int{rows, cols}
	if pool.reuseTemps || pool.reuseGrads {
		if available := pool.pooled[key]; len(available) > 0 {
			matrix := available[len(available)-1]
			pool.pooled[key] = available[:len(available)-1]

			return matrix
		}
	}

	matrix, err := mat.NewMatrix(pool.ctx, rows, cols)
	require.NoError(t, err)

	pool.all = append(pool.all, matrix)

	return matrix
}

func (pool *slGradPool) releaseTemp(matrix *mat.Matrix) {
	pool.releaseWith(matrix, pool.reuseTemps)
}

func (pool *slGradPool) releaseGrad(matrix *mat.Matrix) {
	pool.releaseWith(matrix, pool.reuseGrads)
}

func (pool *slGradPool) releaseWith(matrix *mat.Matrix, reuse bool) {
	if !reuse {
		matrix.Release()

		return
	}

	key := [2]int{matrix.Rows(), matrix.Cols()}
	pool.pooled[key] = append(pool.pooled[key], matrix)
}

func (pool *slGradPool) releaseAll() {
	for _, matrix := range pool.all {
		matrix.Release()
	}
}

func runSLGradAccumRound(
	t *testing.T,
	pool *slGradPool,
	one, left, right, gradL, gradR *mat.Matrix,
) (*mat.Matrix, *mat.Matrix) {
	t.Helper()

	seed := pool.acquire(t, 1, 1)
	dProduct := pool.acquire(t, 2, 2)
	rightT := pool.acquire(t, 2, 2)
	leftT := pool.acquire(t, 2, 2)
	deltaL := pool.acquire(t, 2, 2)
	deltaR := pool.acquire(t, 2, 2)
	committedL := pool.acquire(t, 2, 2)
	committedR := pool.acquire(t, 2, 2)

	require.NoError(t, mat.BroadcastTo(one, seed))
	require.NoError(t, mat.BroadcastTo(seed, dProduct))
	require.NoError(t, mat.Transp(right, rightT))
	require.NoError(t, mat.MatMul(dProduct, rightT, deltaL))
	require.NoError(t, mat.Transp(left, leftT))
	require.NoError(t, mat.MatMul(leftT, dProduct, deltaR))
	require.NoError(t, mat.Add(gradL, deltaL, committedL))
	require.NoError(t, mat.Add(gradR, deltaR, committedR))

	// Commit phase (mirrors go-nn backward.go): old gradients close
	// immediately; round temporaries close at backward return.
	pool.releaseGrad(gradL)
	pool.releaseGrad(gradR)
	pool.releaseTemp(seed)
	pool.releaseTemp(dProduct)
	pool.releaseTemp(rightT)
	pool.releaseTemp(leftT)
	pool.releaseTemp(deltaL)
	pool.releaseTemp(deltaR)

	return committedL, committedR
}

func runSLGradAccumMetal(t *testing.T, reuseTemps, reuseGrads, intermediateRead bool) {
	t.Helper()

	if os.Getenv("GO_WGPU_MAT_GPU") != "1" {
		t.Skip("set GO_WGPU_MAT_GPU=1 to require the local Metal gate")
	}

	ctx, err := mat.NewContext(mat.UseGPU)
	require.NoError(t, err, "SL Metal gate requires a hardware adapter")
	t.Cleanup(ctx.Release)

	pool := newSLGradPool(ctx, reuseTemps, reuseGrads)
	t.Cleanup(pool.releaseAll)

	one := newP4Matrix(t, ctx, 1, 1, []float32{1})
	left := newP4Matrix(t, ctx, 2, 2, []float32{1, 2, 3, 4})
	right := newP4Matrix(t, ctx, 2, 2, []float32{5, 6, 7, 8})

	gradL := pool.acquire(t, 2, 2)
	gradR := pool.acquire(t, 2, 2)
	require.NoError(t, gradL.Write([]float32{0, 0, 0, 0}))
	require.NoError(t, gradR.Write([]float32{0, 0, 0, 0}))

	gradL, gradR = runSLGradAccumRound(t, pool, one, left, right, gradL, gradR)

	if intermediateRead {
		// Intermediate host read between rounds, as in go-nn's contract test.
		assert.InDeltaSlice(t, []float32{11, 15, 11, 15}, readP4Matrix(t, gradL), 1e-4)
		assert.InDeltaSlice(t, []float32{4, 4, 6, 6}, readP4Matrix(t, gradR), 1e-4)
	}

	gradL, gradR = runSLGradAccumRound(t, pool, one, left, right, gradL, gradR)

	assert.InDeltaSlice(t, []float32{22, 30, 22, 30}, readP4Matrix(t, gradL), 1e-4)
	assert.InDeltaSlice(t, []float32{8, 8, 12, 12}, readP4Matrix(t, gradR), 1e-4)
}

func TestReproMetalGradAccumPooledSequence(t *testing.T) {
	runSLGradAccumMetal(t, true, true, true)
}

func TestReproMetalGradAccumFreshSequence(t *testing.T) {
	runSLGradAccumMetal(t, false, false, true)
}

func TestReproMetalGradAccumPooledSequenceNoRead(t *testing.T) {
	runSLGradAccumMetal(t, true, true, false)
}

func TestReproMetalGradAccumFreshSequenceNoRead(t *testing.T) {
	runSLGradAccumMetal(t, false, false, false)
}

func TestReproMetalGradAccumTempsOnlyNoRead(t *testing.T) {
	runSLGradAccumMetal(t, true, false, false)
}

func TestReproMetalGradAccumGradsOnlyNoRead(t *testing.T) {
	runSLGradAccumMetal(t, false, true, false)
}

// TestReproMetalGradAccumPooledForensics is a diagnostic variant of the pooled
// no-read repro: after the second round it reads every intermediate buffer to
// identify the first op whose output diverges. Round temporaries are still
// pooled (never destroyed), so their final content is observable.
//nolint:funlen // Forensic reads stay adjacent to the op sequence they diagnose.
func TestReproMetalGradAccumPooledForensics(t *testing.T) {
	if os.Getenv("GO_WGPU_MAT_GPU") != "1" {
		t.Skip("set GO_WGPU_MAT_GPU=1 to require the local Metal gate")
	}

	ctx, err := mat.NewContext(mat.UseGPU)
	require.NoError(t, err, "SL Metal gate requires a hardware adapter")
	t.Cleanup(ctx.Release)

	pool := newSLGradPool(ctx, true, true)
	t.Cleanup(pool.releaseAll)

	one := newP4Matrix(t, ctx, 1, 1, []float32{1})
	left := newP4Matrix(t, ctx, 2, 2, []float32{1, 2, 3, 4})
	right := newP4Matrix(t, ctx, 2, 2, []float32{5, 6, 7, 8})

	gradL := pool.acquire(t, 2, 2)
	gradR := pool.acquire(t, 2, 2)
	require.NoError(t, gradL.Write([]float32{0, 0, 0, 0}))
	require.NoError(t, gradR.Write([]float32{0, 0, 0, 0}))

	gradL, gradR = runSLGradAccumRound(t, pool, one, left, right, gradL, gradR)

	// Second round with each intermediate retained for forensics.
	seed := pool.acquire(t, 1, 1)
	dProduct := pool.acquire(t, 2, 2)
	rightT := pool.acquire(t, 2, 2)
	leftT := pool.acquire(t, 2, 2)
	deltaL := pool.acquire(t, 2, 2)
	deltaR := pool.acquire(t, 2, 2)
	committedL := pool.acquire(t, 2, 2)
	committedR := pool.acquire(t, 2, 2)

	require.NoError(t, mat.BroadcastTo(one, seed))
	require.NoError(t, mat.BroadcastTo(seed, dProduct))
	require.NoError(t, mat.Transp(right, rightT))
	require.NoError(t, mat.MatMul(dProduct, rightT, deltaL))
	require.NoError(t, mat.Transp(left, leftT))
	require.NoError(t, mat.MatMul(leftT, dProduct, deltaR))
	require.NoError(t, mat.Add(gradL, deltaL, committedL))
	require.NoError(t, mat.Add(gradR, deltaR, committedR))

	t.Logf("seed      = %v (want [1])", readP4Matrix(t, seed))
	t.Logf("dProduct  = %v (want [1 1 1 1])", readP4Matrix(t, dProduct))
	t.Logf("rightT    = %v (want [5 7 6 8])", readP4Matrix(t, rightT))
	t.Logf("leftT     = %v (want [1 3 2 4])", readP4Matrix(t, leftT))
	t.Logf("deltaL    = %v (want [11 15 11 15])", readP4Matrix(t, deltaL))
	t.Logf("deltaR    = %v (want [4 4 6 6])", readP4Matrix(t, deltaR))
	t.Logf("gradL(r1) = %v (want [11 15 11 15])", readP4Matrix(t, gradL))
	t.Logf("gradR(r1) = %v (want [4 4 6 6])", readP4Matrix(t, gradR))
	t.Logf("commitL   = %v (want [22 30 22 30])", readP4Matrix(t, committedL))
	t.Logf("commitR   = %v (want [8 8 12 12])", readP4Matrix(t, committedR))
}
