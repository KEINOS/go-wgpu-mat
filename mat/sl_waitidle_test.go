//nolint:paralleltest // WGPU backend selection is process-global and Metal tests are serialized.
package mat

import (
	"os"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// Diagnostic for the grad-accum pooled-sequence corruption (quarantined under
// the TestRepro prefix; expected to fail while the upstream bug stands).
// Runs the same pooled two-round shape but calls Device.WaitIdle() after every
// operation in round 2. Full per-op synchronization does NOT remove the
// corruption, which rules out completion/timing races and points at wrong
// kernel outputs when writing into previously used buffers.
//nolint:funlen // The pooled two-round shape and per-op sync stay in one linear sequence.
func TestReproMetalGradAccumPooledWaitIdle(t *testing.T) {
	if os.Getenv("GO_WGPU_MAT_GPU") != "1" {
		t.Skip("set GO_WGPU_MAT_GPU=1 to require the local Metal gate")
	}

	ctx, err := NewContext(UseGPU)
	require.NoError(t, err, "SL Metal gate requires a hardware adapter")
	t.Cleanup(ctx.Release)

	mustMatrix := func(rows, cols int, data []float32) *Matrix {
		t.Helper()

		matrix, err := NewMatrix(ctx, rows, cols)
		require.NoError(t, err)

		if data != nil {
			require.NoError(t, matrix.Write(data))
		}

		return matrix
	}

	one := mustMatrix(1, 1, []float32{1})
	left := mustMatrix(2, 2, []float32{1, 2, 3, 4})
	right := mustMatrix(2, 2, []float32{5, 6, 7, 8})

	pooled := make([]*Matrix, 0, 8)
	acquire := func() *Matrix {
		if len(pooled) > 0 {
			matrix := pooled[len(pooled)-1]
			pooled = pooled[:len(pooled)-1]

			return matrix
		}

		return mustMatrix(2, 2, nil)
	}

	gradL := mustMatrix(2, 2, []float32{0, 0, 0, 0})
	gradR := mustMatrix(2, 2, []float32{0, 0, 0, 0})

	runRound := func(gradL, gradR *Matrix, sync bool) (*Matrix, *Matrix) {
		t.Helper()

		seed := mustMatrix(1, 1, nil)
		dProduct := acquire()
		rightT := acquire()
		leftT := acquire()
		deltaL := acquire()
		deltaR := acquire()
		committedL := acquire()
		committedR := acquire()

		wait := func() {
			if sync {
				require.NoError(t, ctx.device.WaitIdle())
			}
		}

		require.NoError(t, BroadcastTo(one, seed))
		wait()
		require.NoError(t, BroadcastTo(seed, dProduct))
		wait()
		require.NoError(t, Transp(right, rightT))
		wait()
		require.NoError(t, MatMul(dProduct, rightT, deltaL))
		wait()
		require.NoError(t, Transp(left, leftT))
		wait()
		require.NoError(t, MatMul(leftT, dProduct, deltaR))
		wait()
		require.NoError(t, Add(gradL, deltaL, committedL))
		wait()
		require.NoError(t, Add(gradR, deltaR, committedR))
		wait()

		pooled = append(pooled, gradL, gradR, dProduct, rightT, leftT, deltaL, deltaR)

		seed.Release()

		return committedL, committedR
	}

	gradL, gradR = runRound(gradL, gradR, false)
	gradL, gradR = runRound(gradL, gradR, true)

	gotL, err := gradL.Read()
	require.NoError(t, err)
	gotR, err := gradR.Read()
	require.NoError(t, err)

	assert.InDeltaSlice(t, []float32{22, 30, 22, 30}, gotL, 1e-4)
	assert.InDeltaSlice(t, []float32{8, 8, 12, 12}, gotR, 1e-4)

	// Cross-check: copy the committed results to brand-new matrices with a
	// device-to-device kernel and read those instead. If the fresh-buffer
	// reads disagree with the direct reads, the readback path (not the
	// compute kernels) is returning the wrong buffer contents.
	freshL := mustMatrix(2, 2, nil)
	freshR := mustMatrix(2, 2, nil)

	require.NoError(t, ReshapeTo(gradL, freshL))
	require.NoError(t, ReshapeTo(gradR, freshR))

	crossL, err := freshL.Read()
	require.NoError(t, err)
	crossR, err := freshR.Read()
	require.NoError(t, err)

	t.Logf("direct read: L=%v R=%v", gotL, gotR)
	t.Logf("copied read: L=%v R=%v", crossL, crossR)
	assert.InDeltaSlice(t, []float32{22, 30, 22, 30}, crossL, 1e-4)
	assert.InDeltaSlice(t, []float32{8, 8, 12, 12}, crossR, 1e-4)
}
