//go:build !cgo

package mat_test

import (
	"testing"

	"github.com/KEINOS/go-wgpu-mat/mat"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestNewContext_smoke verifies NewContext returns a non-nil
// context and Release does not panic.
func TestNewContext_smoke(t *testing.T) {
	t.Parallel()

	ctx, err := mat.NewContext()
	require.NoError(t, err, "NewContext should succeed")
	require.NotNil(t, ctx)
	require.NotPanics(t, func() { ctx.Release() })
}

func TestNewContext_modes(t *testing.T) {
	t.Parallel()

	ctxCPU, err := mat.NewContext(mat.UseCPU)
	require.NoError(t, err)
	require.NotNil(t, ctxCPU)
	require.NotPanics(t, func() { ctxCPU.Release() })

	ctxGPU, err := mat.NewContext(mat.UseGPU)
	require.NoError(t, err)
	require.NotNil(t, ctxGPU)
	require.NotPanics(t, func() { ctxGPU.Release() })
}

// TestNewMatrix_dimensions verifies Rows and Cols are correct.
func TestNewMatrix_dimensions(t *testing.T) {
	t.Parallel()

	ctx, err := mat.NewContext()
	require.NoError(t, err)

	defer ctx.Release()

	matrix, err := mat.NewMatrix(ctx, 3, 4)
	require.NoError(t, err)

	defer matrix.Release()

	assert.Equal(t, 3, matrix.Rows, "Rows mismatch")
	assert.Equal(t, 4, matrix.Cols, "Cols mismatch")
}

// TestMatrix_Write_Read_roundtrip writes a known pattern and
// reads it back, expecting byte-exact equality.
func TestMatrix_Write_Read_roundtrip(t *testing.T) {
	t.Parallel()

	ctx, err := mat.NewContext()
	require.NoError(t, err)

	defer ctx.Release()

	const rows, cols = 2, 3

	matrix, err := mat.NewMatrix(ctx, rows, cols)
	require.NoError(t, err)

	defer matrix.Release()

	want := []float32{1, 2, 3, 4, 5, 6}
	require.NoError(t, matrix.Write(want))

	got, err := matrix.Read()
	require.NoError(t, err)
	require.Len(t, got, rows*cols)

	for i, v := range want {
		assert.InDelta(t, v, got[i], 1e-6, "element %d mismatch", i)
	}
}

// TestMatrix_Release_idempotent verifies that calling Release
// twice does not panic.
func TestMatrix_Release_idempotent(t *testing.T) {
	t.Parallel()

	ctx, err := mat.NewContext()
	require.NoError(t, err)

	defer ctx.Release()

	m, err := mat.NewMatrix(ctx, 1, 1)
	require.NoError(t, err)

	require.NotPanics(t, func() {
		m.Release()
		m.Release() // second call must be a no-op
	})
}

// TestContext_Release_nil verifies that calling Release on a nil
// *Context pointer does not panic.
func TestContext_Release_nil(t *testing.T) {
	t.Parallel()

	var ctx *mat.Context

	require.NotPanics(t, func() { ctx.Release() })
}

// TestMatrix_Write_length_mismatch verifies that Write returns an
// error when given the wrong number of elements.
func TestMatrix_Write_length_mismatch(t *testing.T) {
	t.Parallel()

	ctx, err := mat.NewContext()
	require.NoError(t, err)

	defer ctx.Release()

	matrix, err := mat.NewMatrix(ctx, 2, 3)
	require.NoError(t, err)

	defer matrix.Release()

	// 3 elements provided; 6 expected
	err = matrix.Write([]float32{1, 2, 3})
	assert.ErrorContains(t, err, "mat: fail to write")
}
