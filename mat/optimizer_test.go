package mat_test

import (
	"math"
	"os"
	"testing"

	"github.com/KEINOS/go-wgpu-mat/mat"
	"github.com/stretchr/testify/require"
)

func TestOptimizerPrimitivesCPU(t *testing.T) {
	t.Parallel()

	context, err := mat.NewContext(mat.UseCPU)
	require.NoError(t, err)

	defer context.Release()

	moment := optimizerMatrix(t, context, []float32{0, 0})
	gradient := optimizerMatrix(t, context, []float32{0.1, -0.2})
	first := optimizerMatrix(t, context, []float32{0, 0})
	second := optimizerMatrix(t, context, []float32{0, 0})
	delta := optimizerMatrix(t, context, []float32{0, 0})

	require.NoError(t, mat.AdamFirstMoment(moment, gradient, 0.9, first))
	require.NoError(t, mat.AdamSecondMoment(moment, gradient, 0.999, second))
	require.NoError(t, mat.AdamDelta(first, second, 0.1, 1, delta))
	requireFloat32Close(t, readOptimizerMatrix(t, first), []float32{0.01, -0.02})
	requireFloat32Close(t, readOptimizerMatrix(t, second), []float32{0.00001, 0.00004})

	flag := optimizerMatrix(t, context, []float32{1})
	require.NoError(t, mat.AllFiniteAccumulate(first, flag))
	require.Equal(t, []float32{1}, readOptimizerMatrix(t, flag))
	nonfinite := optimizerMatrix(t, context, []float32{float32(math.Inf(1)), 0})
	require.NoError(t, mat.AllFiniteAccumulate(nonfinite, flag))
	require.Equal(t, []float32{0}, readOptimizerMatrix(t, flag))
}

func TestOptimizerPrimitiveContracts(t *testing.T) {
	t.Parallel()

	context, err := mat.NewContext(mat.UseCPU)
	require.NoError(t, err)

	defer context.Release()

	otherContext, err := mat.NewContext(mat.UseCPU)
	require.NoError(t, err)

	defer otherContext.Release()

	left := optimizerMatrix(t, context, []float32{1, 2})
	right := optimizerMatrix(t, context, []float32{3, 4})
	out := optimizerMatrix(t, context, []float32{0, 0})
	wrongShape := optimizerMatrix(t, context, []float32{0})
	other := optimizerMatrix(t, otherContext, []float32{0, 0})

	for _, beta := range []float32{-0.1, 1, float32(math.NaN())} {
		require.ErrorIs(t, mat.AdamFirstMoment(left, right, beta, out), mat.ErrInvalidProbability)
		require.ErrorIs(t, mat.AdamSecondMoment(left, right, beta, out), mat.ErrInvalidProbability)
	}

	require.ErrorIs(t, mat.AdamFirstMoment(left, wrongShape, 0.9, out), mat.ErrDimensionMismatch)
	require.ErrorIs(t, mat.AdamSecondMoment(left, right, 0.9, left), mat.ErrAliasedOutput)
	require.ErrorIs(t, mat.AdamFirstMoment(left, other, 0.9, out), mat.ErrContextMismatch)
	require.ErrorIs(t, mat.AdamDelta(left, right, 0.1, 0, out), mat.ErrInvalidState)
	require.ErrorIs(t, mat.AdamDelta(left, right, float32(math.Inf(1)), 1e-8, out), mat.ErrInvalidState)

	flag := optimizerMatrix(t, context, []float32{1})
	invalidFlag := optimizerMatrix(t, context, []float32{1, 1})
	otherFlag := optimizerMatrix(t, otherContext, []float32{1})
	require.ErrorIs(t, mat.AllFiniteAccumulate(left, invalidFlag), mat.ErrDimensionMismatch)
	require.ErrorIs(t, mat.AllFiniteAccumulate(flag, flag), mat.ErrAliasedOutput)
	require.ErrorIs(t, mat.AllFiniteAccumulate(left, otherFlag), mat.ErrContextMismatch)

	require.NoError(t, mat.SelectFinite(left, right, flag, out))
	require.Equal(t, []float32{1, 2}, readOptimizerMatrix(t, out))
	require.NoError(t, flag.Write([]float32{0}))
	require.NoError(t, mat.SelectFinite(left, right, flag, out))
	require.Equal(t, []float32{3, 4}, readOptimizerMatrix(t, out))
	require.ErrorIs(t, mat.SelectFinite(left, right, invalidFlag, out), mat.ErrDimensionMismatch)
	require.ErrorIs(t, mat.SelectFinite(left, right, otherFlag, out), mat.ErrContextMismatch)
}

//nolint:paralleltest // Hardware adapter selection is process-global.
func TestOptimizerPrimitivesMetalMatchCPU(t *testing.T) {
	if os.Getenv("GO_WGPU_MAT_GPU") != "1" {
		t.Skip("set GO_WGPU_MAT_GPU=1 to require the local Metal gate")
	}

	cpu, err := mat.NewContext(mat.UseCPU)
	require.NoError(t, err)

	defer cpu.Release()

	gpu, err := mat.NewContext(mat.UseGPU)
	require.NoError(t, err)

	defer gpu.Release()

	cpuFirst, cpuSecond, cpuDelta, cpuFlag := optimizerPrimitiveResults(t, cpu)
	gpuFirst, gpuSecond, gpuDelta, gpuFlag := optimizerPrimitiveResults(t, gpu)
	require.Equal(t, cpuFirst, gpuFirst)
	require.Equal(t, cpuSecond, gpuSecond)
	require.Equal(t, cpuDelta, gpuDelta)
	require.Equal(t, cpuFlag, gpuFlag)
}

func optimizerPrimitiveResults(
	t *testing.T,
	context *mat.Context,
) ([]float32, []float32, []float32, []float32) {
	t.Helper()
	moment := optimizerMatrix(t, context, []float32{0, 0})
	gradient := optimizerMatrix(t, context, []float32{0.1, -0.2})
	first := optimizerMatrix(t, context, []float32{0, 0})
	second := optimizerMatrix(t, context, []float32{0, 0})
	delta := optimizerMatrix(t, context, []float32{0, 0})
	flag := optimizerMatrix(t, context, []float32{1})
	require.NoError(t, mat.AdamFirstMoment(moment, gradient, 0.9, first))
	require.NoError(t, mat.AdamSecondMoment(moment, gradient, 0.999, second))
	require.NoError(t, mat.AdamDelta(first, second, 0.1, 1, delta))
	require.NoError(t, mat.AllFiniteAccumulate(first, flag))

	return readOptimizerMatrix(t, first), readOptimizerMatrix(t, second),
		readOptimizerMatrix(t, delta), readOptimizerMatrix(t, flag)
}

func optimizerMatrix(t *testing.T, context *mat.Context, data []float32) *mat.Matrix {
	t.Helper()

	value, err := mat.NewMatrix(context, 1, len(data))
	require.NoError(t, err)
	t.Cleanup(value.Release)
	require.NoError(t, value.Write(data))

	return value
}

func readOptimizerMatrix(t *testing.T, value *mat.Matrix) []float32 {
	t.Helper()

	data, err := value.Read()
	require.NoError(t, err)

	return data
}

func requireFloat32Close(t *testing.T, got, want []float32) {
	t.Helper()
	require.Len(t, got, len(want))

	for index := range want {
		require.InDelta(t, want[index], got[index], 1e-7)
	}
}
