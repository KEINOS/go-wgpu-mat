package mat_test

import (
	"math"
	"os"
	"testing"

	"github.com/KEINOS/go-wgpu-mat/mat"
	"github.com/stretchr/testify/require"
)

func TestDropoutCounterRandomContract(t *testing.T) {
	t.Parallel()

	context, err := mat.NewContext(mat.UseCPU)
	require.NoError(t, err)

	defer context.Release()

	input, err := mat.NewMatrix(context, 1, 4)
	require.NoError(t, err)

	defer input.Release()

	require.NoError(t, input.Write([]float32{1, 2, 3, 4}))

	output, err := mat.NewMatrix(context, 1, 4)
	require.NoError(t, err)

	defer output.Release()

	err = mat.Dropout(input, 0.5, mat.RandomState{Seed: 42, StreamID: 7, Counter: 0}, output)
	require.NoError(t, err)
	data, err := output.Read()
	require.NoError(t, err)
	require.Equal(t, []float32{2, 0, 0, 8}, data)
}

//nolint:paralleltest // Hardware adapter selection is process-global.
func TestDropoutMetalMatchesCPUWords(t *testing.T) {
	if os.Getenv("GO_WGPU_MAT_GPU") != "1" {
		t.Skip("set GO_WGPU_MAT_GPU=1 to require the local Metal gate")
	}

	gpu, err := mat.NewContext(mat.UseGPU)
	require.NoError(t, err, "Dropout Metal gate requires a hardware adapter")

	defer gpu.Release()

	cpu, err := mat.NewContext(mat.UseCPU)
	require.NoError(t, err)

	defer cpu.Release()

	tests := []struct {
		probability float32
		state       mat.RandomState
	}{
		{probability: 0, state: mat.RandomState{Seed: 42, StreamID: 7, Counter: 0}},
		{probability: 0.001, state: mat.RandomState{Seed: 42, StreamID: 7, Counter: 0}},
		{probability: 0.25, state: mat.RandomState{
			Seed: 1<<63 | 17, StreamID: 1<<62 | 9, Counter: math.MaxUint32 - 3,
		}},
		{probability: 0.5, state: mat.RandomState{
			Seed: math.MaxUint64, StreamID: math.MaxUint64, Counter: math.MaxUint32,
		}},
		{probability: math.Float32frombits(0x3f7fffff), state: mat.RandomState{
			Seed: 5, StreamID: 8, Counter: math.MaxUint64 - 7,
		}},
	}

	for _, test := range tests {
		cpuData := runDropoutContract(t, cpu, test.probability, test.state)
		gpuData := runDropoutContract(t, gpu, test.probability, test.state)
		require.Equal(t, cpuData, gpuData)
	}
}

func runDropoutContract(
	t *testing.T,
	context *mat.Context,
	probability float32,
	state mat.RandomState,
) []float32 {
	t.Helper()

	input, err := mat.NewMatrix(context, 1, 8)
	require.NoError(t, err)

	defer input.Release()

	require.NoError(t, input.Write([]float32{1, 2, 3, 4, 5, 6, 7, 8}))

	output, err := mat.NewMatrix(context, 1, 8)
	require.NoError(t, err)

	defer output.Release()

	require.NoError(t, mat.Dropout(input, probability, state, output))
	data, err := output.Read()
	require.NoError(t, err)

	return data
}

func TestDropoutRejectsInvalidContract(t *testing.T) {
	t.Parallel()

	context, err := mat.NewContext(mat.UseCPU)
	require.NoError(t, err)

	defer context.Release()

	input, err := mat.NewMatrix(context, 1, 1)
	require.NoError(t, err)

	defer input.Release()

	output, err := mat.NewMatrix(context, 1, 1)
	require.NoError(t, err)

	defer output.Release()

	emptyState := mat.RandomState{Seed: 0, StreamID: 0, Counter: 0}
	require.ErrorIs(t, mat.Dropout(input, -0.1, emptyState, output), mat.ErrInvalidProbability)
	require.ErrorIs(t, mat.Dropout(input, 1, emptyState, output), mat.ErrInvalidProbability)
	require.ErrorIs(t, mat.Dropout(input, 0.5, emptyState, input), mat.ErrAliasedOutput)
}
