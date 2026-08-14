package mat

import (
	"runtime"
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestHostWorkerPoolRowsMatchSerialForUnevenPartitions(t *testing.T) {
	t.Parallel()

	const (
		rows = 17
		cols = 13
	)

	input := make([]float32, rows*cols)
	for index := range input {
		input[index] = float32((index%19)-9) * 0.125
	}

	tests := []struct {
		name  string
		apply rowOperationFunc
	}{
		{name: "softmax", apply: applySoftmaxRow},
		{name: "RMS norm", apply: applyRMSNormRow},
	}

	for _, testCase := range tests {
		t.Run(testCase.name, func(t *testing.T) {
			t.Parallel()

			want := make([]float32, len(input))

			for row := range rows {
				testCase.apply(input, want, row*cols, cols)
			}

			got := make([]float32, len(input))

			pool := newHostWorkerPool(4)
			defer pool.close()

			pool.runRows(testCase.apply, input, got, rows, cols, 1)

			assert.Equal(t, want, got)
		})
	}
}

func TestHostWorkerPoolConcurrentRowJobsStayIndependent(t *testing.T) {
	t.Parallel()

	const (
		jobs = 8
		rows = 17
		cols = 13
	)

	input := make([]float32, rows*cols)
	for index := range input {
		input[index] = float32((index%19)-9) * 0.125
	}

	want := make([]float32, len(input))
	for row := range rows {
		applySoftmaxRow(input, want, row*cols, cols)
	}

	pool := newHostWorkerPool(4)
	defer pool.close()

	results := make([][]float32, jobs)

	var callers sync.WaitGroup
	callers.Add(jobs)

	for job := range jobs {
		results[job] = make([]float32, len(input))

		go func() {
			defer callers.Done()

			pool.runRows(applySoftmaxRow, input, results[job], rows, cols, 1)
		}()
	}

	callers.Wait()

	for _, result := range results {
		assert.Equal(t, want, result)
	}
}

func TestHostWorkerPoolReusesRowJobsWithinCapacity(t *testing.T) {
	t.Parallel()

	pool := newHostWorkerPool(1)
	defer pool.close()

	first := pool.acquireRowJob()
	second := pool.acquireRowJob()
	require.NotSame(t, first, second)

	pool.releaseRowJob(first)
	pool.releaseRowJob(second)

	assert.Len(t, pool.rowJobs, 1)
	assert.Same(t, first, pool.acquireRowJob())
}

//nolint:paralleltest // AllocsPerRun is process-wide.
func TestHostWorkerPoolRowsSteadyStateAllocations(t *testing.T) {
	const (
		rows = 128
		cols = 128
	)

	pool := newHostWorkerPool(4)
	defer pool.close()

	input := make([]float32, rows*cols)
	result := make([]float32, len(input))
	pool.runRows(applyRMSNormRow, input, result, rows, cols, 1)

	allocations := testing.AllocsPerRun(10, func() {
		pool.runRows(applyRMSNormRow, input, result, rows, cols, 1)
	})

	assert.Zero(t, allocations)
}

func TestSoftmaxUsesParallelHostRows(t *testing.T) {
	t.Parallel()

	if runtime.GOMAXPROCS(0) < 2 {
		t.Skip("parallel row path requires at least two logical processors")
	}

	const size = 128

	inputData := make([]float32, size*size)
	for index := range inputData {
		inputData[index] = float32((index%19)-9) * 0.125
	}

	input, _ := newMockMatrix(size, size, inputData)
	out, output := newMockMatrix(size, size, make([]float32, len(inputData)))

	shareMockContext(input, out)
	defer input.ctx.releaseHostWorkerPool()

	want := make([]float32, len(inputData))
	for row := range size {
		applySoftmaxRow(inputData, want, row*size, size)
	}

	err := Softmax(input, out)

	require.NoError(t, err)
	require.NotNil(t, input.ctx.hostPool)
	assert.Equal(t, want, output.data)
}

func TestSmallSoftmaxDoesNotCreateHostWorkerPool(t *testing.T) {
	t.Parallel()

	input, _ := newMockMatrix(2, 3, []float32{1, 2, 3, 4, 5, 6})
	out, _ := newMockMatrix(2, 3, make([]float32, 6))

	shareMockContext(input, out)

	err := Softmax(input, out)

	require.NoError(t, err)
	assert.Nil(t, input.ctx.hostPool)
}
