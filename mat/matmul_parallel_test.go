package mat

import (
	"io"
	"runtime"
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestMultiplyMatMulRowsParallelMatchesSerial(t *testing.T) {
	t.Parallel()

	const (
		rows      = 17
		sharedDim = 9
		cols      = 11
	)

	left := make([]float32, rows*sharedDim)
	right := make([]float32, sharedDim*cols)

	for index := range left {
		left[index] = float32((index%13)-6) * 0.25
	}

	for index := range right {
		right[index] = float32((index%11)-5) * 0.125
	}

	serial := make([]float32, rows*cols)
	multiplyMatMulRows(left, right, serial, sharedDim, cols, 0, rows)

	parallel := make([]float32, rows*cols)

	pool := newHostWorkerPool(4)
	defer pool.close()

	pool.runWorkRangesWithConfig(rows, sharedDim*cols, 1, func(start, end int) {
		multiplyMatMulRows(left, right, parallel, sharedDim, cols, start, end)
	})

	assert.Equal(t, serial, parallel)
}

func TestMatMulCPUUsesRangeRunner(t *testing.T) {
	t.Parallel()

	left, _ := newMockMatrix(2, 3, []float32{1, 2, 3, 4, 5, 6})
	right, _ := newMockMatrix(3, 2, []float32{7, 8, 9, 10, 11, 12})
	out, output := newMockMatrix(2, 2, make([]float32, 4))
	shareMockContext(left, right, out)

	called := false
	runner := func(total, workPerItem int, operation workRangeFunc) {
		called = true

		assert.Equal(t, 2, total)
		assert.Equal(t, 6, workPerItem)
		operation(0, total)
	}

	err := matMulCPUWithRunner(left, right, out, runner)

	require.NoError(t, err)
	assert.True(t, called)
	assert.Equal(t, []float32{58, 64, 139, 154}, output.data)
}

func TestHostWorkerPoolMatMulConcurrentJobsMatchSerial(t *testing.T) {
	t.Parallel()

	const (
		jobs      = 8
		rows      = 17
		sharedDim = 9
		cols      = 11
	)

	pool := newHostWorkerPool(4)
	defer pool.close()

	left := make([]float32, rows*sharedDim)
	right := make([]float32, sharedDim*cols)

	for index := range left {
		left[index] = float32((index%13)-6) * 0.25
	}

	for index := range right {
		right[index] = float32((index%11)-5) * 0.125
	}

	want := make([]float32, rows*cols)
	multiplyMatMulRows(left, right, want, sharedDim, cols, 0, rows)

	results := make([][]float32, jobs)

	var callers sync.WaitGroup
	callers.Add(jobs)

	for job := range jobs {
		results[job] = make([]float32, rows*cols)

		go func() {
			defer callers.Done()

			pool.runMatMul(left, right, results[job], rows, sharedDim, cols)
		}()
	}

	callers.Wait()

	for _, result := range results {
		assert.Equal(t, want, result)
	}
}

func TestMatMulCPUParallelPathMatchesSerial(t *testing.T) {
	t.Parallel()

	if runtime.GOMAXPROCS(0) < 2 {
		t.Skip("parallel MatMul path requires at least two logical processors")
	}

	const size = 128

	leftData := make([]float32, size*size)
	rightData := make([]float32, size*size)

	for index := range leftData {
		leftData[index] = float32((index%13)-6) * 0.25
		rightData[index] = float32((index%11)-5) * 0.125
	}

	want := make([]float32, size*size)
	multiplyMatMulRows(leftData, rightData, want, size, size, 0, size)

	left, _ := newMockMatrix(size, size, leftData)
	right, _ := newMockMatrix(size, size, rightData)
	out, output := newMockMatrix(size, size, make([]float32, size*size))
	shareMockContext(left, right, out)
	left.ctx.infoSet = true

	left.ctx.isCPU = true
	defer left.ctx.releaseHostWorkerPool()

	err := MatMul(left, right, out)

	require.NoError(t, err)
	require.NotNil(t, left.ctx.hostPool)
	assert.Equal(t, want, output.data)
}

func TestMatMulCPUWithRunnerReturnsPreparationError(t *testing.T) {
	t.Parallel()

	left, storage := newMockMatrix(1, 1, []float32{1})
	right, _ := newMockMatrix(1, 1, []float32{1})
	out, _ := newMockMatrix(1, 1, []float32{0})
	shareMockContext(left, right, out)

	storage.readErr = io.EOF

	err := matMulCPUWithRunner(left, right, out, runWorkRangesSerial)

	require.ErrorIs(t, err, io.EOF)
}
