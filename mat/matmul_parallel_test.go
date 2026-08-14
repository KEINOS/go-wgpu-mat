package mat

import (
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
