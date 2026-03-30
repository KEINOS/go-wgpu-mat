//go:build !cgo

package mat

import (
	"encoding/binary"
	"io"
	"math"
	"sync/atomic"
	"testing"

	"github.com/gogpu/wgpu"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type mockMatrixIO struct {
	data     []float32
	readErr  error
	writeErr error
}

func newMockMatrix(rows, cols int, values []float32) (*Matrix, *mockMatrixIO) {
	storage := &mockMatrixIO{
		data:     append([]float32(nil), values...),
		readErr:  nil,
		writeErr: nil,
	}

	deps := matrixDeps{
		createBuffer: func(*Context, *wgpu.BufferDescriptor) (*wgpu.Buffer, error) {
			return new(wgpu.Buffer), nil
		},
		writeBuffer: func(_ *Context, _ *wgpu.Buffer, raw []byte) error {
			if storage.writeErr != nil {
				return storage.writeErr
			}

			storage.data = decodeFloat32(raw)

			return nil
		},
		readBuffer: func(_ *Context, _ *wgpu.Buffer, raw []byte) error {
			if storage.readErr != nil {
				return storage.readErr
			}

			encoded := encodeFloat32(storage.data)
			copy(raw, encoded)

			return nil
		},
	}

	matrix := &Matrix{
		Rows: rows,
		Cols: cols,
		buf:  new(wgpu.Buffer),
		ctx:  new(Context),
		released: atomic.Uint32{},
		deps: deps,
	}

	return matrix, storage
}

func encodeFloat32(values []float32) []byte {
	raw := make([]byte, len(values)*bytesPerFloat32Int)
	for idx, value := range values {
		binary.LittleEndian.PutUint32(raw[idx*bytesPerFloat32Int:], math.Float32bits(value))
	}

	return raw
}

func decodeFloat32(raw []byte) []float32 {
	values := make([]float32, len(raw)/bytesPerFloat32Int)
	for idx := range values {
		values[idx] = math.Float32frombits(binary.LittleEndian.Uint32(raw[idx*bytesPerFloat32Int:]))
	}

	return values
}

func TestValidateMatrixInitialized(t *testing.T) {
	t.Parallel()

	err := validateMatrixInitialized("input", nil)
	require.Error(t, err)
	require.ErrorContains(t, err, "input is not initialized")

	matrix := &Matrix{
		Rows:     0,
		Cols:     0,
		buf:      nil,
		ctx:      nil,
		released: atomic.Uint32{},
		deps: matrixDeps{
			createBuffer: nil,
			writeBuffer:  nil,
			readBuffer:   nil,
		},
	}
	err = validateMatrixInitialized("input", matrix)
	require.Error(t, err)
	require.ErrorContains(t, err, "input is not initialized")
}

func TestValidateSameShapeOutMismatch(t *testing.T) {
	t.Parallel()

	leftMatrix, _ := newMockMatrix(2, 2, []float32{1, 2, 3, 4})
	rightMatrix, _ := newMockMatrix(2, 2, []float32{5, 6, 7, 8})
	outMatrix, _ := newMockMatrix(1, 4, []float32{0, 0, 0, 0})

	err := validateSameShape(leftMatrix, rightMatrix, outMatrix)
	require.Error(t, err)
	require.ErrorContains(t, err, "dimension mismatch")
}

func TestRunBinaryElementwiseReadAndWriteErrors(t *testing.T) {
	t.Parallel()

	leftMatrix, leftStorage := newMockMatrix(1, 2, []float32{1, 2})
	rightMatrix, rightStorage := newMockMatrix(1, 2, []float32{3, 4})
	outMatrix, outStorage := newMockMatrix(1, 2, []float32{0, 0})

	leftStorage.readErr = io.EOF
	err := runBinaryElementwise(leftMatrix, rightMatrix, outMatrix,
		func(leftValue, rightValue float32) float32 {
			return leftValue + rightValue
		})
	require.Error(t, err)
	require.ErrorContains(t, err, "failed to read left")

	leftStorage.readErr = nil
	rightStorage.readErr = io.EOF
	err = runBinaryElementwise(leftMatrix, rightMatrix, outMatrix,
		func(leftValue, rightValue float32) float32 {
			return leftValue + rightValue
		})
	require.Error(t, err)
	require.ErrorContains(t, err, "failed to read right")

	rightStorage.readErr = nil
	outStorage.writeErr = io.EOF
	err = runBinaryElementwise(leftMatrix, rightMatrix, outMatrix,
		func(leftValue, rightValue float32) float32 {
			return leftValue + rightValue
		})
	require.Error(t, err)
	require.ErrorContains(t, err, "failed to write out")
}

func TestRunBinaryElementwiseValidationErrors(t *testing.T) {
	t.Parallel()

	rightMatrix, _ := newMockMatrix(1, 2, []float32{3, 4})
	outMatrix, _ := newMockMatrix(1, 2, []float32{0, 0})

	err := runBinaryElementwise(nil, rightMatrix, outMatrix,
		func(leftValue, rightValue float32) float32 {
			return leftValue + rightValue
		})
	require.Error(t, err)
	require.ErrorContains(t, err, "left is not initialized")

	leftMatrix, _ := newMockMatrix(1, 2, []float32{1, 2})
	err = runBinaryElementwise(leftMatrix, nil, outMatrix,
		func(leftValue, rightValue float32) float32 {
			return leftValue + rightValue
		})
	require.Error(t, err)
	require.ErrorContains(t, err, "right is not initialized")

	err = runBinaryElementwise(leftMatrix, rightMatrix, nil,
		func(leftValue, rightValue float32) float32 {
			return leftValue + rightValue
		})
	require.Error(t, err)
	require.ErrorContains(t, err, "out is not initialized")
}

func TestRunUnaryElementwiseReadAndWriteErrors(t *testing.T) {
	t.Parallel()

	inputMatrix, inputStorage := newMockMatrix(1, 2, []float32{1, 2})
	outMatrix, outStorage := newMockMatrix(1, 2, []float32{0, 0})

	inputStorage.readErr = io.EOF
	err := runUnaryElementwise(inputMatrix, outMatrix, func(value float32) float32 {
		return value
	})
	require.Error(t, err)
	require.ErrorContains(t, err, "failed to read input")

	inputStorage.readErr = nil
	outStorage.writeErr = io.EOF
	err = runUnaryElementwise(inputMatrix, outMatrix, func(value float32) float32 {
		return value
	})
	require.Error(t, err)
	require.ErrorContains(t, err, "failed to write out")
}

func TestRunUnaryElementwiseValidationErrors(t *testing.T) {
	t.Parallel()

	outMatrix, _ := newMockMatrix(1, 2, []float32{0, 0})
	err := runUnaryElementwise(nil, outMatrix, func(value float32) float32 {
		return value
	})
	require.Error(t, err)
	require.ErrorContains(t, err, "input is not initialized")

	inputMatrix, _ := newMockMatrix(1, 2, []float32{1, 2})
	err = runUnaryElementwise(inputMatrix, nil, func(value float32) float32 {
		return value
	})
	require.Error(t, err)
	require.ErrorContains(t, err, "out is not initialized")
}

func TestRunRowReductionReadAndWriteErrors(t *testing.T) {
	t.Parallel()

	inputMatrix, inputStorage := newMockMatrix(1, 2, []float32{1, 2})
	outMatrix, outStorage := newMockMatrix(1, 1, []float32{0})

	inputStorage.readErr = io.EOF
	err := runRowReduction(inputMatrix, outMatrix, 0,
		func(accumulator, value float32) float32 {
			return accumulator + value
		})
	require.Error(t, err)
	require.ErrorContains(t, err, "failed to read input")

	inputStorage.readErr = nil
	outStorage.writeErr = io.EOF
	err = runRowReduction(inputMatrix, outMatrix, 0,
		func(accumulator, value float32) float32 {
			return accumulator + value
		})
	require.Error(t, err)
	require.ErrorContains(t, err, "failed to write out")
}

func TestRunRowReductionValidationErrors(t *testing.T) {
	t.Parallel()

	outMatrix, _ := newMockMatrix(1, 1, []float32{0})
	err := runRowReduction(nil, outMatrix, 0,
		func(accumulator, value float32) float32 {
			return accumulator + value
		})
	require.Error(t, err)
	require.ErrorContains(t, err, "input is not initialized")

	inputMatrix, _ := newMockMatrix(1, 2, []float32{1, 2})
	err = runRowReduction(inputMatrix, nil, 0,
		func(accumulator, value float32) float32 {
			return accumulator + value
		})
	require.Error(t, err)
	require.ErrorContains(t, err, "out is not initialized")
}

func TestMatMulReadAndWriteErrors(t *testing.T) {
	t.Parallel()

	leftMatrix, leftStorage := newMockMatrix(2, 2, []float32{1, 2, 3, 4})
	rightMatrix, rightStorage := newMockMatrix(2, 2, []float32{5, 6, 7, 8})
	outMatrix, outStorage := newMockMatrix(2, 2, []float32{0, 0, 0, 0})

	leftStorage.readErr = io.EOF
	err := MatMul(leftMatrix, rightMatrix, outMatrix)
	require.Error(t, err)
	require.ErrorContains(t, err, "failed to read left")

	leftStorage.readErr = nil
	rightStorage.readErr = io.EOF
	err = MatMul(leftMatrix, rightMatrix, outMatrix)
	require.Error(t, err)
	require.ErrorContains(t, err, "failed to read right")

	rightStorage.readErr = nil
	outStorage.writeErr = io.EOF
	err = MatMul(leftMatrix, rightMatrix, outMatrix)
	require.Error(t, err)
	require.ErrorContains(t, err, "failed to write out")
}

func TestMatMulValidationErrors(t *testing.T) {
	t.Parallel()

	rightMatrix, _ := newMockMatrix(2, 2, []float32{5, 6, 7, 8})
	outMatrix, _ := newMockMatrix(2, 2, []float32{0, 0, 0, 0})

	err := MatMul(nil, rightMatrix, outMatrix)
	require.Error(t, err)
	require.ErrorContains(t, err, "left is not initialized")

	leftMatrix, _ := newMockMatrix(2, 2, []float32{1, 2, 3, 4})
	err = MatMul(leftMatrix, nil, outMatrix)
	require.Error(t, err)
	require.ErrorContains(t, err, "right is not initialized")

	err = MatMul(leftMatrix, rightMatrix, nil)
	require.Error(t, err)
	require.ErrorContains(t, err, "out is not initialized")
}

func TestTranspReadAndWriteErrors(t *testing.T) {
	t.Parallel()

	inputMatrix, inputStorage := newMockMatrix(2, 3, []float32{1, 2, 3, 4, 5, 6})
	outMatrix, outStorage := newMockMatrix(3, 2, []float32{0, 0, 0, 0, 0, 0})

	inputStorage.readErr = io.EOF
	err := Transp(inputMatrix, outMatrix)
	require.Error(t, err)
	require.ErrorContains(t, err, "failed to read input")

	inputStorage.readErr = nil
	outStorage.writeErr = io.EOF
	err = Transp(inputMatrix, outMatrix)
	require.Error(t, err)
	require.ErrorContains(t, err, "failed to write out")
}

func TestTranspValidationErrors(t *testing.T) {
	t.Parallel()

	outMatrix, _ := newMockMatrix(3, 2, []float32{0, 0, 0, 0, 0, 0})
	err := Transp(nil, outMatrix)
	require.Error(t, err)
	require.ErrorContains(t, err, "input is not initialized")

	inputMatrix, _ := newMockMatrix(2, 3, []float32{1, 2, 3, 4, 5, 6})
	err = Transp(inputMatrix, nil)
	require.Error(t, err)
	require.ErrorContains(t, err, "out is not initialized")
}

func TestSoftmaxReadAndWriteErrors(t *testing.T) {
	t.Parallel()

	inputMatrix, inputStorage := newMockMatrix(1, 3, []float32{1, 2, 3})
	outMatrix, outStorage := newMockMatrix(1, 3, []float32{0, 0, 0})

	inputStorage.readErr = io.EOF
	err := Softmax(inputMatrix, outMatrix)
	require.Error(t, err)
	require.ErrorContains(t, err, "failed to read input")

	inputStorage.readErr = nil
	outStorage.writeErr = io.EOF
	err = Softmax(inputMatrix, outMatrix)
	require.Error(t, err)
	require.ErrorContains(t, err, "failed to write out")
}

func TestSoftmaxValidationErrors(t *testing.T) {
	t.Parallel()

	outMatrix, _ := newMockMatrix(1, 3, []float32{0, 0, 0})
	err := Softmax(nil, outMatrix)
	require.Error(t, err)
	require.ErrorContains(t, err, "input is not initialized")

	inputMatrix, _ := newMockMatrix(1, 3, []float32{1, 2, 3})
	err = Softmax(inputMatrix, nil)
	require.Error(t, err)
	require.ErrorContains(t, err, "out is not initialized")
}

func TestRMSNormReadAndWriteErrors(t *testing.T) {
	t.Parallel()

	inputMatrix, inputStorage := newMockMatrix(1, 2, []float32{3, 4})
	outMatrix, outStorage := newMockMatrix(1, 2, []float32{0, 0})

	inputStorage.readErr = io.EOF
	err := RMSNorm(inputMatrix, outMatrix)
	require.Error(t, err)
	require.ErrorContains(t, err, "failed to read input")

	inputStorage.readErr = nil
	outStorage.writeErr = io.EOF
	err = RMSNorm(inputMatrix, outMatrix)
	require.Error(t, err)
	require.ErrorContains(t, err, "failed to write out")
}

func TestRMSNormValidationErrors(t *testing.T) {
	t.Parallel()

	outMatrix, _ := newMockMatrix(1, 2, []float32{0, 0})
	err := RMSNorm(nil, outMatrix)
	require.Error(t, err)
	require.ErrorContains(t, err, "input is not initialized")

	inputMatrix, _ := newMockMatrix(1, 2, []float32{3, 4})
	err = RMSNorm(inputMatrix, nil)
	require.Error(t, err)
	require.ErrorContains(t, err, "out is not initialized")
}

func TestApplySoftmaxRowAndRMSNormRow(t *testing.T) {
	t.Parallel()

	softmaxInput := []float32{1, 2, 3}
	softmaxOutput := make([]float32, 3)
	applySoftmaxRow(softmaxInput, softmaxOutput, 0, 3)
	assert.InDelta(t, 1.0, softmaxOutput[0]+softmaxOutput[1]+softmaxOutput[2], 1e-5)
	assert.Greater(t, softmaxOutput[2], softmaxOutput[1])

	rmsInput := []float32{3, 4}
	rmsOutput := make([]float32, 2)
	applyRMSNormRow(rmsInput, rmsOutput, 0, 2)
	assert.InDelta(t, 0.8485, rmsOutput[0], 1e-4)
	assert.InDelta(t, 1.1314, rmsOutput[1], 1e-4)
}
