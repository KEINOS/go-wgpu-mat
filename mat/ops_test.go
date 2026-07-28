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
		rows:     rows,
		cols:     cols,
		buf:      new(wgpu.Buffer),
		ctx:      new(Context),
		released: atomic.Uint32{},
		deps:     deps,
	}

	return matrix, storage
}

func shareMockContext(first *Matrix, matrices ...*Matrix) {
	for _, matrix := range matrices {
		matrix.ctx = first.ctx
	}
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
		rows:     0,
		cols:     0,
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

	matrix, _ = newMockMatrix(1, 1, []float32{1})
	matrix.released.Store(1)
	err = validateMatrixInitialized("input", matrix)
	require.Error(t, err)
	require.ErrorContains(t, err, "input is released")

	matrix, _ = newMockMatrix(1, 1, []float32{1})
	matrix.ctx.released.Store(1)
	err = validateMatrixInitialized("input", matrix)
	require.Error(t, err)
	require.ErrorContains(t, err, "context is released")
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
	shareMockContext(leftMatrix, rightMatrix, outMatrix)

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
	shareMockContext(leftMatrix, rightMatrix, outMatrix)
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

	mismatchedOut, _ := newMockMatrix(2, 1, []float32{0, 0})
	shareMockContext(leftMatrix, mismatchedOut)
	err = runBinaryElementwise(leftMatrix, rightMatrix, mismatchedOut,
		func(leftValue, rightValue float32) float32 {
			return leftValue + rightValue
		})
	require.Error(t, err)
	require.ErrorContains(t, err, "dimension mismatch")
}

func TestRunUnaryElementwiseReadAndWriteErrors(t *testing.T) {
	t.Parallel()

	inputMatrix, inputStorage := newMockMatrix(1, 2, []float32{1, 2})
	outMatrix, outStorage := newMockMatrix(1, 2, []float32{0, 0})
	shareMockContext(inputMatrix, outMatrix)

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

func TestAddDispatchesWithoutHostIO(t *testing.T) { //nolint:dupl // Each operation guards its own host-I/O contract.
	t.Parallel()

	leftMatrix, leftStorage := newMockMatrix(2, 2, []float32{1, 2, 3, 4})
	rightMatrix, rightStorage := newMockMatrix(2, 2, []float32{5, 6, 7, 8})
	outMatrix, outStorage := newMockMatrix(2, 2, []float32{0, 0, 0, 0})
	rightMatrix.ctx = leftMatrix.ctx
	outMatrix.ctx = leftMatrix.ctx
	leftStorage.readErr = io.EOF
	rightStorage.readErr = io.EOF
	outStorage.writeErr = io.EOF

	dispatched := false
	err := add(leftMatrix, rightMatrix, outMatrix, addDeps{
		dispatch: func(left, right, out *Matrix) error {
			dispatched = true

			assert.Same(t, leftMatrix, left)
			assert.Same(t, rightMatrix, right)
			assert.Same(t, outMatrix, out)

			return nil
		},
	})

	require.NoError(t, err)
	assert.True(t, dispatched)
}

func TestAddRejectsAliasAndDifferentContext(t *testing.T) {
	t.Parallel()

	leftMatrix, _ := newMockMatrix(1, 2, []float32{1, 2})
	rightMatrix, _ := newMockMatrix(1, 2, []float32{3, 4})

	err := add(leftMatrix, rightMatrix, leftMatrix, addDeps{
		dispatch: func(*Matrix, *Matrix, *Matrix) error {
			t.Fatal("invalid add must not dispatch")

			return nil
		},
	})
	require.ErrorIs(t, err, ErrContextMismatch)

	shareMockContext(leftMatrix, rightMatrix)
	err = add(leftMatrix, rightMatrix, leftMatrix, addDeps{dispatch: nil})
	require.ErrorIs(t, err, ErrAliasedOutput)
}

func TestAddDispatchError(t *testing.T) {
	t.Parallel()

	leftMatrix, _ := newMockMatrix(1, 1, []float32{1})
	rightMatrix, _ := newMockMatrix(1, 1, []float32{2})
	outMatrix, _ := newMockMatrix(1, 1, []float32{0})
	rightMatrix.ctx = leftMatrix.ctx
	outMatrix.ctx = leftMatrix.ctx

	err := add(leftMatrix, rightMatrix, outMatrix, addDeps{
		dispatch: func(*Matrix, *Matrix, *Matrix) error { return io.EOF },
	})
	require.ErrorContains(t, err, "failed to dispatch add")
}

func TestAddValidationErrors(t *testing.T) {
	t.Parallel()

	matrix, _ := newMockMatrix(1, 1, []float32{1})

	err := add(nil, matrix, matrix, addDeps{dispatch: nil})
	require.ErrorContains(t, err, "left is not initialized")

	err = add(matrix, nil, matrix, addDeps{dispatch: nil})
	require.ErrorContains(t, err, "right is not initialized")

	err = add(matrix, matrix, nil, addDeps{dispatch: nil})
	require.ErrorContains(t, err, "out is not initialized")

	left, _ := newMockMatrix(0, 1, nil)
	right, _ := newMockMatrix(0, 1, nil)
	out, _ := newMockMatrix(0, 1, nil)
	right.ctx = left.ctx
	out.ctx = left.ctx
	err = add(left, right, out, addDeps{
		dispatch: func(*Matrix, *Matrix, *Matrix) error {
			t.Fatal("invalid dimensions must not dispatch")

			return nil
		},
	})
	require.ErrorIs(t, err, ErrInvalidState)
}

func TestRunRowReductionReadAndWriteErrors(t *testing.T) {
	t.Parallel()

	inputMatrix, inputStorage := newMockMatrix(1, 2, []float32{1, 2})
	outMatrix, outStorage := newMockMatrix(1, 1, []float32{0})
	shareMockContext(inputMatrix, outMatrix)

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

func TestMatMulDispatchesWithoutHostIO(t *testing.T) { //nolint:dupl // Each operation guards its own host-I/O contract.
	t.Parallel()

	leftMatrix, leftStorage := newMockMatrix(2, 2, []float32{1, 2, 3, 4})
	rightMatrix, rightStorage := newMockMatrix(2, 2, []float32{5, 6, 7, 8})
	outMatrix, outStorage := newMockMatrix(2, 2, []float32{0, 0, 0, 0})
	rightMatrix.ctx = leftMatrix.ctx
	outMatrix.ctx = leftMatrix.ctx

	// Host I/O must not be used by matrix multiplication. These errors make any
	// accidental Read or Write call fail the test.
	leftStorage.readErr = io.EOF
	rightStorage.readErr = io.EOF
	outStorage.writeErr = io.EOF

	dispatched := false
	err := matMul(leftMatrix, rightMatrix, outMatrix, matMulDeps{
		dispatch: func(left, right, out *Matrix) error {
			dispatched = true

			assert.Same(t, leftMatrix, left)
			assert.Same(t, rightMatrix, right)
			assert.Same(t, outMatrix, out)

			return nil
		},
	})
	require.NoError(t, err)
	assert.True(t, dispatched)
}

func TestMatMulDispatchError(t *testing.T) {
	t.Parallel()

	leftMatrix, _ := newMockMatrix(2, 2, []float32{1, 2, 3, 4})
	rightMatrix, _ := newMockMatrix(2, 2, []float32{5, 6, 7, 8})
	outMatrix, _ := newMockMatrix(2, 2, []float32{0, 0, 0, 0})
	rightMatrix.ctx = leftMatrix.ctx
	outMatrix.ctx = leftMatrix.ctx

	err := matMul(leftMatrix, rightMatrix, outMatrix, matMulDeps{
		dispatch: func(*Matrix, *Matrix, *Matrix) error { return io.EOF },
	})
	require.Error(t, err)
	require.ErrorContains(t, err, "failed to dispatch matmul")
}

func TestMatMulValidationErrors(t *testing.T) { //nolint:funlen // Validation cases are clearest together.
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

	rightMatrix.ctx = leftMatrix.ctx
	outMatrix.ctx = new(Context)
	err = matMul(leftMatrix, rightMatrix, outMatrix, matMulDeps{
		dispatch: func(*Matrix, *Matrix, *Matrix) error {
			t.Fatal("dispatch must not run for matrices from different contexts")

			return nil
		},
	})
	require.Error(t, err)
	require.ErrorContains(t, err, "matrices must use the same context")

	outMatrix.ctx = leftMatrix.ctx
	err = matMul(leftMatrix, rightMatrix, leftMatrix, matMulDeps{
		dispatch: func(*Matrix, *Matrix, *Matrix) error {
			t.Fatal("dispatch must not run when output aliases an input")

			return nil
		},
	})
	require.Error(t, err)
	require.ErrorContains(t, err, "out must not alias an input")

	leftMatrix.rows = math.MaxUint32 + 1
	rightMatrix.rows = leftMatrix.cols
	outMatrix.rows = leftMatrix.rows
	err = matMul(leftMatrix, rightMatrix, outMatrix, matMulDeps{
		dispatch: func(*Matrix, *Matrix, *Matrix) error {
			t.Fatal("dispatch must not run for dimensions above uint32")

			return nil
		},
	})
	require.Error(t, err)
	require.ErrorContains(t, err, "matrix dimensions exceed GPU kernel limits")

	leftMatrix, _ = newMockMatrix(9, 1, make([]float32, 9))
	rightMatrix, _ = newMockMatrix(1, 1, []float32{1})
	outMatrix, _ = newMockMatrix(9, 1, make([]float32, 9))
	rightMatrix.ctx = leftMatrix.ctx
	outMatrix.ctx = leftMatrix.ctx
	leftMatrix.ctx.limits.MaxComputeWorkgroupsPerDimension = 1
	err = matMul(leftMatrix, rightMatrix, outMatrix, matMulDeps{
		dispatch: func(*Matrix, *Matrix, *Matrix) error {
			t.Fatal("dispatch must not exceed device workgroup limits")

			return nil
		},
	})
	require.Error(t, err)
	require.ErrorContains(t, err, "matmul dispatch exceeds device workgroup limits")
}

func TestTranspReadAndWriteErrors(t *testing.T) {
	t.Parallel()

	inputMatrix, inputStorage := newMockMatrix(2, 3, []float32{1, 2, 3, 4, 5, 6})
	outMatrix, outStorage := newMockMatrix(3, 2, []float32{0, 0, 0, 0, 0, 0})
	shareMockContext(inputMatrix, outMatrix)

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
	shareMockContext(inputMatrix, outMatrix)

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
	shareMockContext(inputMatrix, outMatrix)

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
