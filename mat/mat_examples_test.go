//go:build !cgo

package mat_test

import (
	"fmt"
	"os"
	"testing"
	"time"

	"github.com/KEINOS/go-wgpu-mat/mat"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// ============================================================================
//  Examples for documentation (public API tests follows after examples)
// ============================================================================

// Example of basic usage in the README.md. This test ensures that the example
// code compiles and runs without errors.
//
//nolint:varnamelen // example
func Example() {
	panicOnErr := func(err error) {
		if err != nil {
			panic(err)
		}
	}

	// UseGPU (default) or UseCPU
	ctx, err := mat.NewContext(mat.UseGPU)
	panicOnErr(err)

	defer ctx.Release()

	// 2×2 matrices stored on the GPU
	a, err := mat.NewMatrix(ctx, 2, 2)
	panicOnErr(err)

	b, err := mat.NewMatrix(ctx, 2, 2)
	panicOnErr(err)

	c, err := mat.NewMatrix(ctx, 2, 2)
	panicOnErr(err)

	defer a.Release()
	defer b.Release()
	defer c.Release()

	// Upload data (row-major order)
	err = a.Write([]float32{1, 2, 3, 4}) // [[1,2],[3,4]]
	panicOnErr(err)
	err = b.Write([]float32{5, 6, 7, 8}) // [[5,6],[7,8]]
	panicOnErr(err)

	// Compute C = A × B on the GPU
	err = mat.MatMul(a, b, c)
	panicOnErr(err)

	// Read result back to CPU
	data, err := c.Read()
	panicOnErr(err)

	fmt.Println(data)
	// Output:
	// [19 22 43 50]
}

// Example of creating a new matrix for a compute context.
func ExampleNewMatrix() {
	ctx, err := mat.NewContext()
	if err != nil {
		panic(err)
	}
	defer ctx.Release()

	mtx, err := mat.NewMatrix(ctx, 2, 3)
	if err != nil {
		panic(err)
	}
	defer mtx.Release()

	fmt.Printf("Type: %T\n", mtx)
	fmt.Printf("Matrix: %dx%d\n", mtx.Rows, mtx.Cols)
	// Output:
	// Type: *mat.Matrix
	// Matrix: 2x3
}

//nolint:dupl // allow dup for clarity in examples
func ExampleMatMul() {
	ctx, err := mat.NewContext()
	if err != nil {
		panic(err)
	}
	defer ctx.Release()

	leftMatrix, err := mat.NewMatrix(ctx, 2, 2)
	if err != nil {
		panic(err)
	}
	defer leftMatrix.Release()

	rightMatrix, err := mat.NewMatrix(ctx, 2, 2)
	if err != nil {
		panic(err)
	}
	defer rightMatrix.Release()

	out, err := mat.NewMatrix(ctx, 2, 2)
	if err != nil {
		panic(err)
	}
	defer out.Release()

	err = leftMatrix.Write([]float32{1, 2, 3, 4})
	if err != nil {
		panic(err)
	}

	err = rightMatrix.Write([]float32{5, 6, 7, 8})
	if err != nil {
		panic(err)
	}

	err = mat.MatMul(leftMatrix, rightMatrix, out)
	if err != nil {
		panic(err)
	}

	data, err := out.Read()
	if err != nil {
		panic(err)
	}

	fmt.Println(data)
	// Output:
	// [19 22 43 50]
}

//nolint:dupl // allow dup for clarity in examples
func ExampleAdd() {
	ctx, err := mat.NewContext()
	if err != nil {
		panic(err)
	}
	defer ctx.Release()

	leftMatrix, err := mat.NewMatrix(ctx, 2, 2)
	if err != nil {
		panic(err)
	}
	defer leftMatrix.Release()

	rightMatrix, err := mat.NewMatrix(ctx, 2, 2)
	if err != nil {
		panic(err)
	}
	defer rightMatrix.Release()

	out, err := mat.NewMatrix(ctx, 2, 2)
	if err != nil {
		panic(err)
	}
	defer out.Release()

	err = leftMatrix.Write([]float32{1, 2, 3, 4})
	if err != nil {
		panic(err)
	}

	err = rightMatrix.Write([]float32{5, 6, 7, 8})
	if err != nil {
		panic(err)
	}

	err = mat.Add(leftMatrix, rightMatrix, out)
	if err != nil {
		panic(err)
	}

	data, err := out.Read()
	if err != nil {
		panic(err)
	}

	fmt.Println(data)
	// Output:
	// [6 8 10 12]
}

func ExampleScale() {
	ctx, err := mat.NewContext()
	if err != nil {
		panic(err)
	}

	defer ctx.Release()

	sourceMatrix, err := mat.NewMatrix(ctx, 2, 2)
	if err != nil {
		panic(err)
	}

	defer sourceMatrix.Release()

	out, err := mat.NewMatrix(ctx, 2, 2)
	if err != nil {
		panic(err)
	}

	defer out.Release()

	err = sourceMatrix.Write([]float32{1, -2, 3, -4})
	if err != nil {
		panic(err)
	}

	err = mat.Scale(sourceMatrix, 0.5, out)
	if err != nil {
		panic(err)
	}

	data, err := out.Read()
	if err != nil {
		panic(err)
	}

	fmt.Println(data)
	// Output:
	// [0.5 -1 1.5 -2]
}

func ExampleTransp() {
	ctx, err := mat.NewContext()
	if err != nil {
		panic(err)
	}

	defer ctx.Release()

	inputMatrix, err := mat.NewMatrix(ctx, 2, 3)
	if err != nil {
		panic(err)
	}

	defer inputMatrix.Release()

	out, err := mat.NewMatrix(ctx, 3, 2)
	if err != nil {
		panic(err)
	}

	defer out.Release()

	err = inputMatrix.Write([]float32{1, 2, 3, 4, 5, 6})
	if err != nil {
		panic(err)
	}

	err = mat.Transp(inputMatrix, out)
	if err != nil {
		panic(err)
	}

	data, err := out.Read()
	if err != nil {
		panic(err)
	}

	fmt.Println(data)
	// Output:
	// [1 4 2 5 3 6]
}

func ExampleReduceSum() {
	ctx, err := mat.NewContext()
	if err != nil {
		panic(err)
	}

	defer ctx.Release()

	inputMatrix, err := mat.NewMatrix(ctx, 2, 3)
	if err != nil {
		panic(err)
	}

	defer inputMatrix.Release()

	out, err := mat.NewMatrix(ctx, 2, 1)
	if err != nil {
		panic(err)
	}

	defer out.Release()

	err = inputMatrix.Write([]float32{1, 2, 3, 4, 5, 6})
	if err != nil {
		panic(err)
	}

	err = mat.ReduceSum(inputMatrix, out)
	if err != nil {
		panic(err)
	}

	data, err := out.Read()
	if err != nil {
		panic(err)
	}

	fmt.Println(data)
	// Output:
	// [6 15]
}

func ExampleReduceMax() {
	ctx, err := mat.NewContext()
	if err != nil {
		panic(err)
	}

	defer ctx.Release()

	inputMatrix, err := mat.NewMatrix(ctx, 2, 3)
	if err != nil {
		panic(err)
	}

	defer inputMatrix.Release()

	out, err := mat.NewMatrix(ctx, 2, 1)
	if err != nil {
		panic(err)
	}

	defer out.Release()

	err = inputMatrix.Write([]float32{-1, -3, -2, 4, 0, 1})
	if err != nil {
		panic(err)
	}

	err = mat.ReduceMax(inputMatrix, out)
	if err != nil {
		panic(err)
	}

	data, err := out.Read()
	if err != nil {
		panic(err)
	}

	fmt.Println(data)
	// Output:
	// [-1 4]
}

func ExampleSoftmax() {
	ctx, err := mat.NewContext()
	if err != nil {
		panic(err)
	}

	defer ctx.Release()

	inputMatrix, err := mat.NewMatrix(ctx, 1, 3)
	if err != nil {
		panic(err)
	}

	defer inputMatrix.Release()

	out, err := mat.NewMatrix(ctx, 1, 3)
	if err != nil {
		panic(err)
	}

	defer out.Release()

	err = inputMatrix.Write([]float32{1, 2, 3})
	if err != nil {
		panic(err)
	}

	err = mat.Softmax(inputMatrix, out)
	if err != nil {
		panic(err)
	}

	data, err := out.Read()
	if err != nil {
		panic(err)
	}

	fmt.Printf("%.4f %.4f %.4f\n", data[0], data[1], data[2])
	// Output:
	// 0.0900 0.2447 0.6652
}

func ExampleRMSNorm() {
	ctx, err := mat.NewContext()
	if err != nil {
		panic(err)
	}

	defer ctx.Release()

	inputMatrix, err := mat.NewMatrix(ctx, 1, 2)
	if err != nil {
		panic(err)
	}

	defer inputMatrix.Release()

	out, err := mat.NewMatrix(ctx, 1, 2)
	if err != nil {
		panic(err)
	}

	defer out.Release()

	err = inputMatrix.Write([]float32{3, 4})
	if err != nil {
		panic(err)
	}

	err = mat.RMSNorm(inputMatrix, out)
	if err != nil {
		panic(err)
	}

	data, err := out.Read()
	if err != nil {
		panic(err)
	}

	fmt.Printf("%.4f %.4f\n", data[0], data[1])
	// Output:
	// 0.8485 1.1314
}

// ============================================================================
//  Tests for Public API (for internal tests, see mat_unit_test.go)
// ============================================================================

func serializeGPUTest(t *testing.T) {
	t.Helper()

	const lockDirPath = "/tmp/go-wgpu-mat-test.lockdir"

	for {
		err := os.Mkdir(lockDirPath, 0o700)
		if err == nil {
			break
		}

		if os.IsExist(err) {
			time.Sleep(10 * time.Millisecond)

			continue
		}

		require.NoError(t, err)
	}

	t.Cleanup(func() {
		require.NoError(t, os.Remove(lockDirPath))
	})
}

// TestNewContext_smoke verifies NewContext returns a non-nil
// context and Release does not panic.
func TestNewContext_smoke(t *testing.T) {
	t.Parallel()
	serializeGPUTest(t)

	ctx, err := mat.NewContext()
	require.NoError(t, err, "NewContext should succeed")
	require.NotNil(t, ctx)
	require.NotPanics(t, func() { ctx.Release() })
}

func TestNewContext_modes(t *testing.T) {
	t.Parallel()
	serializeGPUTest(t)

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
	serializeGPUTest(t)

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
	serializeGPUTest(t)

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
	serializeGPUTest(t)

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
	serializeGPUTest(t)

	var ctx *mat.Context

	require.NotPanics(t, func() { ctx.Release() })
}

// TestMatrix_Write_length_mismatch verifies that Write returns an
// error when given the wrong number of elements.
func TestMatrix_Write_length_mismatch(t *testing.T) {
	t.Parallel()
	serializeGPUTest(t)

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

func TestMatMul_success(t *testing.T) {
	t.Parallel()
	serializeGPUTest(t)

	ctx, err := mat.NewContext()
	require.NoError(t, err)

	defer ctx.Release()

	leftMatrix, err := mat.NewMatrix(ctx, 2, 3)
	require.NoError(t, err)

	defer leftMatrix.Release()

	rightMatrix, err := mat.NewMatrix(ctx, 3, 2)
	require.NoError(t, err)

	defer rightMatrix.Release()

	out, err := mat.NewMatrix(ctx, 2, 2)
	require.NoError(t, err)

	defer out.Release()

	require.NoError(t, leftMatrix.Write([]float32{1, 2, 3, 4, 5, 6}))
	require.NoError(t, rightMatrix.Write([]float32{7, 8, 9, 10, 11, 12}))

	require.NoError(t, mat.MatMul(leftMatrix, rightMatrix, out))

	got, err := out.Read()
	require.NoError(t, err)

	want := []float32{58, 64, 139, 154}
	for i := range want {
		assert.InDelta(t, want[i], got[i], 1e-6)
	}
}

func TestMatMul_dimensionMismatch(t *testing.T) {
	t.Parallel()
	serializeGPUTest(t)

	ctx, err := mat.NewContext()
	require.NoError(t, err)

	defer ctx.Release()

	leftMatrix, err := mat.NewMatrix(ctx, 2, 3)
	require.NoError(t, err)

	defer leftMatrix.Release()

	rightMatrix, err := mat.NewMatrix(ctx, 2, 2)
	require.NoError(t, err)

	defer rightMatrix.Release()

	out, err := mat.NewMatrix(ctx, 2, 2)
	require.NoError(t, err)

	defer out.Release()

	err = mat.MatMul(leftMatrix, rightMatrix, out)
	require.Error(t, err)
	assert.ErrorContains(t, err, "mat: dimension mismatch")
}

func TestAdd_success(t *testing.T) {
	t.Parallel()
	serializeGPUTest(t)

	ctx, err := mat.NewContext()
	require.NoError(t, err)

	defer ctx.Release()

	leftMatrix, err := mat.NewMatrix(ctx, 2, 2)
	require.NoError(t, err)

	defer leftMatrix.Release()

	rightMatrix, err := mat.NewMatrix(ctx, 2, 2)
	require.NoError(t, err)

	defer rightMatrix.Release()

	out, err := mat.NewMatrix(ctx, 2, 2)
	require.NoError(t, err)

	defer out.Release()

	require.NoError(t, leftMatrix.Write([]float32{1, 2, 3, 4}))
	require.NoError(t, rightMatrix.Write([]float32{10, 20, 30, 40}))

	require.NoError(t, mat.Add(leftMatrix, rightMatrix, out))

	got, err := out.Read()
	require.NoError(t, err)

	want := []float32{11, 22, 33, 44}
	for i := range want {
		assert.InDelta(t, want[i], got[i], 1e-6)
	}
}

func TestAdd_dimensionMismatch(t *testing.T) {
	t.Parallel()
	serializeGPUTest(t)

	ctx, err := mat.NewContext()
	require.NoError(t, err)

	defer ctx.Release()

	leftMatrix, err := mat.NewMatrix(ctx, 2, 3)
	require.NoError(t, err)

	defer leftMatrix.Release()

	rightMatrix, err := mat.NewMatrix(ctx, 2, 2)
	require.NoError(t, err)

	defer rightMatrix.Release()

	out, err := mat.NewMatrix(ctx, 2, 3)
	require.NoError(t, err)

	defer out.Release()

	err = mat.Add(leftMatrix, rightMatrix, out)
	require.Error(t, err)
	assert.ErrorContains(t, err, "mat: dimension mismatch")
}

func TestScale_success(t *testing.T) {
	t.Parallel()
	serializeGPUTest(t)

	ctx, err := mat.NewContext()
	require.NoError(t, err)

	defer ctx.Release()

	sourceMatrix, err := mat.NewMatrix(ctx, 2, 2)
	require.NoError(t, err)

	defer sourceMatrix.Release()

	out, err := mat.NewMatrix(ctx, 2, 2)
	require.NoError(t, err)

	defer out.Release()

	require.NoError(t, sourceMatrix.Write([]float32{1, -2, 3, -4}))
	require.NoError(t, mat.Scale(sourceMatrix, 0.5, out))

	got, err := out.Read()
	require.NoError(t, err)

	want := []float32{0.5, -1, 1.5, -2}
	for i := range want {
		assert.InDelta(t, want[i], got[i], 1e-6)
	}
}

func TestScale_dimensionMismatch(t *testing.T) {
	t.Parallel()
	serializeGPUTest(t)

	ctx, err := mat.NewContext()
	require.NoError(t, err)

	defer ctx.Release()

	sourceMatrix, err := mat.NewMatrix(ctx, 2, 2)
	require.NoError(t, err)

	defer sourceMatrix.Release()

	out, err := mat.NewMatrix(ctx, 1, 4)
	require.NoError(t, err)

	defer out.Release()

	err = mat.Scale(sourceMatrix, 2, out)
	require.Error(t, err)
	assert.ErrorContains(t, err, "mat: dimension mismatch")
}

func TestTransp_success(t *testing.T) {
	t.Parallel()
	serializeGPUTest(t)

	ctx, err := mat.NewContext()
	require.NoError(t, err)

	defer ctx.Release()

	inputMatrix, err := mat.NewMatrix(ctx, 2, 3)
	require.NoError(t, err)

	defer inputMatrix.Release()

	out, err := mat.NewMatrix(ctx, 3, 2)
	require.NoError(t, err)

	defer out.Release()

	require.NoError(t, inputMatrix.Write([]float32{1, 2, 3, 4, 5, 6}))
	require.NoError(t, mat.Transp(inputMatrix, out))

	got, err := out.Read()
	require.NoError(t, err)

	want := []float32{1, 4, 2, 5, 3, 6}
	for i := range want {
		assert.InDelta(t, want[i], got[i], 1e-6)
	}
}

func TestTransp_dimensionMismatch(t *testing.T) {
	t.Parallel()
	serializeGPUTest(t)

	ctx, err := mat.NewContext()
	require.NoError(t, err)

	defer ctx.Release()

	inputMatrix, err := mat.NewMatrix(ctx, 2, 3)
	require.NoError(t, err)

	defer inputMatrix.Release()

	out, err := mat.NewMatrix(ctx, 2, 3)
	require.NoError(t, err)

	defer out.Release()

	err = mat.Transp(inputMatrix, out)
	require.Error(t, err)
	assert.ErrorContains(t, err, "mat: dimension mismatch")
}

func TestReduceSum_success(t *testing.T) {
	t.Parallel()
	serializeGPUTest(t)

	ctx, err := mat.NewContext()
	require.NoError(t, err)

	defer ctx.Release()

	inputMatrix, err := mat.NewMatrix(ctx, 2, 3)
	require.NoError(t, err)

	defer inputMatrix.Release()

	out, err := mat.NewMatrix(ctx, 2, 1)
	require.NoError(t, err)

	defer out.Release()

	require.NoError(t, inputMatrix.Write([]float32{1, 2, 3, 4, 5, 6}))
	require.NoError(t, mat.ReduceSum(inputMatrix, out))

	got, err := out.Read()
	require.NoError(t, err)

	want := []float32{6, 15}
	for i := range want {
		assert.InDelta(t, want[i], got[i], 1e-6)
	}
}

func TestReduceMax_success(t *testing.T) {
	t.Parallel()
	serializeGPUTest(t)

	ctx, err := mat.NewContext()
	require.NoError(t, err)

	defer ctx.Release()

	inputMatrix, err := mat.NewMatrix(ctx, 2, 3)
	require.NoError(t, err)

	defer inputMatrix.Release()

	out, err := mat.NewMatrix(ctx, 2, 1)
	require.NoError(t, err)

	defer out.Release()

	require.NoError(t, inputMatrix.Write([]float32{-1, -3, -2, 4, 0, 1}))
	require.NoError(t, mat.ReduceMax(inputMatrix, out))

	got, err := out.Read()
	require.NoError(t, err)

	want := []float32{-1, 4}
	for i := range want {
		assert.InDelta(t, want[i], got[i], 1e-6)
	}
}

func TestReduce_dimensionMismatch(t *testing.T) {
	t.Parallel()
	serializeGPUTest(t)

	ctx, err := mat.NewContext()
	require.NoError(t, err)

	defer ctx.Release()

	inputMatrix, err := mat.NewMatrix(ctx, 2, 3)
	require.NoError(t, err)

	defer inputMatrix.Release()

	out, err := mat.NewMatrix(ctx, 2, 2)
	require.NoError(t, err)

	defer out.Release()

	err = mat.ReduceSum(inputMatrix, out)
	require.Error(t, err)
	require.ErrorContains(t, err, "mat: dimension mismatch")

	err = mat.ReduceMax(inputMatrix, out)
	require.Error(t, err)
	require.ErrorContains(t, err, "mat: dimension mismatch")
}

func TestSoftmax_success(t *testing.T) {
	t.Parallel()
	serializeGPUTest(t)

	ctx, err := mat.NewContext()
	require.NoError(t, err)

	defer ctx.Release()

	inputMatrix, err := mat.NewMatrix(ctx, 2, 3)
	require.NoError(t, err)

	defer inputMatrix.Release()

	out, err := mat.NewMatrix(ctx, 2, 3)
	require.NoError(t, err)

	defer out.Release()

	require.NoError(t, inputMatrix.Write([]float32{1, 2, 3, 1000, 1000, 1000}))
	require.NoError(t, mat.Softmax(inputMatrix, out))

	got, err := out.Read()
	require.NoError(t, err)

	firstRowSum := got[0] + got[1] + got[2]
	secondRowSum := got[3] + got[4] + got[5]

	assert.InDelta(t, 1.0, firstRowSum, 1e-4)
	assert.InDelta(t, 1.0, secondRowSum, 1e-4)

	assert.Greater(t, got[2], got[1])
	assert.Greater(t, got[1], got[0])

	assert.InDelta(t, 1.0/3.0, got[3], 1e-4)
	assert.InDelta(t, 1.0/3.0, got[4], 1e-4)
	assert.InDelta(t, 1.0/3.0, got[5], 1e-4)
}

func TestSoftmax_dimensionMismatch(t *testing.T) {
	t.Parallel()
	serializeGPUTest(t)

	ctx, err := mat.NewContext()
	require.NoError(t, err)

	defer ctx.Release()

	inputMatrix, err := mat.NewMatrix(ctx, 2, 3)
	require.NoError(t, err)

	defer inputMatrix.Release()

	out, err := mat.NewMatrix(ctx, 3, 2)
	require.NoError(t, err)

	defer out.Release()

	err = mat.Softmax(inputMatrix, out)
	require.Error(t, err)
	require.ErrorContains(t, err, "mat: dimension mismatch")
}

func TestRMSNorm_success(t *testing.T) {
	t.Parallel()
	serializeGPUTest(t)

	ctx, err := mat.NewContext()
	require.NoError(t, err)

	defer ctx.Release()

	inputMatrix, err := mat.NewMatrix(ctx, 2, 2)
	require.NoError(t, err)

	defer inputMatrix.Release()

	out, err := mat.NewMatrix(ctx, 2, 2)
	require.NoError(t, err)

	defer out.Release()

	require.NoError(t, inputMatrix.Write([]float32{3, 4, 0, 0}))
	require.NoError(t, mat.RMSNorm(inputMatrix, out))

	got, err := out.Read()
	require.NoError(t, err)

	assert.InDelta(t, 0.8485, got[0], 1e-4)
	assert.InDelta(t, 1.1314, got[1], 1e-4)
	assert.InDelta(t, 0, got[2], 1e-6)
	assert.InDelta(t, 0, got[3], 1e-6)
}

func TestRMSNorm_dimensionMismatch(t *testing.T) {
	t.Parallel()
	serializeGPUTest(t)

	ctx, err := mat.NewContext()
	require.NoError(t, err)

	defer ctx.Release()

	inputMatrix, err := mat.NewMatrix(ctx, 2, 2)
	require.NoError(t, err)

	defer inputMatrix.Release()

	out, err := mat.NewMatrix(ctx, 1, 4)
	require.NoError(t, err)

	defer out.Release()

	err = mat.RMSNorm(inputMatrix, out)
	require.Error(t, err)
	require.ErrorContains(t, err, "mat: dimension mismatch")
}
