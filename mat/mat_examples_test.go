//go:build !cgo

package mat_test

import (
	"fmt"

	"github.com/KEINOS/go-wgpu-mat/mat"
)

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

func ExampleMatMul() {
	ctx, leftMatrix, rightMatrix, out := mustCreate2x2ExampleMatrices()
	defer ctx.Release()
	defer leftMatrix.Release()
	defer rightMatrix.Release()
	defer out.Release()

	mustWrite2x2Inputs(leftMatrix, rightMatrix)
	must(mat.MatMul(leftMatrix, rightMatrix, out))

	data, err := out.Read()
	must(err)

	fmt.Println(data)
	// Output:
	// [19 22 43 50]
}

func ExampleAdd() {
	ctx, leftMatrix, rightMatrix, out := mustCreate2x2ExampleMatrices()
	defer ctx.Release()
	defer leftMatrix.Release()
	defer rightMatrix.Release()
	defer out.Release()

	mustWrite2x2Inputs(leftMatrix, rightMatrix)
	must(mat.Add(leftMatrix, rightMatrix, out))

	data, err := out.Read()
	must(err)

	fmt.Println(data)
	// Output:
	// [6 8 10 12]
}

func ExampleScale() {
	ctx, err := mat.NewContext()
	must(err)

	defer ctx.Release()

	sourceMatrix, err := mat.NewMatrix(ctx, 2, 2)
	must(err)

	defer sourceMatrix.Release()

	out, err := mat.NewMatrix(ctx, 2, 2)
	must(err)

	defer out.Release()

	must(sourceMatrix.Write([]float32{1, -2, 3, -4}))
	must(mat.Scale(sourceMatrix, 0.5, out))

	data, err := out.Read()
	must(err)

	fmt.Println(data)
	// Output:
	// [0.5 -1 1.5 -2]
}

func ExampleTransp() {
	ctx, err := mat.NewContext()
	must(err)

	defer ctx.Release()

	inputMatrix, err := mat.NewMatrix(ctx, 2, 3)
	must(err)

	defer inputMatrix.Release()

	out, err := mat.NewMatrix(ctx, 3, 2)
	must(err)

	defer out.Release()

	must(inputMatrix.Write([]float32{1, 2, 3, 4, 5, 6}))
	must(mat.Transp(inputMatrix, out))

	data, err := out.Read()
	must(err)

	fmt.Println(data)
	// Output:
	// [1 4 2 5 3 6]
}

func ExampleReduceSum() {
	ctx, err := mat.NewContext()
	must(err)

	defer ctx.Release()

	inputMatrix, err := mat.NewMatrix(ctx, 2, 3)
	must(err)

	defer inputMatrix.Release()

	out, err := mat.NewMatrix(ctx, 2, 1)
	must(err)

	defer out.Release()

	must(inputMatrix.Write([]float32{1, 2, 3, 4, 5, 6}))
	must(mat.ReduceSum(inputMatrix, out))

	data, err := out.Read()
	must(err)

	fmt.Println(data)
	// Output:
	// [6 15]
}

func ExampleReduceMax() {
	ctx, err := mat.NewContext()
	must(err)

	defer ctx.Release()

	inputMatrix, err := mat.NewMatrix(ctx, 2, 3)
	must(err)

	defer inputMatrix.Release()

	out, err := mat.NewMatrix(ctx, 2, 1)
	must(err)

	defer out.Release()

	must(inputMatrix.Write([]float32{-1, -3, -2, 4, 0, 1}))
	must(mat.ReduceMax(inputMatrix, out))

	data, err := out.Read()
	must(err)

	fmt.Println(data)
	// Output:
	// [-1 4]
}

func ExampleSoftmax() {
	ctx, err := mat.NewContext()
	must(err)

	defer ctx.Release()

	inputMatrix, err := mat.NewMatrix(ctx, 1, 3)
	must(err)

	defer inputMatrix.Release()

	out, err := mat.NewMatrix(ctx, 1, 3)
	must(err)

	defer out.Release()

	must(inputMatrix.Write([]float32{1, 2, 3}))
	must(mat.Softmax(inputMatrix, out))

	data, err := out.Read()
	must(err)

	fmt.Printf("%.4f %.4f %.4f\n", data[0], data[1], data[2])
	// Output:
	// 0.0900 0.2447 0.6652
}

func ExampleRMSNorm() {
	ctx, err := mat.NewContext()
	must(err)

	defer ctx.Release()

	inputMatrix, err := mat.NewMatrix(ctx, 1, 2)
	must(err)

	defer inputMatrix.Release()

	out, err := mat.NewMatrix(ctx, 1, 2)
	must(err)

	defer out.Release()

	must(inputMatrix.Write([]float32{3, 4}))
	must(mat.RMSNorm(inputMatrix, out))

	data, err := out.Read()
	must(err)

	fmt.Printf("%.4f %.4f\n", data[0], data[1])
	// Output:
	// 0.8485 1.1314
}

func must(err error) {
	if err != nil {
		panic(err)
	}
}

func mustCreate2x2ExampleMatrices() (
	*mat.Context,
	*mat.Matrix,
	*mat.Matrix,
	*mat.Matrix,
) {
	ctx, err := mat.NewContext()
	must(err)

	leftMatrix, err := mat.NewMatrix(ctx, 2, 2)
	must(err)

	rightMatrix, err := mat.NewMatrix(ctx, 2, 2)
	must(err)

	out, err := mat.NewMatrix(ctx, 2, 2)
	must(err)

	return ctx, leftMatrix, rightMatrix, out
}

func mustWrite2x2Inputs(leftMatrix, rightMatrix *mat.Matrix) {
	must(leftMatrix.Write([]float32{1, 2, 3, 4}))
	must(rightMatrix.Write([]float32{5, 6, 7, 8}))
}
