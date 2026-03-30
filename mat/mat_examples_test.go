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

	// Read result back to CPU
	data, err := c.Read()
	panicOnErr(err)

	fmt.Println(data)
	// Output:
	// [0 0 0 0]
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
