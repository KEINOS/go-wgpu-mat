//go:build !cgo

package mat_test

import (
	"fmt"

	"github.com/KEINOS/go-wgpu-mat/mat"
)

// Example of basic usage in the README.md. This test ensures that the example
// code compiles and runs without errors.
func Example() {
	// UseGPU (default) or UseCPU
	ctx, err := mat.NewContext(mat.UseGPU)
	if err != nil {
		panic(err)
	}
	defer ctx.Release()

	// 2×2 matrices stored on the GPU
	a, _ := mat.NewMatrix(ctx, 2, 2)
	b, _ := mat.NewMatrix(ctx, 2, 2)
	c, _ := mat.NewMatrix(ctx, 2, 2)
	defer a.Release()
	defer b.Release()
	defer c.Release()

	// Upload data (row-major order)
	a.Write([]float32{1, 2, 3, 4}) // [[1,2],[3,4]]
	b.Write([]float32{5, 6, 7, 8}) // [[5,6],[7,8]]

	// Read result back to CPU
	data, err := c.Read()
	if err != nil {
		panic(err)
	}

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
