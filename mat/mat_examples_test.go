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
