//nolint:noinlineerr,wsl_v5 // Compact examples keep each operation next to its error check.
package mat_test

import (
	"fmt"

	"github.com/KEINOS/go-wgpu-mat/mat"
)

func ExampleBroadcastTo() {
	ctx, err := mat.NewContext(mat.UseCPU)
	if err != nil {
		panic(err)
	}
	defer ctx.Release()

	input, err := mat.NewMatrix(ctx, 1, 2)
	if err != nil {
		panic(err)
	}
	defer input.Release()

	out, err := mat.NewMatrix(ctx, 3, 2)
	if err != nil {
		panic(err)
	}
	defer out.Release()

	if err = input.Write([]float32{4, 7}); err != nil {
		panic(err)
	}
	if err = mat.BroadcastTo(input, out); err != nil {
		panic(err)
	}

	data, err := out.Read()
	if err != nil {
		panic(err)
	}

	fmt.Println(data)
	// Output: [4 7 4 7 4 7]
}

func ExampleContext_Stats() {
	ctx, err := mat.NewContext(mat.UseCPU)
	if err != nil {
		panic(err)
	}
	defer ctx.Release()

	before := ctx.Stats()

	input, err := mat.NewMatrix(ctx, 1, 1)
	if err != nil {
		panic(err)
	}
	defer input.Release()

	if err = input.Write([]float32{42}); err != nil {
		panic(err)
	}
	if _, err = input.Read(); err != nil {
		panic(err)
	}

	after := ctx.Stats()

	fmt.Println(after.HostReads - before.HostReads)
	fmt.Println(after.HostWrites - before.HostWrites)
	// Output:
	// 1
	// 1
}

func ExampleMul() {
	ctx, err := mat.NewContext(mat.UseCPU)
	if err != nil {
		panic(err)
	}
	defer ctx.Release()

	left, err := mat.NewMatrix(ctx, 2, 2)
	if err != nil {
		panic(err)
	}
	defer left.Release()

	right, err := mat.NewMatrix(ctx, 1, 2)
	if err != nil {
		panic(err)
	}
	defer right.Release()

	out, err := mat.NewMatrix(ctx, 2, 2)
	if err != nil {
		panic(err)
	}
	defer out.Release()

	if err = left.Write([]float32{1, 2, 3, 4}); err != nil {
		panic(err)
	}
	if err = right.Write([]float32{10, 100}); err != nil {
		panic(err)
	}
	if err = mat.Mul(left, right, out); err != nil {
		panic(err)
	}

	data, err := out.Read()
	if err != nil {
		panic(err)
	}

	fmt.Println(data)
	// Output: [10 200 30 400]
}

func ExampleReduceSumTo() {
	ctx, err := mat.NewContext(mat.UseCPU)
	if err != nil {
		panic(err)
	}
	defer ctx.Release()

	input, err := mat.NewMatrix(ctx, 2, 3)
	if err != nil {
		panic(err)
	}
	defer input.Release()

	out, err := mat.NewMatrix(ctx, 1, 3)
	if err != nil {
		panic(err)
	}
	defer out.Release()

	if err = input.Write([]float32{1, 2, 3, 4, 5, 6}); err != nil {
		panic(err)
	}
	if err = mat.ReduceSumTo(input, out); err != nil {
		panic(err)
	}

	data, err := out.Read()
	if err != nil {
		panic(err)
	}

	fmt.Println(data)
	// Output: [5 7 9]
}

func ExampleReshapeTo() {
	ctx, err := mat.NewContext(mat.UseCPU)
	if err != nil {
		panic(err)
	}
	defer ctx.Release()

	input, err := mat.NewMatrix(ctx, 2, 3)
	if err != nil {
		panic(err)
	}
	defer input.Release()

	out, err := mat.NewMatrix(ctx, 3, 2)
	if err != nil {
		panic(err)
	}
	defer out.Release()

	if err = input.Write([]float32{1, 2, 3, 4, 5, 6}); err != nil {
		panic(err)
	}
	if err = mat.ReshapeTo(input, out); err != nil {
		panic(err)
	}

	data, err := out.Read()
	if err != nil {
		panic(err)
	}

	fmt.Println(data)
	// Output: [1 2 3 4 5 6]
}
