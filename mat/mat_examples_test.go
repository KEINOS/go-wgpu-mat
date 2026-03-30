//go:build !cgo

package mat_test

import "github.com/KEINOS/go-wgpu-mat/mat"

// ExampleNewMatrix documents the basic allocation pattern.
func ExampleNewMatrix() {
	ctx, err := mat.NewContext()
	if err != nil {
		panic(err)
	}
	defer ctx.Release()

	m, err := mat.NewMatrix(ctx, 2, 3)
	if err != nil {
		panic(err)
	}
	defer m.Release()

	// Output:
}
