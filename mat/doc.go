// Package mat provides GPU-accelerated 2D matrix operations for Go.
//
// It uses WebGPU (via github.com/gogpu/wgpu) to execute compute
// shaders on the GPU. All matrices store float32 values in
// row-major order: element (r, c) is at index r*Cols + c.
//
// Build with CGO_ENABLED=0 (gogpu/wgpu is Pure Go):
//
// CGO_ENABLED=0 go build ./...
//
// Import path:
//
// github.com/KEINOS/go-wgpu-mat/mat
//
// Backends are registered internally. Select execution mode with
// NewContext:
//
// ctx, _ := mat.NewContext(mat.UseGPU) // high-performance adapter
// ctx, _ := mat.NewContext(mat.UseCPU) // software/fallback adapter
package mat
