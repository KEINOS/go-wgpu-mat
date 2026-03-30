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
// Import a backend before calling NewContext:
//
// import _ "github.com/gogpu/wgpu/hal/allbackends" // Vulkan/Metal/DX12
// import _ "github.com/gogpu/wgpu/hal/software"    // CPU fallback
package mat
