// Package mat provides WebGPU-buffer-backed 2D matrix operations for Go.
//
// It uses WebGPU (via github.com/gogpu/wgpu) for matrix storage. MatMul, Add,
// Mul, Scale, Transp, ReduceSumTo, BroadcastTo, and ReshapeTo execute as WGSL
// compute kernels when a GPU is available; they fall back to pure Go on a CPU
// adapter. ReduceMax, Softmax, and RMSNorm use a host compatibility path. All
// matrices store float32 values in row-major order: element (r, c) is at index
// r*m.Cols()+c.
//
// Matrix shapes are fixed at construction time. Every operation requires all
// operands to belong to the same Context, and the output must not alias an
// input. Add and Mul support 2D singleton-axis broadcasting. Context.Stats can
// be used to observe host transfers, command submissions, and buffer lifetime.
// Validation and lifecycle failures can be classified with errors.Is and the
// exported Err* sentinel errors.
//
// Both CGO modes are supported. Use CGO_ENABLED=0 when a C toolchain is not
// available, or CGO_ENABLED=1 when combining mat with CGO dependencies:
//
//	CGO_ENABLED=0 go build ./...
//	CGO_ENABLED=1 go build ./...
//
// Import path:
//
//	github.com/KEINOS/go-wgpu-mat/mat
//
// Backends are registered internally. Use NewContext to select the execution
// mode. Without arguments UseAuto is selected by default:
//
//	ctx, _ := mat.NewContext()           // UseAuto — try GPU, then CPU
//	ctx, _ := mat.NewContext(mat.UseGPU) // high-performance GPU adapter
//	ctx, _ := mat.NewContext(mat.UseCPU) // software/fallback adapter
package mat
