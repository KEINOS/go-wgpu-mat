//go:build !cgo

// Package backends registers WGPU backends for mat context creation.
package backends

import (
	// Register all available platform backends for runtime adapter selection.
	_ "github.com/gogpu/wgpu/hal/allbackends"
	// Register software backend explicitly for ForceFallbackAdapter mode.
	_ "github.com/gogpu/wgpu/hal/software"
)

// UseGPU ensures GPU-capable backends are registered.
func UseGPU() {}

// UseCPU ensures fallback-capable backends are registered.
func UseCPU() {}
