//go:build !cgo

package mat

import (
	"fmt"

	"github.com/KEINOS/go-wgpu-mat/mat/internal/backends"
	"github.com/gogpu/wgpu"
)

// Context holds a live WGPU Instance, Adapter, and Device.
// Create one via NewContext; release it with Release when done.
type Context struct {
	instance *wgpu.Instance
	adapter  *wgpu.Adapter
	device   *wgpu.Device
	pipes    *pipelineCache
}

// ContextMode specifies which adapter type NewContext should prefer.
type ContextMode uint8

const (
	// UseGPU prefers a high-performance GPU adapter.
	UseGPU ContextMode = iota
	// UseCPU forces a fallback adapter (software backend).
	UseCPU
)

type contextDeps struct {
	createInstance func(*wgpu.InstanceDescriptor) (*wgpu.Instance, error)
	requestAdapter func(*wgpu.Instance, *wgpu.RequestAdapterOptions) (
		*wgpu.Adapter, error,
	)
	requestDevice func(*wgpu.Adapter, *wgpu.DeviceDescriptor) (
		*wgpu.Device, error,
	)
	releaseInstance func(*wgpu.Instance)
	releaseAdapter  func(*wgpu.Adapter)
}

func defaultContextDeps() contextDeps {
	deps := new(contextDeps)
	deps.createInstance = wgpu.CreateInstance
	deps.requestAdapter = func(
		inst *wgpu.Instance,
		opts *wgpu.RequestAdapterOptions,
	) (*wgpu.Adapter, error) {
		return inst.RequestAdapter(opts)
	}
	deps.requestDevice = func(
		adapter *wgpu.Adapter,
		desc *wgpu.DeviceDescriptor,
	) (*wgpu.Device, error) {
		return adapter.RequestDevice(desc)
	}
	deps.releaseInstance = func(inst *wgpu.Instance) {
		if inst != nil {
			inst.Release()
		}
	}
	deps.releaseAdapter = func(adapter *wgpu.Adapter) {
		if adapter != nil {
			adapter.Release()
		}
	}

	return *deps
}

// NewContext creates a compute context.
//
// The package registers required backends internally, so callers do
// not need blank-import backend packages.
//
// When no mode is provided, UseGPU is selected by default.
//
//	ctx, err := NewContext()       // same as NewContext(UseGPU)
//	ctx, err := NewContext(UseCPU) // force software/fallback adapter
func NewContext(modes ...ContextMode) (*Context, error) {
	mode, err := resolveContextMode(modes)
	if err != nil {
		return nil, err
	}

	return newContext(defaultContextDeps(), mode)
}

func newContext(deps contextDeps, mode ContextMode) (*Context, error) {
	adapterOptions, err := adapterOptionsForMode(mode)
	if err != nil {
		return nil, err
	}

	inst, err := deps.createInstance(nil)
	if err != nil {
		return nil, fmt.Errorf("mat: create instance: %w", err)
	}

	adapter, err := deps.requestAdapter(inst, adapterOptions)
	if err != nil {
		deps.releaseInstance(inst)

		return nil, fmt.Errorf("mat: request adapter: %w", err)
	}

	dev, err := deps.requestDevice(adapter, nil)
	if err != nil {
		deps.releaseAdapter(adapter)
		deps.releaseInstance(inst)

		return nil, fmt.Errorf("mat: request device: %w", err)
	}

	return &Context{
		instance: inst,
		adapter:  adapter,
		device:   dev,
		pipes:    newPipelineCache(defaultReleaseComputePipeline),
	}, nil
}

func resolveContextMode(modes []ContextMode) (ContextMode, error) {
	if len(modes) == 0 {
		return UseGPU, nil
	}

	if len(modes) > 1 {
		return 0, newError("only one context mode can be specified")
	}

	mode := modes[0]
	if mode != UseGPU && mode != UseCPU {
		return 0, newError("invalid context mode: %d", mode)
	}

	return mode, nil
}

func adapterOptionsForMode(mode ContextMode) (*wgpu.RequestAdapterOptions, error) {
	options := new(wgpu.RequestAdapterOptions)

	switch mode {
	case UseGPU:
		backends.UseGPU()

		options.PowerPreference = wgpu.PowerPreferenceHighPerformance

	case UseCPU:
		backends.UseCPU()

		options.PowerPreference = wgpu.PowerPreferenceLowPower
		options.ForceFallbackAdapter = true

	default:
		return nil, newError("invalid context mode: %d", mode)
	}

	return options, nil
}

// Release frees the Device, Adapter, and Instance in reverse order.
// It is a no-op when called on a nil receiver.
func (c *Context) Release() {
	if c == nil {
		return
	}

	if c.pipes != nil {
		c.pipes.releaseAll()
		c.pipes = nil
	}

	if c.device != nil {
		c.device.Release()
	}

	if c.adapter != nil {
		c.adapter.Release()
	}

	if c.instance != nil {
		c.instance.Release()
	}
}

func (c *Context) getOrCreatePipeline(
	key string,
	factory func() (*wgpu.ComputePipeline, error),
) (*wgpu.ComputePipeline, error) {
	if c == nil {
		return nil, newError("context is nil")
	}

	if c.pipes == nil {
		c.pipes = newPipelineCache(defaultReleaseComputePipeline)
	}

	return c.pipes.getOrCreate(key, factory)
}
