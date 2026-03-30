package mat

import (
	"fmt"

	"github.com/gogpu/wgpu"
)

// Context holds a live WGPU Instance, Adapter, and Device.
// Create one via NewContext; release it with Release when done.
type Context struct {
	instance *wgpu.Instance
	adapter  *wgpu.Adapter
	device   *wgpu.Device
}

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

// NewContext creates a GPU context backed by the highest-performance
// adapter available. At least one backend must be registered before
// calling this function, for example:
//
//	import _ "github.com/gogpu/wgpu/hal/allbackends" // real GPU
//	import _ "github.com/gogpu/wgpu/hal/software"    // CPU fallback
func NewContext() (*Context, error) {
	return newContext(defaultContextDeps())
}

func newContext(deps contextDeps) (*Context, error) {
	inst, err := deps.createInstance(nil)
	if err != nil {
		return nil, fmt.Errorf("mat: create instance: %w", err)
	}

	adapterOptions := new(wgpu.RequestAdapterOptions)
	adapterOptions.PowerPreference = wgpu.PowerPreferenceHighPerformance

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
	}, nil
}

// Release frees the Device, Adapter, and Instance in reverse order.
// It is a no-op when called on a nil receiver.
func (c *Context) Release() {
	if c == nil {
		return
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
