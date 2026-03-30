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

// NewContext creates a GPU context backed by the highest-performance
// adapter available. At least one backend must be registered before
// calling this function, for example:
//
//	import _ "github.com/gogpu/wgpu/hal/allbackends" // real GPU
//	import _ "github.com/gogpu/wgpu/hal/software"    // CPU fallback
func NewContext() (*Context, error) {
	inst, err := wgpu.CreateInstance(nil)
	if err != nil {
		return nil, fmt.Errorf("mat: create instance: %w", err)
	}

	adapterOptions := new(wgpu.RequestAdapterOptions)
	adapterOptions.PowerPreference = wgpu.PowerPreferenceHighPerformance

	adapter, err := inst.RequestAdapter(adapterOptions)
	if err != nil {
		inst.Release()

		return nil, fmt.Errorf("mat: request adapter: %w", err)
	}

	dev, err := adapter.RequestDevice(nil)
	if err != nil {
		adapter.Release()
		inst.Release()

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

	c.device.Release()
	c.adapter.Release()
	c.instance.Release()
}
