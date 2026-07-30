package mat

import (
	"errors"
	"fmt"
	"strconv"
	"sync"
	"sync/atomic"

	"github.com/KEINOS/go-wgpu-mat/mat/internal/backends"
	"github.com/gogpu/gputypes"
	"github.com/gogpu/wgpu"
)

var (
	errNilBackendInstance = errors.New("backend returned a nil instance")
	errNilBackendAdapter  = errors.New("backend returned a nil adapter")
	errNilBackendDevice   = errors.New("backend returned a nil device")
)

// Context holds a live WGPU Instance, Adapter, and Device.
// Create one via NewContext; release it with Release when done.
type Context struct {
	instance *wgpu.Instance
	adapter  *wgpu.Adapter
	device   *wgpu.Device
	pipes    *pipelineCache
	limits   gputypes.Limits
	mode     ContextMode
	isCPU    bool
	infoSet  bool
	released atomic.Uint32
	queueMu  sync.Mutex
	stats    contextStats
}

type contextStats struct {
	hostReads          atomic.Uint64
	hostWrites         atomic.Uint64
	commandSubmissions atomic.Uint64
	bufferAllocations  atomic.Uint64
	liveBuffers        atomic.Uint64
	peakLiveBuffers    atomic.Uint64
}

// Stats is an immutable-by-copy snapshot of Context activity.
//
// HostReads and HostWrites include only completed public Matrix.Read and
// Matrix.Write payload transfers. Internal staging and uniform-buffer traffic
// is excluded from those two counters.
type Stats struct {
	HostReads          uint64
	HostWrites         uint64
	CommandSubmissions uint64
	BufferAllocations  uint64
	LiveBuffers        uint64
	PeakLiveBuffers    uint64
}

// ContextMode specifies which adapter type NewContext should prefer.
type ContextMode uint8

const (
	// UseGPU requires a non-CPU, high-performance adapter.
	UseGPU ContextMode = iota
	// UseCPU forces a fallback adapter (software backend).
	UseCPU
	// UseAuto tries a high-performance GPU adapter first, then retries with a
	// software/fallback adapter if no GPU adapter is available.
	UseAuto
)

// String returns the stable name of the context mode.
func (m ContextMode) String() string {
	switch m {
	case UseGPU:
		return "gpu"
	case UseCPU:
		return "cpu"
	case UseAuto:
		return "auto"
	default:
		return "ContextMode(" + strconv.FormatUint(uint64(m), 10) + ")"
	}
}

type contextDeps struct {
	createInstance func(*wgpu.InstanceDescriptor) (*wgpu.Instance, error)
	requestAdapter func(*wgpu.Instance, *wgpu.RequestAdapterOptions) (
		*wgpu.Adapter, error,
	)
	requestDevice func(*wgpu.Adapter, *wgpu.DeviceDescriptor) (
		*wgpu.Device, error,
	)
	adapterInfo     func(*wgpu.Adapter) gputypes.AdapterInfo
	deviceLimits    func(*wgpu.Device) gputypes.Limits
	releaseDevice   func(*wgpu.Device)
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
	deps.adapterInfo = func(adapter *wgpu.Adapter) gputypes.AdapterInfo {
		return adapter.Info()
	}
	deps.deviceLimits = func(device *wgpu.Device) gputypes.Limits {
		return device.Limits()
	}
	deps.releaseDevice = func(device *wgpu.Device) {
		if device != nil {
			device.Release()
		}
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
// When no mode is provided, UseAuto is selected by default.
//
//	ctx, err := NewContext()       // same as NewContext(UseAuto)
//	ctx, err := NewContext(UseCPU) // force software/fallback adapter
//	ctx, err := NewContext(UseGPU) // high-performance GPU adapter
func NewContext(modes ...ContextMode) (*Context, error) {
	mode, err := resolveContextMode(modes)
	if err != nil {
		return nil, err
	}

	return newContext(defaultContextDeps(), mode)
}

func newContext(deps contextDeps, mode ContextMode) (*Context, error) {
	adapterOptions, err := contextAdapterOptions(mode)
	if err != nil {
		return nil, err
	}

	inst, err := deps.createInstance(nil)
	if err != nil {
		deps.releaseInstance(inst)

		return nil, fmt.Errorf("mat: create instance: %w", err)
	}

	if inst == nil {
		return nil, sentinelWrapError(
			ErrBackendUnavailable,
			errNilBackendInstance,
			"create instance",
		)
	}

	adapter, dev, backendErr := requestFirstAvailableDevice(deps, inst, adapterOptions)
	if backendErr != nil {
		deps.releaseInstance(inst)

		return nil, backendErr
	}

	adapterInfo := deps.adapterInfo(adapter)
	if mode == UseGPU && adapterInfo.DeviceType == gputypes.DeviceTypeCPU {
		deps.releaseDevice(dev)
		deps.releaseAdapter(adapter)
		deps.releaseInstance(inst)

		return nil, sentinelError(
			ErrBackendUnavailable,
			"request GPU adapter: backend selected a CPU adapter",
		)
	}

	return &Context{
		instance: inst,
		adapter:  adapter,
		device:   dev,
		pipes:    newPipelineCache(defaultReleaseComputePipeline),
		limits:   deps.deviceLimits(dev),
		mode:     mode,
		isCPU:    adapterInfo.DeviceType == gputypes.DeviceTypeCPU,
		infoSet:  true,
		released: atomic.Uint32{},
		queueMu:  sync.Mutex{},
		stats: contextStats{
			hostReads:          atomic.Uint64{},
			hostWrites:         atomic.Uint64{},
			commandSubmissions: atomic.Uint64{},
			bufferAllocations:  atomic.Uint64{},
			liveBuffers:        atomic.Uint64{},
			peakLiveBuffers:    atomic.Uint64{},
		},
	}, nil
}

func contextAdapterOptions(mode ContextMode) ([]*wgpu.RequestAdapterOptions, error) {
	gpu := &wgpu.RequestAdapterOptions{
		PowerPreference:      wgpu.PowerPreferenceHighPerformance,
		ForceFallbackAdapter: false,
		CompatibleSurface:    nil,
	}
	cpu := &wgpu.RequestAdapterOptions{
		PowerPreference:      wgpu.PowerPreferenceLowPower,
		ForceFallbackAdapter: true,
		CompatibleSurface:    nil,
	}

	switch mode {
	case UseGPU:
		backends.UseGPU()

		return []*wgpu.RequestAdapterOptions{gpu}, nil
	case UseCPU:
		backends.UseCPU()

		return []*wgpu.RequestAdapterOptions{cpu}, nil
	case UseAuto:
		backends.UseGPU()
		backends.UseCPU()

		return []*wgpu.RequestAdapterOptions{gpu, cpu}, nil
	default:
		return nil, sentinelError(ErrInvalidMode, "invalid context mode: %d", mode)
	}
}

func requestFirstAvailableDevice(
	deps contextDeps,
	instance *wgpu.Instance,
	options []*wgpu.RequestAdapterOptions,
) (*wgpu.Adapter, *wgpu.Device, error) {
	var lastErr error

	lastStage := "request adapter"

	for _, option := range options {
		adapter, err := deps.requestAdapter(instance, option)
		if err != nil {
			lastErr = err

			if adapter != nil {
				deps.releaseAdapter(adapter)
			}

			continue
		}

		if adapter == nil {
			lastErr = errNilBackendAdapter

			continue
		}

		device, err := deps.requestDevice(adapter, nil)
		if err == nil && device != nil {
			return adapter, device, nil
		}

		if device != nil {
			deps.releaseDevice(device)
		}

		deps.releaseAdapter(adapter)

		lastStage = "request device"

		if err != nil {
			lastErr = err
		} else {
			lastErr = errNilBackendDevice
		}
	}

	if lastErr == nil {
		lastErr = ErrBackendUnavailable
	}

	return nil, nil, sentinelWrapError(
		ErrBackendUnavailable,
		lastErr,
		"%s",
		lastStage,
	)
}

func resolveContextMode(modes []ContextMode) (ContextMode, error) {
	if len(modes) == 0 {
		return UseAuto, nil
	}

	if len(modes) > 1 {
		return 0, sentinelError(
			ErrInvalidMode,
			"only one context mode can be specified",
		)
	}

	mode := modes[0]
	if mode != UseGPU && mode != UseCPU && mode != UseAuto {
		return 0, sentinelError(ErrInvalidMode, "invalid context mode: %d", mode)
	}

	return mode, nil
}

// Mode reports the mode requested when this Context was created. For UseAuto,
// the actual adapter may be either hardware or software.
func (c *Context) Mode() ContextMode {
	if c == nil {
		return UseAuto
	}

	return c.mode
}

// Released reports whether Release has been called.
func (c *Context) Released() bool {
	return c == nil || c.released.Load() != 0
}

// Stats returns a concurrency-safe snapshot of cumulative context activity.
// It can be called before or after Release.
func (c *Context) Stats() Stats {
	if c == nil {
		return Stats{
			HostReads:          0,
			HostWrites:         0,
			CommandSubmissions: 0,
			BufferAllocations:  0,
			LiveBuffers:        0,
			PeakLiveBuffers:    0,
		}
	}

	return Stats{
		HostReads:          c.stats.hostReads.Load(),
		HostWrites:         c.stats.hostWrites.Load(),
		CommandSubmissions: c.stats.commandSubmissions.Load(),
		BufferAllocations:  c.stats.bufferAllocations.Load(),
		LiveBuffers:        c.stats.liveBuffers.Load(),
		PeakLiveBuffers:    c.stats.peakLiveBuffers.Load(),
	}
}

// Release frees the Device, Adapter, and Instance in reverse order.
// It is a no-op when called on a nil receiver or more than once.
// Release must not run concurrently with matrix operations using this Context.
func (c *Context) Release() {
	if c == nil || !c.released.CompareAndSwap(0, 1) {
		return
	}

	if c.pipes != nil {
		c.pipes.releaseAll()
		c.pipes = nil
	}

	if c.device != nil {
		c.device.Release()
		c.device = nil
	}

	if c.adapter != nil {
		c.adapter.Release()
		c.adapter = nil
	}

	if c.instance != nil {
		c.instance.Release()
		c.instance = nil
	}
}

// Close releases the context and always returns nil. It allows Context to be
// used as an io.Closer while preserving the idempotent Release API.
func (c *Context) Close() error {
	c.Release()

	return nil
}

func (c *Context) recordBufferAllocation() {
	if c == nil {
		return
	}

	c.stats.bufferAllocations.Add(1)
	live := c.stats.liveBuffers.Add(1)

	for {
		peak := c.stats.peakLiveBuffers.Load()
		if live <= peak || c.stats.peakLiveBuffers.CompareAndSwap(peak, live) {
			return
		}
	}
}

func (c *Context) recordBufferRelease() {
	if c != nil {
		c.stats.liveBuffers.Add(^uint64(0))
	}
}

func (c *Context) recordHostRead() {
	if c != nil {
		c.stats.hostReads.Add(1)
	}
}

func (c *Context) recordHostWrite() {
	if c != nil {
		c.stats.hostWrites.Add(1)
	}
}

func (c *Context) recordSubmission() {
	if c != nil {
		c.stats.commandSubmissions.Add(1)
	}
}

func (c *Context) withQueue(operation func() error) error {
	c.queueMu.Lock()
	defer c.queueMu.Unlock()

	return operation()
}

func (c *Context) getOrCreatePipeline(
	key string,
	factory func() (*wgpu.ComputePipeline, error),
) (*wgpu.ComputePipeline, error) {
	if c == nil {
		return nil, sentinelError(ErrNilContext, "context is nil")
	}

	if c.released.Load() != 0 {
		return nil, sentinelError(ErrContextReleased, "context is released")
	}

	if c.pipes == nil {
		c.pipes = newPipelineCache(defaultReleaseComputePipeline)
	}

	return c.pipes.getOrCreate(key, factory)
}
