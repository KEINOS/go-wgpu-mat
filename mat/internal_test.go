//go:build !cgo

package mat

import (
	"encoding/binary"
	"io"
	"math"
	"testing"

	"github.com/gogpu/wgpu"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func newTestContextDeps() contextDeps {
	deps := new(contextDeps)
	deps.createInstance = func(*wgpu.InstanceDescriptor) (*wgpu.Instance, error) {
		return new(wgpu.Instance), nil
	}
	deps.requestAdapter = func(
		*wgpu.Instance,
		*wgpu.RequestAdapterOptions,
	) (*wgpu.Adapter, error) {
		return new(wgpu.Adapter), nil
	}
	deps.requestDevice = func(
		*wgpu.Adapter,
		*wgpu.DeviceDescriptor,
	) (*wgpu.Device, error) {
		return new(wgpu.Device), nil
	}
	deps.releaseInstance = func(*wgpu.Instance) {}
	deps.releaseAdapter = func(*wgpu.Adapter) {}

	return *deps
}

func TestWrapErrorNil(t *testing.T) {
	t.Parallel()

	err := wrapError(nil, "ignored")

	assert.NoError(t, err)
}

func TestWrapErrorWrapsOriginal(t *testing.T) {
	t.Parallel()

	err := wrapError(io.EOF, "failed to run %s", "op")

	require.Error(t, err)
	require.ErrorContains(t, err, "mat: failed to run op")
	require.ErrorIs(t, err, io.EOF)
}

func TestNewContextCreateInstanceError(t *testing.T) {
	t.Parallel()

	deps := new(contextDeps)
	*deps = newTestContextDeps()
	deps.createInstance = func(*wgpu.InstanceDescriptor) (*wgpu.Instance, error) {
		return nil, io.EOF
	}

	ctx, err := newContext(*deps, UseGPU)

	assert.Nil(t, ctx)
	require.Error(t, err)
	require.ErrorContains(t, err, "mat: create instance")
}

func TestNewContextRequestAdapterError(t *testing.T) {
	t.Parallel()

	deps := new(contextDeps)
	*deps = newTestContextDeps()
	deps.requestAdapter = func(
		*wgpu.Instance,
		*wgpu.RequestAdapterOptions,
	) (*wgpu.Adapter, error) {
		return nil, io.EOF
	}

	ctx, err := newContext(*deps, UseGPU)

	assert.Nil(t, ctx)
	require.Error(t, err)
	require.ErrorContains(t, err, "mat: request adapter")
}

func TestNewContextRequestDeviceError(t *testing.T) {
	t.Parallel()

	deps := new(contextDeps)
	*deps = newTestContextDeps()
	deps.requestDevice = func(
		*wgpu.Adapter,
		*wgpu.DeviceDescriptor,
	) (*wgpu.Device, error) {
		return nil, io.EOF
	}

	ctx, err := newContext(*deps, UseGPU)

	assert.Nil(t, ctx)
	require.Error(t, err)
	require.ErrorContains(t, err, "mat: request device")
}

func TestNewContextSuccessWithInjectedDeps(t *testing.T) {
	t.Parallel()

	deps := new(contextDeps)
	*deps = newTestContextDeps()

	ctx, err := newContext(*deps, UseGPU)

	require.NoError(t, err)
	require.NotNil(t, ctx)
	assert.NotNil(t, ctx.instance)
	assert.NotNil(t, ctx.adapter)
	assert.NotNil(t, ctx.device)
}

func TestContextReleaseWithNilFields(t *testing.T) {
	t.Parallel()

	ctx := new(Context)

	require.NotPanics(t, func() { ctx.Release() })
}

func TestDefaultContextDepsReleaseHelpers(t *testing.T) {
	t.Parallel()

	deps := defaultContextDeps()

	require.NotPanics(t, func() {
		deps.releaseInstance(nil)
		deps.releaseAdapter(nil)
	})

	ctx, err := NewContext()
	require.NoError(t, err)
	require.NotNil(t, ctx)
	require.NotNil(t, ctx.device)
	require.NotNil(t, ctx.adapter)
	require.NotNil(t, ctx.instance)

	ctx.device.Release()
	ctx.device = nil

	require.NotPanics(t, func() {
		deps.releaseAdapter(ctx.adapter)
		deps.releaseInstance(ctx.instance)
	})

	ctx.adapter = nil
	ctx.instance = nil
}

func TestResolveContextMode(t *testing.T) {
	t.Parallel()

	mode, err := resolveContextMode(nil)
	require.NoError(t, err)
	assert.Equal(t, UseGPU, mode)

	mode, err = resolveContextMode([]ContextMode{UseCPU})
	require.NoError(t, err)
	assert.Equal(t, UseCPU, mode)

	_, err = resolveContextMode([]ContextMode{UseGPU, UseCPU})
	require.Error(t, err)
	require.ErrorContains(t, err, "only one context mode")

	_, err = resolveContextMode([]ContextMode{ContextMode(99)})
	require.Error(t, err)
	require.ErrorContains(t, err, "invalid context mode")
}

func TestAdapterOptionsForMode(t *testing.T) {
	t.Parallel()

	options, err := adapterOptionsForMode(UseGPU)
	require.NoError(t, err)
	assert.Equal(t, wgpu.PowerPreferenceHighPerformance, options.PowerPreference)
	assert.False(t, options.ForceFallbackAdapter)

	options, err = adapterOptionsForMode(UseCPU)
	require.NoError(t, err)
	assert.Equal(t, wgpu.PowerPreferenceLowPower, options.PowerPreference)
	assert.True(t, options.ForceFallbackAdapter)

	_, err = adapterOptionsForMode(ContextMode(77))
	require.Error(t, err)
	require.ErrorContains(t, err, "invalid context mode")
}

func TestNewContext_invalidModes(t *testing.T) {
	t.Parallel()

	_, err := NewContext(UseCPU, UseGPU)
	require.Error(t, err)
	require.ErrorContains(t, err, "only one context mode")

	_, err = NewContext(ContextMode(99))
	require.Error(t, err)
	require.ErrorContains(t, err, "invalid context mode")
}

func TestNewContext_internalInvalidMode(t *testing.T) {
	t.Parallel()

	ctx, err := newContext(newTestContextDeps(), ContextMode(88))
	assert.Nil(t, ctx)
	require.Error(t, err)
	require.ErrorContains(t, err, "invalid context mode")
}

func TestNewMatrixContextNil(t *testing.T) {
	t.Parallel()

	_, err := NewMatrix(nil, 1, 1)

	require.Error(t, err)
	require.ErrorContains(t, err, "mat: context is nil")
}

func TestNewMatrixDimensionValidation(t *testing.T) {
	t.Parallel()

	ctx := new(Context)
	ctx.device = new(wgpu.Device)

	_, err := NewMatrix(ctx, 0, 1)

	require.Error(t, err)
	require.ErrorContains(t, err, "matrix dimensions must be positive")
}

func TestNewMatrixDimensionOverflow(t *testing.T) {
	t.Parallel()

	ctx := new(Context)
	ctx.device = new(wgpu.Device)
	maxInt := int(^uint(0) >> 1)

	_, err := newMatrix(ctx, maxInt, maxInt, defaultMatrixDeps())

	require.Error(t, err)
	require.ErrorContains(t, err, "matrix dimensions overflow")
}

func TestNewMatrixByteSizeOverflow(t *testing.T) {
	t.Parallel()

	ctx := new(Context)
	ctx.device = new(wgpu.Device)
	maxInt := int(^uint(0) >> 1)

	_, err := newMatrix(ctx, maxInt, 1, defaultMatrixDeps())

	require.Error(t, err)
	require.ErrorContains(t, err, "matrix byte size overflow")
}

func TestNewMatrixCreateBufferError(t *testing.T) {
	t.Parallel()

	ctx := new(Context)
	ctx.device = new(wgpu.Device)

	deps := new(matrixDeps)
	deps.createBuffer = func(
		*Context,
		*wgpu.BufferDescriptor,
	) (*wgpu.Buffer, error) {
		return nil, io.EOF
	}
	deps.writeBuffer = func(*Context, *wgpu.Buffer, []byte) error { return nil }
	deps.readBuffer = func(*Context, *wgpu.Buffer, []byte) error { return nil }

	_, err := newMatrix(ctx, 2, 2, *deps)

	require.Error(t, err)
	require.ErrorContains(t, err, "mat: failed to create buffer")
}

func TestNewMatrixSuccessInjectedDeps(t *testing.T) {
	t.Parallel()

	ctx := new(Context)
	ctx.device = new(wgpu.Device)
	buffer := new(wgpu.Buffer)

	deps := new(matrixDeps)
	deps.createBuffer = func(
		*Context,
		*wgpu.BufferDescriptor,
	) (*wgpu.Buffer, error) {
		return buffer, nil
	}
	deps.writeBuffer = func(*Context, *wgpu.Buffer, []byte) error { return nil }
	deps.readBuffer = func(*Context, *wgpu.Buffer, []byte) error { return nil }

	matrix, err := newMatrix(ctx, 2, 2, *deps)

	require.NoError(t, err)
	assert.Equal(t, 2, matrix.Rows)
	assert.Equal(t, 2, matrix.Cols)
	assert.Equal(t, buffer, matrix.buf)
}

func TestMatrixWriteUninitialized(t *testing.T) {
	t.Parallel()

	var matrix *Matrix

	err := matrix.Write([]float32{1})

	require.Error(t, err)
	require.ErrorContains(t, err, "mat: matrix is not initialized")
}

func TestMatrixWriteLenMismatch(t *testing.T) {
	t.Parallel()

	matrix := new(Matrix)
	matrix.Rows = 2
	matrix.Cols = 2
	matrix.ctx = new(Context)
	matrix.buf = new(wgpu.Buffer)

	deps := new(matrixDeps)
	deps.createBuffer = func(
		*Context,
		*wgpu.BufferDescriptor,
	) (*wgpu.Buffer, error) {
		return new(wgpu.Buffer), nil
	}
	deps.writeBuffer = func(*Context, *wgpu.Buffer, []byte) error { return nil }
	deps.readBuffer = func(*Context, *wgpu.Buffer, []byte) error { return nil }
	matrix.deps = *deps

	err := matrix.Write([]float32{1, 2, 3})

	require.Error(t, err)
	require.ErrorContains(t, err, "mat: fail to write")
}

func TestMatrixWriteBackendError(t *testing.T) {
	t.Parallel()

	matrix := new(Matrix)
	matrix.Rows = 1
	matrix.Cols = 1
	matrix.ctx = new(Context)
	matrix.buf = new(wgpu.Buffer)

	deps := new(matrixDeps)
	deps.createBuffer = func(
		*Context,
		*wgpu.BufferDescriptor,
	) (*wgpu.Buffer, error) {
		return new(wgpu.Buffer), nil
	}
	deps.writeBuffer = func(*Context, *wgpu.Buffer, []byte) error {
		return io.EOF
	}
	deps.readBuffer = func(*Context, *wgpu.Buffer, []byte) error { return nil }
	matrix.deps = *deps

	err := matrix.Write([]float32{1})

	require.Error(t, err)
	require.ErrorContains(t, err, "mat: failed to write buffer")
}

func TestMatrixWriteSuccessConvertsToBytes(t *testing.T) {
	t.Parallel()

	matrix := new(Matrix)
	matrix.Rows = 1
	matrix.Cols = 2
	matrix.ctx = new(Context)
	matrix.buf = new(wgpu.Buffer)

	deps := new(matrixDeps)
	deps.createBuffer = func(
		*Context,
		*wgpu.BufferDescriptor,
	) (*wgpu.Buffer, error) {
		return new(wgpu.Buffer), nil
	}
	deps.writeBuffer = func(_ *Context, _ *wgpu.Buffer, data []byte) error {
		require.Len(t, data, 2*bytesPerFloat32Int)

		first := math.Float32frombits(binary.LittleEndian.Uint32(data[0:4]))
		second := math.Float32frombits(binary.LittleEndian.Uint32(data[4:8]))

		assert.InDelta(t, 1.5, first, 1e-6)
		assert.InDelta(t, -2.25, second, 1e-6)

		return nil
	}
	deps.readBuffer = func(*Context, *wgpu.Buffer, []byte) error { return nil }
	matrix.deps = *deps

	err := matrix.Write([]float32{1.5, -2.25})

	require.NoError(t, err)
}

func TestMatrixReadUninitialized(t *testing.T) {
	t.Parallel()

	var matrix *Matrix

	data, err := matrix.Read()

	assert.Nil(t, data)
	require.Error(t, err)
	require.ErrorContains(t, err, "mat: matrix is not initialized")
}

func TestMatrixReadBackendError(t *testing.T) {
	t.Parallel()

	matrix := new(Matrix)
	matrix.Rows = 1
	matrix.Cols = 1
	matrix.ctx = new(Context)
	matrix.buf = new(wgpu.Buffer)

	deps := new(matrixDeps)
	deps.createBuffer = func(
		*Context,
		*wgpu.BufferDescriptor,
	) (*wgpu.Buffer, error) {
		return new(wgpu.Buffer), nil
	}
	deps.writeBuffer = func(*Context, *wgpu.Buffer, []byte) error { return nil }
	deps.readBuffer = func(*Context, *wgpu.Buffer, []byte) error {
		return io.EOF
	}
	matrix.deps = *deps

	data, err := matrix.Read()

	assert.Nil(t, data)
	require.Error(t, err)
	require.ErrorContains(t, err, "mat: failed to read buffer")
}

func TestMatrixReadSuccessConvertsFromBytes(t *testing.T) {
	t.Parallel()

	matrix := new(Matrix)
	matrix.Rows = 1
	matrix.Cols = 2
	matrix.ctx = new(Context)
	matrix.buf = new(wgpu.Buffer)

	deps := new(matrixDeps)
	deps.createBuffer = func(
		*Context,
		*wgpu.BufferDescriptor,
	) (*wgpu.Buffer, error) {
		return new(wgpu.Buffer), nil
	}
	deps.writeBuffer = func(*Context, *wgpu.Buffer, []byte) error { return nil }
	deps.readBuffer = func(_ *Context, _ *wgpu.Buffer, data []byte) error {
		binary.LittleEndian.PutUint32(data[0:4], math.Float32bits(3.5))
		binary.LittleEndian.PutUint32(data[4:8], math.Float32bits(-4.5))

		return nil
	}
	matrix.deps = *deps

	data, err := matrix.Read()

	require.NoError(t, err)
	require.Len(t, data, 2)
	assert.InDelta(t, 3.5, data[0], 1e-6)
	assert.InDelta(t, -4.5, data[1], 1e-6)
}

func TestMatrixReleaseNilBuffer(t *testing.T) {
	t.Parallel()

	matrix := new(Matrix)

	require.NotPanics(t, func() { matrix.Release() })

	require.NotPanics(t, func() { matrix.Release() })
}
