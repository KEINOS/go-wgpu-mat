package mat

import (
	"encoding/binary"
	"io"
	"math"
	"testing"

	"github.com/gogpu/gputypes"
	"github.com/gogpu/wgpu"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

const (
	testCaseFinishEncoder = "finish encoder"
	testCaseSubmit        = "submit"
)

// ============================================================================
//  Internal/Private function tests
//  (for public API tests, see mat_examples_test.go)
// ============================================================================

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
	deps.adapterInfo = func(*wgpu.Adapter) gputypes.AdapterInfo {
		var info gputypes.AdapterInfo

		return info
	}
	deps.deviceLimits = func(*wgpu.Device) gputypes.Limits {
		return gputypes.DefaultLimits()
	}
	deps.releaseDevice = func(*wgpu.Device) {}
	deps.releaseInstance = func(*wgpu.Instance) {}
	deps.releaseAdapter = func(*wgpu.Adapter) {}

	return *deps
}

func newTestReadBufferDeps(mappedData []byte) readBufferDeps {
	deps := new(readBufferDeps)
	deps.createStaging = func(*Context, uint64) (*wgpu.Buffer, error) {
		return new(wgpu.Buffer), nil
	}
	deps.releaseBuffer = func(*wgpu.Buffer) {}
	deps.createEncoder = func(*Context) (*wgpu.CommandEncoder, error) {
		return new(wgpu.CommandEncoder), nil
	}
	deps.copyBuffer = func(
		*wgpu.CommandEncoder,
		*wgpu.Buffer,
		*wgpu.Buffer,
		uint64,
	) {
	}
	deps.finishEncoder = func(*wgpu.CommandEncoder) (*wgpu.CommandBuffer, error) {
		return new(wgpu.CommandBuffer), nil
	}
	deps.releaseCommandBuffer = func(*wgpu.CommandBuffer) {}
	deps.submit = func(*Context, *wgpu.CommandBuffer) error { return nil }
	deps.mapBuffer = func(*wgpu.Buffer, uint64) error { return nil }
	deps.mappedRange = func(*wgpu.Buffer, uint64) (*wgpu.MappedRange, error) {
		return new(wgpu.MappedRange), nil
	}
	deps.mappedBytes = func(*wgpu.MappedRange) []byte { return mappedData }
	deps.releaseMappedRange = func(*wgpu.MappedRange) {}
	deps.unmapBuffer = func(*wgpu.Buffer) error { return nil }

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

//nolint:funlen // The assertions cover one ordered readback ownership flow.
func TestReadBufferCopiesMappedDataAndReleasesResources(t *testing.T) {
	t.Parallel()

	want := []byte{1, 2, 3, 4}
	deps := newTestReadBufferDeps(want)
	src := new(wgpu.Buffer)
	staging := new(wgpu.Buffer)
	commandBuffer := new(wgpu.CommandBuffer)
	releasedBuffer := false
	releasedRange := false
	releasedCommandBuffer := false
	copyCalled := false
	submitCalled := false
	unmapped := false
	deps.createStaging = func(*Context, uint64) (*wgpu.Buffer, error) {
		return staging, nil
	}
	deps.releaseBuffer = func(*wgpu.Buffer) { releasedBuffer = true }
	deps.copyBuffer = func(
		_ *wgpu.CommandEncoder,
		gotSrc, gotDst *wgpu.Buffer,
		gotSize uint64,
	) {
		copyCalled = true

		assert.Same(t, src, gotSrc)
		assert.Same(t, staging, gotDst)
		assert.Equal(t, uint64(len(want)), gotSize)
	}
	deps.finishEncoder = func(*wgpu.CommandEncoder) (*wgpu.CommandBuffer, error) {
		return commandBuffer, nil
	}
	deps.releaseCommandBuffer = func(*wgpu.CommandBuffer) {
		releasedCommandBuffer = true
	}
	deps.submit = func(_ *Context, got *wgpu.CommandBuffer) error {
		submitCalled = true

		assert.Same(t, commandBuffer, got)

		return nil
	}
	deps.releaseMappedRange = func(*wgpu.MappedRange) { releasedRange = true }
	deps.unmapBuffer = func(*wgpu.Buffer) error {
		unmapped = true

		return nil
	}
	got := make([]byte, len(want))

	err := readBuffer(new(Context), src, got, deps)

	require.NoError(t, err)
	assert.Equal(t, want, got)
	assert.True(t, releasedBuffer)
	assert.True(t, releasedRange)
	assert.True(t, releasedCommandBuffer)
	assert.True(t, copyCalled)
	assert.True(t, submitCalled)
	assert.True(t, unmapped)
}

func TestReadBufferReleasesCommandBufferWhenSubmitFails(t *testing.T) {
	t.Parallel()

	deps := newTestReadBufferDeps(make([]byte, bytesPerFloat32Int))
	commandBuffer := new(wgpu.CommandBuffer)
	releasedCommandBuffer := false
	deps.finishEncoder = func(*wgpu.CommandEncoder) (*wgpu.CommandBuffer, error) {
		return commandBuffer, nil
	}
	deps.submit = func(*Context, *wgpu.CommandBuffer) error { return io.EOF }
	deps.releaseCommandBuffer = func(got *wgpu.CommandBuffer) {
		releasedCommandBuffer = true

		assert.Same(t, commandBuffer, got)
	}

	err := readBuffer(
		new(Context),
		new(wgpu.Buffer),
		make([]byte, bytesPerFloat32Int),
		deps,
	)

	require.Error(t, err)
	assert.True(t, releasedCommandBuffer)
}

// TestReadBufferErrors keeps the readback error contract in one table so that
// every injected operation is checked consistently.
//
//nolint:funlen // Splitting this table would obscure the operation coverage.
func TestReadBufferErrors(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name    string
		wantErr string
		mutate  func(*readBufferDeps)
	}{
		{
			name:    "create staging",
			wantErr: "create readback buffer",
			mutate: func(deps *readBufferDeps) {
				deps.createStaging = func(*Context, uint64) (*wgpu.Buffer, error) {
					return nil, io.EOF
				}
			},
		},
		{
			name:    "create encoder",
			wantErr: "create readback encoder",
			mutate: func(deps *readBufferDeps) {
				deps.createEncoder = func(*Context) (*wgpu.CommandEncoder, error) {
					return nil, io.EOF
				}
			},
		},
		{
			name:    testCaseFinishEncoder,
			wantErr: "finish readback encoder",
			mutate: func(deps *readBufferDeps) {
				deps.finishEncoder = func(*wgpu.CommandEncoder) (*wgpu.CommandBuffer, error) {
					return nil, io.EOF
				}
			},
		},
		{
			name:    testCaseSubmit,
			wantErr: "submit readback",
			mutate: func(deps *readBufferDeps) {
				deps.submit = func(*Context, *wgpu.CommandBuffer) error { return io.EOF }
			},
		},
		{
			name:    "map",
			wantErr: "map readback buffer",
			mutate: func(deps *readBufferDeps) {
				deps.mapBuffer = func(*wgpu.Buffer, uint64) error { return io.EOF }
			},
		},
		{
			name:    "mapped range",
			wantErr: "get mapped readback range",
			mutate: func(deps *readBufferDeps) {
				deps.mappedRange = func(*wgpu.Buffer, uint64) (*wgpu.MappedRange, error) {
					return nil, io.EOF
				}
			},
		},
		{
			name:    "mapped size",
			wantErr: "mapped readback size mismatch",
			mutate: func(deps *readBufferDeps) {
				deps.mappedBytes = func(*wgpu.MappedRange) []byte { return []byte{1} }
			},
		},
		{
			name:    "unmap",
			wantErr: "unmap readback buffer",
			mutate: func(deps *readBufferDeps) {
				deps.unmapBuffer = func(*wgpu.Buffer) error { return io.EOF }
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()

			deps := newTestReadBufferDeps(make([]byte, bytesPerFloat32Int))
			test.mutate(&deps)

			err := readBuffer(
				new(Context),
				new(wgpu.Buffer),
				make([]byte, bytesPerFloat32Int),
				deps,
			)

			require.Error(t, err)
			require.ErrorContains(t, err, test.wantErr)
		})
	}
}

func TestDefaultReadBufferDepsReleaseNilCommandBuffer(t *testing.T) {
	t.Parallel()

	deps := defaultReadBufferDeps()

	require.NotPanics(t, func() { deps.releaseCommandBuffer(nil) })
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

func TestContextReleaseIsIdempotent(t *testing.T) {
	t.Parallel()

	releasedPipelines := 0
	ctx := new(Context)
	ctx.pipes = newPipelineCache(func(*wgpu.ComputePipeline) {
		releasedPipelines++
	})
	_, err := ctx.pipes.getOrCreate("test", func() (*wgpu.ComputePipeline, error) {
		return new(wgpu.ComputePipeline), nil
	})
	require.NoError(t, err)

	ctx.Release()
	ctx.Release()

	assert.Equal(t, uint32(1), ctx.released.Load())
	assert.Equal(t, 1, releasedPipelines)
}

func TestNewMatrixRejectsReleasedContext(t *testing.T) {
	t.Parallel()

	ctx := new(Context)
	ctx.device = new(wgpu.Device)
	ctx.released.Store(1)

	deps := matrixDeps{
		createBuffer: func(*Context, *wgpu.BufferDescriptor) (*wgpu.Buffer, error) {
			t.Fatal("buffer creation must not run for a released context")

			return nil, io.EOF
		},
		writeBuffer: nil,
		readBuffer:  nil,
	}

	matrix, err := newMatrix(ctx, 1, 1, deps)

	assert.Nil(t, matrix)
	require.ErrorContains(t, err, "context is released")
}

func TestNewMatrixRejectsContextWithoutDevice(t *testing.T) {
	t.Parallel()

	deps := defaultMatrixDeps()
	matrix, err := newMatrix(new(Context), 1, 1, deps)

	assert.Nil(t, matrix)
	require.ErrorContains(t, err, "context is not initialized")
	require.ErrorIs(t, err, ErrContextNotInitialized)
}

func TestNewMatrixRejectsDeviceBufferLimits(t *testing.T) {
	t.Parallel()

	bufferLimits := gputypes.DefaultLimits()
	bufferLimits.MaxBufferSize = 3
	bufferLimits.MaxStorageBufferBindingSize = 4
	storageLimits := gputypes.DefaultLimits()
	storageLimits.MaxBufferSize = 4
	storageLimits.MaxStorageBufferBindingSize = 3

	tests := []struct {
		name    string
		limits  gputypes.Limits
		wantErr string
	}{
		{
			name:    "maximum buffer size",
			limits:  bufferLimits,
			wantErr: "exceeds device maximum buffer size",
		},
		{
			name:    "maximum storage binding size",
			limits:  storageLimits,
			wantErr: "exceeds device maximum storage buffer binding size",
		},
	}

	for _, testCase := range tests {
		t.Run(testCase.name, func(t *testing.T) {
			t.Parallel()

			ctx := new(Context)
			ctx.device = new(wgpu.Device)
			ctx.limits = testCase.limits
			deps := matrixDeps{
				createBuffer: func(*Context, *wgpu.BufferDescriptor) (*wgpu.Buffer, error) {
					t.Fatal("buffer creation must not exceed device limits")

					return nil, io.EOF
				},
				writeBuffer: nil,
				readBuffer:  nil,
			}

			matrix, err := newMatrix(ctx, 1, 1, deps)

			assert.Nil(t, matrix)
			require.ErrorContains(t, err, testCase.wantErr)
		})
	}
}

func TestDefaultContextDepsReleaseHelpers(t *testing.T) {
	t.Parallel()

	deps := defaultContextDeps()

	require.NotPanics(t, func() {
		deps.releaseDevice(nil)
		deps.releaseInstance(nil)
		deps.releaseAdapter(nil)
	})

	ctx, err := NewContext()
	require.NoError(t, err)
	require.NotNil(t, ctx)
	require.NotNil(t, ctx.device)
	require.NotNil(t, ctx.adapter)
	require.NotNil(t, ctx.instance)

	deps.releaseDevice(ctx.device)
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
	assert.Equal(t, UseAuto, mode)

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

func TestNewMatrixRejectsNilBuffer(t *testing.T) {
	t.Parallel()

	ctx := new(Context)
	ctx.device = new(wgpu.Device)
	deps := defaultMatrixDeps()
	deps.createBuffer = func(
		*Context,
		*wgpu.BufferDescriptor,
	) (*wgpu.Buffer, error) {
		return nil, nil //nolint:nilnil // verify defensive nil handling
	}

	matrix, err := newMatrix(ctx, 2, 2, deps)

	assert.Nil(t, matrix)
	require.ErrorIs(t, err, ErrBackendUnavailable)
	require.ErrorContains(t, err, "nil buffer")
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
	assert.Equal(t, 2, matrix.rows)
	assert.Equal(t, 2, matrix.cols)
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
	matrix.rows = 2
	matrix.cols = 2
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
	matrix.rows = 1
	matrix.cols = 1
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
	matrix.rows = 1
	matrix.cols = 2
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
	matrix.rows = 1
	matrix.cols = 1
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
	matrix.rows = 1
	matrix.cols = 2
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

func TestMatrixReadWriteRejectReleasedMatrix(t *testing.T) {
	t.Parallel()

	matrix, _ := newMockMatrix(1, 1, []float32{1})
	matrix.released.Store(1)

	err := matrix.Write([]float32{2})
	require.ErrorContains(t, err, "matrix is released")

	data, err := matrix.Read()
	assert.Nil(t, data)
	require.ErrorContains(t, err, "matrix is released")
}

func TestMatrixReadWriteRejectReleasedContext(t *testing.T) {
	t.Parallel()

	matrix, _ := newMockMatrix(1, 1, []float32{1})
	matrix.ctx.released.Store(1)

	err := matrix.Write([]float32{2})
	require.ErrorContains(t, err, "context is released")

	data, err := matrix.Read()
	assert.Nil(t, data)
	require.ErrorContains(t, err, "context is released")
}
