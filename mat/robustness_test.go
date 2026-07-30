package mat

import (
	"io"
	"math"
	"testing"

	"github.com/gogpu/gputypes"
	"github.com/gogpu/wgpu"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

var (
	_ io.Closer = (*Context)(nil)
	_ io.Closer = (*Matrix)(nil)
)

func TestContextModeString(t *testing.T) {
	t.Parallel()

	assert.Equal(t, "gpu", UseGPU.String())
	assert.Equal(t, "cpu", UseCPU.String())
	assert.Equal(t, "auto", UseAuto.String())
	assert.Equal(t, "ContextMode(99)", ContextMode(99).String())
}

func TestMatrixDiagnosticAPI(t *testing.T) {
	t.Parallel()

	matrix := new(Matrix)
	matrix.rows = 2
	matrix.cols = 3

	assert.Equal(t, 2, matrix.Rows())
	assert.Equal(t, 3, matrix.Cols())
	assert.Equal(t, 6, matrix.Len())
	assert.Equal(t, "Matrix[2x3]", matrix.String())
	assert.False(t, matrix.Released())

	shape := matrix.Shape()
	assert.Equal(t, 2, shape.Rows())
	assert.Equal(t, 3, shape.Cols())
	assert.Equal(t, 6, shape.Len())
	assert.Equal(t, "2x3", shape.String())

	require.NoError(t, matrix.Close())
	assert.True(t, matrix.Released())
	assert.Equal(t, "Matrix[2x3, released]", matrix.String())

	assert.Equal(t, 0, (*Matrix)(nil).Rows())
	assert.Equal(t, 0, (*Matrix)(nil).Cols())
	assert.Equal(t, 0, (*Matrix)(nil).Len())
	assert.Equal(t, Shape{rows: 0, cols: 0}, (*Matrix)(nil).Shape())
	assert.True(t, (*Matrix)(nil).Released())
	assert.Equal(t, "Matrix<nil>", (*Matrix)(nil).String())
	require.NoError(t, (*Matrix)(nil).Close())
}

func TestContextDiagnosticAPI(t *testing.T) {
	t.Parallel()

	ctx := new(Context)
	ctx.mode = UseCPU

	assert.Equal(t, UseCPU, ctx.Mode())
	assert.False(t, ctx.Released())
	require.NoError(t, ctx.Close())
	assert.True(t, ctx.Released())

	assert.Equal(t, UseAuto, (*Context)(nil).Mode())
	assert.True(t, (*Context)(nil).Released())
	require.NoError(t, (*Context)(nil).Close())
}

func TestValidationErrorsSupportErrorsIs(t *testing.T) {
	t.Parallel()

	left, _ := newMockMatrix(2, 3, make([]float32, 6))
	right, _ := newMockMatrix(4, 2, make([]float32, 8))
	out, _ := newMockMatrix(2, 2, make([]float32, 4))
	shareMockContext(left, right, out)

	err := MatMul(left, right, out)
	require.ErrorIs(t, err, ErrDimensionMismatch)
	require.ErrorContains(t, err, "left=2x3")
	require.ErrorContains(t, err, "right=4x2")
	require.ErrorContains(t, err, "out=2x2")

	err = left.Write(make([]float32, 5))
	require.ErrorIs(t, err, ErrLengthMismatch)
	require.ErrorContains(t, err, "2x3")

	otherContext, _ := newMockMatrix(2, 3, make([]float32, 6))
	err = Add(left, otherContext, out)
	require.ErrorIs(t, err, ErrContextMismatch)

	shareMockContext(left, otherContext)
	err = Add(left, otherContext, left)
	require.ErrorIs(t, err, ErrAliasedOutput)

	_, err = NewMatrix(nil, 1, 1)
	require.ErrorIs(t, err, ErrNilContext)

	_, err = NewContext(UseCPU, UseGPU)
	require.ErrorIs(t, err, ErrInvalidMode)

	left.released.Store(1)
	err = left.Write(make([]float32, 6))
	require.ErrorIs(t, err, ErrReleased)
}

type unaryContractCase struct {
	name    string
	newOut  func() *Matrix
	operate func(*Matrix, *Matrix) error
}

func unaryContractCases() []unaryContractCase {
	return []unaryContractCase{
		{
			name: "scale",
			newOut: func() *Matrix {
				out, _ := newMockMatrix(2, 2, make([]float32, 4))

				return out
			},
			operate: func(input, out *Matrix) error {
				return Scale(input, 2, out)
			},
		},
		{
			name: "row reduction",
			newOut: func() *Matrix {
				out, _ := newMockMatrix(2, 1, make([]float32, 2))

				return out
			},
			operate: ReduceSum,
		},
		{
			name: "transpose",
			newOut: func() *Matrix {
				out, _ := newMockMatrix(2, 2, make([]float32, 4))

				return out
			},
			operate: Transp,
		},
		{
			name: "softmax",
			newOut: func() *Matrix {
				out, _ := newMockMatrix(2, 2, make([]float32, 4))

				return out
			},
			operate: Softmax,
		},
		{
			name: "RMSNorm",
			newOut: func() *Matrix {
				out, _ := newMockMatrix(2, 2, make([]float32, 4))

				return out
			},
			operate: RMSNorm,
		},
	}
}

func TestUnaryOperationsRejectContextMismatchAndAliasing(t *testing.T) {
	t.Parallel()

	for _, testCase := range unaryContractCases() {
		t.Run(testCase.name, func(t *testing.T) {
			t.Parallel()
			assertUnaryContract(t, testCase)
		})
	}
}

func assertUnaryContract(t *testing.T, testCase unaryContractCase) {
	t.Helper()

	input, _ := newMockMatrix(2, 2, []float32{1, 2, 3, 4})
	out := testCase.newOut()

	err := testCase.operate(input, out)
	require.ErrorIs(t, err, ErrContextMismatch)

	shareMockContext(input, out)
	err = testCase.operate(input, input)
	require.ErrorIs(t, err, ErrAliasedOutput)
}

func TestValidateSameContextAcceptsFewerThanTwoMatrices(t *testing.T) {
	t.Parallel()

	require.NoError(t, validateSameContext())
	require.NoError(t, validateSameContext(new(Matrix)))
}

func TestAddReturnsKernelValidationError(t *testing.T) {
	t.Parallel()

	left, _ := newMockMatrix(257, 1, make([]float32, 257))
	right, _ := newMockMatrix(257, 1, make([]float32, 257))
	out, _ := newMockMatrix(257, 1, make([]float32, 257))
	shareMockContext(left, right, out)
	left.ctx.limits.MaxComputeWorkgroupsPerDimension = 1

	err := add(left, right, out, addDeps{
		dispatchBroadcast: nil,
		dispatch: func(*Matrix, *Matrix, *Matrix) error {
			t.Fatal("invalid dispatch must not run")

			return nil
		},
	})

	require.ErrorIs(t, err, ErrDeviceLimit)
}

func TestMatMulCPUReadErrors(t *testing.T) {
	t.Parallel()

	left, leftStorage := newMockMatrix(1, 2, []float32{1, 2})
	right, rightStorage := newMockMatrix(2, 1, []float32{3, 4})
	out, _ := newMockMatrix(1, 1, []float32{0})
	shareMockContext(left, right, out)
	left.ctx.infoSet = true
	left.ctx.isCPU = true

	leftStorage.readErr = io.EOF
	err := MatMul(left, right, out)
	require.ErrorContains(t, err, "failed to read left")

	leftStorage.readErr = nil
	rightStorage.readErr = io.EOF
	err = MatMul(left, right, out)
	require.ErrorContains(t, err, "failed to read right")
}

func TestNewContextUseAutoFallsBackAfterAdapterFailure(t *testing.T) {
	t.Parallel()

	deps := newTestContextDeps()
	calls := 0

	deps.requestAdapter = func(
		_ *wgpu.Instance,
		options *wgpu.RequestAdapterOptions,
	) (*wgpu.Adapter, error) {
		calls++

		if calls == 1 {
			assert.False(t, options.ForceFallbackAdapter)

			return nil, io.EOF
		}

		assert.True(t, options.ForceFallbackAdapter)

		return new(wgpu.Adapter), nil
	}
	deps.adapterInfo = func(*wgpu.Adapter) gputypes.AdapterInfo {
		info := new(gputypes.AdapterInfo)
		info.DeviceType = gputypes.DeviceTypeCPU

		return *info
	}

	ctx, err := newContext(deps, UseAuto)

	require.NoError(t, err)
	require.NotNil(t, ctx)
	assert.Equal(t, 2, calls)
	assert.Equal(t, UseAuto, ctx.Mode())
	assert.True(t, ctx.infoSet)
	assert.True(t, ctx.isCPU)
}

func TestNewContextReleasesAdapterReturnedWithError(t *testing.T) {
	t.Parallel()

	deps := newTestContextDeps()
	adapter := new(wgpu.Adapter)
	releases := 0
	deps.requestAdapter = func(
		*wgpu.Instance,
		*wgpu.RequestAdapterOptions,
	) (*wgpu.Adapter, error) {
		return adapter, io.EOF
	}
	deps.releaseAdapter = func(got *wgpu.Adapter) {
		assert.Same(t, adapter, got)

		releases++
	}

	ctx, err := newContext(deps, UseGPU)

	assert.Nil(t, ctx)
	require.ErrorIs(t, err, ErrBackendUnavailable)
	assert.Equal(t, 1, releases)
}

func TestNewContextReleasesDeviceReturnedWithError(t *testing.T) {
	t.Parallel()

	deps := newTestContextDeps()
	device := new(wgpu.Device)
	deviceReleases := 0
	adapterReleases := 0
	deps.requestDevice = func(
		*wgpu.Adapter,
		*wgpu.DeviceDescriptor,
	) (*wgpu.Device, error) {
		return device, io.EOF
	}
	deps.releaseDevice = func(got *wgpu.Device) {
		assert.Same(t, device, got)

		deviceReleases++
	}
	deps.releaseAdapter = func(*wgpu.Adapter) {
		adapterReleases++
	}

	ctx, err := newContext(deps, UseGPU)

	assert.Nil(t, ctx)
	require.ErrorIs(t, err, ErrBackendUnavailable)
	assert.Equal(t, 1, deviceReleases)
	assert.Equal(t, 1, adapterReleases)
}

func TestRequestFirstAvailableDeviceRejectsNoOptions(t *testing.T) {
	t.Parallel()

	adapter, device, err := requestFirstAvailableDevice(
		newTestContextDeps(),
		new(wgpu.Instance),
		nil,
	)

	assert.Nil(t, adapter)
	assert.Nil(t, device)
	require.ErrorIs(t, err, ErrBackendUnavailable)
}

func TestNewContextUseAutoFallsBackAfterDeviceFailure(t *testing.T) {
	t.Parallel()

	deps := newTestContextDeps()
	firstAdapter := new(wgpu.Adapter)
	secondAdapter := new(wgpu.Adapter)
	adapterCalls := 0
	deviceCalls := 0
	releasedAdapters := 0

	deps.requestAdapter = func(
		*wgpu.Instance,
		*wgpu.RequestAdapterOptions,
	) (*wgpu.Adapter, error) {
		adapterCalls++
		if adapterCalls == 1 {
			return firstAdapter, nil
		}

		return secondAdapter, nil
	}
	deps.requestDevice = func(
		adapter *wgpu.Adapter,
		_ *wgpu.DeviceDescriptor,
	) (*wgpu.Device, error) {
		deviceCalls++

		if adapter == firstAdapter {
			return nil, io.EOF
		}

		return new(wgpu.Device), nil
	}
	deps.releaseAdapter = func(adapter *wgpu.Adapter) {
		assert.Same(t, firstAdapter, adapter)

		releasedAdapters++
	}

	ctx, err := newContext(deps, UseAuto)

	require.NoError(t, err)
	require.NotNil(t, ctx)
	assert.Same(t, secondAdapter, ctx.adapter)
	assert.Equal(t, 2, adapterCalls)
	assert.Equal(t, 2, deviceCalls)
	assert.Equal(t, 1, releasedAdapters)
}

func TestNewContextUseGPUDoesNotRetryFallback(t *testing.T) {
	t.Parallel()

	deps := newTestContextDeps()
	calls := 0
	deps.requestAdapter = func(
		*wgpu.Instance,
		*wgpu.RequestAdapterOptions,
	) (*wgpu.Adapter, error) {
		calls++

		return nil, io.EOF
	}

	ctx, err := newContext(deps, UseGPU)

	assert.Nil(t, ctx)
	require.ErrorIs(t, err, ErrBackendUnavailable)
	require.ErrorIs(t, err, io.EOF)
	assert.Equal(t, 1, calls)
}

func TestNewContextUseGPURejectsCPUAdapter(t *testing.T) {
	t.Parallel()

	deps := newTestContextDeps()
	deviceReleases := 0
	adapterReleases := 0
	instanceReleases := 0
	deps.adapterInfo = func(*wgpu.Adapter) gputypes.AdapterInfo {
		info := new(gputypes.AdapterInfo)
		info.DeviceType = gputypes.DeviceTypeCPU

		return *info
	}
	deps.releaseDevice = func(*wgpu.Device) {
		deviceReleases++
	}
	deps.releaseAdapter = func(*wgpu.Adapter) {
		adapterReleases++
	}
	deps.releaseInstance = func(*wgpu.Instance) {
		instanceReleases++
	}

	ctx, err := newContext(deps, UseGPU)

	assert.Nil(t, ctx)
	require.ErrorIs(t, err, ErrBackendUnavailable)
	require.ErrorContains(t, err, "selected a CPU adapter")
	assert.Equal(t, 1, deviceReleases)
	assert.Equal(t, 1, adapterReleases)
	assert.Equal(t, 1, instanceReleases)
}

func TestNewContextRejectsNilBackendObjects(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name   string
		mutate func(*contextDeps)
	}{
		{
			name: "instance",
			mutate: func(deps *contextDeps) {
				deps.createInstance = func(*wgpu.InstanceDescriptor) (*wgpu.Instance, error) {
					return nil, nil //nolint:nilnil // verify defensive nil handling
				}
			},
		},
		{
			name: "adapter",
			mutate: func(deps *contextDeps) {
				deps.requestAdapter = func(
					*wgpu.Instance,
					*wgpu.RequestAdapterOptions,
				) (*wgpu.Adapter, error) {
					return nil, nil //nolint:nilnil // verify defensive nil handling
				}
			},
		},
		{
			name: "device",
			mutate: func(deps *contextDeps) {
				deps.requestDevice = func(
					*wgpu.Adapter,
					*wgpu.DeviceDescriptor,
				) (*wgpu.Device, error) {
					return nil, nil //nolint:nilnil // verify defensive nil handling
				}
			},
		},
	}

	for _, testCase := range tests {
		t.Run(testCase.name, func(t *testing.T) {
			t.Parallel()

			deps := newTestContextDeps()
			testCase.mutate(&deps)

			ctx, err := newContext(deps, UseGPU)

			assert.Nil(t, ctx)
			require.ErrorIs(t, err, ErrBackendUnavailable)
		})
	}
}

func TestReadBufferAlwaysReleasesFinishedCommandBuffer(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name   string
		mutate func(*readBufferDeps)
	}{
		{name: "success", mutate: func(*readBufferDeps) {}},
		{
			name: "submit error",
			mutate: func(deps *readBufferDeps) {
				deps.submit = func(*Context, *wgpu.CommandBuffer) error {
					return io.EOF
				}
			},
		},
		{
			name: "map error",
			mutate: func(deps *readBufferDeps) {
				deps.mapBuffer = func(*wgpu.Buffer, uint64) error {
					return io.EOF
				}
			},
		},
	}

	for _, testCase := range tests {
		t.Run(testCase.name, func(t *testing.T) {
			t.Parallel()

			releases := 0
			deps := newTestReadBufferDeps(make([]byte, bytesPerFloat32Int))
			deps.releaseCommandBuffer = func(*wgpu.CommandBuffer) {
				releases++
			}
			testCase.mutate(&deps)

			_ = readBuffer(
				new(Context),
				new(wgpu.Buffer),
				make([]byte, bytesPerFloat32Int),
				deps,
			)

			assert.Equal(t, 1, releases)
		})
	}
}

func TestSentinelWrapPreservesBothCauses(t *testing.T) {
	t.Parallel()

	err := sentinelWrapError(ErrBackendUnavailable, io.EOF, "request adapter")

	require.ErrorIs(t, err, ErrBackendUnavailable)
	require.ErrorIs(t, err, io.EOF)
	assert.Equal(t, "mat: request adapter: EOF", err.Error())
}

func TestSentinelWrapAcceptsNilCause(t *testing.T) {
	t.Parallel()

	err := sentinelWrapError(ErrBackendUnavailable, nil, "request adapter")

	require.ErrorIs(t, err, ErrBackendUnavailable)
	assert.Equal(t, "mat: request adapter", err.Error())
}

func TestSoftmaxHandlesInfiniteAndNaNInputs(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name  string
		input []float32
		check func(*testing.T, []float32)
	}{
		{
			name:  "positive infinities share probability",
			input: []float32{float32(math.Inf(1)), 1, float32(math.Inf(1))},
			check: func(t *testing.T, output []float32) {
				t.Helper()
				assert.Equal(t, []float32{0.5, 0, 0.5}, output)
			},
		},
		{
			name: "all negative infinity becomes uniform",
			input: []float32{
				float32(math.Inf(-1)),
				float32(math.Inf(-1)),
			},
			check: func(t *testing.T, output []float32) {
				t.Helper()
				assert.Equal(t, []float32{0.5, 0.5}, output)
			},
		},
		{
			name:  "NaN propagates to the row",
			input: []float32{1, float32(math.NaN()), 2},
			check: func(t *testing.T, output []float32) {
				t.Helper()

				for _, value := range output {
					assert.True(t, math.IsNaN(float64(value)))
				}
			},
		},
	}

	for _, testCase := range tests {
		t.Run(testCase.name, func(t *testing.T) {
			t.Parallel()

			output := make([]float32, len(testCase.input))
			applySoftmaxRow(testCase.input, output, 0, len(testCase.input))
			testCase.check(t, output)
		})
	}
}

func TestRMSNormAvoidsFloat32SquareOverflow(t *testing.T) {
	t.Parallel()

	input := []float32{math.MaxFloat32, -math.MaxFloat32}
	output := make([]float32, len(input))

	applyRMSNormRow(input, output, 0, len(input))

	assert.InDelta(t, 1, output[0], 1e-6)
	assert.InDelta(t, -1, output[1], 1e-6)
}

func TestReduceMaxPreservesNegativeInfinity(t *testing.T) {
	t.Parallel()

	input, _ := newMockMatrix(
		1,
		2,
		[]float32{float32(math.Inf(-1)), float32(math.Inf(-1))},
	)
	out, storage := newMockMatrix(1, 1, []float32{0})
	shareMockContext(input, out)

	err := ReduceMax(input, out)

	require.NoError(t, err)
	require.Len(t, storage.data, 1)
	assert.True(t, math.IsInf(float64(storage.data[0]), -1))
}
