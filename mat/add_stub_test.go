package mat

import (
	"testing"

	"github.com/gogpu/gputypes"
	"github.com/gogpu/wgpu"
	"github.com/stretchr/testify/require"
)

// stubContext creates a Context that simulates a no-GPU stub environment:
// adapter is nil but pipes is initialized (not a mock). This covers the
// isCPUAdapter stub/fallback detection path (line 120-123).
func stubContext() *Context {
	return &Context{ //nolint:exhaustruct // stub context intentionally omits optional fields for testing.
		adapter: nil,
		pipes:   newPipelineCache(defaultReleaseComputePipeline),
		limits:  gputypes.Limits{}, //nolint:exhaustruct // stub limits are intentionally zeroed for testing.
	}
}

func TestIsCPUAdapter_stubContext(t *testing.T) {
	t.Parallel()

	ctx := stubContext()

	// Stub context (adapter==nil, pipes!=nil) should be detected as CPU.
	require.True(t, isCPUAdapter(ctx), "stub context should be detected as CPU adapter")
}

func TestIsCPUAdapter_realAdapter(t *testing.T) {
	t.Parallel()

	ctx := &Context{ //nolint:exhaustruct // real adapter test intentionally omits optional fields for testing.
		adapter: new(wgpu.Adapter),
		pipes:   newPipelineCache(defaultReleaseComputePipeline),
	}

	// Real adapter with DeviceTypeUnknown should NOT be detected as CPU.
	require.False(t, isCPUAdapter(ctx), "adapter with default DeviceType should not be detected as CPU")
}

func TestAdd_cpuFallback(t *testing.T) {
	t.Parallel()

	// Create stub context (adapter==nil, pipes!=nil) that triggers isCPUAdapter==true.
	stubCtx := stubContext()

	// Use mock matrices (ctx.adapter==nil && ctx.pipes==nil) so they pass as mocks.
	// Then manually assign the stub context to force the CPU fallback path.
	left, _ := newMockMatrix(2, 2, []float32{1, 2, 3, 4})
	right, _ := newMockMatrix(2, 2, []float32{5, 6, 7, 8})
	out, outStorage := newMockMatrix(2, 2, []float32{0, 0, 0, 0})

	// Replace the mock context with the stub context to trigger isCPUAdapter==true.
	left.ctx = stubCtx
	right.ctx = stubCtx
	out.ctx = stubCtx

	err := add(left, right, out, defaultAddDeps())
	require.NoError(t, err)

	// Should have executed the runBinaryElementwise CPU fallback, not dispatchAdd.
	require.Equal(t, []float32{6, 8, 10, 12}, outStorage.data,
		"stub context should have taken CPU fallback path")
}
