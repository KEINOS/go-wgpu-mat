//nolint:lll,nilnil,paralleltest,tparallel,wsl_v5 // Sequential fault injection returns invalid pairs intentionally.
package mat

import (
	"io"
	"testing"

	"github.com/gogpu/wgpu"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestCreateComputePipelinePartialAndNilResources(t *testing.T) { //nolint:funlen // Resource stages are audited together.
	t.Parallel()

	t.Run("shader partial error", func(t *testing.T) {
		deps := successfulMatMulWGPUDeps()
		released := 0
		deps.createShaderModule = func(*wgpu.Device, *wgpu.ShaderModuleDescriptor) (*wgpu.ShaderModule, error) {
			return new(wgpu.ShaderModule), io.EOF
		}
		deps.releaseShaderModule = func(*wgpu.ShaderModule) { released++ }
		_, err := createComputePipeline(nil, nil, "test", "", deps)
		require.ErrorIs(t, err, io.EOF)
		assert.Equal(t, 1, released)
	})

	t.Run("shader nil success", func(t *testing.T) {
		deps := successfulMatMulWGPUDeps()
		deps.createShaderModule = func(*wgpu.Device, *wgpu.ShaderModuleDescriptor) (*wgpu.ShaderModule, error) {
			return nil, nil
		}
		_, err := createComputePipeline(nil, nil, "test", "", deps)
		require.ErrorIs(t, err, ErrBackendUnavailable)
	})

	t.Run("layout partial error", func(t *testing.T) {
		deps := successfulMatMulWGPUDeps()
		released := 0
		deps.createPipelineLayout = func(*wgpu.Device, *wgpu.PipelineLayoutDescriptor) (*wgpu.PipelineLayout, error) {
			return new(wgpu.PipelineLayout), io.EOF
		}
		deps.releasePipelineLayout = func(*wgpu.PipelineLayout) { released++ }
		_, err := createComputePipeline(nil, nil, "test", "", deps)
		require.ErrorIs(t, err, io.EOF)
		assert.Equal(t, 1, released)
	})

	t.Run("layout nil success", func(t *testing.T) {
		deps := successfulMatMulWGPUDeps()
		deps.createPipelineLayout = func(*wgpu.Device, *wgpu.PipelineLayoutDescriptor) (*wgpu.PipelineLayout, error) {
			return nil, nil
		}
		_, err := createComputePipeline(nil, nil, "test", "", deps)
		require.ErrorIs(t, err, ErrBackendUnavailable)
	})

	t.Run("pipeline partial error", func(t *testing.T) {
		deps := successfulMatMulWGPUDeps()
		released := 0
		deps.createComputePipeline = func(*wgpu.Device, *wgpu.ComputePipelineDescriptor) (*wgpu.ComputePipeline, error) {
			return new(wgpu.ComputePipeline), io.EOF
		}
		deps.releaseComputePipeline = func(*wgpu.ComputePipeline) { released++ }
		_, err := createComputePipeline(nil, nil, "test", "", deps)
		require.ErrorIs(t, err, io.EOF)
		assert.Equal(t, 1, released)
	})

	t.Run("pipeline nil success", func(t *testing.T) {
		deps := successfulMatMulWGPUDeps()
		deps.createComputePipeline = func(*wgpu.Device, *wgpu.ComputePipelineDescriptor) (*wgpu.ComputePipeline, error) {
			return nil, nil
		}
		_, err := createComputePipeline(nil, nil, "test", "", deps)
		require.ErrorIs(t, err, ErrBackendUnavailable)
	})
}

func TestEncodeComputePartialAndNilResources(t *testing.T) { //nolint:funlen // Resource stages are audited together.
	t.Parallel()

	ctx := new(Context)
	ctx.device = new(wgpu.Device)
	call := func(deps matMulWGPUDeps) error {
		return encodeAndSubmitCompute(
			ctx, new(wgpu.ComputePipeline), new(wgpu.BindGroup),
			computeDispatch{x: 1, y: 1, z: 1}, "test", deps,
		)
	}

	t.Run("encoder partial error", func(t *testing.T) {
		deps := successfulMatMulWGPUDeps()
		discarded := 0
		deps.createCommandEncoder = func(*wgpu.Device, *wgpu.CommandEncoderDescriptor) (*wgpu.CommandEncoder, error) {
			return new(wgpu.CommandEncoder), io.EOF
		}
		deps.discardCommandEncoder = func(*wgpu.CommandEncoder) { discarded++ }
		require.ErrorIs(t, call(deps), io.EOF)
		assert.Equal(t, 1, discarded)
	})

	t.Run("encoder nil success", func(t *testing.T) {
		deps := successfulMatMulWGPUDeps()
		deps.createCommandEncoder = func(*wgpu.Device, *wgpu.CommandEncoderDescriptor) (*wgpu.CommandEncoder, error) {
			return nil, nil
		}
		require.ErrorIs(t, call(deps), ErrBackendUnavailable)
	})

	t.Run("pass partial error", func(t *testing.T) {
		deps := successfulMatMulWGPUDeps()
		ended := 0
		discarded := 0
		deps.beginComputePass = func(*wgpu.CommandEncoder, *wgpu.ComputePassDescriptor) (*wgpu.ComputePassEncoder, error) {
			return new(wgpu.ComputePassEncoder), io.EOF
		}
		deps.endComputePass = func(*wgpu.ComputePassEncoder) error {
			ended++

			return nil
		}
		deps.discardCommandEncoder = func(*wgpu.CommandEncoder) { discarded++ }
		require.ErrorIs(t, call(deps), io.EOF)
		assert.Equal(t, 1, ended)
		assert.Equal(t, 1, discarded)
	})

	t.Run("pass nil success", func(t *testing.T) {
		deps := successfulMatMulWGPUDeps()
		discarded := 0
		deps.beginComputePass = func(*wgpu.CommandEncoder, *wgpu.ComputePassDescriptor) (*wgpu.ComputePassEncoder, error) {
			return nil, nil
		}
		deps.discardCommandEncoder = func(*wgpu.CommandEncoder) { discarded++ }
		require.ErrorIs(t, call(deps), ErrBackendUnavailable)
		assert.Equal(t, 1, discarded)
	})

	t.Run("finish partial error", func(t *testing.T) {
		deps := successfulMatMulWGPUDeps()
		released := 0
		deps.finishCommandEncoder = func(*wgpu.CommandEncoder) (*wgpu.CommandBuffer, error) {
			return new(wgpu.CommandBuffer), io.EOF
		}
		deps.releaseCommandBuffer = func(*wgpu.CommandBuffer) { released++ }
		require.ErrorIs(t, call(deps), io.EOF)
		assert.Equal(t, 1, released)
	})

	t.Run("finish nil success", func(t *testing.T) {
		deps := successfulMatMulWGPUDeps()
		deps.finishCommandEncoder = func(*wgpu.CommandEncoder) (*wgpu.CommandBuffer, error) {
			return nil, nil
		}
		require.ErrorIs(t, call(deps), ErrBackendUnavailable)
	})
}

func TestTensorOperationPartialAndNilResources(t *testing.T) { //nolint:funlen // Resource stages are audited together.
	t.Parallel()

	left, right, out := matMulTestMatrices()
	markHardwareMock(left, right, out)

	t.Run("layout partial and nil", func(t *testing.T) {
		deps := successfulMatMulWGPUDeps()
		released := 0
		deps.releaseBindGroupLayout = func(*wgpu.BindGroupLayout) { released++ }
		deps.createBindGroupLayout = func(*wgpu.Device, *wgpu.BindGroupLayoutDescriptor) (*wgpu.BindGroupLayout, error) {
			return new(wgpu.BindGroupLayout), io.EOF
		}
		_, err := createTensorOpBindGroupLayout(nil, deps)
		require.ErrorIs(t, err, io.EOF)
		assert.Equal(t, 1, released)
		deps.createBindGroupLayout = func(*wgpu.Device, *wgpu.BindGroupLayoutDescriptor) (*wgpu.BindGroupLayout, error) {
			return nil, nil
		}
		_, err = createTensorOpBindGroupLayout(nil, deps)
		require.ErrorIs(t, err, ErrBackendUnavailable)
	})

	t.Run("uniform partial nil and write", func(t *testing.T) {
		deps := successfulMatMulWGPUDeps()
		released := 0
		deps.releaseBuffer = func(*wgpu.Buffer) { released++ }
		deps.createBuffer = func(*wgpu.Device, *wgpu.BufferDescriptor) (*wgpu.Buffer, error) {
			return new(wgpu.Buffer), io.EOF
		}
		_, err := createTensorOpUniform(left.ctx, tensorOpMul, left, right, out, 0, deps)
		require.ErrorIs(t, err, io.EOF)
		assert.Equal(t, 1, released)

		deps.createBuffer = func(*wgpu.Device, *wgpu.BufferDescriptor) (*wgpu.Buffer, error) { return nil, nil }
		_, err = createTensorOpUniform(left.ctx, tensorOpMul, left, right, out, 0, deps)
		require.ErrorIs(t, err, ErrBackendUnavailable)

		deps.createBuffer = func(*wgpu.Device, *wgpu.BufferDescriptor) (*wgpu.Buffer, error) {
			return new(wgpu.Buffer), nil
		}
		deps.writeBuffer = func(*wgpu.Device, *wgpu.Buffer, uint64, []byte) error { return io.EOF }
		_, err = createTensorOpUniform(left.ctx, tensorOpMul, left, right, out, 0, deps)
		require.ErrorIs(t, err, io.EOF)
		assert.Equal(t, 2, released)
	})

	t.Run("bind group partial and nil", func(t *testing.T) {
		deps := successfulMatMulWGPUDeps()
		released := 0
		deps.releaseBindGroup = func(*wgpu.BindGroup) { released++ }
		deps.createBindGroup = func(*wgpu.Device, *wgpu.BindGroupDescriptor) (*wgpu.BindGroup, error) {
			return new(wgpu.BindGroup), io.EOF
		}
		_, err := createTensorOpBindGroup(nil, nil, nil, left, right, out, deps)
		require.ErrorIs(t, err, io.EOF)
		assert.Equal(t, 1, released)
		deps.createBindGroup = func(*wgpu.Device, *wgpu.BindGroupDescriptor) (*wgpu.BindGroup, error) {
			return nil, nil
		}
		_, err = createTensorOpBindGroup(nil, nil, nil, left, right, out, deps)
		require.ErrorIs(t, err, ErrBackendUnavailable)
	})

	t.Run("dispatch stages", func(t *testing.T) {
		badOut, _ := newMockMatrix(0, 1, nil)
		badOut.ctx = left.ctx
		err := dispatchTensorOperationWithDeps(tensorOpMul, left, right, badOut, 0, successfulMatMulWGPUDeps())
		require.ErrorIs(t, err, ErrKernelLimit)

		deps := successfulMatMulWGPUDeps()
		deps.createBindGroupLayout = func(*wgpu.Device, *wgpu.BindGroupLayoutDescriptor) (*wgpu.BindGroupLayout, error) {
			return nil, io.EOF
		}
		err = dispatchTensorOperationWithDeps(tensorOpMul, left, right, out, 0, deps)
		require.ErrorIs(t, err, io.EOF)

		deps = successfulMatMulWGPUDeps()
		deps.getOrCreatePipeline = func(*Context, string, func() (*wgpu.ComputePipeline, error)) (*wgpu.ComputePipeline, error) {
			return new(wgpu.ComputePipeline), io.EOF
		}
		released := 0
		deps.releaseComputePipeline = func(*wgpu.ComputePipeline) { released++ }
		err = dispatchTensorOperationWithDeps(tensorOpMul, left, right, out, 0, deps)
		require.ErrorIs(t, err, io.EOF)
		assert.Equal(t, 1, released)

		deps = successfulMatMulWGPUDeps()
		deps.getOrCreatePipeline = func(*Context, string, func() (*wgpu.ComputePipeline, error)) (*wgpu.ComputePipeline, error) {
			return nil, nil
		}
		err = dispatchTensorOperationWithDeps(tensorOpMul, left, right, out, 0, deps)
		require.ErrorIs(t, err, ErrBackendUnavailable)

		deps = successfulMatMulWGPUDeps()
		deps.createBuffer = func(*wgpu.Device, *wgpu.BufferDescriptor) (*wgpu.Buffer, error) { return nil, io.EOF }
		err = dispatchTensorOperationWithDeps(tensorOpMul, left, right, out, 0, deps)
		require.ErrorIs(t, err, io.EOF)

		deps = successfulMatMulWGPUDeps()
		deps.createBindGroup = func(*wgpu.Device, *wgpu.BindGroupDescriptor) (*wgpu.BindGroup, error) {
			return nil, io.EOF
		}
		err = dispatchTensorOperationWithDeps(tensorOpMul, left, right, out, 0, deps)
		require.ErrorIs(t, err, io.EOF)
	})
}

func TestExistingKernelPartialAndNilResources(t *testing.T) { //nolint:funlen // Add and MatMul preserve the same cleanup contract.
	t.Parallel()

	left, right, out := matMulTestMatrices()
	markHardwareMock(left, right, out)

	for _, dispatch := range []struct {
		name string
		call func(matMulWGPUDeps) error
	}{
		{name: "add", call: func(deps matMulWGPUDeps) error { return dispatchAddWithDeps(left, right, out, deps) }},
		{name: "matmul", call: func(deps matMulWGPUDeps) error { return dispatchMatMulWithDeps(left, right, out, deps) }},
	} {
		t.Run(dispatch.name+" layout partial", func(t *testing.T) {
			deps := successfulMatMulWGPUDeps()
			deps.createBindGroupLayout = func(*wgpu.Device, *wgpu.BindGroupLayoutDescriptor) (*wgpu.BindGroupLayout, error) {
				return new(wgpu.BindGroupLayout), io.EOF
			}
			released := 0
			deps.releaseBindGroupLayout = func(*wgpu.BindGroupLayout) { released++ }
			require.ErrorIs(t, dispatch.call(deps), io.EOF)
			assert.Equal(t, 1, released)
		})

		t.Run(dispatch.name+" layout nil", func(t *testing.T) {
			deps := successfulMatMulWGPUDeps()
			deps.createBindGroupLayout = func(*wgpu.Device, *wgpu.BindGroupLayoutDescriptor) (*wgpu.BindGroupLayout, error) {
				return nil, nil
			}
			require.ErrorIs(t, dispatch.call(deps), ErrBackendUnavailable)
		})

		t.Run(dispatch.name+" pipeline partial", func(t *testing.T) {
			deps := successfulMatMulWGPUDeps()
			deps.getOrCreatePipeline = func(*Context, string, func() (*wgpu.ComputePipeline, error)) (*wgpu.ComputePipeline, error) {
				return new(wgpu.ComputePipeline), io.EOF
			}
			released := 0
			deps.releaseComputePipeline = func(*wgpu.ComputePipeline) { released++ }
			require.ErrorIs(t, dispatch.call(deps), io.EOF)
			assert.Equal(t, 1, released)
		})

		t.Run(dispatch.name+" pipeline nil", func(t *testing.T) {
			deps := successfulMatMulWGPUDeps()
			deps.getOrCreatePipeline = func(*Context, string, func() (*wgpu.ComputePipeline, error)) (*wgpu.ComputePipeline, error) {
				return nil, nil
			}
			require.ErrorIs(t, dispatch.call(deps), ErrBackendUnavailable)
		})

		t.Run(dispatch.name+" bind group partial", func(t *testing.T) {
			deps := successfulMatMulWGPUDeps()
			deps.createBindGroup = func(*wgpu.Device, *wgpu.BindGroupDescriptor) (*wgpu.BindGroup, error) {
				return new(wgpu.BindGroup), io.EOF
			}
			released := 0
			deps.releaseBindGroup = func(*wgpu.BindGroup) { released++ }
			require.ErrorIs(t, dispatch.call(deps), io.EOF)
			assert.Equal(t, 1, released)
		})

		t.Run(dispatch.name+" bind group nil", func(t *testing.T) {
			deps := successfulMatMulWGPUDeps()
			deps.createBindGroup = func(*wgpu.Device, *wgpu.BindGroupDescriptor) (*wgpu.BindGroup, error) {
				return nil, nil
			}
			require.ErrorIs(t, dispatch.call(deps), ErrBackendUnavailable)
		})
	}

	t.Run("matmul uniform partial", func(t *testing.T) {
		deps := successfulMatMulWGPUDeps()
		deps.createBuffer = func(*wgpu.Device, *wgpu.BufferDescriptor) (*wgpu.Buffer, error) {
			return new(wgpu.Buffer), io.EOF
		}
		released := 0
		deps.releaseBuffer = func(*wgpu.Buffer) { released++ }
		require.ErrorIs(t, dispatchMatMulWithDeps(left, right, out, deps), io.EOF)
		assert.Equal(t, 1, released)
	})

	t.Run("matmul uniform nil", func(t *testing.T) {
		deps := successfulMatMulWGPUDeps()
		deps.createBuffer = func(*wgpu.Device, *wgpu.BufferDescriptor) (*wgpu.Buffer, error) {
			return nil, nil
		}
		require.ErrorIs(t, dispatchMatMulWithDeps(left, right, out, deps), ErrBackendUnavailable)
	})
}

func TestMatrixAndReadbackPartialResources(t *testing.T) {
	t.Parallel()

	ctx := new(Context)
	ctx.device = new(wgpu.Device)
	released := 0
	deps := matrixDeps{
		createBuffer: func(*Context, *wgpu.BufferDescriptor) (*wgpu.Buffer, error) {
			return new(wgpu.Buffer), io.EOF
		},
		releaseBuffer: func(*wgpu.Buffer) { released++ },
		writeBuffer:   nil,
		readBuffer:    nil,
	}
	_, err := newMatrix(ctx, 1, 1, deps)
	require.ErrorIs(t, err, io.EOF)
	assert.Equal(t, 1, released)

	readDeps := newTestReadBufferDeps(nil)
	readDeps.createStaging = func(*Context, uint64) (*wgpu.Buffer, error) {
		return new(wgpu.Buffer), io.EOF
	}
	readDeps.releaseBuffer = func(*wgpu.Buffer) { released++ }
	err = readBuffer(ctx, new(wgpu.Buffer), make([]byte, 4), readDeps)
	require.ErrorIs(t, err, io.EOF)
	assert.Equal(t, 2, released)

	readDeps.createStaging = func(*Context, uint64) (*wgpu.Buffer, error) { return nil, nil }
	err = readBuffer(ctx, new(wgpu.Buffer), make([]byte, 4), readDeps)
	require.ErrorIs(t, err, ErrBackendUnavailable)
}
