package mat

import (
	"io"
	"testing"

	"github.com/gogpu/gputypes"
	"github.com/gogpu/wgpu"
	"github.com/stretchr/testify/require"
)

//nolint:funlen // Complete declarative WGPU test double.
func successfulMatMulWGPUDeps() matMulWGPUDeps {
	return matMulWGPUDeps{
		createBindGroupLayout: func(
			*wgpu.Device, *wgpu.BindGroupLayoutDescriptor,
		) (*wgpu.BindGroupLayout, error) {
			return new(wgpu.BindGroupLayout), nil
		},
		getOrCreatePipeline: func(
			_ *Context,
			_ string,
			factory func() (*wgpu.ComputePipeline, error),
		) (*wgpu.ComputePipeline, error) {
			return factory()
		},
		createShaderModule: func(
			*wgpu.Device, *wgpu.ShaderModuleDescriptor,
		) (*wgpu.ShaderModule, error) {
			return new(wgpu.ShaderModule), nil
		},
		createPipelineLayout: func(
			*wgpu.Device, *wgpu.PipelineLayoutDescriptor,
		) (*wgpu.PipelineLayout, error) {
			return new(wgpu.PipelineLayout), nil
		},
		createComputePipeline: func(
			*wgpu.Device, *wgpu.ComputePipelineDescriptor,
		) (*wgpu.ComputePipeline, error) {
			return new(wgpu.ComputePipeline), nil
		},
		createBuffer: func(
			*wgpu.Device, *wgpu.BufferDescriptor,
		) (*wgpu.Buffer, error) {
			return new(wgpu.Buffer), nil
		},
		writeBuffer: func(*wgpu.Device, *wgpu.Buffer, uint64, []byte) error {
			return nil
		},
		createBindGroup: func(
			*wgpu.Device, *wgpu.BindGroupDescriptor,
		) (*wgpu.BindGroup, error) {
			return new(wgpu.BindGroup), nil
		},
		createCommandEncoder: func(
			*wgpu.Device, *wgpu.CommandEncoderDescriptor,
		) (*wgpu.CommandEncoder, error) {
			return new(wgpu.CommandEncoder), nil
		},
		beginComputePass: func(
			*wgpu.CommandEncoder, *wgpu.ComputePassDescriptor,
		) (*wgpu.ComputePassEncoder, error) {
			return new(wgpu.ComputePassEncoder), nil
		},
		setPipeline:    func(*wgpu.ComputePassEncoder, *wgpu.ComputePipeline) {},
		setBindGroup:   func(*wgpu.ComputePassEncoder, uint32, *wgpu.BindGroup, []uint32) {},
		dispatch:       func(*wgpu.ComputePassEncoder, uint32, uint32, uint32) {},
		endComputePass: func(*wgpu.ComputePassEncoder) error { return nil },
		finishCommandEncoder: func(*wgpu.CommandEncoder) (*wgpu.CommandBuffer, error) {
			return new(wgpu.CommandBuffer), nil
		},
		discardCommandEncoder:  func(*wgpu.CommandEncoder) {},
		submit:                 func(*wgpu.Device, *wgpu.CommandBuffer) error { return nil },
		releaseBindGroupLayout: func(*wgpu.BindGroupLayout) {},
		releaseShaderModule:    func(*wgpu.ShaderModule) {},
		releasePipelineLayout:  func(*wgpu.PipelineLayout) {},
		releaseComputePipeline: func(*wgpu.ComputePipeline) {},
		releaseBuffer:          func(*wgpu.Buffer) {},
		releaseBindGroup:       func(*wgpu.BindGroup) {},
		releaseCommandBuffer:   func(*wgpu.CommandBuffer) {},
	}
}

func matMulTestMatrices() (*Matrix, *Matrix, *Matrix) {
	left, _ := newMockMatrix(2, 2, []float32{1, 2, 3, 4})
	right, _ := newMockMatrix(2, 2, []float32{5, 6, 7, 8})
	out, _ := newMockMatrix(2, 2, []float32{0, 0, 0, 0})
	right.ctx = left.ctx
	out.ctx = left.ctx

	return left, right, out
}

//nolint:funlen // Error cases stay together for auditability.
func TestDispatchMatMulWGPUErrors(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name    string
		wantErr string
		mutate  func(*matMulWGPUDeps)
	}{
		{
			name:    "bind group layout",
			wantErr: "create matmul bind group layout",
			mutate: func(deps *matMulWGPUDeps) {
				deps.createBindGroupLayout = func(
					*wgpu.Device, *wgpu.BindGroupLayoutDescriptor,
				) (*wgpu.BindGroupLayout, error) {
					return nil, io.EOF
				}
			},
		},
		{
			name:    "pipeline cache",
			wantErr: "create matmul pipeline",
			mutate: func(deps *matMulWGPUDeps) {
				deps.getOrCreatePipeline = func(
					*Context, string, func() (*wgpu.ComputePipeline, error),
				) (*wgpu.ComputePipeline, error) {
					return nil, io.EOF
				}
			},
		},
		{
			name:    "uniform",
			wantErr: "create matmul uniform buffer",
			mutate: func(deps *matMulWGPUDeps) {
				deps.createBuffer = func(
					*wgpu.Device, *wgpu.BufferDescriptor,
				) (*wgpu.Buffer, error) {
					return nil, io.EOF
				}
			},
		},
		{
			name:    "bind group",
			wantErr: "create matmul bind group",
			mutate: func(deps *matMulWGPUDeps) {
				deps.createBindGroup = func(
					*wgpu.Device, *wgpu.BindGroupDescriptor,
				) (*wgpu.BindGroup, error) {
					return nil, io.EOF
				}
			},
		},
	}

	for _, testCase := range tests {
		t.Run(testCase.name, func(t *testing.T) {
			t.Parallel()

			left, right, out := matMulTestMatrices()
			deps := successfulMatMulWGPUDeps()
			testCase.mutate(&deps)

			err := dispatchMatMulWithDeps(left, right, out, deps)
			require.ErrorContains(t, err, testCase.wantErr)
		})
	}
}

func TestCreateMatMulPipelineErrors(t *testing.T) { //nolint:dupl // MatMul errors form an independent API contract.
	t.Parallel()

	tests := []struct {
		name    string
		wantErr string
		mutate  func(*matMulWGPUDeps)
	}{
		{
			name:    "shader",
			wantErr: "create matmul shader",
			mutate: func(deps *matMulWGPUDeps) {
				deps.createShaderModule = func(
					*wgpu.Device, *wgpu.ShaderModuleDescriptor,
				) (*wgpu.ShaderModule, error) {
					return nil, io.EOF
				}
			},
		},
		{
			name:    "layout",
			wantErr: "create matmul pipeline layout",
			mutate: func(deps *matMulWGPUDeps) {
				deps.createPipelineLayout = func(
					*wgpu.Device, *wgpu.PipelineLayoutDescriptor,
				) (*wgpu.PipelineLayout, error) {
					return nil, io.EOF
				}
			},
		},
		{
			name:    "compute pipeline",
			wantErr: "create matmul compute pipeline",
			mutate: func(deps *matMulWGPUDeps) {
				deps.createComputePipeline = func(
					*wgpu.Device, *wgpu.ComputePipelineDescriptor,
				) (*wgpu.ComputePipeline, error) {
					return nil, io.EOF
				}
			},
		},
	}

	for _, testCase := range tests {
		t.Run(testCase.name, func(t *testing.T) {
			t.Parallel()

			deps := successfulMatMulWGPUDeps()
			testCase.mutate(&deps)

			_, err := createMatMulPipeline(nil, new(wgpu.BindGroupLayout), deps)
			require.ErrorContains(t, err, testCase.wantErr)
		})
	}
}

func TestCreateMatMulUniformWriteError(t *testing.T) {
	t.Parallel()

	left, right, _ := matMulTestMatrices()
	deps := successfulMatMulWGPUDeps()
	deps.writeBuffer = func(*wgpu.Device, *wgpu.Buffer, uint64, []byte) error {
		return io.EOF
	}

	_, err := createMatMulUniform(left.ctx, left, right, deps)
	require.ErrorContains(t, err, "write matmul dimensions")
}

//nolint:funlen // Error cases stay together for auditability.
func TestEncodeAndSubmitMatMulErrors(t *testing.T) { //nolint:dupl // MatMul errors form an independent API contract.
	t.Parallel()

	tests := []struct {
		name    string
		wantErr string
		mutate  func(*matMulWGPUDeps)
	}{
		{
			name:    "command encoder",
			wantErr: "create matmul command encoder",
			mutate: func(deps *matMulWGPUDeps) {
				deps.createCommandEncoder = func(
					*wgpu.Device, *wgpu.CommandEncoderDescriptor,
				) (*wgpu.CommandEncoder, error) {
					return nil, io.EOF
				}
			},
		},
		{
			name:    "compute pass",
			wantErr: "begin matmul compute pass",
			mutate: func(deps *matMulWGPUDeps) {
				deps.beginComputePass = func(
					*wgpu.CommandEncoder, *wgpu.ComputePassDescriptor,
				) (*wgpu.ComputePassEncoder, error) {
					return nil, io.EOF
				}
			},
		},
		{
			name:    "end pass",
			wantErr: "end matmul compute pass",
			mutate: func(deps *matMulWGPUDeps) {
				deps.endComputePass = func(*wgpu.ComputePassEncoder) error { return io.EOF }
			},
		},
		{
			name:    testCaseFinishEncoder,
			wantErr: "finish matmul command encoder",
			mutate: func(deps *matMulWGPUDeps) {
				deps.finishCommandEncoder = func(
					*wgpu.CommandEncoder,
				) (*wgpu.CommandBuffer, error) {
					return nil, io.EOF
				}
			},
		},
		{
			name:    testCaseSubmit,
			wantErr: "submit matmul command buffer",
			mutate: func(deps *matMulWGPUDeps) {
				deps.submit = func(*wgpu.Device, *wgpu.CommandBuffer) error { return io.EOF }
			},
		},
	}

	for _, testCase := range tests {
		t.Run(testCase.name, func(t *testing.T) {
			t.Parallel()

			_, _, out := matMulTestMatrices()
			deps := successfulMatMulWGPUDeps()
			testCase.mutate(&deps)

			err := encodeAndSubmitMatMul(
				out.ctx,
				new(wgpu.ComputePipeline),
				new(wgpu.BindGroup),
				out,
				deps,
			)
			require.ErrorContains(t, err, testCase.wantErr)
		})
	}
}

func TestMatMulDescriptorHelpers(t *testing.T) {
	t.Parallel()

	deps := successfulMatMulWGPUDeps()
	_, err := createMatMulBindGroupLayout(nil, deps)
	require.NoError(t, err)

	left, right, out := matMulTestMatrices()
	_, err = createMatMulBindGroup(
		nil,
		new(wgpu.BindGroupLayout),
		new(wgpu.Buffer),
		left,
		right,
		out,
		deps,
	)
	require.NoError(t, err)

	entry := matMulLayoutEntry(1, gputypes.BufferBindingTypeStorage, 4)
	require.Equal(t, uint32(1), entry.Binding)
}
