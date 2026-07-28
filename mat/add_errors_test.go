package mat

import (
	"io"
	"math"
	"testing"

	"github.com/gogpu/wgpu"
	"github.com/stretchr/testify/require"
)

//nolint:funlen // Error stages stay together for auditability.
func TestDispatchAddWGPUStages(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name    string
		wantErr string
		mutate  func(*matMulWGPUDeps)
	}{
		{
			name:    "success",
			wantErr: "",
			mutate:  func(*matMulWGPUDeps) {},
		},
		{
			name:    "bind group layout",
			wantErr: "create add bind group layout",
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
			wantErr: "create add pipeline",
			mutate: func(deps *matMulWGPUDeps) {
				deps.getOrCreatePipeline = func(
					*Context, string, func() (*wgpu.ComputePipeline, error),
				) (*wgpu.ComputePipeline, error) {
					return nil, io.EOF
				}
			},
		},
		{
			name:    "bind group",
			wantErr: "create add bind group",
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

			err := dispatchAddWithDeps(left, right, out, deps)
			if testCase.wantErr == "" {
				require.NoError(t, err)

				return
			}

			require.ErrorContains(t, err, testCase.wantErr)
		})
	}
}

func TestCreateAddPipelineErrors(t *testing.T) { //nolint:dupl // Add errors form an independent API contract.
	t.Parallel()

	tests := []struct {
		name    string
		wantErr string
		mutate  func(*matMulWGPUDeps)
	}{
		{
			name:    "shader",
			wantErr: "create add shader",
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
			wantErr: "create add pipeline layout",
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
			wantErr: "create add compute pipeline",
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

			_, err := createAddPipeline(nil, new(wgpu.BindGroupLayout), deps)
			require.ErrorContains(t, err, testCase.wantErr)
		})
	}
}

//nolint:funlen // Error stages stay together for auditability.
func TestEncodeAndSubmitAddErrors(t *testing.T) { //nolint:dupl // Add errors form an independent API contract.
	t.Parallel()

	tests := []struct {
		name    string
		wantErr string
		mutate  func(*matMulWGPUDeps)
	}{
		{
			name:    "command encoder",
			wantErr: "create add command encoder",
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
			wantErr: "begin add compute pass",
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
			wantErr: "end add compute pass",
			mutate: func(deps *matMulWGPUDeps) {
				deps.endComputePass = func(*wgpu.ComputePassEncoder) error { return io.EOF }
			},
		},
		{
			name:    testCaseFinishEncoder,
			wantErr: "finish add command encoder",
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
			wantErr: "submit add command buffer",
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

			err := encodeAndSubmitAdd(
				nil,
				new(wgpu.ComputePipeline),
				new(wgpu.BindGroup),
				out,
				deps,
			)
			require.ErrorContains(t, err, testCase.wantErr)
		})
	}
}

func TestAddKernelLimits(t *testing.T) {
	t.Parallel()

	out, _ := newMockMatrix(0, 1, nil)
	err := validateAddKernelContract(out)
	require.ErrorContains(t, err, "matrix dimensions exceed add kernel limits")

	out.rows = math.MaxUint32
	out.cols = 2
	err = validateAddKernelContract(out)
	require.ErrorContains(t, err, "matrix dimensions exceed add kernel limits")

	out.rows = 257
	out.cols = 1
	out.ctx.limits.MaxComputeWorkgroupsPerDimension = 1
	err = validateAddKernelContract(out)
	require.ErrorContains(t, err, "add dispatch exceeds device workgroup limits")
}
