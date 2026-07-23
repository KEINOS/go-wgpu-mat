package mat

import (
	"encoding/binary"
	"math"

	"github.com/gogpu/gputypes"
	"github.com/gogpu/wgpu"
)

const (
	matMulPipelineKey       = "matmul:f32"
	matMulUniformSize       = uint64(16)
	matMulWorkgroup         = uint32(8)
	matMulLeftBinding       = uint32(0)
	matMulRightBinding      = uint32(1)
	matMulOutputBinding     = uint32(2)
	matMulDimensionsBinding = uint32(3)
)

const matMulWGSL = `
struct Dimensions {
    rows: u32,
    shared: u32,
    cols: u32,
    padding: u32,
}

@group(0) @binding(0) var<storage, read> left: array<f32>;
@group(0) @binding(1) var<storage, read> right: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> dimensions: Dimensions;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let col = id.x;
    let row = id.y;
    if (row >= dimensions.rows || col >= dimensions.cols) {
        return;
    }

    var sum = 0.0;
    for (var k = 0u; k < dimensions.shared; k++) {
        sum += left[row * dimensions.shared + k] *
            right[k * dimensions.cols + col];
    }
    output[row * dimensions.cols + col] = sum;
}
`

type matMulDeps struct {
	dispatch func(left, right, out *Matrix) error
}

func defaultMatMulDeps() matMulDeps {
	return matMulDeps{dispatch: dispatchMatMul}
}

func matMul(left, right, out *Matrix, deps matMulDeps) error {
	err := validateMatMul(left, right, out)
	if err != nil {
		return err
	}

	err = deps.dispatch(left, right, out)
	if err != nil {
		return wrapError(err, "failed to dispatch matmul")
	}

	return nil
}

func validateMatMul(left, right, out *Matrix) error {
	err := validateMatrixInitialized("left", left)
	if err != nil {
		return err
	}

	err = validateMatrixInitialized("right", right)
	if err != nil {
		return err
	}

	err = validateMatrixInitialized("out", out)
	if err != nil {
		return err
	}

	err = validateMatMulDims(left, right, out)
	if err != nil {
		return err
	}

	err = validateMatMulKernelContract(left, right, out)
	if err != nil {
		return err
	}

	return nil
}

func validateMatMulKernelContract(left, right, out *Matrix) error {
	if left.ctx != right.ctx || left.ctx != out.ctx {
		return newError("matrices must use the same context")
	}

	if out == left || out == right {
		return newError("out must not alias an input")
	}

	if left.Rows > math.MaxUint32 || left.Cols > math.MaxUint32 ||
		right.Cols > math.MaxUint32 {
		return newError("matrix dimensions exceed GPU kernel limits")
	}

	return nil
}

func dispatchMatMul(left, right, out *Matrix) error {
	ctx := left.ctx
	device := ctx.device

	bindGroupLayout, err := createMatMulBindGroupLayout(device)
	if err != nil {
		return err
	}
	defer bindGroupLayout.Release()

	pipeline, err := ctx.getOrCreatePipeline(matMulPipelineKey, func() (*wgpu.ComputePipeline, error) {
		return createMatMulPipeline(device, bindGroupLayout)
	})
	if err != nil {
		return wrapError(err, "create matmul pipeline")
	}

	uniform, err := createMatMulUniform(device, left, right)
	if err != nil {
		return err
	}
	defer uniform.Release()

	bindGroup, err := createMatMulBindGroup(device, bindGroupLayout, uniform, left, right, out)
	if err != nil {
		return err
	}
	defer bindGroup.Release()

	return encodeAndSubmitMatMul(device, pipeline, bindGroup, out)
}

func encodeAndSubmitMatMul(
	device *wgpu.Device,
	pipeline *wgpu.ComputePipeline,
	bindGroup *wgpu.BindGroup,
	out *Matrix,
) error {
	encoder, err := device.CreateCommandEncoder(&wgpu.CommandEncoderDescriptor{
		Label: "go-wgpu-mat-matmul-encoder",
	})
	if err != nil {
		return wrapError(err, "create matmul command encoder")
	}

	pass, err := encoder.BeginComputePass(nil)
	if err != nil {
		return wrapError(err, "begin matmul compute pass")
	}

	pass.SetPipeline(pipeline)
	pass.SetBindGroup(0, bindGroup, nil)
	pass.Dispatch(
		ceilDiv(dimensionU32(out.Cols), matMulWorkgroup),
		ceilDiv(dimensionU32(out.Rows), matMulWorkgroup),
		1,
	)

	err = pass.End()
	if err != nil {
		return wrapError(err, "end matmul compute pass")
	}

	commandBuffer, err := encoder.Finish()
	if err != nil {
		return wrapError(err, "finish matmul command encoder")
	}
	defer commandBuffer.Release()

	_, err = device.Queue().Submit(commandBuffer)
	if err != nil {
		return wrapError(err, "submit matmul command buffer")
	}

	return nil
}

func createMatMulBindGroupLayout(device *wgpu.Device) (*wgpu.BindGroupLayout, error) {
	layout, err := device.CreateBindGroupLayout(&wgpu.BindGroupLayoutDescriptor{
		Label: "go-wgpu-mat-matmul-bind-group-layout",
		Entries: []wgpu.BindGroupLayoutEntry{
			matMulLayoutEntry(matMulLeftBinding, gputypes.BufferBindingTypeReadOnlyStorage, 0),
			matMulLayoutEntry(matMulRightBinding, gputypes.BufferBindingTypeReadOnlyStorage, 0),
			matMulLayoutEntry(matMulOutputBinding, gputypes.BufferBindingTypeStorage, 0),
			matMulLayoutEntry(matMulDimensionsBinding, gputypes.BufferBindingTypeUniform, matMulUniformSize),
		},
	})
	if err != nil {
		return nil, wrapError(err, "create matmul bind group layout")
	}

	return layout, nil
}

func createMatMulPipeline(
	device *wgpu.Device,
	bindGroupLayout *wgpu.BindGroupLayout,
) (*wgpu.ComputePipeline, error) {
	shader, err := device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label: "go-wgpu-mat-matmul-shader",
		WGSL:  matMulWGSL,
		SPIRV: nil,
	})
	if err != nil {
		return nil, wrapError(err, "create matmul shader")
	}
	defer shader.Release()

	pipelineLayout, err := device.CreatePipelineLayout(&wgpu.PipelineLayoutDescriptor{
		Label:            "go-wgpu-mat-matmul-pipeline-layout",
		BindGroupLayouts: []*wgpu.BindGroupLayout{bindGroupLayout},
	})
	if err != nil {
		return nil, wrapError(err, "create matmul pipeline layout")
	}
	defer pipelineLayout.Release()

	pipeline, err := device.CreateComputePipeline(&wgpu.ComputePipelineDescriptor{
		Label:                         "go-wgpu-mat-matmul-pipeline",
		Layout:                        pipelineLayout,
		Module:                        shader,
		EntryPoint:                    "main",
		Constants:                     nil,
		ZeroInitializeWorkgroupMemory: nil,
	})
	if err != nil {
		return nil, wrapError(err, "create matmul compute pipeline")
	}

	return pipeline, nil
}

func matrixByteSize(matrix *Matrix) uint64 {
	rows := uint64(matrix.Rows) //nolint:gosec // NewMatrix requires positive dimensions.
	cols := uint64(matrix.Cols) //nolint:gosec // NewMatrix requires positive dimensions.

	return rows * cols * bytesPerFloat32U64
}

func ceilDiv(value, divisor uint32) uint32 {
	return (value-1)/divisor + 1
}

func dimensionU32(value int) uint32 {
	return uint32(value) //nolint:gosec // validateMatMul rejects values above uint32.
}

func createMatMulUniform(device *wgpu.Device, left, right *Matrix) (*wgpu.Buffer, error) {
	uniform, err := device.CreateBuffer(&wgpu.BufferDescriptor{
		Label:            "go-wgpu-mat-matmul-dimensions",
		Size:             matMulUniformSize,
		Usage:            wgpu.BufferUsageUniform | wgpu.BufferUsageCopyDst,
		MappedAtCreation: false,
	})
	if err != nil {
		return nil, wrapError(err, "create matmul uniform buffer")
	}

	dimensions := make([]byte, matMulUniformSize)
	binary.LittleEndian.PutUint32(dimensions[0:4], dimensionU32(left.Rows))
	binary.LittleEndian.PutUint32(dimensions[4:8], dimensionU32(left.Cols))
	binary.LittleEndian.PutUint32(dimensions[8:12], dimensionU32(right.Cols))

	err = device.Queue().WriteBuffer(uniform, 0, dimensions)
	if err != nil {
		uniform.Release()

		return nil, wrapError(err, "write matmul dimensions")
	}

	return uniform, nil
}

func createMatMulBindGroup(
	device *wgpu.Device,
	layout *wgpu.BindGroupLayout,
	uniform *wgpu.Buffer,
	left, right, out *Matrix,
) (*wgpu.BindGroup, error) {
	bindGroup, err := device.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Label:  "go-wgpu-mat-matmul-bind-group",
		Layout: layout,
		Entries: []wgpu.BindGroupEntry{
			matMulBufferEntry(matMulLeftBinding, left.buf, matrixByteSize(left)),
			matMulBufferEntry(matMulRightBinding, right.buf, matrixByteSize(right)),
			matMulBufferEntry(matMulOutputBinding, out.buf, matrixByteSize(out)),
			matMulBufferEntry(matMulDimensionsBinding, uniform, matMulUniformSize),
		},
	})
	if err != nil {
		return nil, wrapError(err, "create matmul bind group")
	}

	return bindGroup, nil
}

func matMulBufferEntry(binding uint32, buffer *wgpu.Buffer, size uint64) wgpu.BindGroupEntry {
	return wgpu.BindGroupEntry{
		Binding:     binding,
		Buffer:      buffer,
		Offset:      0,
		Size:        size,
		Sampler:     nil,
		TextureView: nil,
	}
}

func matMulLayoutEntry(
	binding uint32,
	bindingType gputypes.BufferBindingType,
	minSize uint64,
) wgpu.BindGroupLayoutEntry {
	return wgpu.BindGroupLayoutEntry{
		Binding:    binding,
		Visibility: wgpu.ShaderStageCompute,
		Buffer: &gputypes.BufferBindingLayout{
			Type:             bindingType,
			HasDynamicOffset: false,
			MinBindingSize:   minSize,
		},
		Sampler:        nil,
		Texture:        nil,
		StorageTexture: nil,
	}
}
