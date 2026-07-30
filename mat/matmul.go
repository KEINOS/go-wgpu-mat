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

type matMulWGPUDeps struct {
	createBindGroupLayout func(
		*wgpu.Device, *wgpu.BindGroupLayoutDescriptor,
	) (*wgpu.BindGroupLayout, error)
	getOrCreatePipeline func(
		*Context, string, func() (*wgpu.ComputePipeline, error),
	) (*wgpu.ComputePipeline, error)
	createShaderModule func(
		*wgpu.Device, *wgpu.ShaderModuleDescriptor,
	) (*wgpu.ShaderModule, error)
	createPipelineLayout func(
		*wgpu.Device, *wgpu.PipelineLayoutDescriptor,
	) (*wgpu.PipelineLayout, error)
	createComputePipeline func(
		*wgpu.Device, *wgpu.ComputePipelineDescriptor,
	) (*wgpu.ComputePipeline, error)
	createBuffer func(
		*wgpu.Device, *wgpu.BufferDescriptor,
	) (*wgpu.Buffer, error)
	writeBuffer     func(*wgpu.Device, *wgpu.Buffer, uint64, []byte) error
	createBindGroup func(
		*wgpu.Device, *wgpu.BindGroupDescriptor,
	) (*wgpu.BindGroup, error)
	createCommandEncoder func(
		*wgpu.Device, *wgpu.CommandEncoderDescriptor,
	) (*wgpu.CommandEncoder, error)
	beginComputePass func(
		*wgpu.CommandEncoder, *wgpu.ComputePassDescriptor,
	) (*wgpu.ComputePassEncoder, error)
	setPipeline            func(*wgpu.ComputePassEncoder, *wgpu.ComputePipeline)
	setBindGroup           func(*wgpu.ComputePassEncoder, uint32, *wgpu.BindGroup, []uint32)
	dispatch               func(*wgpu.ComputePassEncoder, uint32, uint32, uint32)
	endComputePass         func(*wgpu.ComputePassEncoder) error
	finishCommandEncoder   func(*wgpu.CommandEncoder) (*wgpu.CommandBuffer, error)
	discardCommandEncoder  func(*wgpu.CommandEncoder)
	submit                 func(*wgpu.Device, *wgpu.CommandBuffer) error
	releaseBindGroupLayout func(*wgpu.BindGroupLayout)
	releaseShaderModule    func(*wgpu.ShaderModule)
	releasePipelineLayout  func(*wgpu.PipelineLayout)
	releaseComputePipeline func(*wgpu.ComputePipeline)
	releaseBuffer          func(*wgpu.Buffer)
	releaseBindGroup       func(*wgpu.BindGroup)
	releaseCommandBuffer   func(*wgpu.CommandBuffer)
}

func defaultMatMulDeps() matMulDeps {
	return matMulDeps{dispatch: dispatchMatMul}
}

func defaultMatMulWGPUDeps() matMulWGPUDeps {
	deps := new(matMulWGPUDeps)
	setMatMulResourceDeps(deps)
	setMatMulCommandDeps(deps)
	setMatMulReleaseDeps(deps)

	return *deps
}

func setMatMulResourceDeps(deps *matMulWGPUDeps) {
	deps.createBindGroupLayout = func(
		device *wgpu.Device,
		descriptor *wgpu.BindGroupLayoutDescriptor,
	) (*wgpu.BindGroupLayout, error) {
		return device.CreateBindGroupLayout(descriptor)
	}
	deps.getOrCreatePipeline = func(
		ctx *Context,
		key string,
		factory func() (*wgpu.ComputePipeline, error),
	) (*wgpu.ComputePipeline, error) {
		return ctx.getOrCreatePipeline(key, factory)
	}
	deps.createShaderModule = func(
		device *wgpu.Device,
		descriptor *wgpu.ShaderModuleDescriptor,
	) (*wgpu.ShaderModule, error) {
		return device.CreateShaderModule(descriptor)
	}
	deps.createPipelineLayout = func(
		device *wgpu.Device,
		descriptor *wgpu.PipelineLayoutDescriptor,
	) (*wgpu.PipelineLayout, error) {
		return device.CreatePipelineLayout(descriptor)
	}
	deps.createComputePipeline = func(
		device *wgpu.Device,
		descriptor *wgpu.ComputePipelineDescriptor,
	) (*wgpu.ComputePipeline, error) {
		return device.CreateComputePipeline(descriptor)
	}
	deps.createBuffer = func(
		device *wgpu.Device,
		descriptor *wgpu.BufferDescriptor,
	) (*wgpu.Buffer, error) {
		return device.CreateBuffer(descriptor)
	}
	deps.writeBuffer = func(
		device *wgpu.Device,
		buffer *wgpu.Buffer,
		offset uint64,
		data []byte,
	) error {
		return device.Queue().WriteBuffer(buffer, offset, data)
	}
	deps.createBindGroup = func(
		device *wgpu.Device,
		descriptor *wgpu.BindGroupDescriptor,
	) (*wgpu.BindGroup, error) {
		return device.CreateBindGroup(descriptor)
	}
}

func setMatMulCommandDeps(deps *matMulWGPUDeps) {
	deps.createCommandEncoder = func(
		device *wgpu.Device,
		descriptor *wgpu.CommandEncoderDescriptor,
	) (*wgpu.CommandEncoder, error) {
		return device.CreateCommandEncoder(descriptor)
	}
	deps.beginComputePass = func(
		encoder *wgpu.CommandEncoder,
		descriptor *wgpu.ComputePassDescriptor,
	) (*wgpu.ComputePassEncoder, error) {
		return encoder.BeginComputePass(descriptor)
	}
	deps.setPipeline = func(pass *wgpu.ComputePassEncoder, pipeline *wgpu.ComputePipeline) {
		pass.SetPipeline(pipeline)
	}
	deps.setBindGroup = func(
		pass *wgpu.ComputePassEncoder,
		index uint32,
		bindGroup *wgpu.BindGroup,
		dynamicOffsets []uint32,
	) {
		pass.SetBindGroup(index, bindGroup, dynamicOffsets)
	}
	deps.dispatch = func(pass *wgpu.ComputePassEncoder, x, y, z uint32) {
		pass.Dispatch(x, y, z)
	}
	deps.endComputePass = func(pass *wgpu.ComputePassEncoder) error {
		return pass.End()
	}
	deps.finishCommandEncoder = func(
		encoder *wgpu.CommandEncoder,
	) (*wgpu.CommandBuffer, error) {
		return encoder.Finish()
	}
	deps.discardCommandEncoder = (*wgpu.CommandEncoder).DiscardEncoding
	deps.submit = func(device *wgpu.Device, commandBuffer *wgpu.CommandBuffer) error {
		_, err := device.Queue().Submit(commandBuffer)

		return wrapError(err, "submit command buffer")
	}
}

func setMatMulReleaseDeps(deps *matMulWGPUDeps) {
	deps.releaseBindGroupLayout = func(layout *wgpu.BindGroupLayout) { layout.Release() }
	deps.releaseShaderModule = func(shader *wgpu.ShaderModule) { shader.Release() }
	deps.releasePipelineLayout = func(layout *wgpu.PipelineLayout) { layout.Release() }
	deps.releaseComputePipeline = (*wgpu.ComputePipeline).Release
	deps.releaseBuffer = func(buffer *wgpu.Buffer) { buffer.Release() }
	deps.releaseBindGroup = func(bindGroup *wgpu.BindGroup) { bindGroup.Release() }
	deps.releaseCommandBuffer = func(commandBuffer *wgpu.CommandBuffer) { commandBuffer.Release() }
}

func matMul(left, right, out *Matrix, deps matMulDeps) error {
	err := validateMatMul(left, right, out)
	if err != nil {
		return err
	}

	// Detect GPU unavailability and fall back to CPU.
	if isCPUAdapter(left.ctx) {
		return matMulCPU(left, right, out)
	}

	err = validateMatMulKernelContract(left, right, out)
	if err != nil {
		return err
	}

	err = deps.dispatch(left, right, out)
	if err != nil {
		return wrapError(err, "failed to dispatch matmul")
	}

	return nil
}

func matMulCPU(left, right, out *Matrix) error {
	leftData, err := left.Read()
	if err != nil {
		return wrapError(err, "failed to read left")
	}

	rightData, err := right.Read()
	if err != nil {
		return wrapError(err, "failed to read right")
	}

	result := make([]float32, out.rows*out.cols)
	for row := range left.rows {
		for col := range right.cols {
			var sum float32
			for k := range left.cols {
				sum += leftData[row*left.cols+k] * rightData[k*right.cols+col]
			}

			result[row*right.cols+col] = sum
		}
	}

	return out.Write(result)
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

	err = validateSameContext(left, right, out)
	if err != nil {
		return err
	}

	err = validateOutputNotAliased(out, left, right)
	if err != nil {
		return err
	}

	return nil
}

func validateMatMulKernelContract(left, right, out *Matrix) error {
	if left.rows > math.MaxUint32 || left.cols > math.MaxUint32 ||
		right.cols > math.MaxUint32 {
		return sentinelError(
			ErrKernelLimit,
			"matrix dimensions exceed GPU kernel limits: left=%s right=%s",
			left.Shape(),
			right.Shape(),
		)
	}

	return validateMatMulDispatchLimits(left.ctx, out)
}

func validateMatMulDispatchLimits(ctx *Context, out *Matrix) error {
	maxWorkgroups := ctx.limits.MaxComputeWorkgroupsPerDimension
	xWorkgroups := ceilDiv(dimensionU32(out.cols), matMulWorkgroup)
	yWorkgroups := ceilDiv(dimensionU32(out.rows), matMulWorkgroup)

	if maxWorkgroups > 0 &&
		(xWorkgroups > maxWorkgroups || yWorkgroups > maxWorkgroups) {
		return sentinelError(
			ErrDeviceLimit,
			"matmul dispatch exceeds device workgroup limits: need %dx%d, max %d per dimension",
			xWorkgroups,
			yWorkgroups,
			maxWorkgroups,
		)
	}

	return nil
}

func dispatchMatMul(left, right, out *Matrix) error {
	return dispatchMatMulWithDeps(left, right, out, defaultMatMulWGPUDeps())
}

func dispatchMatMulWithDeps(left, right, out *Matrix, deps matMulWGPUDeps) error {
	ctx := left.ctx
	device := ctx.device

	bindGroupLayout, err := createMatMulBindGroupLayout(device, deps)
	if err != nil {
		return err
	}
	defer deps.releaseBindGroupLayout(bindGroupLayout)

	pipeline, err := deps.getOrCreatePipeline(ctx, matMulPipelineKey, func() (*wgpu.ComputePipeline, error) {
		return createMatMulPipeline(device, bindGroupLayout, deps)
	})
	if err != nil {
		if pipeline != nil {
			deps.releaseComputePipeline(pipeline)
		}

		return wrapError(err, "create matmul pipeline")
	}

	if pipeline == nil {
		return sentinelError(ErrBackendUnavailable, "create matmul pipeline returned nil")
	}

	uniform, err := createMatMulUniform(ctx, left, right, deps)
	if err != nil {
		return err
	}
	defer func() {
		deps.releaseBuffer(uniform)
		ctx.recordBufferRelease()
	}()

	bindGroup, err := createMatMulBindGroup(device, bindGroupLayout, uniform, left, right, out, deps)
	if err != nil {
		return err
	}
	defer deps.releaseBindGroup(bindGroup)

	return encodeAndSubmitMatMul(ctx, pipeline, bindGroup, out, deps)
}

func encodeAndSubmitMatMul(
	ctx *Context,
	pipeline *wgpu.ComputePipeline,
	bindGroup *wgpu.BindGroup,
	out *Matrix,
	deps matMulWGPUDeps,
) error {
	return encodeAndSubmitCompute(
		ctx,
		pipeline,
		bindGroup,
		computeDispatch{
			x: ceilDiv(dimensionU32(out.cols), matMulWorkgroup),
			y: ceilDiv(dimensionU32(out.rows), matMulWorkgroup),
			z: 1,
		},
		"matmul",
		deps,
	)
}

func createMatMulBindGroupLayout(
	device *wgpu.Device,
	deps matMulWGPUDeps,
) (*wgpu.BindGroupLayout, error) {
	layout, err := deps.createBindGroupLayout(device, &wgpu.BindGroupLayoutDescriptor{
		Label: "go-wgpu-mat-matmul-bind-group-layout",
		Entries: []wgpu.BindGroupLayoutEntry{
			matMulLayoutEntry(matMulLeftBinding, gputypes.BufferBindingTypeReadOnlyStorage, 0),
			matMulLayoutEntry(matMulRightBinding, gputypes.BufferBindingTypeReadOnlyStorage, 0),
			matMulLayoutEntry(matMulOutputBinding, gputypes.BufferBindingTypeStorage, 0),
			matMulLayoutEntry(matMulDimensionsBinding, gputypes.BufferBindingTypeUniform, matMulUniformSize),
		},
	})
	if err != nil {
		if layout != nil {
			deps.releaseBindGroupLayout(layout)
		}

		return nil, wrapError(err, "create matmul bind group layout")
	}

	if layout == nil {
		return nil, sentinelError(ErrBackendUnavailable, "create matmul bind group layout returned nil")
	}

	return layout, nil
}

func createMatMulPipeline(
	device *wgpu.Device,
	bindGroupLayout *wgpu.BindGroupLayout,
	deps matMulWGPUDeps,
) (*wgpu.ComputePipeline, error) {
	return createComputePipeline(device, bindGroupLayout, "matmul", matMulWGSL, deps)
}

func matrixByteSize(matrix *Matrix) uint64 {
	rows := uint64(matrix.rows) //nolint:gosec // NewMatrix requires positive dimensions.
	cols := uint64(matrix.cols) //nolint:gosec // NewMatrix requires positive dimensions.

	return rows * cols * bytesPerFloat32U64
}

func ceilDiv(value, divisor uint32) uint32 {
	return (value-1)/divisor + 1
}

func dimensionU32(value int) uint32 {
	return uint32(value) //nolint:gosec // validateMatMul rejects values above uint32.
}

func createMatMulUniform(
	ctx *Context,
	left, right *Matrix,
	deps matMulWGPUDeps,
) (*wgpu.Buffer, error) {
	uniform, err := deps.createBuffer(ctx.device, &wgpu.BufferDescriptor{
		Label:            "go-wgpu-mat-matmul-dimensions",
		Size:             matMulUniformSize,
		Usage:            wgpu.BufferUsageUniform | wgpu.BufferUsageCopyDst,
		MappedAtCreation: false,
	})
	if err != nil {
		if uniform != nil {
			deps.releaseBuffer(uniform)
		}

		return nil, wrapError(err, "create matmul uniform buffer")
	}

	if uniform == nil {
		return nil, sentinelError(ErrBackendUnavailable, "create matmul uniform buffer returned nil")
	}

	ctx.recordBufferAllocation()

	dimensions := make([]byte, matMulUniformSize)
	binary.LittleEndian.PutUint32(dimensions[0:4], dimensionU32(left.rows))
	binary.LittleEndian.PutUint32(dimensions[4:8], dimensionU32(left.cols))
	binary.LittleEndian.PutUint32(dimensions[8:12], dimensionU32(right.cols))

	err = ctx.withQueue(func() error {
		return deps.writeBuffer(ctx.device, uniform, 0, dimensions)
	})
	if err != nil {
		deps.releaseBuffer(uniform)
		ctx.recordBufferRelease()

		return nil, wrapError(err, "write matmul dimensions")
	}

	return uniform, nil
}

func createMatMulBindGroup(
	device *wgpu.Device,
	layout *wgpu.BindGroupLayout,
	uniform *wgpu.Buffer,
	left, right, out *Matrix,
	deps matMulWGPUDeps,
) (*wgpu.BindGroup, error) {
	bindGroup, err := deps.createBindGroup(device, &wgpu.BindGroupDescriptor{
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
		if bindGroup != nil {
			deps.releaseBindGroup(bindGroup)
		}

		return nil, wrapError(err, "create matmul bind group")
	}

	if bindGroup == nil {
		return nil, sentinelError(ErrBackendUnavailable, "create matmul bind group returned nil")
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
