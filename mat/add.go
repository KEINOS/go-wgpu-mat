package mat

import (
	"math"

	"github.com/gogpu/gputypes"
	"github.com/gogpu/wgpu"
)

const (
	addPipelineKey   = "add:f32"
	addWorkgroupSize = uint32(256)
	addLeftBinding   = uint32(0)
	addRightBinding  = uint32(1)
	addOutputBinding = uint32(2)
)

const addWGSL = `
@group(0) @binding(0) var<storage, read> left: array<f32>;
@group(0) @binding(1) var<storage, read> right: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    if (index >= arrayLength(&output)) {
        return;
    }

    output[index] = left[index] + right[index];
}
`

type addDeps struct {
	dispatch func(left, right, out *Matrix) error
}

func defaultAddDeps() addDeps {
	return addDeps{dispatch: dispatchAdd}
}

func add(left, right, out *Matrix, deps addDeps) error {
	err := validateAdd(left, right, out)
	if err != nil {
		return err
	}

	if left.ctx != right.ctx || left.ctx != out.ctx || out == left || out == right {
		return runBinaryElementwise(left, right, out, func(a, b float32) float32 {
			return a + b
		})
	}

	// Detect GPU unavailability (e.g., no GPU on CI runner) and fall back to CPU.
	// When the adapter is a software/CPU adapter, the WGSL kernel silently returns zeros.
	// In this case, we use the software path which reads/writes buffers directly.
	if isCPUAdapter(left.ctx) {
		return runBinaryElementwise(left, right, out, func(a, b float32) float32 {
			return a + b
		})
	}

	err = validateAddKernelContract(out)
	if err != nil {
		return err
	}

	err = deps.dispatch(left, right, out)
	if err != nil {
		return wrapError(err, "failed to dispatch add")
	}

	return nil
}

func validateAdd(left, right, out *Matrix) error {
	matrices := []struct {
		name   string
		matrix *Matrix
	}{
		{name: "left", matrix: left},
		{name: "right", matrix: right},
		{name: "out", matrix: out},
	}

	for _, item := range matrices {
		err := validateMatrixInitialized(item.name, item.matrix)
		if err != nil {
			return err
		}
	}

	return validateSameShape(left, right, out)
}

func validateAddKernelContract(out *Matrix) error {
	rows := uint64(out.Rows) //nolint:gosec // NewMatrix requires positive dimensions.
	cols := uint64(out.Cols) //nolint:gosec // NewMatrix requires positive dimensions.

	elementCount := rows * cols
	if elementCount == 0 || elementCount > math.MaxUint32 {
		return newError("matrix dimensions exceed add kernel limits")
	}

	workgroups := ceilDiv(uint32(elementCount), addWorkgroupSize)

	maxWorkgroups := out.ctx.limits.MaxComputeWorkgroupsPerDimension
	if maxWorkgroups > 0 && workgroups > maxWorkgroups {
		return newError("add dispatch exceeds device workgroup limits")
	}

	return nil
}

func isCPUAdapter(ctx *Context) bool {
	if ctx.adapter == nil {
		// Not a real GPU context (e.g., test mock) — keep original behavior.
		return false
	}

	info := ctx.adapter.Info()

	return info.DeviceType == gputypes.DeviceTypeCPU
}

func dispatchAdd(left, right, out *Matrix) error {
	return dispatchAddWithDeps(left, right, out, defaultMatMulWGPUDeps())
}

func dispatchAddWithDeps(left, right, out *Matrix, deps matMulWGPUDeps) error {
	device := left.ctx.device

	bindGroupLayout, err := createAddBindGroupLayout(device, deps)
	if err != nil {
		return err
	}
	defer deps.releaseBindGroupLayout(bindGroupLayout)

	pipeline, err := deps.getOrCreatePipeline(
		left.ctx,
		addPipelineKey,
		func() (*wgpu.ComputePipeline, error) {
			return createAddPipeline(device, bindGroupLayout, deps)
		},
	)
	if err != nil {
		return wrapError(err, "create add pipeline")
	}

	bindGroup, err := createAddBindGroup(device, bindGroupLayout, left, right, out, deps)
	if err != nil {
		return err
	}
	defer deps.releaseBindGroup(bindGroup)

	return encodeAndSubmitAdd(device, pipeline, bindGroup, out, deps)
}

func createAddBindGroupLayout(
	device *wgpu.Device,
	deps matMulWGPUDeps,
) (*wgpu.BindGroupLayout, error) {
	layout, err := deps.createBindGroupLayout(device, &wgpu.BindGroupLayoutDescriptor{
		Label: "go-wgpu-mat-add-bind-group-layout",
		Entries: []wgpu.BindGroupLayoutEntry{
			matMulLayoutEntry(addLeftBinding, gputypes.BufferBindingTypeReadOnlyStorage, 0),
			matMulLayoutEntry(addRightBinding, gputypes.BufferBindingTypeReadOnlyStorage, 0),
			matMulLayoutEntry(addOutputBinding, gputypes.BufferBindingTypeStorage, 0),
		},
	})
	if err != nil {
		return nil, wrapError(err, "create add bind group layout")
	}

	return layout, nil
}

func createAddPipeline(
	device *wgpu.Device,
	bindGroupLayout *wgpu.BindGroupLayout,
	deps matMulWGPUDeps,
) (*wgpu.ComputePipeline, error) {
	return createComputePipeline(device, bindGroupLayout, "add", addWGSL, deps)
}

func createAddBindGroup(
	device *wgpu.Device,
	layout *wgpu.BindGroupLayout,
	left, right, out *Matrix,
	deps matMulWGPUDeps,
) (*wgpu.BindGroup, error) {
	bindGroup, err := deps.createBindGroup(device, &wgpu.BindGroupDescriptor{
		Label:  "go-wgpu-mat-add-bind-group",
		Layout: layout,
		Entries: []wgpu.BindGroupEntry{
			matMulBufferEntry(addLeftBinding, left.buf, matrixByteSize(left)),
			matMulBufferEntry(addRightBinding, right.buf, matrixByteSize(right)),
			matMulBufferEntry(addOutputBinding, out.buf, matrixByteSize(out)),
		},
	})
	if err != nil {
		return nil, wrapError(err, "create add bind group")
	}

	return bindGroup, nil
}

func encodeAndSubmitAdd(
	device *wgpu.Device,
	pipeline *wgpu.ComputePipeline,
	bindGroup *wgpu.BindGroup,
	out *Matrix,
	deps matMulWGPUDeps,
) error {
	rows := uint64(out.Rows)            //nolint:gosec // validateAddKernelContract checks this conversion.
	cols := uint64(out.Cols)            //nolint:gosec // validateAddKernelContract checks this conversion.
	elementCount := uint32(rows * cols) //nolint:gosec // Dispatch follows validateAddKernelContract.

	return encodeAndSubmitCompute(
		device,
		pipeline,
		bindGroup,
		computeDispatch{x: ceilDiv(elementCount, addWorkgroupSize), y: 1, z: 1},
		"add",
		deps,
	)
}
