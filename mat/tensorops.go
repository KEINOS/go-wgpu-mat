package mat

import (
	"encoding/binary"
	"math"

	"github.com/gogpu/gputypes"
	"github.com/gogpu/wgpu"
)

type tensorOperation uint32

type tensorOpExecutionDeps struct {
	dispatch func(tensorOperation, *Matrix, *Matrix, *Matrix, float32) error
}

const (
	tensorOpAdd tensorOperation = iota
	tensorOpMul
	tensorOpScale
	tensorOpTranspose
	tensorOpReduceSumTo
	tensorOpBroadcastTo
	tensorOpReshapeTo
)

const (
	tensorOpPipelineKey   = "tensorops:f32"
	tensorOpUniformSize   = uint64(32)
	tensorOpWorkgroupSize = uint32(256)
	tensorOpLeftBinding   = uint32(0)
	tensorOpRightBinding  = uint32(1)
	tensorOpOutputBinding = uint32(2)
	tensorOpParamsBinding = uint32(3)
)

//nolint:gosec // WGSL shader source contains no credential.
const tensorOpWGSL = `
struct Params {
    operation: u32,
    output_rows: u32,
    output_cols: u32,
    left_rows: u32,
    left_cols: u32,
    right_rows: u32,
    right_cols: u32,
    scalar: f32,
}

@group(0) @binding(0) var<storage, read> left: array<f32>;
@group(0) @binding(1) var<storage, read> right: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

fn source_index(rows: u32, cols: u32, row: u32, col: u32) -> u32 {
    let source_row = select(row, 0u, rows == 1u);
    let source_col = select(col, 0u, cols == 1u);
    return source_row * cols + source_col;
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    let output_len = params.output_rows * params.output_cols;
    if (index >= output_len) {
        return;
    }

    let row = index / params.output_cols;
    let col = index % params.output_cols;

    if (params.operation == 0u || params.operation == 1u) {
        let left_index = source_index(params.left_rows, params.left_cols, row, col);
        let right_index = source_index(params.right_rows, params.right_cols, row, col);
        if (params.operation == 0u) {
            output[index] = left[left_index] + right[right_index];
        } else {
            output[index] = left[left_index] * right[right_index];
        }
        return;
    }

    if (params.operation == 2u) {
        output[index] = left[index] * params.scalar;
        return;
    }

    if (params.operation == 3u) {
        output[index] = left[col * params.left_cols + row];
        return;
    }

    if (params.operation == 4u) {
        let row_start = select(row, 0u, params.output_rows == 1u);
        let row_end = select(row + 1u, params.left_rows, params.output_rows == 1u);
        let col_start = select(col, 0u, params.output_cols == 1u);
        let col_end = select(col + 1u, params.left_cols, params.output_cols == 1u);
        var sum = 0.0;
        for (var source_row = row_start; source_row < row_end; source_row++) {
            for (var source_col = col_start; source_col < col_end; source_col++) {
                sum += left[source_row * params.left_cols + source_col];
            }
        }
        output[index] = sum;
        return;
    }

    if (params.operation == 5u) {
        output[index] = left[source_index(params.left_rows, params.left_cols, row, col)];
        return;
    }

    output[index] = left[index];
}
`

func mul(left, right, out *Matrix) error {
	return mulWithDeps(left, right, out, defaultTensorOpExecutionDeps())
}

func mulWithDeps(left, right, out *Matrix, deps tensorOpExecutionDeps) error {
	err := validateBinaryBroadcastOperation(left, right, out)
	if err != nil {
		return err
	}

	if useHostCompatibility(left.ctx) {
		return runBinaryBroadcast(left, right, out, func(a, b float32) float32 {
			return a * b
		})
	}

	return deps.dispatch(tensorOpMul, left, right, out, 0)
}

func scale(input *Matrix, scalar float32, out *Matrix) error {
	return scaleWithDeps(input, scalar, out, defaultTensorOpExecutionDeps())
}

func scaleWithDeps(input *Matrix, scalar float32, out *Matrix, deps tensorOpExecutionDeps) error {
	err := validateUnaryTensorOperation(input, out, validateUnaryShape)
	if err != nil {
		return err
	}

	if useHostCompatibility(input.ctx) {
		return runUnaryElementwise(input, out, func(value float32) float32 {
			return value * scalar
		})
	}

	return deps.dispatch(tensorOpScale, input, input, out, scalar)
}

func transp(input, out *Matrix) error {
	return transpWithDeps(input, out, defaultTensorOpExecutionDeps())
}

func transpWithDeps(input, out *Matrix, deps tensorOpExecutionDeps) error {
	err := validateUnaryTensorOperation(input, out, validateTransposeShape)
	if err != nil {
		return err
	}

	if !useHostCompatibility(input.ctx) {
		return deps.dispatch(tensorOpTranspose, input, input, out, 0)
	}

	inputData, err := input.Read()
	if err != nil {
		return wrapError(err, "failed to read input")
	}

	result := make([]float32, out.Len())

	for row := range input.rows {
		for col := range input.cols {
			result[col*out.cols+row] = inputData[row*input.cols+col]
		}
	}

	err = out.Write(result)
	if err != nil {
		return wrapError(err, "failed to write out")
	}

	return nil
}

func reduceSumTo(input, out *Matrix) error {
	return reduceSumToWithDeps(input, out, defaultTensorOpExecutionDeps())
}

//nolint:dupl // Reduction and broadcasting mirror each other but have different index contracts.
func reduceSumToWithDeps(input, out *Matrix, deps tensorOpExecutionDeps) error {
	err := validateUnaryTensorOperation(input, out, validateReduceSumToShape)
	if err != nil {
		return err
	}

	if !useHostCompatibility(input.ctx) {
		return deps.dispatch(tensorOpReduceSumTo, input, input, out, 0)
	}

	inputData, err := input.Read()
	if err != nil {
		return wrapError(err, "failed to read input")
	}

	result := make([]float32, out.Len())

	for row := range input.rows {
		outRow := row
		if out.rows == 1 {
			outRow = 0
		}

		for col := range input.cols {
			outCol := col
			if out.cols == 1 {
				outCol = 0
			}

			result[outRow*out.cols+outCol] += inputData[row*input.cols+col]
		}
	}

	err = out.Write(result)
	if err != nil {
		return wrapError(err, "failed to write out")
	}

	return nil
}

func broadcastTo(input, out *Matrix) error {
	return broadcastToWithDeps(input, out, defaultTensorOpExecutionDeps())
}

//nolint:dupl // Reduction and broadcasting mirror each other but have different index contracts.
func broadcastToWithDeps(input, out *Matrix, deps tensorOpExecutionDeps) error {
	err := validateUnaryTensorOperation(input, out, validateBroadcastToShape)
	if err != nil {
		return err
	}

	if !useHostCompatibility(input.ctx) {
		return deps.dispatch(tensorOpBroadcastTo, input, input, out, 0)
	}

	inputData, err := input.Read()
	if err != nil {
		return wrapError(err, "failed to read input")
	}

	result := make([]float32, out.Len())
	for row := range out.rows {
		inputRow := row
		if input.rows == 1 {
			inputRow = 0
		}

		for col := range out.cols {
			inputCol := col
			if input.cols == 1 {
				inputCol = 0
			}

			result[row*out.cols+col] = inputData[inputRow*input.cols+inputCol]
		}
	}

	err = out.Write(result)
	if err != nil {
		return wrapError(err, "failed to write out")
	}

	return nil
}

func reshapeTo(input, out *Matrix) error {
	return reshapeToWithDeps(input, out, defaultTensorOpExecutionDeps())
}

func reshapeToWithDeps(input, out *Matrix, deps tensorOpExecutionDeps) error {
	err := validateUnaryTensorOperation(input, out, validateReshapeToShape)
	if err != nil {
		return err
	}

	if !useHostCompatibility(input.ctx) {
		return deps.dispatch(tensorOpReshapeTo, input, input, out, 0)
	}

	inputData, err := input.Read()
	if err != nil {
		return wrapError(err, "failed to read input")
	}

	err = out.Write(inputData)
	if err != nil {
		return wrapError(err, "failed to write out")
	}

	return nil
}

func useHostCompatibility(ctx *Context) bool {
	return ctx == nil || ctx.device == nil || isCPUAdapter(ctx)
}

func defaultTensorOpExecutionDeps() tensorOpExecutionDeps {
	return tensorOpExecutionDeps{dispatch: dispatchTensorOperation}
}

func validateUnaryTensorOperation(
	input, out *Matrix,
	validateShape func(*Matrix, *Matrix) error,
) error {
	err := validateMatrixInitialized("input", input)
	if err != nil {
		return err
	}

	err = validateMatrixInitialized("out", out)
	if err != nil {
		return err
	}

	err = validateSameContext(input, out)
	if err != nil {
		return err
	}

	err = validateOutputNotAliased(out, input)
	if err != nil {
		return err
	}

	return validateShape(input, out)
}

func validateReduceSumToShape(input, out *Matrix) error {
	if (out.rows != input.rows && out.rows != 1) ||
		(out.cols != input.cols && out.cols != 1) {
		return sentinelError(
			ErrDimensionMismatch,
			"dimension mismatch: input=%s out=%s; out axes must equal input or 1",
			input.Shape(), out.Shape(),
		)
	}

	return nil
}

func validateBroadcastToShape(input, out *Matrix) error {
	if (input.rows != out.rows && input.rows != 1) ||
		(input.cols != out.cols && input.cols != 1) {
		return sentinelError(
			ErrDimensionMismatch,
			"dimension mismatch: input=%s out=%s; input axes must equal out or 1",
			input.Shape(), out.Shape(),
		)
	}

	return nil
}

func validateReshapeToShape(input, out *Matrix) error {
	if input.Len() != out.Len() {
		return sentinelError(
			ErrDimensionMismatch,
			"dimension mismatch: input=%s out=%s; want equal element counts",
			input.Shape(), out.Shape(),
		)
	}

	return nil
}

func dispatchTensorOperation(
	operation tensorOperation,
	left, right, out *Matrix,
	scalar float32,
) error {
	return dispatchTensorOperationWithDeps(
		operation, left, right, out, scalar, defaultMatMulWGPUDeps(),
	)
}

func dispatchTensorOperationWithDeps(
	operation tensorOperation,
	left, right, out *Matrix,
	scalar float32,
	deps matMulWGPUDeps,
) error {
	ctx := left.ctx

	err := validateTensorOpKernelContract(out)
	if err != nil {
		return err
	}

	layout, err := createTensorOpBindGroupLayout(ctx.device, deps)
	if err != nil {
		return err
	}
	defer deps.releaseBindGroupLayout(layout)

	pipeline, err := deps.getOrCreatePipeline(ctx, tensorOpPipelineKey, func() (*wgpu.ComputePipeline, error) {
		return createComputePipeline(ctx.device, layout, "tensorops", tensorOpWGSL, deps)
	})
	if err != nil {
		if pipeline != nil {
			deps.releaseComputePipeline(pipeline)
		}

		return wrapError(err, "create tensor operation pipeline")
	}

	if pipeline == nil {
		return sentinelError(ErrBackendUnavailable, "create tensor operation pipeline returned nil")
	}

	uniform, err := createTensorOpUniform(ctx, operation, left, right, out, scalar, deps)
	if err != nil {
		return err
	}
	defer func() {
		deps.releaseBuffer(uniform)
		ctx.recordBufferRelease()
	}()

	bindGroup, err := createTensorOpBindGroup(ctx.device, layout, uniform, left, right, out, deps)
	if err != nil {
		return err
	}
	defer deps.releaseBindGroup(bindGroup)

	return encodeAndSubmitCompute(
		ctx,
		pipeline,
		bindGroup,
		computeDispatch{x: ceilDiv(dimensionU32(out.Len()), tensorOpWorkgroupSize), y: 1, z: 1},
		"tensor operation",
		deps,
	)
}

func validateTensorOpKernelContract(out *Matrix) error {
	elements := uint64(out.Len()) //nolint:gosec // Matrix dimensions are positive and bounded.
	if elements == 0 || elements > math.MaxUint32 {
		return sentinelError(ErrKernelLimit, "matrix dimensions exceed tensor operation kernel limits: out=%s", out.Shape())
	}

	workgroups := ceilDiv(uint32(elements), tensorOpWorkgroupSize)

	limit := out.ctx.limits.MaxComputeWorkgroupsPerDimension
	if limit > 0 && workgroups > limit {
		return sentinelError(
			ErrDeviceLimit,
			"tensor operation dispatch exceeds device workgroup limits: need %d, max %d",
			workgroups, limit,
		)
	}

	return nil
}

func createTensorOpBindGroupLayout(
	device *wgpu.Device,
	deps matMulWGPUDeps,
) (*wgpu.BindGroupLayout, error) {
	layout, err := deps.createBindGroupLayout(device, &wgpu.BindGroupLayoutDescriptor{
		Label: "go-wgpu-mat-tensor-operation-bind-group-layout",
		Entries: []wgpu.BindGroupLayoutEntry{
			matMulLayoutEntry(tensorOpLeftBinding, gputypes.BufferBindingTypeReadOnlyStorage, 0),
			matMulLayoutEntry(tensorOpRightBinding, gputypes.BufferBindingTypeReadOnlyStorage, 0),
			matMulLayoutEntry(tensorOpOutputBinding, gputypes.BufferBindingTypeStorage, 0),
			matMulLayoutEntry(tensorOpParamsBinding, gputypes.BufferBindingTypeUniform, tensorOpUniformSize),
		},
	})
	if err != nil {
		if layout != nil {
			deps.releaseBindGroupLayout(layout)
		}

		return nil, wrapError(err, "create tensor operation bind group layout")
	}

	if layout == nil {
		return nil, sentinelError(ErrBackendUnavailable, "create tensor operation bind group layout returned nil")
	}

	return layout, nil
}

func createTensorOpUniform(
	ctx *Context,
	operation tensorOperation,
	left, right, out *Matrix,
	scalar float32,
	deps matMulWGPUDeps,
) (*wgpu.Buffer, error) {
	uniform, err := deps.createBuffer(ctx.device, &wgpu.BufferDescriptor{
		Label:            "go-wgpu-mat-tensor-operation-params",
		Size:             tensorOpUniformSize,
		Usage:            wgpu.BufferUsageUniform | wgpu.BufferUsageCopyDst,
		MappedAtCreation: false,
	})
	if err != nil {
		if uniform != nil {
			deps.releaseBuffer(uniform)
		}

		return nil, wrapError(err, "create tensor operation uniform buffer")
	}

	if uniform == nil {
		return nil, sentinelError(ErrBackendUnavailable, "create tensor operation uniform buffer returned nil")
	}

	ctx.recordBufferAllocation()

	params := make([]byte, tensorOpUniformSize)
	binary.LittleEndian.PutUint32(params[0:4], uint32(operation))
	binary.LittleEndian.PutUint32(params[4:8], dimensionU32(out.rows))
	binary.LittleEndian.PutUint32(params[8:12], dimensionU32(out.cols))
	binary.LittleEndian.PutUint32(params[12:16], dimensionU32(left.rows))
	binary.LittleEndian.PutUint32(params[16:20], dimensionU32(left.cols))
	binary.LittleEndian.PutUint32(params[20:24], dimensionU32(right.rows))
	binary.LittleEndian.PutUint32(params[24:28], dimensionU32(right.cols))
	binary.LittleEndian.PutUint32(params[28:32], math.Float32bits(scalar))

	err = ctx.withQueue(func() error {
		return deps.writeBuffer(ctx.device, uniform, 0, params)
	})
	if err != nil {
		deps.releaseBuffer(uniform)
		ctx.recordBufferRelease()

		return nil, wrapError(err, "write tensor operation parameters")
	}

	return uniform, nil
}

func createTensorOpBindGroup(
	device *wgpu.Device,
	layout *wgpu.BindGroupLayout,
	uniform *wgpu.Buffer,
	left, right, out *Matrix,
	deps matMulWGPUDeps,
) (*wgpu.BindGroup, error) {
	bindGroup, err := deps.createBindGroup(device, &wgpu.BindGroupDescriptor{
		Label:  "go-wgpu-mat-tensor-operation-bind-group",
		Layout: layout,
		Entries: []wgpu.BindGroupEntry{
			matMulBufferEntry(tensorOpLeftBinding, left.buf, matrixByteSize(left)),
			matMulBufferEntry(tensorOpRightBinding, right.buf, matrixByteSize(right)),
			matMulBufferEntry(tensorOpOutputBinding, out.buf, matrixByteSize(out)),
			matMulBufferEntry(tensorOpParamsBinding, uniform, tensorOpUniformSize),
		},
	})
	if err != nil {
		if bindGroup != nil {
			deps.releaseBindGroup(bindGroup)
		}

		return nil, wrapError(err, "create tensor operation bind group")
	}

	if bindGroup == nil {
		return nil, sentinelError(ErrBackendUnavailable, "create tensor operation bind group returned nil")
	}

	return bindGroup, nil
}
