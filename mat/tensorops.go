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
	tensorOpDropout
	tensorOpAllFinite
	tensorOpAdamFirst
	tensorOpAdamSecond
	tensorOpAdamDelta
	tensorOpSelectFinite
)

const (
	tensorOpPipelineKey   = "tensorops:f32:v2"
	tensorOpUniformSize   = uint64(32)
	tensorOpWorkgroupSize = uint32(256)
	tensorOpLeftBinding   = uint32(0)
	tensorOpRightBinding  = uint32(1)
	tensorOpOutputBinding = uint32(2)
	tensorOpParamsBinding = uint32(3)
	tensorOpAuxBinding    = uint32(4)
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
@group(0) @binding(4) var<storage, read> aux: array<f32>;

fn source_index(rows: u32, cols: u32, row: u32, col: u32) -> u32 {
    let source_row = select(row, 0u, rows == 1u);
    let source_col = select(col, 0u, cols == 1u);
    return source_row * cols + source_col;
}

fn rotate_left(value: u32, shift: u32) -> u32 {
    return (value << shift) | (value >> (32u - shift));
}

fn mix32(initial: u32) -> u32 {
    var value = initial + 0x9e3779b9u;
    value = value ^ (value >> 16u);
    value = value * 0x21f0aaadu;
    value = value ^ (value >> 15u);
    value = value * 0x735a2d97u;
    return value ^ (value >> 15u);
}

fn random_word(index: u32) -> u32 {
	// Dropout aliases the six shape slots to seed, stream, and counter low/high
	// words. createTensorOpUniform writes this ABI when random state is present.
    let counter_low = params.right_rows + index;
    let carry = select(0u, 1u, counter_low < params.right_rows);
    let counter_high = params.right_cols + carry;
    let value = mix32(params.output_rows) ^
        rotate_left(mix32(params.output_cols), 7u) ^
        rotate_left(mix32(params.left_rows), 13u) ^
        rotate_left(mix32(params.left_cols), 21u) ^
        counter_low ^ rotate_left(mix32(counter_high), 11u);
    return mix32(value);
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    let output_len = arrayLength(&output);
    if (index >= output_len) {
        return;
    }

    if (params.operation == 7u) {
		let threshold = u32(params.scalar * 4294967296.0);
        let keep = random_word(index) >= threshold;
        output[index] = select(0.0, left[index] / (1.0 - params.scalar), keep);
        return;
    }

	if (params.operation == 8u) {
		if (index != 0u) {
			return;
		}
		var finite = output[0] != 0.0;
		for (var source = 0u; source < arrayLength(&left); source++) {
			let value = left[source];
			finite = finite && value == value && abs(value) <= 3.4028234663852886e38;
		}
		output[0] = select(0.0, 1.0, finite);
		return;
	}

	if (params.operation == 9u) {
		output[index] = params.scalar * left[index] + (1.0 - params.scalar) * right[index];
		return;
	}

	if (params.operation == 10u) {
		output[index] = params.scalar * left[index] +
			(1.0 - params.scalar) * right[index] * right[index];
		return;
	}

	if (params.operation == 11u) {
		let epsilon = bitcast<f32>(params.output_rows);
		output[index] = -params.scalar * left[index] / (sqrt(right[index]) + epsilon);
		return;
	}

	if (params.operation == 12u) {
		output[index] = select(right[index], left[index], aux[0] != 0.0);
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

// RandomState identifies the first counter-based random word for a device operation.
type RandomState struct {
	Seed     uint64
	StreamID uint64
	Counter  uint64
}

func dropout(input *Matrix, probability float32, state RandomState, out *Matrix) error {
	err := validateUnaryTensorOperation(input, out, validateUnaryShape)
	if err != nil {
		return err
	}

	if probability < 0 || probability >= 1 || math.IsNaN(float64(probability)) {
		return sentinelError(ErrInvalidProbability, "invalid dropout probability %g", probability)
	}

	if useHostCompatibility(input.ctx) {
		return runDropoutCompatibility(input, probability, state, out)
	}

	return dispatchTensorDropout(input, probability, state, out, defaultMatMulWGPUDeps())
}

//nolint:cyclop // Validation and host/device paths implement one operation contract.
func allFiniteAccumulate(input, flag *Matrix) error {
	err := validateMatrixInitialized("input", input)
	if err != nil {
		return err
	}

	err = validateMatrixInitialized("flag", flag)
	if err != nil {
		return err
	}

	err = validateSameContext(input, flag)
	if err != nil {
		return err
	}

	if flag.Len() != 1 {
		return sentinelError(ErrDimensionMismatch, "all-finite flag must have one element")
	}

	if input == flag {
		return sentinelError(ErrAliasedOutput, "all-finite flag aliases input")
	}

	if useHostCompatibility(input.ctx) {
		data, readErr := input.Read()
		if readErr != nil {
			return wrapError(readErr, "failed to read all-finite input")
		}

		flagData, readErr := flag.Read()
		if readErr != nil {
			return wrapError(readErr, "failed to read all-finite flag")
		}

		finite := flagData[0] != 0
		for _, value := range data {
			finite = finite && !float32NonFinite(value)
		}

		if finite {
			return flag.Write([]float32{1})
		}

		return flag.Write([]float32{0})
	}

	return dispatchTensorOperation(tensorOpAllFinite, input, input, flag, 0)
}

func adamFirstMoment(moment, gradient *Matrix, beta float32, out *Matrix) error {
	return adamMoment(moment, gradient, beta, out, tensorOpAdamFirst)
}

func adamSecondMoment(moment, gradient *Matrix, beta float32, out *Matrix) error {
	return adamMoment(moment, gradient, beta, out, tensorOpAdamSecond)
}

//nolint:cyclop // Validation and host/device paths implement one operation contract.
func adamMoment(moment, gradient *Matrix, beta float32, out *Matrix, operation tensorOperation) error {
	err := validateBinaryBroadcastOperation(moment, gradient, out)
	if err != nil {
		return err
	}

	if moment.Shape() != gradient.Shape() || moment.Shape() != out.Shape() {
		return sentinelError(ErrDimensionMismatch, "Adam moment tensors must have equal shapes")
	}

	if beta < 0 || beta >= 1 || math.IsNaN(float64(beta)) {
		return sentinelError(ErrInvalidProbability, "invalid Adam beta %g", beta)
	}

	if useHostCompatibility(moment.ctx) {
		momentData, readErr := moment.Read()
		if readErr != nil {
			return readErr
		}

		gradientData, readErr := gradient.Read()
		if readErr != nil {
			return readErr
		}

		result := make([]float32, len(momentData))
		for index := range result {
			if operation == tensorOpAdamFirst {
				result[index] = beta*momentData[index] + (1-beta)*gradientData[index]
			} else {
				result[index] = beta*momentData[index] + (1-beta)*gradientData[index]*gradientData[index]
			}
		}

		return out.Write(result)
	}

	return dispatchTensorOperation(operation, moment, gradient, out, beta)
}

//nolint:cyclop // Validation and host/device paths implement one operation contract.
func adamDelta(first, second *Matrix, scale, epsilon float32, out *Matrix) error {
	err := validateBinaryBroadcastOperation(first, second, out)
	if err != nil {
		return err
	}

	if first.Shape() != second.Shape() || first.Shape() != out.Shape() {
		return sentinelError(ErrDimensionMismatch, "Adam delta tensors must have equal shapes")
	}

	if float32NonFinite(scale) || float32NonFinite(epsilon) || epsilon <= 0 {
		return sentinelError(ErrInvalidState, "invalid Adam delta config")
	}

	if useHostCompatibility(first.ctx) {
		firstData, readErr := first.Read()
		if readErr != nil {
			return readErr
		}

		secondData, readErr := second.Read()
		if readErr != nil {
			return readErr
		}

		result := make([]float32, len(firstData))
		for index := range result {
			result[index] = -scale * firstData[index] / (float32(math.Sqrt(float64(secondData[index]))) + epsilon)
		}

		return out.Write(result)
	}

	override := RandomState{Seed: uint64(math.Float32bits(epsilon)), StreamID: 0, Counter: 0}

	return dispatchTensorOperationWithRandom(
		tensorOpAdamDelta, first, second, out, scale, &override, defaultMatMulWGPUDeps(),
	)
}

func float32NonFinite(value float32) bool {
	return math.IsNaN(float64(value)) || math.IsInf(float64(value), 0)
}

//nolint:cyclop // Validation and host/device paths implement one operation contract.
func selectFinite(candidate, original, flag, out *Matrix) error {
	err := validateBinaryBroadcastOperation(candidate, original, out)
	if err != nil {
		return err
	}

	if candidate.Shape() != original.Shape() || candidate.Shape() != out.Shape() {
		return sentinelError(ErrDimensionMismatch, "select tensors must have equal shapes")
	}

	err = validateMatrixInitialized("flag", flag)
	if err != nil {
		return err
	}

	err = validateSameContext(candidate, flag)
	if err != nil {
		return err
	}

	if flag.Len() != 1 {
		return sentinelError(ErrDimensionMismatch, "select flag must have one element")
	}

	if useHostCompatibility(candidate.ctx) {
		candidateData, readErr := candidate.Read()
		if readErr != nil {
			return readErr
		}

		originalData, readErr := original.Read()
		if readErr != nil {
			return readErr
		}

		flagData, readErr := flag.Read()
		if readErr != nil {
			return readErr
		}

		if flagData[0] != 0 {
			return out.Write(candidateData)
		}

		return out.Write(originalData)
	}

	return dispatchTensorOperationWithAux(
		tensorOpSelectFinite, candidate, original, flag, out, 0, nil, defaultMatMulWGPUDeps(),
	)
}

//nolint:mnd // The 32-bit word space is the CPU/WGSL parity contract.
func runDropoutCompatibility(input *Matrix, probability float32, state RandomState, out *Matrix) error {
	data, err := input.Read()
	if err != nil {
		return wrapError(err, "failed to read input")
	}

	threshold := uint64(float64(probability) * (1 << 32))
	scale := float32(1) / (1 - probability)

	result := make([]float32, len(data))
	for index, value := range data {
		if uint64(randomWord(state.Seed, state.StreamID, state.Counter+uint64(index))) >= threshold {
			result[index] = value * scale
		}
	}

	err = out.Write(result)
	if err != nil {
		return wrapError(err, "failed to write out")
	}

	return nil
}

//nolint:mnd // Rotation constants and mixers are the CPU/WGSL RNG contract.
func randomWord(seed, streamID, counter uint64) uint32 {
	value := mixRandom32(uint32(seed)) ^ //nolint:gosec // Low word extraction is intentional.
		rotateRandom32(mixRandom32(uint32(seed>>32)), 7) ^
		rotateRandom32(mixRandom32(uint32(streamID)), 13) ^ //nolint:gosec // Low word extraction is intentional.
		rotateRandom32(mixRandom32(uint32(streamID>>32)), 21) ^
		uint32(counter) ^ //nolint:gosec // Low word extraction is intentional.
		rotateRandom32(mixRandom32(uint32(counter>>32)), 11)

	return mixRandom32(value)
}

//nolint:mnd // Constants define the published mixer.
func mixRandom32(value uint32) uint32 {
	value += 0x9e3779b9
	value ^= value >> 16
	value *= 0x21f0aaad
	value ^= value >> 15
	value *= 0x735a2d97

	return value ^ value>>15
}

//nolint:mnd // Uint32 word width is fixed.
func rotateRandom32(value uint32, shift int) uint32 {
	return value<<shift | value>>(32-shift)
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
	return dispatchTensorOperationWithRandom(
		operation, left, right, out, scalar, nil, deps,
	)
}

func dispatchTensorDropout(
	input *Matrix,
	probability float32,
	state RandomState,
	out *Matrix,
	deps matMulWGPUDeps,
) error {
	return dispatchTensorOperationWithRandom(
		tensorOpDropout, input, input, out, probability, &state, deps,
	)
}

func dispatchTensorOperationWithRandom(
	operation tensorOperation,
	left, right, out *Matrix,
	scalar float32,
	random *RandomState,
	deps matMulWGPUDeps,
) error {
	return dispatchTensorOperationWithAux(operation, left, right, nil, out, scalar, random, deps)
}

func dispatchTensorOperationWithAux(
	operation tensorOperation,
	left, right, aux, out *Matrix,
	scalar float32,
	random *RandomState,
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

	uniform, err := createTensorOpUniform(ctx, operation, left, right, out, scalar, random, deps)
	if err != nil {
		return err
	}
	defer func() {
		deps.releaseBuffer(uniform)
	}()

	bindGroup, err := createTensorOpBindGroup(ctx.device, layout, uniform, left, right, aux, out, deps)
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
			matMulLayoutEntry(tensorOpAuxBinding, gputypes.BufferBindingTypeReadOnlyStorage, 0),
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

//nolint:mnd // Byte offsets and 32-bit halves define the WGSL uniform ABI.
func createTensorOpUniform(
	ctx *Context,
	operation tensorOperation,
	left, right, out *Matrix,
	scalar float32,
	random *RandomState,
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

	params := make([]byte, tensorOpUniformSize)
	binary.LittleEndian.PutUint32(params[0:4], uint32(operation))
	binary.LittleEndian.PutUint32(params[4:8], dimensionU32(out.rows))
	binary.LittleEndian.PutUint32(params[8:12], dimensionU32(out.cols))
	binary.LittleEndian.PutUint32(params[12:16], dimensionU32(left.rows))
	binary.LittleEndian.PutUint32(params[16:20], dimensionU32(left.cols))
	binary.LittleEndian.PutUint32(params[20:24], dimensionU32(right.rows))
	binary.LittleEndian.PutUint32(params[24:28], dimensionU32(right.cols))
	binary.LittleEndian.PutUint32(params[28:32], math.Float32bits(scalar))

	if random != nil {
		binary.LittleEndian.PutUint32(params[4:8], uint32(random.Seed)) //nolint:gosec // Low word ABI.
		binary.LittleEndian.PutUint32(params[8:12], uint32(random.Seed>>32))
		binary.LittleEndian.PutUint32(params[12:16], uint32(random.StreamID)) //nolint:gosec // Low word ABI.
		binary.LittleEndian.PutUint32(params[16:20], uint32(random.StreamID>>32))
		binary.LittleEndian.PutUint32(params[20:24], uint32(random.Counter)) //nolint:gosec // Low word ABI.
		binary.LittleEndian.PutUint32(params[24:28], uint32(random.Counter>>32))
	}

	err = ctx.withQueue(func() error {
		return deps.writeBuffer(ctx.device, uniform, 0, params)
	})
	if err != nil {
		deps.releaseBuffer(uniform)

		return nil, wrapError(err, "write tensor operation parameters")
	}

	return uniform, nil
}

func createTensorOpBindGroup(
	device *wgpu.Device,
	layout *wgpu.BindGroupLayout,
	uniform *wgpu.Buffer,
	left, right, aux, out *Matrix,
	deps matMulWGPUDeps,
) (*wgpu.BindGroup, error) {
	if aux == nil {
		aux = left
	}

	bindGroup, err := deps.createBindGroup(device, &wgpu.BindGroupDescriptor{
		Label:  "go-wgpu-mat-tensor-operation-bind-group",
		Layout: layout,
		Entries: []wgpu.BindGroupEntry{
			matMulBufferEntry(tensorOpLeftBinding, left.buf, matrixByteSize(left)),
			matMulBufferEntry(tensorOpRightBinding, right.buf, matrixByteSize(right)),
			matMulBufferEntry(tensorOpOutputBinding, out.buf, matrixByteSize(out)),
			matMulBufferEntry(tensorOpParamsBinding, uniform, tensorOpUniformSize),
			matMulBufferEntry(tensorOpAuxBinding, aux.buf, matrixByteSize(aux)),
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
