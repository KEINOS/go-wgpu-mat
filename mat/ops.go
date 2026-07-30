package mat

import "slices"

import "math"

const rmsNormEpsilon float32 = 1e-5
const minimumMatricesForContextCheck = 2

func validateMatrixInitialized(name string, matrix *Matrix) error {
	if matrix == nil || matrix.ctx == nil || matrix.buf == nil {
		return sentinelError(ErrNotInitialized, "%s is not initialized", name)
	}

	if matrix.released.Load() != 0 {
		return sentinelError(ErrReleased, "%s is released", name)
	}

	if matrix.ctx.released.Load() != 0 {
		return sentinelError(ErrContextReleased, "context is released")
	}

	if matrix.rows <= 0 || matrix.cols <= 0 {
		return sentinelError(
			ErrInvalidState,
			"%s has invalid shape %s",
			name,
			matrix.Shape(),
		)
	}

	return nil
}

func validateMatMulDims(left, right, out *Matrix) error {
	if left.cols != right.rows || out.rows != left.rows || out.cols != right.cols {
		return sentinelError(
			ErrDimensionMismatch,
			"dimension mismatch: left=%s right=%s out=%s; want left.cols == right.rows and out=%dx%d",
			left.Shape(),
			right.Shape(),
			out.Shape(),
			left.rows,
			right.cols,
		)
	}

	return nil
}

func validateSameShape(left, right, out *Matrix) error {
	if left.rows != right.rows || left.cols != right.cols {
		return sentinelError(
			ErrDimensionMismatch,
			"dimension mismatch: left=%s right=%s; want equal shapes",
			left.Shape(),
			right.Shape(),
		)
	}

	if out.rows != left.rows || out.cols != left.cols {
		return sentinelError(
			ErrDimensionMismatch,
			"dimension mismatch: inputs=%s out=%s; want out=%s",
			left.Shape(),
			out.Shape(),
			left.Shape(),
		)
	}

	return nil
}

func broadcastDimension(left, right int) (int, bool) {
	switch {
	case left == right:
		return left, true
	case left == 1:
		return right, true
	case right == 1:
		return left, true
	default:
		return 0, false
	}
}

func validateBroadcastShape(left, right, out *Matrix) error {
	rows, rowsOK := broadcastDimension(left.rows, right.rows)

	cols, colsOK := broadcastDimension(left.cols, right.cols)
	if !rowsOK || !colsOK || out.rows != rows || out.cols != cols {
		return sentinelError(
			ErrDimensionMismatch,
			"dimension mismatch: left=%s right=%s out=%s; want 2D broadcast out",
			left.Shape(),
			right.Shape(),
			out.Shape(),
		)
	}

	return nil
}

func validateSameContext(matrices ...*Matrix) error {
	if len(matrices) < minimumMatricesForContextCheck {
		return nil
	}

	context := matrices[0].ctx
	for _, matrix := range matrices[1:] {
		if matrix.ctx != context {
			return sentinelError(
				ErrContextMismatch,
				"matrices must use the same context",
			)
		}
	}

	return nil
}

func validateOutputNotAliased(out *Matrix, inputs ...*Matrix) error {
	if slices.Contains(inputs, out) {
		return sentinelError(
			ErrAliasedOutput,
			"out must not alias an input",
		)
	}

	return nil
}

func validateBinaryOperation(left, right, out *Matrix) error {
	return validateBinaryOperationShape(left, right, out, validateSameShape)
}

func validateBinaryBroadcastOperation(left, right, out *Matrix) error {
	return validateBinaryOperationShape(left, right, out, validateBroadcastShape)
}

func validateBinaryOperationShape(
	left, right, out *Matrix,
	validateShape func(*Matrix, *Matrix, *Matrix) error,
) error {
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

	err := validateSameContext(left, right, out)
	if err != nil {
		return err
	}

	err = validateOutputNotAliased(out, left, right)
	if err != nil {
		return err
	}

	return validateShape(left, right, out)
}

//nolint:cyclop // Explicit singleton-axis indexing keeps the CPU reference readable.
func runBinaryBroadcast(
	left, right, out *Matrix,
	operation func(float32, float32) float32,
) error {
	err := validateBinaryBroadcastOperation(left, right, out)
	if err != nil {
		return err
	}

	leftData, err := left.Read()
	if err != nil {
		return wrapError(err, "failed to read left")
	}

	rightData, err := right.Read()
	if err != nil {
		return wrapError(err, "failed to read right")
	}

	result := make([]float32, out.Len())
	for row := range out.rows {
		for col := range out.cols {
			leftRow := row
			if left.rows == 1 {
				leftRow = 0
			}

			leftCol := col
			if left.cols == 1 {
				leftCol = 0
			}

			rightRow := row
			if right.rows == 1 {
				rightRow = 0
			}

			rightCol := col
			if right.cols == 1 {
				rightCol = 0
			}

			result[row*out.cols+col] = operation(
				leftData[leftRow*left.cols+leftCol],
				rightData[rightRow*right.cols+rightCol],
			)
		}
	}

	err = out.Write(result)
	if err != nil {
		return wrapError(err, "failed to write out")
	}

	return nil
}

func runBinaryElementwise(
	left, right, out *Matrix,
	operation func(float32, float32) float32,
) error {
	err := validateBinaryOperation(left, right, out)
	if err != nil {
		return err
	}

	leftData, err := left.Read()
	if err != nil {
		return wrapError(err, "failed to read left")
	}

	rightData, err := right.Read()
	if err != nil {
		return wrapError(err, "failed to read right")
	}

	result := make([]float32, len(leftData))
	for i := range result {
		result[i] = operation(leftData[i], rightData[i])
	}

	err = out.Write(result)
	if err != nil {
		return wrapError(err, "failed to write out")
	}

	return nil
}

func validateUnaryShape(input, out *Matrix) error {
	if out.rows != input.rows || out.cols != input.cols {
		return sentinelError(
			ErrDimensionMismatch,
			"dimension mismatch: input=%s out=%s; want equal shapes",
			input.Shape(),
			out.Shape(),
		)
	}

	return nil
}

func runUnaryElementwise(
	input, out *Matrix,
	operation func(float32) float32,
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

	err = validateUnaryShape(input, out)
	if err != nil {
		return err
	}

	inputData, err := input.Read()
	if err != nil {
		return wrapError(err, "failed to read input")
	}

	result := make([]float32, len(inputData))
	for i := range result {
		result[i] = operation(inputData[i])
	}

	err = out.Write(result)
	if err != nil {
		return wrapError(err, "failed to write out")
	}

	return nil
}

func validateTransposeShape(input, out *Matrix) error {
	if out.rows != input.cols || out.cols != input.rows {
		return sentinelError(
			ErrDimensionMismatch,
			"dimension mismatch: input=%s out=%s; want out=%dx%d",
			input.Shape(),
			out.Shape(),
			input.cols,
			input.rows,
		)
	}

	return nil
}

func validateRowReductionShape(input, out *Matrix) error {
	if out.rows != input.rows || out.cols != 1 {
		return sentinelError(
			ErrDimensionMismatch,
			"dimension mismatch: input=%s out=%s; want out=%dx1",
			input.Shape(),
			out.Shape(),
			input.rows,
		)
	}

	return nil
}

type softmaxRowStats struct {
	maxValue           float32
	positiveInfinity   int
	containsNotANumber bool
}

func applySoftmaxRow(inputData, outputData []float32, offset, cols int) {
	inputRow := inputData[offset : offset+cols]
	outputRow := outputData[offset : offset+cols]
	stats := inspectSoftmaxRow(inputRow)

	if applySpecialSoftmaxRow(inputRow, outputRow, stats) {
		return
	}

	applyFiniteSoftmaxRow(inputRow, outputRow, stats.maxValue)
}

func inspectSoftmaxRow(input []float32) softmaxRowStats {
	stats := new(softmaxRowStats)
	stats.maxValue = input[0]

	for _, value := range input {
		if math.IsNaN(float64(value)) {
			stats.containsNotANumber = true

			return *stats
		}

		if math.IsInf(float64(value), 1) {
			stats.positiveInfinity++
		}

		if value > stats.maxValue {
			stats.maxValue = value
		}
	}

	return *stats
}

func applySpecialSoftmaxRow(
	input, output []float32,
	stats softmaxRowStats,
) bool {
	if stats.containsNotANumber {
		fillNaN(output)

		return true
	}

	if stats.positiveInfinity > 0 {
		fillPositiveInfinitySoftmax(input, output, stats.positiveInfinity)

		return true
	}

	if math.IsInf(float64(stats.maxValue), -1) {
		fillUniform(output)

		return true
	}

	return false
}

func applyFiniteSoftmaxRow(input, output []float32, maxValue float32) {
	sumExp := float64(0)

	for index, value := range input {
		expValue := math.Exp(float64(value - maxValue))
		output[index] = float32(expValue)
		sumExp += expValue
	}

	for index := range output {
		output[index] = float32(float64(output[index]) / sumExp)
	}
}

func fillPositiveInfinitySoftmax(input, output []float32, count int) {
	probability := float32(1) / float32(count)

	for index, value := range input {
		if math.IsInf(float64(value), 1) {
			output[index] = probability
		} else {
			output[index] = 0
		}
	}
}

func fillUniform(values []float32) {
	probability := float32(1) / float32(len(values))
	for index := range values {
		values[index] = probability
	}
}

func fillNaN(values []float32) {
	nan := float32(math.NaN())
	for index := range values {
		values[index] = nan
	}
}

func applyRMSNormRow(inputData, outputData []float32, offset, cols int) {
	sumSquares := float64(0)

	for col := range cols {
		value := float64(inputData[offset+col])
		sumSquares += value * value
	}

	meanSquare := sumSquares / float64(cols)
	denominator := math.Sqrt(meanSquare + float64(rmsNormEpsilon))

	for col := range cols {
		outputData[offset+col] = float32(float64(inputData[offset+col]) / denominator)
	}
}

func runRowReduction(
	input, out *Matrix,
	initialValue float32,
	combine func(float32, float32) float32,
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

	err = validateRowReductionShape(input, out)
	if err != nil {
		return err
	}

	inputData, err := input.Read()
	if err != nil {
		return wrapError(err, "failed to read input")
	}

	result := make([]float32, out.rows)
	for row := range input.rows {
		acc := initialValue
		for col := range input.cols {
			acc = combine(acc, inputData[row*input.cols+col])
		}

		result[row] = acc
	}

	err = out.Write(result)
	if err != nil {
		return wrapError(err, "failed to write out")
	}

	return nil
}

// MatMul computes out = left x right with a WGSL compute kernel.
// All matrices must belong to the same Context, and out must not alias either
// input. The result remains in its device buffer until Read is called.
//
//nolint:revive // keep explicit API name for consistency with roadmap/docs.
func MatMul(left, right, out *Matrix) error {
	return matMul(left, right, out, defaultMatMulDeps())
}

// Add computes out = left + right.
func Add(left, right, out *Matrix) error {
	return add(left, right, out, defaultAddDeps())
}

// Scale computes out = input * scalar.
func Scale(input *Matrix, scalar float32, out *Matrix) error {
	return scale(input, scalar, out)
}

// Transp computes out = input^T.
func Transp(input, out *Matrix) error {
	return transp(input, out)
}

// ReduceSum computes row-wise sum and stores the result in out.
func ReduceSum(input, out *Matrix) error {
	return ReduceSumTo(input, out)
}

// Mul computes an elementwise product with 2D broadcasting.
func Mul(left, right, out *Matrix) error {
	return mul(left, right, out)
}

// ReduceSumTo sums input axes whose corresponding out dimension is 1.
func ReduceSumTo(input, out *Matrix) error {
	return reduceSumTo(input, out)
}

// BroadcastTo expands singleton input axes to the shape of out.
func BroadcastTo(input, out *Matrix) error {
	return broadcastTo(input, out)
}

// ReshapeTo copies row-major input data to an equal-length output shape.
func ReshapeTo(input, out *Matrix) error {
	return reshapeTo(input, out)
}

// ReduceMax computes row-wise max and stores the result in out.
func ReduceMax(input, out *Matrix) error {
	return runRowReduction(input, out, float32(math.Inf(-1)),
		func(accumulator, value float32) float32 {
			if value > accumulator {
				return value
			}

			return accumulator
		})
}

// Softmax computes row-wise softmax for input and stores it in out.
func Softmax(input, out *Matrix) error {
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

	err = validateUnaryShape(input, out)
	if err != nil {
		return err
	}

	inputData, err := input.Read()
	if err != nil {
		return wrapError(err, "failed to read input")
	}

	result := make([]float32, len(inputData))

	for row := range input.rows {
		rowOffset := row * input.cols
		applySoftmaxRow(inputData, result, rowOffset, input.cols)
	}

	err = out.Write(result)
	if err != nil {
		return wrapError(err, "failed to write out")
	}

	return nil
}

// RMSNorm computes row-wise root-mean-square normalization.
func RMSNorm(input, out *Matrix) error {
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

	err = validateUnaryShape(input, out)
	if err != nil {
		return err
	}

	inputData, err := input.Read()
	if err != nil {
		return wrapError(err, "failed to read input")
	}

	result := make([]float32, len(inputData))

	for row := range input.rows {
		rowOffset := row * input.cols
		applyRMSNormRow(inputData, result, rowOffset, input.cols)
	}

	err = out.Write(result)
	if err != nil {
		return wrapError(err, "failed to write out")
	}

	return nil
}
