//go:build !cgo

package mat

import "math"

func validateMatrixInitialized(name string, m *Matrix) error {
	if m == nil || m.ctx == nil || m.buf == nil {
		return newError("%s is not initialized", name)
	}

	return nil
}

func validateMatMulDims(a, b, out *Matrix) error {
	if a.Cols != b.Rows || out.Rows != a.Rows || out.Cols != b.Cols {
		return newError("dimension mismatch")
	}

	return nil
}

func matMulCPU(aData, bData []float32, rows, shared, cols int) []float32 {
	result := make([]float32, rows*cols)
	for row := range rows {
		for col := range cols {
			sum := float32(0)
			for k := range shared {
				sum += aData[row*shared+k] * bData[k*cols+col]
			}

			result[row*cols+col] = sum
		}
	}

	return result
}

func validateSameShape(left, right, out *Matrix) error {
	if left.Rows != right.Rows || left.Cols != right.Cols {
		return newError("dimension mismatch")
	}

	if out.Rows != left.Rows || out.Cols != left.Cols {
		return newError("dimension mismatch")
	}

	return nil
}

func runBinaryElementwise(
	left, right, out *Matrix,
	operation func(float32, float32) float32,
) error {
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

	err = validateSameShape(left, right, out)
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
	if out.Rows != input.Rows || out.Cols != input.Cols {
		return newError("dimension mismatch")
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
	if out.Rows != input.Cols || out.Cols != input.Rows {
		return newError("dimension mismatch")
	}

	return nil
}

func validateRowReductionShape(input, out *Matrix) error {
	if out.Rows != input.Rows || out.Cols != 1 {
		return newError("dimension mismatch")
	}

	return nil
}

func applySoftmaxRow(inputData, outputData []float32, offset, cols int) {
	maxValue := inputData[offset]
	for col := 1; col < cols; col++ {
		value := inputData[offset+col]
		if value > maxValue {
			maxValue = value
		}
	}

	sumExp := float32(0)

	for col := range cols {
		expValue := float32(math.Exp(float64(inputData[offset+col] - maxValue)))
		outputData[offset+col] = expValue
		sumExp += expValue
	}

	for col := range cols {
		outputData[offset+col] /= sumExp
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

	err = validateRowReductionShape(input, out)
	if err != nil {
		return err
	}

	inputData, err := input.Read()
	if err != nil {
		return wrapError(err, "failed to read input")
	}

	result := make([]float32, out.Rows)
	for row := range input.Rows {
		acc := initialValue
		for col := range input.Cols {
			acc = combine(acc, inputData[row*input.Cols+col])
		}

		result[row] = acc
	}

	err = out.Write(result)
	if err != nil {
		return wrapError(err, "failed to write out")
	}

	return nil
}

// MatMul computes out = a x b.
//
//nolint:revive // keep explicit API name for consistency with roadmap/docs.
func MatMul(left, right, out *Matrix) error {
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

	aData, err := left.Read()
	if err != nil {
		return wrapError(err, "failed to read left")
	}

	bData, err := right.Read()
	if err != nil {
		return wrapError(err, "failed to read right")
	}

	result := matMulCPU(aData, bData, out.Rows, left.Cols, out.Cols)

	err = out.Write(result)
	if err != nil {
		return wrapError(err, "failed to write out")
	}

	return nil
}

// Add computes out = left + right.
func Add(left, right, out *Matrix) error {
	return runBinaryElementwise(left, right, out, func(a, b float32) float32 {
		return a + b
	})
}

// Scale computes out = input * scalar.
func Scale(input *Matrix, scalar float32, out *Matrix) error {
	return runUnaryElementwise(input, out, func(value float32) float32 {
		return value * scalar
	})
}

// Transp computes out = input^T.
func Transp(input, out *Matrix) error {
	err := validateMatrixInitialized("input", input)
	if err != nil {
		return err
	}

	err = validateMatrixInitialized("out", out)
	if err != nil {
		return err
	}

	err = validateTransposeShape(input, out)
	if err != nil {
		return err
	}

	inputData, err := input.Read()
	if err != nil {
		return wrapError(err, "failed to read input")
	}

	result := make([]float32, out.Rows*out.Cols)
	for row := range input.Rows {
		for col := range input.Cols {
			result[col*out.Cols+row] = inputData[row*input.Cols+col]
		}
	}

	err = out.Write(result)
	if err != nil {
		return wrapError(err, "failed to write out")
	}

	return nil
}

// ReduceSum computes row-wise sum and stores the result in out.
func ReduceSum(input, out *Matrix) error {
	return runRowReduction(input, out, 0, func(accumulator, value float32) float32 {
		return accumulator + value
	})
}

// ReduceMax computes row-wise max and stores the result in out.
func ReduceMax(input, out *Matrix) error {
	return runRowReduction(input, out, -float32(math.MaxFloat32),
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

	err = validateUnaryShape(input, out)
	if err != nil {
		return err
	}

	inputData, err := input.Read()
	if err != nil {
		return wrapError(err, "failed to read input")
	}

	result := make([]float32, len(inputData))

	for row := range input.Rows {
		rowOffset := row * input.Cols
		applySoftmaxRow(inputData, result, rowOffset, input.Cols)
	}

	err = out.Write(result)
	if err != nil {
		return wrapError(err, "failed to write out")
	}

	return nil
}
