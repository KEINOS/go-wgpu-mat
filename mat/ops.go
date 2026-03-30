//go:build !cgo

package mat

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

// MatMul computes out = a x b.
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
