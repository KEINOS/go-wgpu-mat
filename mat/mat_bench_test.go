package mat_test

import (
	"testing"

	"github.com/KEINOS/go-wgpu-mat/mat"
)

type binaryOperation func(*mat.Matrix, *mat.Matrix, *mat.Matrix) error
type unaryOperation func(*mat.Matrix, *mat.Matrix) error

func benchmarkData(length int) []float32 {
	data := make([]float32, length)
	for index := range data {
		data[index] = float32((index%17)-8) * 0.25
	}

	return data
}

func benchmarkBinaryOperation( //nolint:cyclop,funlen // Benchmark setup and errors remain readable together.
	b *testing.B,
	operation binaryOperation,
	syncEachIteration bool,
) {
	b.Helper()

	const rows, cols = 256, 256

	ctx, err := mat.NewContext()
	if err != nil {
		b.Fatal(err)
	}
	defer ctx.Release()

	leftMatrix, err := mat.NewMatrix(ctx, rows, cols)
	if err != nil {
		b.Fatal(err)
	}
	defer leftMatrix.Release()

	rightMatrix, err := mat.NewMatrix(ctx, rows, cols)
	if err != nil {
		b.Fatal(err)
	}
	defer rightMatrix.Release()

	outMatrix, err := mat.NewMatrix(ctx, rows, cols)
	if err != nil {
		b.Fatal(err)
	}
	defer outMatrix.Release()

	err = leftMatrix.Write(benchmarkData(rows * cols))
	if err != nil {
		b.Fatal(err)
	}

	err = rightMatrix.Write(benchmarkData(rows * cols))
	if err != nil {
		b.Fatal(err)
	}

	b.ReportAllocs()
	b.ResetTimer()

	for range b.N {
		err = operation(leftMatrix, rightMatrix, outMatrix)
		if err != nil {
			b.Fatal(err)
		}

		if syncEachIteration {
			_, err = outMatrix.Read()
			if err != nil {
				b.Fatal(err)
			}
		}
	}

	if !syncEachIteration {
		_, err = outMatrix.Read()
		if err != nil {
			b.Fatal(err)
		}
	}
}

func benchmarkMatMulOperation( //nolint:cyclop,funlen // Benchmark setup and errors remain readable together.
	b *testing.B,
	rows int,
	sharedDim int,
	cols int,
	syncEachIteration bool,
) {
	b.Helper()

	ctx, err := mat.NewContext()
	if err != nil {
		b.Fatal(err)
	}
	defer ctx.Release()

	leftMatrix, err := mat.NewMatrix(ctx, rows, sharedDim)
	if err != nil {
		b.Fatal(err)
	}
	defer leftMatrix.Release()

	rightMatrix, err := mat.NewMatrix(ctx, sharedDim, cols)
	if err != nil {
		b.Fatal(err)
	}
	defer rightMatrix.Release()

	outMatrix, err := mat.NewMatrix(ctx, rows, cols)
	if err != nil {
		b.Fatal(err)
	}
	defer outMatrix.Release()

	err = leftMatrix.Write(benchmarkData(rows * sharedDim))
	if err != nil {
		b.Fatal(err)
	}

	err = rightMatrix.Write(benchmarkData(sharedDim * cols))
	if err != nil {
		b.Fatal(err)
	}

	b.ReportAllocs()
	b.ResetTimer()

	for range b.N {
		err = mat.MatMul(leftMatrix, rightMatrix, outMatrix)
		if err != nil {
			b.Fatal(err)
		}

		if syncEachIteration {
			_, err = outMatrix.Read()
			if err != nil {
				b.Fatal(err)
			}
		}
	}

	if !syncEachIteration {
		_, err = outMatrix.Read()
		if err != nil {
			b.Fatal(err)
		}
	}
}

func benchmarkUnaryOperation( //nolint:cyclop // Benchmark setup and synchronization errors must fail locally.
	b *testing.B,
	operation unaryOperation,
	syncEachIteration bool,
) {
	b.Helper()

	const rows, cols = 128, 128

	ctx, err := mat.NewContext()
	if err != nil {
		b.Fatal(err)
	}
	defer ctx.Release()

	inputMatrix, err := mat.NewMatrix(ctx, rows, cols)
	if err != nil {
		b.Fatal(err)
	}
	defer inputMatrix.Release()

	outMatrix, err := mat.NewMatrix(ctx, rows, cols)
	if err != nil {
		b.Fatal(err)
	}
	defer outMatrix.Release()

	err = inputMatrix.Write(benchmarkData(rows * cols))
	if err != nil {
		b.Fatal(err)
	}

	b.ReportAllocs()
	b.ResetTimer()

	for range b.N {
		err = operation(inputMatrix, outMatrix)
		if err != nil {
			b.Fatal(err)
		}

		if syncEachIteration {
			_, err = outMatrix.Read()
			if err != nil {
				b.Fatal(err)
			}
		}
	}

	if !syncEachIteration {
		_, err = outMatrix.Read()
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkMatMul64x64(b *testing.B) {
	b.Run("EndToEnd", func(b *testing.B) {
		benchmarkMatMulOperation(b, 64, 64, 64, true)
	})
	b.Run("DeviceResident", func(b *testing.B) {
		benchmarkMatMulOperation(b, 64, 64, 64, false)
	})
}

func BenchmarkAdd256x256(b *testing.B) {
	b.Run("EndToEnd", func(b *testing.B) {
		benchmarkBinaryOperation(b, mat.Add, true)
	})
	b.Run("DeviceResident", func(b *testing.B) {
		benchmarkBinaryOperation(b, mat.Add, false)
	})
}

func BenchmarkMul256x256(b *testing.B) {
	b.Run("EndToEnd", func(b *testing.B) {
		benchmarkBinaryOperation(b, mat.Mul, true)
	})
	b.Run("DeviceResident", func(b *testing.B) {
		benchmarkBinaryOperation(b, mat.Mul, false)
	})
}

func BenchmarkP4DeviceResidentChain(b *testing.B) { //nolint:cyclop,funlen // Setup names every device-resident stage.
	const rows, cols = 64, 64

	ctx, err := mat.NewContext()
	if err != nil {
		b.Fatal(err)
	}

	b.Cleanup(ctx.Release)

	newMatrix := func(matrixRows, matrixCols int) *mat.Matrix {
		matrix, matrixErr := mat.NewMatrix(ctx, matrixRows, matrixCols)
		if matrixErr != nil {
			b.Fatal(matrixErr)
		}

		b.Cleanup(matrix.Release)

		return matrix
	}

	input := newMatrix(rows, cols)
	row := newMatrix(1, cols)
	column := newMatrix(rows, 1)
	added := newMatrix(rows, cols)
	multiplied := newMatrix(rows, cols)
	scaled := newMatrix(rows, cols)
	transposed := newMatrix(cols, rows)
	reduced := newMatrix(1, rows)
	broadcast := newMatrix(cols, rows)
	reshaped := newMatrix(rows, cols)

	err = input.Write(benchmarkData(rows * cols))
	if err != nil {
		b.Fatal(err)
	}

	err = row.Write(benchmarkData(cols))
	if err != nil {
		b.Fatal(err)
	}

	err = column.Write(benchmarkData(rows))
	if err != nil {
		b.Fatal(err)
	}

	b.ReportAllocs()
	b.ResetTimer()

	for range b.N {
		err = mat.Add(input, row, added)
		if err != nil {
			b.Fatal(err)
		}

		err = mat.Mul(added, column, multiplied)
		if err != nil {
			b.Fatal(err)
		}

		err = mat.Scale(multiplied, 0.5, scaled)
		if err != nil {
			b.Fatal(err)
		}

		err = mat.Transp(scaled, transposed)
		if err != nil {
			b.Fatal(err)
		}

		err = mat.ReduceSumTo(transposed, reduced)
		if err != nil {
			b.Fatal(err)
		}

		err = mat.BroadcastTo(reduced, broadcast)
		if err != nil {
			b.Fatal(err)
		}

		err = mat.ReshapeTo(broadcast, reshaped)
		if err != nil {
			b.Fatal(err)
		}

		_, err = reshaped.Read()
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkSoftmax128x128(b *testing.B) {
	b.Run("EndToEnd", func(b *testing.B) {
		benchmarkUnaryOperation(b, mat.Softmax, true)
	})
	b.Run("DeviceResident", func(b *testing.B) {
		benchmarkUnaryOperation(b, mat.Softmax, false)
	})
}

func BenchmarkRMSNorm128x128(b *testing.B) {
	b.Run("EndToEnd", func(b *testing.B) {
		benchmarkUnaryOperation(b, mat.RMSNorm, true)
	})
	b.Run("DeviceResident", func(b *testing.B) {
		benchmarkUnaryOperation(b, mat.RMSNorm, false)
	})
}
