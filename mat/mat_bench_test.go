//go:build !cgo

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

func benchmarkBinaryOperation(
	b *testing.B,
	rows int,
	cols int,
	operation binaryOperation,
) {
	b.Helper()

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

		_, err = outMatrix.Read()
		if err != nil {
			b.Fatal(err)
		}
	}
}

func benchmarkMatMulOperation(
	b *testing.B,
	rows int,
	sharedDim int,
	cols int,
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

		_, err = outMatrix.Read()
		if err != nil {
			b.Fatal(err)
		}
	}
}

func benchmarkUnaryOperation(
	b *testing.B,
	rows int,
	cols int,
	operation unaryOperation,
) {
	b.Helper()

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

		_, err = outMatrix.Read()
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkMatMul64x64(b *testing.B) {
	benchmarkMatMulOperation(b, 64, 64, 64)
}

func BenchmarkAdd256x256(b *testing.B) {
	benchmarkBinaryOperation(b, 256, 256, mat.Add)
}

func BenchmarkSoftmax128x128(b *testing.B) {
	benchmarkUnaryOperation(b, 128, 128, mat.Softmax)
}

func BenchmarkRMSNorm128x128(b *testing.B) {
	benchmarkUnaryOperation(b, 128, 128, mat.RMSNorm)
}