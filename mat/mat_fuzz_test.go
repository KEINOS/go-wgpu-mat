//go:build !cgo

package mat_test

import (
	"math"
	"testing"

	"github.com/KEINOS/go-wgpu-mat/mat"
)

func fuzzData(length int, seed uint64) []float32 {
	data := make([]float32, length)

	x := seed
	for index := range data {
		x ^= x << 13
		x ^= x >> 7
		x ^= x << 17
		data[index] = (float32(int64(x%2001)-1000) / 100.0)
	}

	return data
}

func FuzzMatrixWriteReadRoundTrip(f *testing.F) {
	f.Add(uint8(2), uint8(3), uint64(1))
	f.Add(uint8(4), uint8(4), uint64(42))

	f.Fuzz(func(t *testing.T, rowsU8 uint8, colsU8 uint8, seed uint64) {
		serializeGPUTest(t)

		rows := int(rowsU8%8) + 1
		cols := int(colsU8%8) + 1

		ctx, err := mat.NewContext(mat.UseCPU)
		if err != nil {
			t.Fatal(err)
		}
		defer ctx.Release()

		matrix, err := mat.NewMatrix(ctx, rows, cols)
		if err != nil {
			t.Fatal(err)
		}
		defer matrix.Release()

		input := fuzzData(rows*cols, seed)

		err = matrix.Write(input)
		if err != nil {
			t.Fatal(err)
		}

		output, err := matrix.Read()
		if err != nil {
			t.Fatal(err)
		}

		if len(output) != len(input) {
			t.Fatalf("len mismatch: got %d want %d", len(output), len(input))
		}

		for index := range input {
			if math.Abs(float64(input[index]-output[index])) > 1e-5 {
				t.Fatalf(
					"value mismatch at %d: got %f want %f",
					index,
					output[index],
					input[index],
				)
			}
		}
	})
}

func FuzzSoftmaxRowSums(f *testing.F) {
	f.Add(uint8(2), uint8(3), uint64(3))
	f.Add(uint8(3), uint8(5), uint64(9))

	f.Fuzz(func(t *testing.T, rowsU8 uint8, colsU8 uint8, seed uint64) {
		serializeGPUTest(t)

		rows := int(rowsU8%6) + 1
		cols := int(colsU8%6) + 1

		ctx, err := mat.NewContext(mat.UseCPU)
		if err != nil {
			t.Fatal(err)
		}
		defer ctx.Release()

		inputMatrix, err := mat.NewMatrix(ctx, rows, cols)
		if err != nil {
			t.Fatal(err)
		}
		defer inputMatrix.Release()

		outMatrix, err := mat.NewMatrix(ctx, rows, cols)
		if err != nil {
			t.Fatal(err)
		}
		defer outMatrix.Release()

		input := fuzzData(rows*cols, seed)

		err = inputMatrix.Write(input)
		if err != nil {
			t.Fatal(err)
		}

		err = mat.Softmax(inputMatrix, outMatrix)
		if err != nil {
			t.Fatal(err)
		}

		output, err := outMatrix.Read()
		if err != nil {
			t.Fatal(err)
		}

		assertSoftmaxRows(t, output, rows, cols)
	})
}

func assertSoftmaxRows(t *testing.T, output []float32, rows int, cols int) {
	t.Helper()

	for row := range rows {
		sum := 0.0

		for col := range cols {
			value := output[row*cols+col]
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
				t.Fatalf("invalid softmax value at row=%d col=%d", row, col)
			}

			sum += float64(value)
		}

		if math.Abs(sum-1.0) > 1e-4 {
			t.Fatalf("row sum mismatch at row=%d: got %f", row, sum)
		}
	}
}
