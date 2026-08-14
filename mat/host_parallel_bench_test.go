package mat

import (
	"fmt"
	"runtime"
	"testing"
)

func BenchmarkMatMulCPUExecution(b *testing.B) {
	benchmarks := []struct {
		name string
		size int
	}{
		{name: "64x64", size: 64},
		{name: "128x128", size: 128},
		{name: "256x256", size: 256},
	}

	for _, benchmark := range benchmarks {
		b.Run(benchmark.name+"/original", func(b *testing.B) {
			benchmarkMatMulCPUExecution(b, benchmark.size, func(left, right, out *Matrix) error {
				return matMulCPUWithRunner(left, right, out, runWorkRangesSerial)
			}, nil)
		})
		b.Run(benchmark.name+"/enhanced", func(b *testing.B) {
			benchmarkMatMulCPUExecution(b, benchmark.size, matMulCPU, func(ctx *Context) {
				ctx.hostWorkerPool()
			})
		})
	}
}

func BenchmarkMatMulCPUKernel(b *testing.B) {
	for _, size := range []int{128, 256} {
		b.Run(fmt.Sprintf("%dx%d/original", size, size), func(b *testing.B) {
			benchmarkMatMulCPUKernel(b, size, runWorkRangesSerial)
		})
		b.Run(fmt.Sprintf("%dx%d/enhanced", size, size), func(b *testing.B) {
			pool := newHostWorkerPool(runtime.GOMAXPROCS(0))
			defer pool.close()

			benchmarkMatMulCPUKernel(b, size, pool.runWorkRanges)
		})
	}
}

func benchmarkMatMulCPUKernel(b *testing.B, size int, runner workRangeRunner) {
	b.Helper()

	left := make([]float32, size*size)
	right := make([]float32, size*size)
	result := make([]float32, size*size)

	b.ReportAllocs()
	b.ResetTimer()

	for range b.N {
		runner(size, size*size, func(start, end int) {
			multiplyMatMulRows(left, right, result, size, size, start, end)
		})
	}
}

func benchmarkMatMulCPUExecution( //nolint:cyclop // Benchmark setup and errors remain readable together.
	b *testing.B,
	size int,
	execute func(left, right, out *Matrix) error,
	prepare func(ctx *Context),
) {
	b.Helper()

	ctx, err := NewContext(UseCPU)
	if err != nil {
		b.Fatal(err)
	}
	defer ctx.Release()

	left, err := NewMatrix(ctx, size, size)
	if err != nil {
		b.Fatal(err)
	}
	defer left.Release()

	right, err := NewMatrix(ctx, size, size)
	if err != nil {
		b.Fatal(err)
	}
	defer right.Release()

	out, err := NewMatrix(ctx, size, size)
	if err != nil {
		b.Fatal(err)
	}
	defer out.Release()

	if prepare != nil {
		prepare(ctx)
	}

	data := make([]float32, size*size)
	for index := range data {
		data[index] = float32((index%17)-8) * 0.25
	}

	err = left.Write(data)
	if err != nil {
		b.Fatal(err)
	}

	err = right.Write(data)
	if err != nil {
		b.Fatal(err)
	}

	b.ReportAllocs()
	b.ResetTimer()

	for range b.N {
		err = execute(left, right, out)
		if err != nil {
			b.Fatal(err)
		}
	}
}

func runWorkRangesSerial(total, _ int, operation workRangeFunc) {
	operation(0, total)
}
