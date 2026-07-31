package mat

import (
	"io"
	"sync"
	"testing"

	"github.com/gogpu/wgpu"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestStatsMatrixTransferAndLifetimeContract(t *testing.T) {
	t.Parallel()

	ctx := new(Context)
	ctx.device = new(wgpu.Device)

	deps := new(matrixDeps)
	deps.createBuffer = func(*Context, *wgpu.BufferDescriptor) (*wgpu.Buffer, error) {
		return new(wgpu.Buffer), nil
	}
	deps.releaseBuffer = func(*wgpu.Buffer) {}
	deps.writeBuffer = func(*Context, *wgpu.Buffer, []byte) error { return nil }
	deps.readBuffer = func(*Context, *wgpu.Buffer, []byte) error { return nil }

	matrix, err := newMatrix(ctx, 2, 3, *deps)
	require.NoError(t, err)
	secondary, err := newMatrix(ctx, 1, 1, *deps)
	require.NoError(t, err)

	afterAllocation := ctx.Stats()
	assert.Equal(t, uint64(2), afterAllocation.MatrixAllocationCount)
	assert.Equal(t, uint64(0), afterAllocation.MatrixReleaseCount)
	assert.Equal(t, uint64(28), afterAllocation.LiveMatrixBytes)
	assert.Equal(t, uint64(28), afterAllocation.PeakLiveMatrixBytes)

	require.NoError(t, matrix.Write([]float32{1, 2, 3, 4, 5, 6}))
	_, err = matrix.Read()
	require.NoError(t, err)

	afterTransfer := ctx.Stats()
	assert.Equal(t, uint64(1), afterTransfer.HostWriteCount)
	assert.Equal(t, uint64(24), afterTransfer.HostWriteBytes)
	assert.Equal(t, uint64(1), afterTransfer.HostReadCount)
	assert.Equal(t, uint64(24), afterTransfer.HostReadBytes)

	secondary.Release()
	matrix.Release()
	matrix.Release()

	afterRelease := ctx.Stats()
	assert.Equal(t, uint64(2), afterRelease.MatrixAllocationCount)
	assert.Equal(t, uint64(2), afterRelease.MatrixReleaseCount)
	assert.Equal(t, uint64(0), afterRelease.LiveMatrixBytes)
	assert.Equal(t, uint64(28), afterRelease.PeakLiveMatrixBytes)
}

func TestStatsSubmissionContract(t *testing.T) {
	t.Parallel()

	successCtx := new(Context)
	readDeps := newTestReadBufferDeps(make([]byte, bytesPerFloat32Int))
	require.NoError(t, readBuffer(
		successCtx,
		new(wgpu.Buffer),
		make([]byte, bytesPerFloat32Int),
		readDeps,
	))
	assert.Equal(t, uint64(1), successCtx.Stats().ReadbackSubmissionCount)
	assert.Equal(t, uint64(0), successCtx.Stats().ComputeSubmissionCount)

	failureCtx := new(Context)
	readDeps.submit = func(*Context, *wgpu.CommandBuffer) error { return io.EOF }
	require.ErrorIs(t, readBuffer(
		failureCtx,
		new(wgpu.Buffer),
		make([]byte, bytesPerFloat32Int),
		readDeps,
	), io.EOF)
	assert.Equal(t, uint64(0), failureCtx.Stats().ReadbackSubmissionCount)

	left, right, out := matMulTestMatrices()
	markHardwareMock(left, right, out)

	computeDeps := successfulMatMulWGPUDeps()
	computeDeps.submit = func(*wgpu.Device, *wgpu.CommandBuffer) error { return io.EOF }
	require.ErrorIs(t, dispatchTensorOperationWithDeps(
		tensorOpMul,
		left,
		right,
		out,
		0,
		computeDeps,
	), io.EOF)
	assert.Equal(t, uint64(0), left.ctx.Stats().ComputeSubmissionCount)
	assert.Equal(t, uint64(0), left.ctx.Stats().MatrixAllocationCount)
}

func TestStatsFailedMatrixOperationsAreNotCounted(t *testing.T) {
	t.Parallel()

	ctx := new(Context)
	ctx.device = new(wgpu.Device)
	deps := new(matrixDeps)
	deps.createBuffer = func(*Context, *wgpu.BufferDescriptor) (*wgpu.Buffer, error) {
		return nil, io.EOF
	}
	deps.releaseBuffer = func(*wgpu.Buffer) {}

	_, err := newMatrix(ctx, 1, 1, *deps)
	require.ErrorIs(t, err, io.EOF)

	var zero Stats
	assert.Equal(t, zero, ctx.Stats())

	matrix, matrixIO := newMockMatrix(1, 1, []float32{1})
	matrixIO.writeErr = io.EOF
	require.ErrorIs(t, matrix.Write([]float32{2}), io.EOF)

	matrixIO.writeErr = nil
	matrixIO.readErr = io.EOF
	_, err = matrix.Read()
	require.ErrorIs(t, err, io.EOF)
	assert.Equal(t, zero, matrix.ctx.Stats())
}

func TestStatsConcurrentAccounting(t *testing.T) {
	t.Parallel()

	ctx := new(Context)

	const (
		workers     = 32
		matrixBytes = uint64(16)
	)

	var group sync.WaitGroup
	group.Add(workers)

	for range workers {
		go func() {
			defer group.Done()

			ctx.recordHostRead(matrixBytes)
			ctx.recordHostWrite(matrixBytes)
			ctx.recordComputeSubmission()
			ctx.recordReadbackSubmission()
			ctx.recordMatrixAllocation(matrixBytes)
			ctx.recordMatrixRelease(matrixBytes)
		}()
	}

	group.Wait()

	stats := ctx.Stats()

	assert.Equal(t, uint64(workers), stats.HostReadCount)
	assert.Equal(t, uint64(workers)*matrixBytes, stats.HostReadBytes)
	assert.Equal(t, uint64(workers), stats.HostWriteCount)
	assert.Equal(t, uint64(workers)*matrixBytes, stats.HostWriteBytes)
	assert.Equal(t, uint64(workers), stats.ComputeSubmissionCount)
	assert.Equal(t, uint64(workers), stats.ReadbackSubmissionCount)
	assert.Equal(t, uint64(workers), stats.MatrixAllocationCount)
	assert.Equal(t, uint64(workers), stats.MatrixReleaseCount)
	assert.Equal(t, uint64(0), stats.LiveMatrixBytes)
	assert.GreaterOrEqual(t, stats.PeakLiveMatrixBytes, matrixBytes)
}
