package mat

import (
	"context"
	"encoding/binary"
	"fmt"
	"math"
	"math/bits"
	"sync/atomic"

	"github.com/gogpu/wgpu"
)

// ============================================================================
//  Constants/Magic Numbers
// ============================================================================

const (
	bytesPerFloat32Int = 4
	bytesPerFloat32U64 = uint64(4)
)

// ============================================================================
//  Type: Matrix
// ============================================================================

// Shape is the immutable-by-copy shape of a Matrix.
type Shape struct {
	rows int
	cols int
}

// String formats a shape as "rowsxcols".
func (s Shape) String() string {
	return fmt.Sprintf("%dx%d", s.rows, s.cols)
}

// Rows returns the number of rows.
func (s Shape) Rows() int {
	return s.rows
}

// Cols returns the number of columns.
func (s Shape) Cols() int {
	return s.cols
}

// Len returns the number of elements in the shape. Shapes returned by Matrix
// are always valid and cannot overflow int.
func (s Shape) Len() int {
	return s.rows * s.cols
}

// Matrix represents a 2D float32 array stored in a WGPU storage buffer.
//
// Data is stored in row-major order: element (r, c) is at
// index r*Cols() + c within the underlying GPU buffer.
//
// Kernelized GPU operations submit commands to the device queue. CPU fallback
// and host compatibility operations complete synchronously. Read waits for
// pending device work before returning host data.
type Matrix struct {
	rows     int
	cols     int
	buf      *wgpu.Buffer
	ctx      *Context
	released atomic.Uint32
	deps     matrixDeps
}

// ----------------------------------------------------------------------------
//  Constructors
// ----------------------------------------------------------------------------

// NewMatrix allocates a WGPU buffer for a rows x cols float32 matrix.
// The initial buffer contents are undefined; call Write to upload
// data before performing calculations.
func NewMatrix(ctx *Context, rows, cols int) (*Matrix, error) {
	return newMatrix(ctx, rows, cols, defaultMatrixDeps())
}

func newMatrix(
	ctx *Context,
	rows, cols int,
	deps matrixDeps,
) (*Matrix, error) {
	err := validateContextForMatrix(ctx)
	if err != nil {
		return nil, err
	}

	size, err := matrixBufferSize(rows, cols)
	if err != nil {
		return nil, err
	}

	err = validateDeviceBufferSize(ctx, size)
	if err != nil {
		return nil, err
	}

	bufferDescriptor := new(wgpu.BufferDescriptor)
	bufferDescriptor.Label = fmt.Sprintf("go-wgpu-mat-%dx%d", rows, cols)
	bufferDescriptor.Size = size
	bufferDescriptor.Usage = wgpu.BufferUsageStorage |
		wgpu.BufferUsageCopyDst |
		wgpu.BufferUsageCopySrc

	buf, err := deps.createBuffer(ctx, bufferDescriptor)
	if err != nil {
		if buf != nil {
			deps.releaseBuffer(buf)
		}

		return nil, wrapError(err, "failed to create buffer")
	}

	if buf == nil {
		return nil, sentinelError(
			ErrBackendUnavailable,
			"failed to create buffer: backend returned a nil buffer",
		)
	}

	ctx.recordBufferAllocation()

	matrix := new(Matrix)
	matrix.rows = rows
	matrix.cols = cols
	matrix.buf = buf
	matrix.ctx = ctx
	matrix.deps = deps

	return matrix, nil
}

func validateContextForMatrix(ctx *Context) error {
	if ctx == nil {
		return sentinelError(ErrNilContext, "context is nil")
	}

	if ctx.released.Load() != 0 {
		return sentinelError(ErrContextReleased, "context is released")
	}

	if ctx.device == nil {
		return sentinelError(
			ErrContextNotInitialized,
			"context is not initialized",
		)
	}

	return nil
}

func matrixBufferSize(rows, cols int) (uint64, error) {
	if rows <= 0 || cols <= 0 {
		return 0, sentinelError(
			ErrInvalidDimension,
			"matrix dimensions must be positive: got %dx%d",
			rows,
			cols,
		)
	}

	high, elementCount := bits.Mul64(uint64(rows), uint64(cols))
	if high != 0 {
		return 0, sentinelError(
			ErrOverflow,
			"matrix dimensions overflow: %dx%d",
			rows,
			cols,
		)
	}

	high, size := bits.Mul64(elementCount, bytesPerFloat32U64)
	if high != 0 {
		return 0, sentinelError(
			ErrOverflow,
			"matrix byte size overflow: %d elements",
			elementCount,
		)
	}

	return size, nil
}

func validateDeviceBufferSize(ctx *Context, size uint64) error {
	if ctx.limits.MaxBufferSize > 0 && size > ctx.limits.MaxBufferSize {
		return sentinelError(
			ErrDeviceLimit,
			"matrix byte size %d exceeds device maximum buffer size %d",
			size,
			ctx.limits.MaxBufferSize,
		)
	}

	if ctx.limits.MaxStorageBufferBindingSize > 0 &&
		size > ctx.limits.MaxStorageBufferBindingSize {
		return sentinelError(
			ErrDeviceLimit,
			"matrix byte size %d exceeds device maximum storage buffer binding size %d",
			size,
			ctx.limits.MaxStorageBufferBindingSize,
		)
	}

	return nil
}

// ----------------------------------------------------------------------------
//  Methods
// ----------------------------------------------------------------------------

// Rows returns the number of rows.
func (m *Matrix) Rows() int {
	if m == nil {
		return 0
	}

	return m.rows
}

// Cols returns the number of columns.
func (m *Matrix) Cols() int {
	if m == nil {
		return 0
	}

	return m.cols
}

// Shape returns the matrix shape as an immutable-by-copy value.
func (m *Matrix) Shape() Shape {
	if m == nil {
		return Shape{rows: 0, cols: 0}
	}

	return Shape{rows: m.rows, cols: m.cols}
}

// Len returns the number of matrix elements.
func (m *Matrix) Len() int {
	if m == nil {
		return 0
	}

	return m.rows * m.cols
}

// Released reports whether Release has been called.
func (m *Matrix) Released() bool {
	return m == nil || m.released.Load() != 0
}

// String returns a compact diagnostic representation of the matrix.
func (m *Matrix) String() string {
	if m == nil {
		return "Matrix<nil>"
	}

	if m.Released() {
		return fmt.Sprintf("Matrix[%s, released]", m.Shape())
	}

	return fmt.Sprintf("Matrix[%s]", m.Shape())
}

// Write uploads data to the GPU buffer.
// data must have exactly m.Len() elements.
func (m *Matrix) Write(data []float32) error {
	if m == nil || m.ctx == nil || m.buf == nil {
		return sentinelError(ErrNotInitialized, "matrix is not initialized")
	}

	if m.released.Load() != 0 {
		return sentinelError(ErrReleased, "matrix is released")
	}

	if m.ctx.released.Load() != 0 {
		return sentinelError(ErrContextReleased, "context is released")
	}

	want := m.Len()
	if len(data) != want {
		return sentinelError(
			ErrLengthMismatch,
			"fail to write %s: got %d elements, want %d",
			m.Shape(),
			len(data),
			want,
		)
	}

	raw := make([]byte, len(data)*bytesPerFloat32Int)
	for i, v := range data {
		binary.LittleEndian.PutUint32(
			raw[i*bytesPerFloat32Int:], math.Float32bits(v),
		)
	}

	err := m.ctx.withQueue(func() error {
		return m.deps.writeBuffer(m.ctx, m.buf, raw)
	})
	if err != nil {
		return wrapError(err, "failed to write buffer")
	}

	m.ctx.recordHostWrite()

	return nil
}

// Read downloads the matrix data from the GPU and returns it as a
// flat float32 slice in row-major order (length = m.Len()).
func (m *Matrix) Read() ([]float32, error) {
	if m == nil || m.ctx == nil || m.buf == nil {
		return nil, sentinelError(ErrNotInitialized, "matrix is not initialized")
	}

	if m.released.Load() != 0 {
		return nil, sentinelError(ErrReleased, "matrix is released")
	}

	if m.ctx.released.Load() != 0 {
		return nil, sentinelError(ErrContextReleased, "context is released")
	}

	elementCount := m.Len()

	raw := make([]byte, elementCount*bytesPerFloat32Int)

	err := m.ctx.withQueue(func() error {
		return m.deps.readBuffer(m.ctx, m.buf, raw)
	})
	if err != nil {
		return nil, wrapError(err, "failed to read buffer")
	}

	m.ctx.recordHostRead()

	result := make([]float32, elementCount)
	for i := range result {
		result[i] = math.Float32frombits(
			binary.LittleEndian.Uint32(raw[i*bytesPerFloat32Int:]),
		)
	}

	return result, nil
}

// Release frees the GPU buffer held by this matrix.
// Calling Release more than once is safe (subsequent calls are no-ops).
// Release must not run concurrently with operations using the matrix.
func (m *Matrix) Release() {
	if m == nil || !m.released.CompareAndSwap(0, 1) {
		return
	}

	if m.buf != nil {
		m.deps.releaseBuffer(m.buf)
		m.ctx.recordBufferRelease()
	}
}

// Close releases the matrix and always returns nil. It allows Matrix to be
// used as an io.Closer while preserving the idempotent Release API.
func (m *Matrix) Close() error {
	m.Release()

	return nil
}

// ============================================================================
//  Type: matrixDeps
// ============================================================================

type matrixDeps struct {
	createBuffer  func(*Context, *wgpu.BufferDescriptor) (*wgpu.Buffer, error)
	releaseBuffer func(*wgpu.Buffer)
	writeBuffer   func(*Context, *wgpu.Buffer, []byte) error
	readBuffer    func(*Context, *wgpu.Buffer, []byte) error
}

type readBufferDeps struct {
	createStaging        func(*Context, uint64) (*wgpu.Buffer, error)
	releaseBuffer        func(*wgpu.Buffer)
	createEncoder        func(*Context) (*wgpu.CommandEncoder, error)
	copyBuffer           func(*wgpu.CommandEncoder, *wgpu.Buffer, *wgpu.Buffer, uint64)
	finishEncoder        func(*wgpu.CommandEncoder) (*wgpu.CommandBuffer, error)
	releaseCommandBuffer func(*wgpu.CommandBuffer)
	submit               func(*Context, *wgpu.CommandBuffer) error
	mapBuffer            func(*wgpu.Buffer, uint64) error
	mappedRange          func(*wgpu.Buffer, uint64) (*wgpu.MappedRange, error)
	mappedBytes          func(*wgpu.MappedRange) []byte
	releaseMappedRange   func(*wgpu.MappedRange)
	unmapBuffer          func(*wgpu.Buffer) error
}

// ============================================================================
//  Functions
// ============================================================================

func defaultMatrixDeps() matrixDeps {
	deps := new(matrixDeps)
	deps.createBuffer = func(
		ctx *Context,
		desc *wgpu.BufferDescriptor,
	) (*wgpu.Buffer, error) {
		return ctx.device.CreateBuffer(desc)
	}
	deps.releaseBuffer = func(buf *wgpu.Buffer) { buf.Release() }
	deps.writeBuffer = func(ctx *Context, buf *wgpu.Buffer, data []byte) error {
		return ctx.device.Queue().WriteBuffer(buf, 0, data)
	}
	deps.readBuffer = func(ctx *Context, buf *wgpu.Buffer, data []byte) error {
		return readBuffer(ctx, buf, data, defaultReadBufferDeps())
	}

	return *deps
}

func defaultReadBufferDeps() readBufferDeps {
	deps := new(readBufferDeps)
	deps.createStaging = func(ctx *Context, size uint64) (*wgpu.Buffer, error) {
		desc := new(wgpu.BufferDescriptor)
		desc.Label = "go-wgpu-mat-readback"
		desc.Size = size
		desc.Usage = wgpu.BufferUsageCopyDst | wgpu.BufferUsageMapRead

		return ctx.device.CreateBuffer(desc)
	}
	deps.releaseBuffer = func(buf *wgpu.Buffer) { buf.Release() }
	deps.createEncoder = func(ctx *Context) (*wgpu.CommandEncoder, error) {
		return ctx.device.CreateCommandEncoder(nil)
	}
	deps.copyBuffer = func(
		encoder *wgpu.CommandEncoder,
		src, dst *wgpu.Buffer,
		size uint64,
	) {
		encoder.CopyBufferToBuffer(src, 0, dst, 0, size)
	}
	deps.finishEncoder = func(
		encoder *wgpu.CommandEncoder,
	) (*wgpu.CommandBuffer, error) {
		return encoder.Finish()
	}
	deps.releaseCommandBuffer = func(cmd *wgpu.CommandBuffer) { cmd.Release() }
	deps.submit = func(ctx *Context, cmd *wgpu.CommandBuffer) error {
		_, err := ctx.device.Queue().Submit(cmd)

		return wrapError(err, "submit command buffer")
	}
	deps.mapBuffer = func(buf *wgpu.Buffer, size uint64) error {
		return buf.Map(context.Background(), wgpu.MapModeRead, 0, size)
	}
	deps.mappedRange = func(
		buf *wgpu.Buffer,
		size uint64,
	) (*wgpu.MappedRange, error) {
		return buf.MappedRange(0, size)
	}
	deps.mappedBytes = func(mapped *wgpu.MappedRange) []byte { return mapped.Bytes() }
	deps.releaseMappedRange = func(mapped *wgpu.MappedRange) { mapped.Release() }
	deps.unmapBuffer = func(buf *wgpu.Buffer) error { return buf.Unmap() }

	return *deps
}

//nolint:cyclop,funlen // Readback stages require local cleanup and error context.
func readBuffer(ctx *Context, src *wgpu.Buffer, data []byte, deps readBufferDeps) error {
	size := uint64(len(data))

	staging, err := deps.createStaging(ctx, size)
	if err != nil {
		if staging != nil {
			deps.releaseBuffer(staging)
		}

		return wrapError(err, "create readback buffer")
	}

	if staging == nil {
		return sentinelError(ErrBackendUnavailable, "create readback buffer returned nil")
	}

	ctx.recordBufferAllocation()

	defer func() {
		deps.releaseBuffer(staging)
		ctx.recordBufferRelease()
	}()

	encoder, err := deps.createEncoder(ctx)
	if err != nil {
		return wrapError(err, "create readback encoder")
	}

	deps.copyBuffer(encoder, src, staging, size)

	commandBuffer, err := deps.finishEncoder(encoder)
	if err != nil {
		return wrapError(err, "finish readback encoder")
	}
	defer deps.releaseCommandBuffer(commandBuffer)

	err = deps.submit(ctx, commandBuffer)
	if err != nil {
		return wrapError(err, "submit readback")
	}

	ctx.recordSubmission()

	err = deps.mapBuffer(staging, size)
	if err != nil {
		return wrapError(err, "map readback buffer")
	}

	mapped, err := deps.mappedRange(staging, size)
	if err != nil {
		_ = deps.unmapBuffer(staging)

		return wrapError(err, "get mapped readback range")
	}

	mappedData := deps.mappedBytes(mapped)
	if len(mappedData) != len(data) {
		deps.releaseMappedRange(mapped)
		_ = deps.unmapBuffer(staging)

		return newError(
			"mapped readback size mismatch: got %d bytes, want %d",
			len(mappedData),
			len(data),
		)
	}

	copy(data, mappedData)
	deps.releaseMappedRange(mapped)

	err = deps.unmapBuffer(staging)
	if err != nil {
		return wrapError(err, "unmap readback buffer")
	}

	return nil
}
