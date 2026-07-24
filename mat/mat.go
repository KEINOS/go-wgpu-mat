package mat

import (
	"context"
	"encoding/binary"
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

// Matrix represents a 2D float32 array stored on the GPU.
//
// Data is stored in row-major order: element (r, c) is at
// index r*Cols + c within the underlying GPU buffer.
//
// All operations on a Matrix submit commands to the GPU queue.
// Results are synchronized on Read.
type Matrix struct {
	// Rows is the number of rows.
	Rows int
	// Cols is the number of columns.
	Cols int

	buf      *wgpu.Buffer
	ctx      *Context
	released atomic.Uint32
	deps     matrixDeps
}

// ----------------------------------------------------------------------------
//  Constructors
// ----------------------------------------------------------------------------

// NewMatrix allocates a GPU buffer for a rows x cols float32 matrix.
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
	if ctx == nil {
		return nil, newError("context is nil")
	}

	if ctx.released.Load() != 0 {
		return nil, newError("context is released")
	}

	if ctx.device == nil {
		return nil, newError("context is nil")
	}

	if rows <= 0 || cols <= 0 {
		return nil, newError("matrix dimensions must be positive")
	}

	rowCount := uint64(rows)
	colCount := uint64(cols)

	high, elementCount := bits.Mul64(rowCount, colCount)
	if high != 0 {
		return nil, newError("matrix dimensions overflow")
	}

	high, size := bits.Mul64(elementCount, bytesPerFloat32U64)
	if high != 0 {
		return nil, newError("matrix byte size overflow")
	}

	err := validateDeviceBufferSize(ctx, size)
	if err != nil {
		return nil, err
	}

	bufferDescriptor := new(wgpu.BufferDescriptor)
	bufferDescriptor.Label = "go-wgpu-mat"
	bufferDescriptor.Size = size
	bufferDescriptor.Usage = wgpu.BufferUsageStorage |
		wgpu.BufferUsageCopyDst |
		wgpu.BufferUsageCopySrc

	buf, err := deps.createBuffer(ctx, bufferDescriptor)
	if err != nil {
		return nil, wrapError(err, "failed to create buffer")
	}

	matrix := new(Matrix)
	matrix.Rows = rows
	matrix.Cols = cols
	matrix.buf = buf
	matrix.ctx = ctx
	matrix.deps = deps

	return matrix, nil
}

func validateDeviceBufferSize(ctx *Context, size uint64) error {
	if ctx.limits.MaxBufferSize > 0 && size > ctx.limits.MaxBufferSize {
		return newError(
			"matrix byte size %d exceeds device maximum buffer size %d",
			size,
			ctx.limits.MaxBufferSize,
		)
	}

	if ctx.limits.MaxStorageBufferBindingSize > 0 &&
		size > ctx.limits.MaxStorageBufferBindingSize {
		return newError(
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

// Write uploads data to the GPU buffer.
// data must have exactly m.Rows*m.Cols elements.
func (m *Matrix) Write(data []float32) error {
	if m == nil || m.ctx == nil || m.buf == nil {
		return newError("matrix is not initialized")
	}

	if m.released.Load() != 0 {
		return newError("matrix is released")
	}

	if m.ctx.released.Load() != 0 {
		return newError("context is released")
	}

	want := m.Rows * m.Cols
	if len(data) != want {
		return newError(
			"fail to write: got %d elements, want %d", len(data), want,
		)
	}

	raw := make([]byte, len(data)*bytesPerFloat32Int)
	for i, v := range data {
		binary.LittleEndian.PutUint32(
			raw[i*bytesPerFloat32Int:], math.Float32bits(v),
		)
	}

	err := m.deps.writeBuffer(m.ctx, m.buf, raw)
	if err != nil {
		return wrapError(err, "failed to write buffer")
	}

	return nil
}

// Read downloads the matrix data from the GPU and returns it as a
// flat float32 slice in row-major order (length = m.Rows*m.Cols).
func (m *Matrix) Read() ([]float32, error) {
	if m == nil || m.ctx == nil || m.buf == nil {
		return nil, newError("matrix is not initialized")
	}

	if m.released.Load() != 0 {
		return nil, newError("matrix is released")
	}

	if m.ctx.released.Load() != 0 {
		return nil, newError("context is released")
	}

	elementCount := m.Rows * m.Cols

	raw := make([]byte, elementCount*bytesPerFloat32Int)

	err := m.deps.readBuffer(m.ctx, m.buf, raw)
	if err != nil {
		return nil, wrapError(err, "failed to read buffer")
	}

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
		m.buf.Release()
	}
}

// ============================================================================
//  Type: matrixDeps
// ============================================================================

type matrixDeps struct {
	createBuffer func(*Context, *wgpu.BufferDescriptor) (*wgpu.Buffer, error)
	writeBuffer  func(*Context, *wgpu.Buffer, []byte) error
	readBuffer   func(*Context, *wgpu.Buffer, []byte) error
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

func readBuffer(ctx *Context, src *wgpu.Buffer, data []byte, deps readBufferDeps) error {
	size := uint64(len(data))

	staging, err := deps.createStaging(ctx, size)
	if err != nil {
		return wrapError(err, "create readback buffer")
	}
	defer deps.releaseBuffer(staging)

	encoder, err := deps.createEncoder(ctx)
	if err != nil {
		return wrapError(err, "create readback encoder")
	}

	deps.copyBuffer(encoder, src, staging, size)

	commandBuffer, err := deps.finishEncoder(encoder)
	if err != nil {
		return wrapError(err, "finish readback encoder")
	}

	err = deps.submit(ctx, commandBuffer)
	if err != nil {
		deps.releaseCommandBuffer(commandBuffer)

		return wrapError(err, "submit readback")
	}

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
