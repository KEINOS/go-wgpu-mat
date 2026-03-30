//go:build !cgo

package mat

import (
	"encoding/binary"
	"math"
	"math/bits"
	"sync/atomic"

	"github.com/gogpu/wgpu"
)

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

type matrixDeps struct {
	createBuffer func(*Context, *wgpu.BufferDescriptor) (*wgpu.Buffer, error)
	writeBuffer  func(*Context, *wgpu.Buffer, []byte) error
	readBuffer   func(*Context, *wgpu.Buffer, []byte) error
}

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
		return ctx.device.Queue().ReadBuffer(buf, 0, data)
	}

	return *deps
}

const (
	bytesPerFloat32Int = 4
	bytesPerFloat32U64 = uint64(4)
)

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
	if ctx == nil || ctx.device == nil {
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

// Write uploads data to the GPU buffer.
// data must have exactly m.Rows*m.Cols elements.
func (m *Matrix) Write(data []float32) error {
	if m == nil || m.ctx == nil || m.buf == nil {
		return newError("matrix is not initialized")
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
func (m *Matrix) Release() {
	if m == nil || !m.released.CompareAndSwap(0, 1) {
		return
	}

	if m.buf != nil {
		m.buf.Release()
	}
}
