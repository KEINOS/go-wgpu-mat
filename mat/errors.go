package mat

import (
	"errors"
	"fmt"
)

// Sentinel errors returned by the package. Callers should compare against
// these with errors.Is rather than matching on error strings, e.g.:
//
//	err := mat.MatMul(a, b, out)
//	if errors.Is(err, mat.ErrDimensionMismatch) {
//		// handle incompatible shapes
//	}
//
// Validation and lifecycle errors wrap one of these sentinels, so their
// classification is stable across releases even though the human-readable
// message may gain extra detail (such as the offending matrix shapes).
var (
	// ErrNilContext indicates a nil or uninitialized *Context was supplied.
	ErrNilContext = errors.New("context is nil")
	// ErrContextReleased indicates the *Context has already been released.
	ErrContextReleased = errors.New("context is released")
	// ErrContextNotInitialized indicates a zero-value or incomplete Context.
	ErrContextNotInitialized = errors.New("context is not initialized")
	// ErrInvalidMode indicates an unknown or conflicting ContextMode.
	ErrInvalidMode = errors.New("invalid context mode")
	// ErrBackendUnavailable indicates no usable WGPU adapter was found.
	ErrBackendUnavailable = errors.New("backend unavailable")

	// ErrNotInitialized indicates a nil or uninitialized *Matrix was supplied.
	ErrNotInitialized = errors.New("not initialized")
	// ErrReleased indicates the *Matrix has already been released.
	ErrReleased = errors.New("released")
	// ErrInvalidState indicates internally inconsistent matrix metadata.
	ErrInvalidState = errors.New("invalid matrix state")

	// ErrInvalidDimension indicates a non-positive matrix dimension.
	ErrInvalidDimension = errors.New("matrix dimensions must be positive")
	// ErrDimensionMismatch indicates operand shapes are incompatible.
	ErrDimensionMismatch = errors.New("dimension mismatch")
	// ErrLengthMismatch indicates host data has the wrong element count.
	ErrLengthMismatch = errors.New("data length mismatch")
	// ErrContextMismatch indicates operands belong to different contexts.
	ErrContextMismatch = errors.New("matrices must use the same context")
	// ErrAliasedOutput indicates the output matrix aliases an input.
	ErrAliasedOutput = errors.New("out must not alias an input")

	// ErrOverflow indicates a size computation overflowed.
	ErrOverflow = errors.New("overflow")
	// ErrDeviceLimit indicates a request exceeds a device/hardware limit.
	ErrDeviceLimit = errors.New("exceeds device limits")
	// ErrKernelLimit indicates a request exceeds a compute-kernel limit.
	ErrKernelLimit = errors.New("exceeds kernel limits")
)

func wrapError(err error, format string, args ...any) error {
	if err == nil {
		return nil
	}

	//nolint:err113 // wrap
	return fmt.Errorf("mat: "+format+": %w", append(args, err)...)
}

func newError(format string, args ...any) error {
	//nolint:err113 // new
	return fmt.Errorf("mat: "+format, args...)
}

// sentinelError formats a "mat: "-prefixed message that wraps sentinel so
// callers can classify it with errors.Is.
func sentinelError(sentinel error, format string, args ...any) error {
	return &classifiedError{
		message: fmt.Sprintf(format, args...),
		cause:   sentinel,
	}
}

func sentinelWrapError(sentinel, err error, format string, args ...any) error {
	if err == nil {
		return sentinelError(sentinel, format, args...)
	}

	return &classifiedError{
		message: fmt.Sprintf(format, args...) + ": " + err.Error(),
		cause:   errors.Join(sentinel, err),
	}
}

type classifiedError struct {
	message string
	cause   error
}

func (e *classifiedError) Error() string { return "mat: " + e.message }
func (e *classifiedError) Unwrap() error { return e.cause }
