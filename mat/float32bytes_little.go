//go:build 386 || amd64 || arm || arm64 || loong64 || mips64le || mipsle || ppc64le || riscv64 || wasm

package mat

import "unsafe"

func encodeFloat32Slice(values []float32) []byte {
	return float32ByteView(values)
}

func newFloat32ReadBuffer(elementCount int) ([]byte, []float32) {
	result := make([]float32, elementCount)

	return float32ByteView(result), result
}

func decodeFloat32ReadBuffer(_ []byte, _ []float32) {}

// float32ByteView avoids a second allocation while the owning float32 slice
// remains alive. Callers must not retain the returned view independently.
//
//nolint:gosec // The view length exactly matches the live, owning float32 slice.
func float32ByteView(values []float32) []byte {
	return unsafe.Slice(
		(*byte)(unsafe.Pointer(unsafe.SliceData(values))),
		len(values)*bytesPerFloat32Int,
	)
}
