//go:build mips || mips64 || ppc64 || s390x

package mat

import (
	"encoding/binary"
	"math"
)

func encodeFloat32Slice(values []float32) []byte {
	raw := make([]byte, len(values)*bytesPerFloat32Int)
	for index, value := range values {
		binary.LittleEndian.PutUint32(
			raw[index*bytesPerFloat32Int:],
			math.Float32bits(value),
		)
	}

	return raw
}

func newFloat32ReadBuffer(elementCount int) ([]byte, []float32) {
	return make([]byte, elementCount*bytesPerFloat32Int), make([]float32, elementCount)
}

func decodeFloat32ReadBuffer(raw []byte, result []float32) {
	for index := range result {
		result[index] = math.Float32frombits(
			binary.LittleEndian.Uint32(raw[index*bytesPerFloat32Int:]),
		)
	}
}
