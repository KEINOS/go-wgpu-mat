//go:build !cgo

package backends

import "testing"

func TestUseGPU(t *testing.T) {
	t.Parallel()

	UseGPU()
}

func TestUseCPU(t *testing.T) {
	t.Parallel()

	UseCPU()
}
