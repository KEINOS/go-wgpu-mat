//go:build cgo

package backends

// Fail fast when cgo is enabled. This package supports CGO_ENABLED=0 only.
var _ = CGO_ENABLED_must_be_0_for_github_com_KEINOS_go_wgpu_mat