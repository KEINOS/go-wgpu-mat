//go:build !cgo

package mat

import "fmt"

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
