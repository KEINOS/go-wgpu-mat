//go:build !cgo

package pipelinecache

import (
	"io"
	"testing"

	"github.com/gogpu/wgpu"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestCacheGetOrCreateCachesByKey(t *testing.T) {
	t.Parallel()

	cache := New(nil)
	calls := 0

	factory := func() (*wgpu.ComputePipeline, error) {
		calls++

		return new(wgpu.ComputePipeline), nil
	}

	first, err := cache.GetOrCreate("matmul:f32", factory)
	require.NoError(t, err)
	require.NotNil(t, first)

	second, err := cache.GetOrCreate("matmul:f32", factory)
	require.NoError(t, err)
	require.NotNil(t, second)

	assert.Same(t, first, second)
	assert.Equal(t, 1, calls)
}

func TestCacheGetOrCreateValidation(t *testing.T) {
	t.Parallel()

	var cache *Cache

	_, err := cache.GetOrCreate("matmul:f32", func() (*wgpu.ComputePipeline, error) {
		return new(wgpu.ComputePipeline), nil
	})
	require.ErrorContains(t, err, "pipeline cache is nil")

	cache = New(nil)

	_, err = cache.GetOrCreate("", func() (*wgpu.ComputePipeline, error) {
		return new(wgpu.ComputePipeline), nil
	})
	require.ErrorContains(t, err, "pipeline key is empty")

	_, err = cache.GetOrCreate("matmul:f32", nil)
	require.ErrorContains(t, err, "pipeline factory is nil")

	_, err = cache.GetOrCreate("matmul:f32", func() (*wgpu.ComputePipeline, error) {
		return nil, io.EOF
	})
	require.ErrorContains(t, err, "failed to create pipeline")

	_, err = cache.GetOrCreate("matmul:f32", nilPipelineFactory)
	require.ErrorContains(t, err, "pipeline factory returned nil pipeline")
}

//nolint:nilnil // Intentional for validation coverage: factory can return nil,nil.
func nilPipelineFactory() (*wgpu.ComputePipeline, error) {
	return nil, nil
}

func TestCacheReleaseAllReleasesAndClears(t *testing.T) {
	t.Parallel()

	released := 0
	cache := New(func(*wgpu.ComputePipeline) {
		released++
	})

	_, err := cache.GetOrCreate("matmul:f32", func() (*wgpu.ComputePipeline, error) {
		return new(wgpu.ComputePipeline), nil
	})
	require.NoError(t, err)

	_, err = cache.GetOrCreate("add:f32", func() (*wgpu.ComputePipeline, error) {
		return new(wgpu.ComputePipeline), nil
	})
	require.NoError(t, err)

	assert.Equal(t, 2, cache.Size())

	cache.ReleaseAll()

	assert.Equal(t, 2, released)
	assert.Equal(t, 0, cache.Size())

	cache.ReleaseAll()
	assert.Equal(t, 2, released)
}

func TestDefaultReleaseComputePipelineNilSafe(t *testing.T) {
	t.Parallel()

	require.NotPanics(t, func() {
		DefaultReleaseComputePipeline(nil)
	})

	require.NotPanics(t, func() {
		DefaultReleaseComputePipeline(new(wgpu.ComputePipeline))
	})
}
