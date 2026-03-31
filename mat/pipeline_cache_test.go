//go:build !cgo

package mat

import (
	"io"
	"testing"

	"github.com/gogpu/wgpu"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNewContextInitializesPipelineCache(t *testing.T) {
	t.Parallel()

	ctx, err := newContext(newTestContextDeps(), UseGPU)
	require.NoError(t, err)
	require.NotNil(t, ctx)
	assert.NotNil(t, ctx.pipes)
}

func TestPipelineCacheGetOrCreateCachesResult(t *testing.T) {
	t.Parallel()

	cache := newPipelineCache(nil)

	factoryCalls := 0
	factory := func() (*wgpu.ComputePipeline, error) {
		factoryCalls++

		return new(wgpu.ComputePipeline), nil
	}

	first, err := cache.getOrCreate("matmul:f32", factory)
	require.NoError(t, err)
	require.NotNil(t, first)

	second, err := cache.getOrCreate("matmul:f32", factory)
	require.NoError(t, err)
	require.NotNil(t, second)

	assert.Same(t, first, second)
	assert.Equal(t, 1, factoryCalls)
}

func TestPipelineCacheGetOrCreateValidation(t *testing.T) {
	t.Parallel()

	var cache *pipelineCache

	_, err := cache.getOrCreate("matmul:f32", func() (*wgpu.ComputePipeline, error) {
		return new(wgpu.ComputePipeline), nil
	})
	require.Error(t, err)
	require.ErrorContains(t, err, "pipeline cache is nil")

	cache = newPipelineCache(nil)

	_, err = cache.getOrCreate("", func() (*wgpu.ComputePipeline, error) {
		return new(wgpu.ComputePipeline), nil
	})
	require.Error(t, err)
	require.ErrorContains(t, err, "pipeline key is empty")

	_, err = cache.getOrCreate("matmul:f32", nil)
	require.Error(t, err)
	require.ErrorContains(t, err, "pipeline factory is nil")

	_, err = cache.getOrCreate("matmul:f32", func() (*wgpu.ComputePipeline, error) {
		return nil, io.EOF
	})
	require.Error(t, err)
	require.ErrorContains(t, err, "failed to create pipeline")
}

func TestPipelineCacheReleaseAllReleasesAndClears(t *testing.T) {
	t.Parallel()

	released := 0
	cache := newPipelineCache(func(*wgpu.ComputePipeline) {
		released++
	})

	_, err := cache.getOrCreate("matmul:f32", func() (*wgpu.ComputePipeline, error) {
		return new(wgpu.ComputePipeline), nil
	})
	require.NoError(t, err)

	_, err = cache.getOrCreate("add:f32", func() (*wgpu.ComputePipeline, error) {
		return new(wgpu.ComputePipeline), nil
	})
	require.NoError(t, err)

	assert.Equal(t, 2, cache.size())

	cache.releaseAll()

	assert.Equal(t, 2, released)
	assert.Equal(t, 0, cache.size())

	cache.releaseAll()
	assert.Equal(t, 2, released)
}

func TestContextGetOrCreatePipeline(t *testing.T) {
	t.Parallel()

	factoryCalls := 0
	ctx := new(Context)
	ctx.pipes = newPipelineCache(nil)

	factory := func() (*wgpu.ComputePipeline, error) {
		factoryCalls++

		return new(wgpu.ComputePipeline), nil
	}

	first, err := ctx.getOrCreatePipeline("matmul:f32", factory)
	require.NoError(t, err)
	require.NotNil(t, first)

	second, err := ctx.getOrCreatePipeline("matmul:f32", factory)
	require.NoError(t, err)
	require.NotNil(t, second)

	third, err := ctx.getOrCreatePipeline("add:f32", factory)
	require.NoError(t, err)
	require.NotNil(t, third)

	assert.Same(t, first, second)
	assert.NotSame(t, second, third)
	assert.Equal(t, 2, factoryCalls)
}

func TestContextGetOrCreatePipelineValidation(t *testing.T) {
	t.Parallel()

	var ctx *Context

	_, err := ctx.getOrCreatePipeline("matmul:f32", func() (*wgpu.ComputePipeline, error) {
		return new(wgpu.ComputePipeline), nil
	})
	require.Error(t, err)
	require.ErrorContains(t, err, "context is nil")

	ctx = new(Context)
	ctx.pipes = new(pipelineCache)
	_, err = ctx.getOrCreatePipeline("matmul:f32", func() (*wgpu.ComputePipeline, error) {
		return new(wgpu.ComputePipeline), nil
	})
	require.Error(t, err)
	require.ErrorContains(t, err, "pipeline cache is nil")
}

func TestContextGetOrCreatePipelineLazyInit(t *testing.T) {
	t.Parallel()

	ctx := new(Context)

	pipeline, err := ctx.getOrCreatePipeline("matmul:f32", func() (*wgpu.ComputePipeline, error) {
		return new(wgpu.ComputePipeline), nil
	})
	require.NoError(t, err)
	require.NotNil(t, pipeline)
	require.NotNil(t, ctx.pipes)
}

func TestContextReleaseReleasesPipelineCache(t *testing.T) {
	t.Parallel()

	released := 0
	ctx := new(Context)
	ctx.pipes = newPipelineCache(func(*wgpu.ComputePipeline) {
		released++
	})

	_, err := ctx.getOrCreatePipeline("matmul:f32", func() (*wgpu.ComputePipeline, error) {
		return new(wgpu.ComputePipeline), nil
	})
	require.NoError(t, err)

	ctx.Release()

	assert.Equal(t, 1, released)
	assert.Nil(t, ctx.pipes)
}

func TestPipelineCacheNilSafeHelpers(t *testing.T) {
	t.Parallel()

	var cache *pipelineCache

	require.NotPanics(t, func() { cache.releaseAll() })
	assert.Equal(t, 0, cache.size())

	require.NotPanics(t, func() { defaultReleaseComputePipeline(nil) })
	require.NotPanics(t, func() {
		defaultReleaseComputePipeline(new(wgpu.ComputePipeline))
	})
}
