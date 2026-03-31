//go:build !cgo

package mat

import (
	"github.com/KEINOS/go-wgpu-mat/mat/internal/pipelinecache"
	"github.com/gogpu/wgpu"
)

// pipelineCache stores compute pipelines keyed by an internal operation key.
// It is internal to mat and not exposed through the public API.
type pipelineCache struct {
	inner *pipelinecache.Cache
}

func newPipelineCache(releasePipeline func(*wgpu.ComputePipeline)) *pipelineCache {
	cache := new(pipelineCache)
	cache.inner = pipelinecache.New(releasePipeline)

	return cache
}

func defaultReleaseComputePipeline(pipeline *wgpu.ComputePipeline) {
	pipelinecache.DefaultReleaseComputePipeline(pipeline)
}

func (c *pipelineCache) getOrCreate(
	key string,
	factory func() (*wgpu.ComputePipeline, error),
) (*wgpu.ComputePipeline, error) {
	if c == nil {
		return nil, newError("pipeline cache is nil")
	}

	if c.inner == nil {
		return nil, newError("pipeline cache is nil")
	}

	pipeline, err := c.inner.GetOrCreate(key, factory)
	if err != nil {
		return nil, err
	}

	return pipeline, nil
}

func (c *pipelineCache) releaseAll() {
	if c == nil || c.inner == nil {
		return
	}

	c.inner.ReleaseAll()
}

func (c *pipelineCache) size() int {
	if c == nil || c.inner == nil {
		return 0
	}

	return c.inner.Size()
}
