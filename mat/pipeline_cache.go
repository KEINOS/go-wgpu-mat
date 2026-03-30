//go:build !cgo

package mat

import (
	"sync"

	"github.com/gogpu/wgpu"
)

// pipelineCache stores compute pipelines keyed by an internal operation key.
// It is internal to mat and not exposed through the public API.
type pipelineCache struct {
	mu              sync.RWMutex
	pipelines       map[string]*wgpu.ComputePipeline
	releasePipeline func(*wgpu.ComputePipeline)
}

func newPipelineCache(releasePipeline func(*wgpu.ComputePipeline)) *pipelineCache {
	if releasePipeline == nil {
		releasePipeline = defaultReleaseComputePipeline
	}

	cache := new(pipelineCache)
	cache.pipelines = make(map[string]*wgpu.ComputePipeline)
	cache.releasePipeline = releasePipeline

	return cache
}

func defaultReleaseComputePipeline(pipeline *wgpu.ComputePipeline) {
	if pipeline == nil {
		return
	}

	// Keep context teardown robust even if a stale/invalid handle exists.
	defer func() {
		_ = recover()
	}()

	pipeline.Release()
}

func (c *pipelineCache) getOrCreate(
	key string,
	factory func() (*wgpu.ComputePipeline, error),
) (*wgpu.ComputePipeline, error) {
	if c == nil {
		return nil, newError("pipeline cache is nil")
	}

	if key == "" {
		return nil, newError("pipeline key is empty")
	}

	if factory == nil {
		return nil, newError("pipeline factory is nil")
	}

	c.mu.Lock()
	defer c.mu.Unlock()

	if pipeline, ok := c.pipelines[key]; ok {
		return pipeline, nil
	}

	pipeline, err := factory()
	if err != nil {
		return nil, wrapError(err, "failed to create pipeline %q", key)
	}

	c.pipelines[key] = pipeline

	return pipeline, nil
}

func (c *pipelineCache) releaseAll() {
	if c == nil {
		return
	}

	c.mu.Lock()
	pipelines := c.pipelines
	c.pipelines = make(map[string]*wgpu.ComputePipeline)
	releasePipeline := c.releasePipeline
	c.mu.Unlock()

	for _, pipeline := range pipelines {
		releasePipeline(pipeline)
	}
}

func (c *pipelineCache) size() int {
	if c == nil {
		return 0
	}

	c.mu.RLock()
	defer c.mu.RUnlock()

	return len(c.pipelines)
}
