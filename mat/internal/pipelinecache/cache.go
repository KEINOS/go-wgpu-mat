//go:build !cgo

package pipelinecache

import (
	"fmt"
	"sync"

	"github.com/gogpu/wgpu"
)

// Cache stores compute pipelines keyed by an operation identifier.
type Cache struct {
	mu              sync.RWMutex
	pipelines       map[string]*wgpu.ComputePipeline
	releasePipeline func(*wgpu.ComputePipeline)
}

// New initializes a pipeline cache.
func New(releasePipeline func(*wgpu.ComputePipeline)) *Cache {
	if releasePipeline == nil {
		releasePipeline = DefaultReleaseComputePipeline
	}

	cache := new(Cache)
	cache.pipelines = make(map[string]*wgpu.ComputePipeline)
	cache.releasePipeline = releasePipeline

	return cache
}

// DefaultReleaseComputePipeline releases a compute pipeline safely.
func DefaultReleaseComputePipeline(pipeline *wgpu.ComputePipeline) {
	if pipeline == nil {
		return
	}

	defer func() {
		_ = recover()
	}()

	pipeline.Release()
}

// GetOrCreate gets a cached pipeline by key or creates and caches it.
func (c *Cache) GetOrCreate(
	key string,
	factory func() (*wgpu.ComputePipeline, error),
) (*wgpu.ComputePipeline, error) {
	if c == nil {
		return nil, fmt.Errorf("pipeline cache is nil")
	}

	if key == "" {
		return nil, fmt.Errorf("pipeline key is empty")
	}

	if factory == nil {
		return nil, fmt.Errorf("pipeline factory is nil")
	}

	c.mu.Lock()
	defer c.mu.Unlock()

	if pipeline, ok := c.pipelines[key]; ok {
		return pipeline, nil
	}

	pipeline, err := factory()
	if err != nil {
		return nil, fmt.Errorf("failed to create pipeline %q: %w", key, err)
	}

	if pipeline == nil {
		return nil, fmt.Errorf("pipeline factory returned nil pipeline")
	}

	c.pipelines[key] = pipeline

	return pipeline, nil
}

// ReleaseAll releases all cached pipelines and clears the cache.
func (c *Cache) ReleaseAll() {
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

// Size returns the number of cached pipelines.
func (c *Cache) Size() int {
	if c == nil {
		return 0
	}

	c.mu.RLock()
	defer c.mu.RUnlock()

	return len(c.pipelines)
}