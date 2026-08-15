package mat_test

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/KEINOS/go-wgpu-mat/mat"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNewEngineAppliesFunctionalOptionsInOrder(t *testing.T) {
	serializeGPUTest(t)

	appCtx, cancel := context.WithCancel(context.Background())
	defer cancel()

	engine, err := mat.NewEngine(
		mat.WithMode(mat.UseGPU),
		mat.WithMode(mat.UseCPU),
		mat.WithContext(appCtx),
		mat.WithChunkRows(7),
	)
	require.NoError(t, err)
	t.Cleanup(func() { require.NoError(t, engine.Close()) })

	assert.Equal(t, mat.UseCPU, engine.Mode())
	assert.False(t, engine.Closed())
}

func TestEngineWithContextClosesWhenApplicationContextEnds(t *testing.T) {
	serializeGPUTest(t)

	appCtx, cancel := context.WithCancel(context.Background())
	engine, err := mat.NewEngine(
		mat.WithMode(mat.UseCPU),
		mat.WithContext(appCtx),
	)
	require.NoError(t, err)
	t.Cleanup(func() { require.NoError(t, engine.Close()) })

	cancel()

	require.Eventually(t, engine.Closed, time.Second, time.Millisecond)
	require.NoError(t, engine.Close(), "Close must remain idempotent after automatic shutdown")
}

func TestEngineCloseIsConcurrentAndRejectsNewMatrices(t *testing.T) {
	serializeGPUTest(t)

	engine, err := mat.NewEngine(mat.WithMode(mat.UseCPU))
	require.NoError(t, err)

	const callers = 16

	start := make(chan struct{})
	errorsByCaller := make([]error, callers)
	var wait sync.WaitGroup
	wait.Add(callers)

	for caller := range callers {
		go func() {
			defer wait.Done()
			<-start
			errorsByCaller[caller] = engine.Close()
		}()
	}

	close(start)
	wait.Wait()

	for _, closeErr := range errorsByCaller {
		assert.NoError(t, closeErr)
	}

	assert.True(t, engine.Closed())

	matrix, err := engine.NewMatrix(1, 1)
	assert.Nil(t, matrix)
	require.ErrorIs(t, err, mat.ErrEngineClosed)
}
