package mat

import (
	"sort"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestRunWorkRangesUsesSerialPath(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name        string
		total       int
		workPerItem int
		minWork     int
		maxWorkers  int
		wantCalls   int
	}{
		{name: "empty", total: 0, workPerItem: 10, minWork: 10, maxWorkers: 4, wantCalls: 0},
		{name: "one item", total: 1, workPerItem: 10, minWork: 10, maxWorkers: 4, wantCalls: 1},
		{name: "one worker", total: 8, workPerItem: 10, minWork: 10, maxWorkers: 1, wantCalls: 1},
		{name: "below threshold", total: 8, workPerItem: 10, minWork: 100, maxWorkers: 4, wantCalls: 1},
		{name: "zero work estimate", total: 8, workPerItem: 0, minWork: 1, maxWorkers: 4, wantCalls: 1},
	}

	for _, testCase := range tests {
		t.Run(testCase.name, func(t *testing.T) {
			t.Parallel()

			pool := newHostWorkerPool(testCase.maxWorkers)
			defer pool.close()

			calls := 0

			pool.runWorkRangesWithConfig(
				testCase.total,
				testCase.workPerItem,
				testCase.minWork,
				func(start, end int) {
					calls++

					assert.Equal(t, 0, start)
					assert.Equal(t, testCase.total, end)
				},
			)

			assert.Equal(t, testCase.wantCalls, calls)
		})
	}
}

func TestRunWorkRangesPartitionsWorkAcrossBoundedWorkers(t *testing.T) {
	t.Parallel()

	type indexRange struct {
		start int
		end   int
	}

	var mutex sync.Mutex

	ranges := make([]indexRange, 0, 4)

	pool := newHostWorkerPool(4)
	defer pool.close()

	pool.runWorkRangesWithConfig(17, 16, 32, func(start, end int) {
		mutex.Lock()

		ranges = append(ranges, indexRange{start: start, end: end})
		mutex.Unlock()
	})

	sort.Slice(ranges, func(first, second int) bool {
		return ranges[first].start < ranges[second].start
	})

	require.Len(t, ranges, 4)
	assert.Equal(t, 0, ranges[0].start)
	assert.Equal(t, 17, ranges[len(ranges)-1].end)

	for index := 1; index < len(ranges); index++ {
		assert.Equal(t, ranges[index-1].end, ranges[index].start)
	}
}

func TestRunWorkRangesRunsWorkersConcurrently(t *testing.T) {
	t.Parallel()

	const workers = 4

	started := make(chan struct{}, workers)
	release := make(chan struct{})
	done := make(chan struct{})

	var (
		active atomic.Int32
		peak   atomic.Int32
	)

	pool := newHostWorkerPool(workers)
	defer pool.close()

	go func() {
		pool.runWorkRangesWithConfig(workers, 1, 1, func(_, _ int) {
			current := active.Add(1)

			for {
				observed := peak.Load()
				if current <= observed || peak.CompareAndSwap(observed, current) {
					break
				}
			}

			started <- struct{}{}

			<-release
			active.Add(-1)
		})
		close(done)
	}()

	for range workers {
		<-started
	}

	close(release)
	<-done

	assert.Equal(t, int32(workers), peak.Load())
}

func TestContextReusesAndReleasesHostWorkerPool(t *testing.T) {
	t.Parallel()

	ctx := new(Context)
	assert.Nil(t, ctx.hostPool)

	ctx.runHostWorkRanges(4, hostParallelMinWork, func(_, _ int) {})
	first := ctx.hostPool
	require.NotNil(t, first)

	ctx.runHostWorkRanges(4, hostParallelMinWork, func(_, _ int) {})
	assert.Same(t, first, ctx.hostPool)

	ctx.Release()
	assert.Nil(t, ctx.hostPool)
	require.NotPanics(t, first.close)
}

func TestHostWorkerPoolKeepsConcurrentJobsIndependent(t *testing.T) {
	t.Parallel()

	const (
		jobs  = 8
		items = 32
	)

	pool := newHostWorkerPool(4)
	defer pool.close()

	results := make([][]int, jobs)

	var callers sync.WaitGroup
	callers.Add(jobs)

	for job := range jobs {
		results[job] = make([]int, items)

		go func() {
			defer callers.Done()

			pool.runWorkRangesWithConfig(items, 1, 1, func(start, end int) {
				for index := start; index < end; index++ {
					results[job][index] = job + 1
				}
			})
		}()
	}

	callers.Wait()

	for job, result := range results {
		assert.Equal(t, makeFilledInts(items, job+1), result)
	}
}

func makeFilledInts(length, value int) []int {
	result := make([]int, length)
	for index := range result {
		result[index] = value
	}

	return result
}
