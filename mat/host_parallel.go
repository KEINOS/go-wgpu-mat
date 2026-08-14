package mat

import (
	"sync"
)

const hostParallelMinWork = 1 << 18

type workRangeFunc func(start, end int)
type workRangeRunner func(total, workPerItem int, operation workRangeFunc)

type hostWork struct {
	start     int
	end       int
	operation workRangeFunc
	done      *sync.WaitGroup
}

type hostWorkerPool struct {
	maxWorkers int
	tasks      chan hostWork
	workers    sync.WaitGroup
	jobs       chan *sync.WaitGroup
	closeOnce  sync.Once
}

func newHostWorkerPool(maxWorkers int) *hostWorkerPool {
	pool := new(hostWorkerPool)
	pool.maxWorkers = max(1, maxWorkers)
	pool.tasks = make(chan hostWork)

	pool.jobs = make(chan *sync.WaitGroup, pool.maxWorkers)
	pool.jobs <- new(sync.WaitGroup)

	pool.workers.Add(pool.maxWorkers)

	for range pool.maxWorkers {
		go pool.worker()
	}

	return pool
}

func (p *hostWorkerPool) runWorkRanges(
	total, workPerItem int,
	operation workRangeFunc,
) {
	p.runWorkRangesWithConfig(total, workPerItem, hostParallelMinWork, operation)
}

func (p *hostWorkerPool) runWorkRangesWithConfig(
	total, workPerItem, minWork int,
	operation workRangeFunc,
) {
	if total <= 0 {
		return
	}

	workerCount := rangeWorkerCount(total, workPerItem, minWork, p.maxWorkers)
	if workerCount == 1 {
		operation(0, total)

		return
	}

	done := p.acquireJob()
	done.Add(workerCount)

	start := 0

	for worker := range workerCount {
		length := total / workerCount
		if worker < total%workerCount {
			length++
		}

		end := start + length
		p.tasks <- hostWork{
			start:     start,
			end:       end,
			operation: operation,
			done:      done,
		}

		start = end
	}

	done.Wait()
	p.releaseJob(done)
}

func (p *hostWorkerPool) acquireJob() *sync.WaitGroup {
	select {
	case job := <-p.jobs:
		return job
	default:
		return new(sync.WaitGroup)
	}
}

func (p *hostWorkerPool) releaseJob(job *sync.WaitGroup) {
	select {
	case p.jobs <- job:
	default:
	}
}

func (p *hostWorkerPool) worker() {
	defer p.workers.Done()

	for work := range p.tasks {
		work.operation(work.start, work.end)
		work.done.Done()
	}
}

func (p *hostWorkerPool) close() {
	if p == nil {
		return
	}

	p.closeOnce.Do(func() {
		close(p.tasks)
		p.workers.Wait()
	})
}

func rangeWorkerCount(total, workPerItem, minWork, maxWorkers int) int {
	if total <= 1 || workPerItem <= 0 || minWork <= 0 || maxWorkers <= 1 {
		return 1
	}

	itemsPerWorker := 1 + (minWork-1)/workPerItem
	workerCount := min(maxWorkers, total/itemsPerWorker)

	return max(1, workerCount)
}
