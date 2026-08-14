package mat

import (
	"runtime"
	"sync"
)

const (
	hostParallelMinWork = 1 << 18
	softmaxMinWork      = 1 << 12
)

type workRangeFunc func(start, end int)
type workRangeRunner func(total, workPerItem int, operation workRangeFunc)

type workRangeOperation interface {
	runWorkRange(start, end int)
}

type rowOperationFunc func(inputData, outputData []float32, offset, cols int)

func (operation workRangeFunc) runWorkRange(start, end int) {
	operation(start, end)
}

type hostWork struct {
	start     int
	end       int
	operation workRangeOperation
	done      *sync.WaitGroup
}

type matMulHostOperation struct {
	leftData  []float32
	rightData []float32
	result    []float32
	sharedDim int
	rightCols int
}

type rowHostOperation struct {
	apply      rowOperationFunc
	inputData  []float32
	outputData []float32
	cols       int
}

func (operation *rowHostOperation) runWorkRange(start, end int) {
	for row := start; row < end; row++ {
		operation.apply(
			operation.inputData,
			operation.outputData,
			row*operation.cols,
			operation.cols,
		)
	}
}

func (operation *matMulHostOperation) runWorkRange(start, end int) {
	multiplyMatMulRows(
		operation.leftData,
		operation.rightData,
		operation.result,
		operation.sharedDim,
		operation.rightCols,
		start,
		end,
	)
}

type hostWorkerPool struct {
	maxWorkers int
	tasks      chan hostWork
	workers    sync.WaitGroup
	jobs       chan *sync.WaitGroup
	matMulJobs chan *matMulHostOperation
	rowJobs    chan *rowHostOperation
	closeOnce  sync.Once
}

func newHostWorkerPool(maxWorkers int) *hostWorkerPool {
	pool := new(hostWorkerPool)
	pool.maxWorkers = max(1, maxWorkers)
	pool.tasks = make(chan hostWork)

	pool.jobs = make(chan *sync.WaitGroup, pool.maxWorkers)
	pool.jobs <- new(sync.WaitGroup)

	pool.matMulJobs = make(chan *matMulHostOperation, pool.maxWorkers)
	for range pool.maxWorkers {
		pool.matMulJobs <- new(matMulHostOperation)
	}

	pool.rowJobs = make(chan *rowHostOperation, pool.maxWorkers)

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
	p.runOperationWithConfig(total, workPerItem, minWork, operation)
}

func (p *hostWorkerPool) runOperationWithConfig(
	total, workPerItem, minWork int,
	operation workRangeOperation,
) {
	if total <= 0 {
		return
	}

	workerCount := rangeWorkerCount(total, workPerItem, minWork, p.maxWorkers)
	if workerCount == 1 {
		operation.runWorkRange(0, total)

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

func (p *hostWorkerPool) runMatMul(
	leftData, rightData, result []float32,
	rows, sharedDim, rightCols int,
) {
	operation := <-p.matMulJobs
	operation.leftData = leftData
	operation.rightData = rightData
	operation.result = result
	operation.sharedDim = sharedDim
	operation.rightCols = rightCols

	p.runOperationWithConfig(rows, sharedDim*rightCols, hostParallelMinWork, operation)

	operation.leftData = nil
	operation.rightData = nil

	operation.result = nil
	p.matMulJobs <- operation
}

func (p *hostWorkerPool) runRows(
	apply rowOperationFunc,
	inputData, outputData []float32,
	rows, cols, minWork int,
) {
	operation := p.acquireRowJob()
	operation.apply = apply
	operation.inputData = inputData
	operation.outputData = outputData
	operation.cols = cols

	p.runOperationWithConfig(rows, cols, minWork, operation)

	operation.apply = nil
	operation.inputData = nil
	operation.outputData = nil
	p.releaseRowJob(operation)
}

func (p *hostWorkerPool) acquireRowJob() *rowHostOperation {
	select {
	case job := <-p.rowJobs:
		return job
	default:
		return new(rowHostOperation)
	}
}

func (p *hostWorkerPool) releaseRowJob(job *rowHostOperation) {
	select {
	case p.rowJobs <- job:
	default:
	}
}

func (c *Context) runHostRows(
	apply rowOperationFunc,
	inputData, outputData []float32,
	rows, cols, minWork int,
) {
	if rangeWorkerCount(
		rows,
		cols,
		minWork,
		runtime.GOMAXPROCS(0),
	) == 1 {
		operation := rowHostOperation{
			apply:      apply,
			inputData:  inputData,
			outputData: outputData,
			cols:       cols,
		}
		operation.runWorkRange(0, rows)

		return
	}

	c.hostWorkerPool().runRows(
		apply,
		inputData,
		outputData,
		rows,
		cols,
		minWork,
	)
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
		work.operation.runWorkRange(work.start, work.end)
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
