
LOCKDIR := /tmp/go-wgpu-mat-test.lockdir
WGPU_GOOS := $(shell go env GOOS)
WGPU_GOARCH := $(shell go env GOARCH)
RACE_FLAGS := -race
RACE_ENV :=

ifeq ($(WGPU_GOOS)/$(WGPU_GOARCH),darwin/arm64)
RACE_FLAGS += -gcflags=all=-d=checkptr=0 -parallel=1 -run=^Test
RACE_ENV := GOMAXPROCS=1 GO_WGPU_MAT_SKIP_GPU_TESTS=1
endif

.PHONY: test lint bench fuzz prep-test-lock

prep-test-lock:
	rm -rf $(LOCKDIR)

test: prep-test-lock
	@echo "* Testing with CGO_ENABLED=0..."
	@CGO_ENABLED=0 go test -cover ./...
	@echo "* Testing with CGO_ENABLED=1 and the race detector..."
	@$(RACE_ENV) CGO_ENABLED=1 go test $(RACE_FLAGS) -cover ./...

lint:
	@echo "* Running markdownlint..."
	markdownlint-cli2 **/*.md
	@echo ""
	@echo "* Running golangci-lint..."
	golangci-lint run --fix

bench:
	go test -run=^$$ -bench=. -benchmem ./mat/...

fuzz: prep-test-lock
	go test -parallel=1 -run=^$$ -fuzz=FuzzMatrixWriteReadRoundTrip -fuzztime=10s ./mat
	go test -parallel=1 -run=^$$ -fuzz=FuzzSoftmaxRowSums -fuzztime=10s ./mat
