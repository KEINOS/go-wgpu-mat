
LOCKDIR := /tmp/go-wgpu-mat-test.lockdir
WGPU_GOOS := $(shell go env GOOS)
WGPU_GOARCH := $(shell go env GOARCH)
RACE_FLAGS := -race
RACE_ENV :=

ifeq ($(WGPU_GOOS)/$(WGPU_GOARCH),darwin/arm64)
# On macOS arm64 the race detector triggers checkptr panics in Metal FFI.
# Disable checkptr, limit parallelism, and exclude Example() to avoid crashes.
RACE_FLAGS += -gcflags=all=-d=checkptr=0 -parallel=1 -run='^Test'
RACE_ENV := GOMAXPROCS=1 GO_WGPU_MAT_SKIP_GPU_TESTS=1
endif

.PHONY: clean
clean:
	rm -rf $(LOCKDIR)

.PHONY: test
test: clean
	@echo "* Testing with CGO_ENABLED=0..."
	@GO_WGPU_MAT_SKIP_GPU_TESTS=1 CGO_ENABLED=0 go test -run='^Test' -cover ./...
	@echo "* Testing with CGO_ENABLED=1 and the race detector..."
	@$(RACE_ENV) CGO_ENABLED=1 go test $(RACE_FLAGS) -cover ./...

.PHONY: lint
lint:
	@echo "* Running markdownlint..."
	markdownlint-cli2 **/*.md
	@echo ""
	@echo "* Running golangci-lint..."
	golangci-lint run --fix

.PHONY: bench bench-isolated
bench:
	go test -run=^$$ -bench=. -benchmem ./mat/...

bench-isolated:
	./scripts/bench-isolated.sh

.PHONY: fuzz
fuzz: clean
	go test -parallel=1 -run=^$$ -fuzz=FuzzMatrixWriteReadRoundTrip -fuzztime=10s ./mat
	go test -parallel=1 -run=^$$ -fuzz=FuzzSoftmaxRowSums -fuzztime=10s ./mat
