LOCKDIR := /tmp/go-wgpu-mat-test.lockdir
WGPU_GOOS := $(shell go env GOOS)
WGPU_GOARCH := $(shell go env GOARCH)

# On darwin/arm64 the race detector triggers checkptr panics in Metal FFI.
# Disable checkptr for local macOS testing; CI (linux) is unaffected.
ifeq ($(WGPU_GOOS)/$(WGPU_GOARCH),darwin/arm64)
RACE_FLAGS := -gcflags=all=-d=checkptr=0 -parallel=1
else
RACE_FLAGS := -race
endif

# CGO_ENABLED=0: Metal FFI RequestAdapter hangs on some platforms.
# Set timeout so integration tests are killed instead of blocking.
CGO0_FLAGS := -timeout=30s

.PHONY: clean
clean:
	rm -rf $(LOCKDIR)
	go clean -testcache

.PHONY: test
test: clean
	@echo "* Testing with CGO_ENABLED=0..."
	@CGO_ENABLED=0 go test $(CGO0_FLAGS) -cover ./...
	@echo "* Testing with CGO_ENABLED=1..."
	@CGO_ENABLED=1 go test $(RACE_FLAGS) -cover ./...

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