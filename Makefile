LOCKDIR := /tmp/go-wgpu-mat-test.lockdir

RACE_FLAGS := -race

# CGO_ENABLED=0: Metal FFI RequestAdapter hangs on some platforms.
# Set timeout and serialize tests so integration tests neither block nor race
# through the process-global backend/driver state.
CGO0_FLAGS := -timeout=30s -parallel=1

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

.PHONY: test-metal
test-metal: clean
	@echo "* Testing Metal contracts with the race detector..."
	@GO_WGPU_MAT_GPU=1 CGO_ENABLED=1 go test -race -count=3 -parallel=1 \
		-run '^(TestP4Metal|TestSLMetal)' ./mat

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
