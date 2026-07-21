
LOCKDIR := /tmp/go-wgpu-mat-test.lockdir
WGPU_GOOS := $(shell go env GOOS)
WGPU_GOARCH := $(shell go env GOARCH)
RACE_FLAGS := -race

ifeq ($(WGPU_GOOS)/$(WGPU_GOARCH),darwin/arm64)
RACE_FLAGS += -gcflags=all=-d=checkptr=0
endif

.PHONY: test lint bench fuzz prep-test-lock

prep-test-lock:
	rm -rf $(LOCKDIR)

test: prep-test-lock
	@for mode in 0 1; do \
		echo "* Testing with CGO_ENABLED=$$mode..."; \
		CGO_ENABLED=$$mode go test $(RACE_FLAGS) -cover ./... || exit 1; \
	done

lint:
	@echo "* Running markdownlint..."
	markdownlint-cli2 **/*.md
	@echo ""
	@echo "* Running golangci-lint..."
	@for mode in 0 1; do \
		echo "* Linting with CGO_ENABLED=$$mode..."; \
		CGO_ENABLED=$$mode golangci-lint run --fix || exit 1; \
	done

bench:
	go test -run=^$$ -bench=. -benchmem ./mat/...

fuzz: prep-test-lock
	go test -parallel=1 -run=^$$ -fuzz=FuzzMatrixWriteReadRoundTrip -fuzztime=10s ./mat
	go test -parallel=1 -run=^$$ -fuzz=FuzzSoftmaxRowSums -fuzztime=10s ./mat
