
LOCKDIR := /tmp/go-wgpu-mat-test.lockdir

.PHONY: test lint bench fuzz prep-test-lock

prep-test-lock:
	rm -rf $(LOCKDIR)

test: prep-test-lock
	CGO_ENABLED=0 go test -cover ./...

lint:
	@echo "* Running markdownlint..."
	markdownlint-cli2 **/*.md
	@echo ""
	@echo "* Running golangci-lint..."
	CGO_ENABLED=0 golangci-lint run --fix

bench:
	CGO_ENABLED=0 go test -run=^$$ -bench=. -benchmem ./mat/...

fuzz: prep-test-lock
	CGO_ENABLED=0 go test -parallel=1 -run=^$$ -fuzz=FuzzMatrixWriteReadRoundTrip -fuzztime=10s ./mat
	CGO_ENABLED=0 go test -parallel=1 -run=^$$ -fuzz=FuzzSoftmaxRowSums -fuzztime=10s ./mat
