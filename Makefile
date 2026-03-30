.PHONY: test lint

test:
	CGO_ENABLED=0 go test -cover -race ./...

lint:
	@echo "* Running markdownlint..."
	markdownlint-cli2 **/*.md
	@echo ""
	@echo "* Running golangci-lint..."
	CGO_ENABLED=0 golangci-lint run --fix
