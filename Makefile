.PHONY: test lint

test:
	CGO_ENABLED=0 go test -cover -race ./...

lint:
	CGO_ENABLED=0 golangci-lint run --fix
