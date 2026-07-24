#!/usr/bin/env bash

set -euo pipefail
IFS=$'\n\t'

benchmark_pattern=${BENCH_PATTERN:-.}
sample_count=${BENCH_SAMPLES:-10}
benchmark_time=${BENCH_TIME:-20x}
cgo_mode=${CGO_ENABLED:-0}
benchmark_dir=${BENCH_OUT:-}

if [[ ! "$sample_count" =~ ^[1-9][0-9]*$ ]]; then
	printf 'BENCH_SAMPLES must be a positive integer\n' >&2
	exit 2
fi

if [[ -z "$benchmark_dir" ]]; then
	benchmark_dir=$(mktemp -d "${TMPDIR:-/tmp}/go-wgpu-mat-bench.XXXXXX")
else
	mkdir -p "$benchmark_dir"
fi

combined_output="$benchmark_dir/combined.txt"
: >"$combined_output"

failed_samples=0

for ((sample = 1; sample <= sample_count; sample++)); do
	sample_output="$benchmark_dir/sample-${sample}.txt"

	if GOMAXPROCS=1 CGO_ENABLED="$cgo_mode" go test \
		-run='^$' \
		-bench="$benchmark_pattern" \
		-benchmem \
		-count=1 \
		-benchtime="$benchmark_time" \
		./mat >"$sample_output" 2>&1; then
		awk '/^Benchmark/ { print }' "$sample_output" >>"$combined_output"
	else
		failed_samples=$((failed_samples + 1))
		printf 'sample %d failed; see %s\n' "$sample" "$sample_output" >&2
	fi
done

printf 'benchmark output: %s\n' "$combined_output"
printf 'successful samples: %d/%d\n' "$((sample_count - failed_samples))" "$sample_count"

if ((failed_samples > 0)); then
	exit 1
fi
