package mat

import (
	"io"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestAllFiniteValidationFailures(t *testing.T) {
	t.Parallel()

	input, _ := newMockMatrix(1, 2, []float32{1, 2})
	flag, _ := newMockMatrix(1, 1, []float32{1})

	err := validateAllFiniteAccumulate(nil, flag)
	require.ErrorIs(t, err, ErrNotInitialized)
	require.ErrorContains(t, err, "input is not initialized")

	err = validateAllFiniteAccumulate(input, nil)
	require.ErrorIs(t, err, ErrNotInitialized)
	require.ErrorContains(t, err, "flag is not initialized")
}

func TestAdamDeltaValidationFailures(t *testing.T) {
	t.Parallel()

	first, _ := newMockMatrix(1, 2, []float32{1, 2})
	second, _ := newMockMatrix(1, 1, []float32{1})
	out, _ := newMockMatrix(1, 2, []float32{0, 0})
	shareMockContext(first, second, out)

	require.ErrorIs(t, validateAdamDelta(nil, second, 1, 1, out), ErrNotInitialized)
	require.ErrorIs(t, validateAdamDelta(first, second, 1, 1, out), ErrDimensionMismatch)
}

func TestSelectFiniteValidationFailures(t *testing.T) {
	t.Parallel()

	candidate, _ := newMockMatrix(1, 2, []float32{1, 2})
	original, _ := newMockMatrix(1, 1, []float32{3})
	out, _ := newMockMatrix(1, 2, []float32{0, 0})
	shareMockContext(candidate, original, out)

	err := validateSelectFinite(nil, original, nil, out)
	require.ErrorIs(t, err, ErrNotInitialized)
	require.ErrorContains(t, err, "left is not initialized")
	require.ErrorIs(t, validateSelectFinite(candidate, original, nil, out), ErrDimensionMismatch)

	original, _ = newMockMatrix(1, 2, []float32{3, 4})
	shareMockContext(candidate, original, out)
	err = validateSelectFinite(candidate, original, nil, out)
	require.ErrorIs(t, err, ErrNotInitialized)
	require.ErrorContains(t, err, "flag is not initialized")
}

func TestAllFiniteCompatibilityIOFailures(t *testing.T) {
	t.Parallel()

	input, inputIO := newMockMatrix(1, 1, []float32{1})
	flag, flagIO := newMockMatrix(1, 1, []float32{1})

	inputIO.readErr = io.EOF
	require.ErrorIs(t, runAllFiniteCompatibility(input, flag), io.EOF)

	inputIO.readErr = nil
	flagIO.readErr = io.EOF
	require.ErrorIs(t, runAllFiniteCompatibility(input, flag), io.EOF)
}

func TestAdamMomentCompatibilityIOFailures(t *testing.T) {
	t.Parallel()

	moment, momentIO := newMockMatrix(1, 1, []float32{0})
	gradient, gradientIO := newMockMatrix(1, 1, []float32{1})
	out, _ := newMockMatrix(1, 1, []float32{0})

	momentIO.readErr = io.EOF
	require.ErrorIs(
		t,
		runAdamMomentCompatibility(moment, gradient, 0.9, out, tensorOpAdamFirst),
		io.EOF,
	)

	momentIO.readErr = nil
	gradientIO.readErr = io.EOF
	require.ErrorIs(
		t,
		runAdamMomentCompatibility(moment, gradient, 0.9, out, tensorOpAdamFirst),
		io.EOF,
	)
}

func TestAdamDeltaCompatibilityIOFailures(t *testing.T) {
	t.Parallel()

	first, firstIO := newMockMatrix(1, 1, []float32{1})
	second, secondIO := newMockMatrix(1, 1, []float32{1})
	out, _ := newMockMatrix(1, 1, []float32{0})

	firstIO.readErr = io.EOF
	require.ErrorIs(t, runAdamDeltaCompatibility(first, second, 1, 1, out), io.EOF)

	firstIO.readErr = nil
	secondIO.readErr = io.EOF
	require.ErrorIs(t, runAdamDeltaCompatibility(first, second, 1, 1, out), io.EOF)
}

func TestSelectFiniteCompatibilityIOFailures(t *testing.T) {
	t.Parallel()

	candidate, candidateIO := newMockMatrix(1, 1, []float32{1})
	original, originalIO := newMockMatrix(1, 1, []float32{2})
	flag, flagIO := newMockMatrix(1, 1, []float32{1})
	out, _ := newMockMatrix(1, 1, []float32{0})

	candidateIO.readErr = io.EOF
	require.ErrorIs(t, runSelectFiniteCompatibility(candidate, original, flag, out), io.EOF)

	candidateIO.readErr = nil
	originalIO.readErr = io.EOF
	require.ErrorIs(t, runSelectFiniteCompatibility(candidate, original, flag, out), io.EOF)

	originalIO.readErr = nil
	flagIO.readErr = io.EOF
	require.ErrorIs(t, runSelectFiniteCompatibility(candidate, original, flag, out), io.EOF)
}

func TestDropoutCompatibilityIOFailures(t *testing.T) {
	t.Parallel()

	input, inputIO := newMockMatrix(1, 1, []float32{1})
	out, outIO := newMockMatrix(1, 1, []float32{0})
	state := RandomState{Seed: 0, StreamID: 0, Counter: 0}

	inputIO.readErr = io.EOF
	require.ErrorIs(t, runDropoutCompatibility(input, 0.5, state, out), io.EOF)

	inputIO.readErr = nil
	outIO.writeErr = io.EOF
	require.ErrorIs(t, runDropoutCompatibility(input, 0.5, state, out), io.EOF)
}
