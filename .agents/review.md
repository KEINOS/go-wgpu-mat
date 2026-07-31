# Codex Review

## Scope

- Reviewed range: `origin/main..70299ca` (`7502d8e`, `ba3ea83`, `70299ca`).
- Production change: `gogpu/wgpu` v0.30.22 to v0.30.29 and its transitive dependency updates.
- Test change: `mat/sl_contract_test.go` submission-lifetime regression coverage.
- Review date: 2026-07-31.

## Open issues

### RV-001 [P1] Live status describes the pre-close-out commit

The committed file at `70299ca:.agents/status.md:8-24` says HEAD is `ba3ea83`, the branch is two commits ahead, the tree still contains pending index changes, Kimi is the active executor, and `SL-009` is active. The actual reviewed base is `70299ca`, three commits ahead of `origin/main`, with `SL-009` complete. Because `AGENTS.md` declares `status.md` authoritative, a fresh Agent receives a false repository state and an obsolete next action. Codex added a provisional status/routing update to the review working tree; keep or improve it during remediation, and close the issue only after checking the final notes together.

Required correction:

- Make the status fields describe the reviewed base `70299ca` and the current review-remediation state.
- Remove the contradictory active executor/task wording and set the next action to resolving this review before push/tag or downstream integration.
- Avoid a field that necessarily becomes false when the notes commit is created. Prefer an explicitly named "reviewed base" or "state before the current notes commit" field, with Git commands remaining authoritative for the exact current HEAD.

Acceptance check:

```sh
git status --short --branch
git log -4 --oneline --decorate
```

The live status must not contradict these results or its own `Last completed`/`Next task` fields.

### RV-002 [P1] Confirmed facts mix the obsolete v0.30.22 baseline with current state

`.agents/findings.md:3-11` currently presents two obsolete statements as current confirmed facts: `go.mod` pins v0.30.22, and `Context.Release` has no drain mechanism. The reviewed tree pins v0.30.29. In that version, `Context.Release` calls `wgpu.Device.Release`, whose native implementation calls its internal `waitIdle` before destroying the device. The later historical sections correctly describe the v0.30.22 baseline and the upstream fixes, so the opening section now contradicts both the repository and the rest of the file.

Required correction:

- Split current facts from the historical v0.30.22 baseline.
- State the current v0.30.29 pin and that drain is provided transitively by `wgpu.Device.Release`, rather than by explicit submission tracking in `go-wgpu-mat`.
- Preserve the old v0.30.22 facts under a clearly historical heading because they are valid RED-before evidence.

Acceptance check:

```sh
rg -n 'v0\.30\.22|v0\.30\.29|drain' .agents/findings.md go.mod
```

Every v0.30.22 occurrence must be explicitly historical; current-state statements must agree with `go.mod` and the pinned module implementation.

## Validation completed by Codex

- `go mod tidy -diff`: no diff.
- `go build ./...`: PASS.
- `make test`: PASS in CGO-disabled and CGO-enabled race modes; `mat` coverage 95.0%.
- Metal selector `-count=10`: PASS for `TestP4Metal*` and `TestSLMetal*`.
- `go vet ./...`: PASS.
- `golangci-lint run`: 0 issues.
- Modified `.agents/*.md` files: `markdownlint-cli2` reports 0 issues.
- `make fuzz`: both 10-second targets PASS.
- `git diff --check origin/main..HEAD`: PASS.

No blocking implementation or dependency issue was found. Do not push, tag, release, or begin `go-nn` integration until RV-001 and RV-002 are corrected and this review is rechecked.

## Remediation(2026-07-31、Kimi Code CLI)

### RV-001対応

`status.md`を修正した。`HEAD`項目を「Reviewed base: `70299ca`」表現へ置き換え、
commitしても偽にならない形とした。正確な現HEADとahead数は
`git status --short --branch`と`git log --oneline --decorate`を正とする注記を
追加した。Active handoverはexecutor・active taskを「なし(remediation完了、
再検証待ち)」へ更新し、Next taskをMaintainerのreview再検証とした。

### RV-002対応

`findings.md`を「Confirmed facts(現行状態)」と「Historical baseline(v0.30.22、
RED-before evidence)」へ分割した。現行状態では`go.mod`がv0.30.29をpinし、
drainは`wgpu.Device.Release`の内部waitIdleによりtransitiveに提供されること
(go-wgpu-mat自身の追跡機構ではないこと)を明記した。旧pinの事実はすべて
historical節または「旧pin」明示の記述へ移した。あわせて、各節のv0.30.22言及が
historicalと読み取れることを点検し、解決済みのQuestions 3と4を現状態へ同期した。

### Acceptance check結果

```text
$ git status --short --branch
## main...origin/main [ahead 3]
(remediationの`.agents/`変更のみ、commit前)

$ git log -4 --oneline --decorate
70299ca (HEAD -> main) chore: resync code indexes and record SL close-out
ba3ea83 fix: bump gogpu/wgpu to v0.30.29 and add submission lifetime regression tests
7502d8e docs: add agent handover notes and code indexes
9949088 (tag: v0.0.2, origin/main, origin/HEAD) feat: add detailed matrix and submission statistics

$ rg -n 'v0\.30\.22|v0\.30\.29|drain' .agents/findings.md go.mod
go.mod:7: github.com/gogpu/wgpu v0.30.29
findings.mdの全`v0.30.22`言及は「旧pin」「Historical baseline」「当時」等で
明示的にhistorical。現行状態の記述(pin v0.30.29、transitive drain)はgo.modと
pinned module実装に一致。

$ markdownlint-cli2 ".agents/*.md"
Summary: 0 issues in 0 files (10 files)

$ git diff --check
PASS(出力なし)
```

`RV-001`と`RV-002`は修正済み。`status.md`のlive statusは上記git結果と矛盾せず、
自身の`Last completed`/`Next task`とも整合する。Maintainer/Codexによる再検証を
待つ。
