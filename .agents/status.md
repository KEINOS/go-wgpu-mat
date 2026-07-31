# Current Status

## Repository

- 更新日: 2026-07-31
- Repository: `github.com/KEINOS/go-wgpu-mat`
- Branch: `main`
- HEAD: `9949088 feat: add detailed matrix and submission statistics`
- Remote: `origin/main`は`9949088`と一致する。
- Release: tag `v0.0.2`は`9949088`を指す。
- Handover開始時のworking tree: clean。
- 現在のworking tree: Handover用の`AGENTS.md`、`.agents/`、`graphify-out/`、
  `.codegraph/`と`plan.md`移動が未commit。これらを誤って破棄しないこと。

## Active handover

- Current executor: Kimi Code CLI。2026-07-31にhandoverをintakeし、着手済み。
- Active task: `SL-001` submission pathとresource ownershipのread-only inventory。
- Last completed: `H-005` Handover notes、Graphify、CodeGraph初期化。
- Next task: `SL-002` `gogpu/wgpu v0.30.22`のownership contract確認と詳細計画。
- Next command: `codegraph explore "Trace compute submission resource ownership"`
- Blocker: なし。原因仮説は未確認のため、`SL-003` RED testより前に修正を断定しない。
- Background workers: managed sub-agent、Kimi、Hermes、Claude、Agy、Copilot CLIは
  稼働していない。VS Code内蔵Copilot processはsub-agentではない。

## Current phase

P4 device-resident kernelsとstatisticsは[`plan.md`](plan.md)のclose-outまで完了し、
Maintainerがcommit、push、tag `v0.0.2`、CI成功を確認済みである。

Downstream `go-nn`のF2 WGPU統合中に、device-resident graphで中間readbackを挟まず
`Backward`を繰り返すと、Metalで2回目の累積gradientが非決定的に破損する問題が
見つかった。1回目は正しい。中間`Matrix.Read`は同期して症状を隠すが、
no-host-transfer contractに違反するため修正として採用しない。

## Next handover point

次のagentは[`findings.md`](findings.md)を確認し、submission resource lifecycleの
read-only reviewと詳細計画から始める。最初の実装gateは、中間readbackなしで複数の
compute submissionを連鎖させ、最終readだけで結果を検証するRED regression testで
ある。

原因は未確定である。command buffer、bind group、uniform bufferなどをsubmit直後に
解放している現行lifecycleは確認済みだが、それが破損原因かはtestで証明する必要が
ある。

## Authority

- Maintainerは次のAI Agentが残作業を進めることを2026-07-31に指示した。
- Read-only調査、詳細計画、test-firstのlocal実装、Metal検証、phaseごとのlocal
  commitは許可済みである。
- Push、tag作成、releaseは禁止。必要な場合はMaintainerの明示的な許可を得る。
- `go-nn`のdependency更新と統合再開は、upstream修正がtestを通り、Maintainerが
  push/tagした後に行う。

## Validation state

このhandover作業ではproduction codeやtestを変更していない。新しいsubmission
lifecycle regression testとMetal検証は未実施であり、過去P4の検証結果を新問題の
合格根拠として扱わない。

Handover indexは2026-07-31に初期化済みである。

- Graphify: 561 nodes、1,575 edges、17 communities。
- CodeGraph: 32 files、575 nodes、2,419 edges。`codegraph status .`はup to date。
- Graphifyは`.vscode/launch.json`と`.vscode/settings.json`がzero-nodeだったと警告した。
  Go code graphの生成は成功しており、この警告は非blockingである。
- Graphify再同期でcommunity setが変わり、一部labelをhub名へ自動更新した。
  LLMによるlabel refreshはhandover成立に不要なため未実行である。
- `AGENTS.md`、`.agents/*.md`、`graphify-out/*.md`のMarkdown lintと
  `git diff --check`は成功した。
