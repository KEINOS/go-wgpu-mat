# Current Status

## Repository

- 更新日: 2026-07-31
- Repository: `github.com/KEINOS/go-wgpu-mat`
- Branch: `main`
- Reviewed base: `d203721 docs: close out review recheck in repo-local notes`
  (Codexが再検証し、Maintainerがtag `v0.0.3`を切ったcommit。正確な現HEADと
  ahead数は`git status --short --branch`と`git log --oneline --decorate`を正とする)
- Remote: `origin/main`は`9949088`のまま。Maintainerはtag `v0.0.3`(=`d203721`)を
  push済みで、tag経由で全commit内容は公開済み。`main` branchのpushは未実施。
- Release: tag `v0.0.3`は`d203721`を指す。CI(`unit test`)はv0.0.3 pushで成功
  (2026-07-31T09:09:32Z、run 30618892290)。
- 現在のworking tree: 本notes更新のみ。最新の正確な状態は
  `git status --short --branch`を正とする。

## Active handover

- Current executor: Kimi Code CLI。Module調査でroot causeを特定し、patch検証まで
  完了した。Upstreamへの届け方はMaintainer判断待ち。
- Active task: なし(Maintainerの判断待ち: upstream issue/patch添付/PR、forkの
  要否、go-nn側暫定対応)。
- Last completed: `gogpu/wgpu` Metal backendのroot cause特定(naga
  `_mslBufferSizes`未bindによるOOB write)とpatch検証(repro 80/80、go-nn
  Metal test 10/10 GREEN)。詳細は[`findings.md`](findings.md)の「Root cause
  identified」、patchは`.agents/wgpu-sizes-buffer-fix.patch`。
- Next task: Maintainer判断後、(a)upstream release取り込み時に両repoのpin更新と
  gate再開、(b)選ばれた届け方の支援。
- Next command: `GO_WGPU_MAT_GPU=1 CGO_ENABLED=1 go test -count=10 -parallel=1 -run '^TestReproMetal' ./mat`
  (upstream未修正の間はreuse/read variantがFAIL=期待値)
- Blocker: go-nn F2-009は、upstream(`gogpu/wgpu`)の`_mslBufferSizes`未bind修正が
  releaseされるまでblocked。暫定選択肢はfindingsの「Root cause identified」節を
  参照。
- Background workers: managed sub-agent、Kimi、Hermes、Claude、Agy、Copilot CLIは
  稼働していない。VS Code内蔵Copilot processはsub-agentではない。

## Current phase

P4 device-resident kernelsとstatisticsは[`plan.md`](plan.md)のclose-outまで完了し、
Maintainerがcommit、push、tag `v0.0.2`、CI成功を確認済みである。

Downstream `go-nn`のF2 WGPU統合中に見つかった、readbackなしで`Backward`を
繰り返すと2回目の累積gradientが非決定的に破損する問題について、SL-001〜SL-008を
実施した。旧pin`gogpu/wgpu v0.30.22`の`BindGroup.Release` use-after-free
(upstream ADR-056、v0.30.28で修正)が最有力原因と特定され、修正は
`go.mod`のpinをv0.30.29へ上げることと確定した。詳細は[`findings.md`](findings.md)の
「Reproduction and isolation evidence」と[`plan-submission-lifetime.md`](
plan-submission-lifetime.md)を参照。

## Next handover point

SL-001〜SL-009、Codex review、Kimi remediation、Codex再検証は完了し、Maintainerが
tag `v0.0.3`をpushしてCI成功を確認した。`SL-010`の前提条件は満たされ、作業は
`go-nn` repoへ移る。go-nn側では`AGENTS.md`と`.agents/README.md`を入口に現在地を
確認してからdependency更新と統合再開を行う。

`go-nn`側で同じ破損が再発した場合は、[`findings.md`](findings.md)の副仮説
(H2-H5)と[`plan-submission-lifetime.md`](plan-submission-lifetime.md)の
切り分けladder rung 3-5(WaitIdle挿入、zero-init、単一op family chain)から
再開する。

## Authority

- Maintainerは次のAI Agentが残作業を進めることを2026-07-31に指示した。
- Read-only調査、詳細計画、test-firstのlocal実装、Metal検証、phaseごとのlocal
  commitは許可済みである。
- Push、tag作成、releaseは禁止。必要な場合はMaintainerの明示的な許可を得る。
- `go-nn`のdependency更新と統合再開は、upstream修正がtestを通り、Maintainerが
  push/tagした後に行う。

## Validation state

- RED(旧pin v0.30.22、historical): `TestSLMetalChainedCompute`が3/3回、同一PCで
  SIGSEGV。
- GREEN(現行pin v0.30.29): 同一testが`-count=1`、`-count=10`(15,360
  submission)、`GO_WGPU_MAT_SL_ROUNDS=1024 -count=3`(18,432 submission)でPASS。
  `TestSLMetalReleaseWithInflightWork`もPASS。
- 既存gate: `TestP4MetalKernels` PASS、`make test`(CGO=0/1、race)GREEN
  (95.0% coverage)、`make lint` 0 issues、`make fuzz`両target PASS、
  software race selector GREEN。
- Production codeは変更していない。変更は`go.mod`/`go.sum`のpin bumpと
  test追加、`.agents/` notesのみである。

Handover indexは2026-07-31に初期化済みである。SL-003のtest追加を反映する
再同期をsession close-outで実施した。

- Graphify: 591 nodes、1,639 edges、25 communities。
- CodeGraph: `codegraph sync .`完了。
- Graphifyは`.vscode/launch.json`と`.vscode/settings.json`がzero-nodeだったと警告した。
  Go code graphの生成は成功しており、この警告は非blockingである。
- Graphify再同期でcommunity setが変わり、一部labelをhub名へ自動更新した。
  LLMによるlabel refreshはhandover成立に不要なため未実行である。
- `AGENTS.md`、`.agents/*.md`、`graphify-out/*.md`のMarkdown lintと
  `git diff --check`は成功した。
- `7ba11af`のCodex再検証で、Kimi remediationの変更対象が`.agents/`だけであること、
  `RV-001`と`RV-002`のacceptance checks、Markdown lint、`git diff --check`の成功を
  確認した。
