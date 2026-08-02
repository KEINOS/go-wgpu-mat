# Current Status

## Repository

- 更新日: 2026-08-02
- Repository: `github.com/KEINOS/go-wgpu-mat`
- Branch: `main`
- Reviewed base: `313ac51 docs: defer upstream PR until KEINOS packages are complete
  (D-010)`。正確な現HEADとahead数は`git status --short --branch`と
  `git log --oneline --decorate`を正とする。
- Remote: `origin/main`は`9949088`のまま。Maintainerはtag `v0.0.3`(=`d203721`)を
  push済みで、tag経由で全commit内容は公開済み。`main` branchのpushは未実施。
- Release: tag `v0.0.3`は`d203721`を指す。CI(`unit test`)はv0.0.3 pushで成功
  (2026-07-31T09:09:32Z、run 30618892290)。
- 現在のworking tree: review済み`SL-011` close-out変更(commit待ち)。最新の正確な状態は
  `git status --short --branch`を正とする。

## Active handover

- Current executor: Codex。Kimiが特定したroot causeとfork修正を引き継ぎ、正式な
  regression gate、checkptr有効race、文書、indexをclose-outした。
- Active task: なし。`SL-011`はfull gateとHermes read-only reviewまでGREEN。
- Last completed: `SL-011`。`TestSLMetalGradAccum*`を正式Metal gateへ昇格し、
  checkptr有効race、software/Metal gate、文書、index、reviewを完了した。
- Next task: close-out変更をcommit/push後、`go-nn` F2-010へ戻る。
- Next command: `git diff --cached --check`を確認してcommitし、KEINOSの`origin/main`
  だけをdry-run後にpushする。
- Blocker: なし。go-nn F2-009はfork修正でlocalではGREEN。upstream本家修正の
  届け方はMaintainer決定(D-008/D-009)。
- Background workers: managed sub-agent、Kimi、Hermes、Claude、Agy、Copilot CLIは
  稼働していない。VS Code内蔵Copilot processはsub-agentではない。

## Current phase

P4 device-resident kernelsとstatisticsは[`plan.md`](plan.md)のclose-outまで完了し、
Maintainerがcommit、push、tag `v0.0.2`、CI成功を確認済みである。

Downstream `go-nn`のF2 WGPU統合中に見つかったgradient破損は、Metal backendが
Nagaの`_mslBufferSizes`をbindせずruntime-sized storage arrayのbounds checkを
壊していたことがroot causeである。`KEINOS/wgpu`のbranch
`fix/metal-msl-buffer-sizes`(`aa07c1d`)をversioned replaceし、再利用、readback、
WaitIdleを含む全再現形状がGREENになった。詳細は[`findings.md`](findings.md)を参照。

## Next handover point

`SL-011`まで完了し、修正済み再現testを正式Metal gateへ昇格した。
`make test-metal`はrace/checkptr有効でP4、submission lifetime、gradient accumulationを
3反復する。全gateとHermes reviewはGREEN。commit/push後は`go-nn` F2-010へ戻る。

`go-nn`側で同じ破損が再発した場合は、[`findings.md`](findings.md)の副仮説
(H2-H5)と[`plan-submission-lifetime.md`](plan-submission-lifetime.md)の
切り分けladder rung 3-5(WaitIdle挿入、zero-init、単一op family chain)から
再開する。

## Authority

- Maintainerは次のAI Agentが残作業を進めることを2026-07-31に指示した。
- Read-only調査、詳細計画、test-firstのlocal実装、Metal検証、phaseごとのlocal
  commitは許可済みである。
- Maintainerは2026-08-02にKEINOS管理repositoryへのcommitとpushを許可した。
  Tagとreleaseは明示許可がないため行わない。本家`gogpu/wgpu`へのpush/PRは禁止。
- `go-nn`のdependency更新と統合再開は、upstream修正がtestを通り、Maintainerが
  push/tagした後に行う。

## Validation state

- RED(旧pin v0.30.22、historical): `TestSLMetalChainedCompute`が3/3回、同一PCで
  SIGSEGV。
- GREEN(現行fork replace): `make test`はCGO 0/1とcheckptr有効raceでGREEN
  (`mat` 95.0%、`pipelinecache` 100.0%)。`make test-metal`は正式化した
  `TestSLMetalGradAccum*`を含め3反復GREEN。`make lint` 0 issues、`make fuzz`両target、
  `go vet ./...`、`go mod tidy -diff`、`git diff --check`もGREEN。

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
