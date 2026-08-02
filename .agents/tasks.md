# Tasks

## Handover setup

- [x] `H-001` Root `AGENTS.md`から`.agents/README.md`へrouteする。
- [x] `H-002` 完了済みroot `plan.md`を`.agents/plan.md`へ移す。
- [x] `H-003` Graphifyを初期化し、`graphify-out/`を生成する。
- [x] `H-004` CodeGraphを`codegraph init`で初期化し、`.codegraph/`を生成する。
- [x] `H-005` Index、working tree、session close-out規約をnotesへ同期する。

## Next implementation candidate: submission lifetime

- [x] `SL-001` 現行submission pathとresource ownershipをread-onlyでinventoryする。
- [x] `SL-002` Pin済み`gogpu/wgpu v0.30.22`のownership contractを確認し、詳細計画、
  alternative hypothesis、rollback、acceptance gateを作成する。
  成果物: [`.agents/plan-submission-lifetime.md`](plan-submission-lifetime.md)。
- [x] `SL-003` 中間readbackなしのchained-compute RED regression testを追加する。
  `mat/sl_contract_test.go`に`TestSLMetalChainedCompute`等を追加し、pin版で
  3/3 SIGSEGVのREDを確認した。
- [x] `SL-004` 切り分けladderを実行する。第1 rungのupgrade仮説testで、v0.30.29へ
  上げると同一testが反復GREENとなり、H1(bind group use-after-free)を有力化した。
- [x] `SL-005` 切り分け結果に基づき修正する。`go.mod`のpinをv0.30.29へ更新
  (goffi v0.6.2、gpucontext v0.23.0、naga v0.17.16、webgpu v0.5.4を伴う)。
- [x] `SL-006` `Context.Release`のdrainを検証する。`TestSLMetalReleaseWithInflightWork`
  でin-flight work下のmatrix/context解放とidempotencyを検証した(upstream修正で
  drainが機能する)。
- [x] `SL-007` Failure path、idempotency、statistics、race、coverageを検証する。
  `make test`両mode、`make lint`、`make fuzz`、software race selectorがGREEN。
- [x] `SL-008` Local Metalで反復、device residency、concurrency、full releaseを検証する。
  `-count=10`と`GO_WGPU_MAT_SL_ROUNDS=1024 -count=3`でGREEN。device residencyは
  test内のStats assertionで検証済み。
- [x] `SL-009` Read-only review後にlocal commitし、session close-outする。
  copilot、hermes、codexの3 reviewerが最終状態でAGREED。
- [x] `SL-010` Maintainerのpush/tag後、`go-nn`統合へ戻る。
- [x] `SL-011` fork修正後のgrad-accum再現testを正式なMetal regression gateへ昇格し、
  checkptr有効のrace、全software/Metal gate、文書、indexをclose-outする。

## Acceptance outline

- v0.0.2でregression testがREDになり、修正後に十分な反復でGREENになる。
- 中間host read/write、operationごとの強制同期、暗黙CPU fallbackを導入しない。
- Submission resourceが完了後およびContext close時にexactly onceで解放される。
- 既存software/Metal contract、race、lint、coverage gateを維持する。
- Maintainerの許可に基づきKEINOS管理repositoryへcommit/pushできる。Tagとrelease、
  本家`gogpu/wgpu`へのpush/PRは行わない。
