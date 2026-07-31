# Tasks

## Handover setup

- [x] `H-001` Root `AGENTS.md`から`.agents/README.md`へrouteする。
- [x] `H-002` 完了済みroot `plan.md`を`.agents/plan.md`へ移す。
- [x] `H-003` Graphifyを初期化し、`graphify-out/`を生成する。
- [x] `H-004` CodeGraphを`codegraph init`で初期化し、`.codegraph/`を生成する。
- [x] `H-005` Index、working tree、session close-out規約をnotesへ同期する。

## Next implementation candidate: submission lifetime

- [ ] `SL-001` 現行submission pathとresource ownershipをread-onlyでinventoryする。
- [ ] `SL-002` Pin済み`gogpu/wgpu v0.30.22`のownership contractを確認し、詳細計画、
  alternative hypothesis、rollback、acceptance gateを作成する。
- [ ] `SL-003` 中間readbackなしのchained-compute RED regression testを追加する。
- [ ] `SL-004` Command bufferだけを保持する切り分けで仮説を検証する。
- [ ] `SL-005` 必要resourceをsubmission完了まで保持し、完了済みだけを回収する。
- [ ] `SL-006` `Context.Release`でoutstanding submissionをdrainして全resourceを解放する。
- [ ] `SL-007` Failure path、idempotency、statistics、race、coverageを検証する。
- [ ] `SL-008` Local Metalで反復、device residency、concurrency、full releaseを検証する。
- [ ] `SL-009` Read-only review後にlocal commitし、session close-outする。
- [ ] `SL-010` Maintainerのpush/tag後、`go-nn`統合へ戻る。

## Acceptance outline

- v0.0.2でregression testがREDになり、修正後に十分な反復でGREENになる。
- 中間host read/write、operationごとの強制同期、暗黙CPU fallbackを導入しない。
- Submission resourceが完了後およびContext close時にexactly onceで解放される。
- 既存software/Metal contract、race、lint、coverage gateを維持する。
- Push、tag、releaseを行わない。
