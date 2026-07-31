# Submission Lifetime 詳細計画(SL-002 成果物)

2026-07-31、Kimi Code CLI。SL-001 inventoryとSL-002 contract調査の結果を
統合し、RED testから修正、検証までの実行計画を定める。

## 前提: 確認済みcontract事実

以下はすべて`gogpu/wgpu v0.30.22`(pin済み)のmodule sourceを直接読んで確認した
事実である(`.agents/findings.md`の「Module contract facts」節にfile/line証拠あり)。

1. `Queue.Submit`はsubmission indexを返し、完了追跡は`Queue.Poll()`(non-blocking、
   最終完了index)と`Device.WaitIdle()`(blocking、全submission完了+deferred破棄の
   triage)で行う。`OnSubmittedWorkDone`相当のcallback APIは存在しない。
2. Compute passの`SetBindGroup`はbind groupと各bound bufferの`ResourceRef`を
   Cloneし、Submit時に`TrackSubmission`へ移し、GPU完了時のTriageでDropする。
   したがって、operation return時にbind groupやuniform bufferをreleaseしても
   HAL resourceはGPU完了まで生きる設計である。
3. Submit後の`CommandBuffer.Release`は事実上no-opである(ownershipはSubmitへ
   移る)。よって「command bufferだけ保持する切り分け」(旧SL-004)はmodule挙動を
   変えないため、実験として意味を持たない。
4. 一方、v0.30.22の`BindGroup.Release`は`dq.Defer(lastSubmissionIndex, ...)`へ
   直接行き、ResourceRef countingをbypassする。upstreamはこれを
   **use-after-free(HAL resource destroyed while GPU still processing commands)**
   としてv0.30.28で修正した(ADR-056、#287)。go-wgpu-matは全compute opのreturn
   時にbind groupをreleaseしており、影響patternに合致する。
5. v0.30.22の`Device.Release`は`released.Store(true)`を先に立ててから
   `WaitIdle()`を呼ぶが、`WaitIdle`はreleased flagで即`ErrReleased`を返すため
   drainはno-opである。upstreamはv0.30.23で内部waitIdleへ修正した(#264)。
   すなわち、pin版では`Context.Release`はin-flight GPU workをdrainしない。
6. その他のpost-pin修正: v0.30.28 DestroyQueue deadlock(Triage→onZero→Defer
   再入)、v0.30.29 `LastSubmissionIndex` deadlock(Submit→Triage→onZero)、
   v0.30.24 Metal checkptr under `-race`修正(`go test -race`がMetal computeで
   動作するようになったと記載)。

## 仮説の状態

- H1(早期releaseが破損原因): 上記2-3により、command buffer / uniform bufferの
  早期releaseはcontract上安全。bind groupの早期releaseだけがv0.30.22の既知
  use-after-free(上記4)に該当する。H1は「bind group経路に限り」有力。
- H2(upstream teardown drain bug): 上記5。mid-runの勾配破損より、終了時の
  crash/hangに関係する可能性が高い。
- H3(upstream Metal HAL / completion tracking bug): upgradeしても再現する場合の
  次候補。
- H4(go-nn側logic bug): 本repoのRED testが一切再現しない場合の候補。
- H5(未定義初期buffer contentsの読み取り): `NewMatrix`は初期内容を未定義と
  明記している。zero-init切り分けで評価する。

引き続きD-004を適用し、RED testと切り分け結果が揃うまでいずれも「原因」と
表現しない。

## 実行計画

### Phase 1: SL-003 RED regression test(production変更なし)

`mat/`にchained-compute regression testを追加する。go-nnのbackward形状を
模倣し、中間readbackなしで複数roundの依存op連鎖を実行する。

- `TestSLMetalChainedCompute`(仮称): `GO_WGPU_MAT_GPU=1` gate、`UseGPU`、
  直列実行。N round(初期値256、envで変更可能)について、
  `MatMul`→`Add`→`Mul`→`Scale`等の依存chainを回し、accumulationを
  ping-pong bufferで更新する(aliasing禁止contractのため)。
  全round終了後に1回だけ`Read`し、並行計算したpure-Go参照と比較する。
  間にhost transferが無いことを`Stats`で検証する。
- `TestSLSoftwareChainedCompute`(仮称): `UseCPU`版。参照mathの妥当性を検証する
  常にGREENのcontrol。
- 期待: v0.30.22 pin上でRED(非決定的な値の破損)。再現しなければH4へ進む
  判断材料とする。

### Phase 2: SL-004 切り分けladder(安い順)

1. **Upgrade仮説test**: 作業treeで`go mod edit -require=github.com/gogpu/wgpu@v0.30.29`
   +`go mod tidy`し、RED testを十分な反復で再実行する。GREENならH1(bind group
   use-after-free)を強く支持。REDのままならrung 2へ。試行後、採用しない場合は
   `go.mod`/`go.sum`をrevertする。
2. **Version bisect(任意)**: 修正同定が必要ならv0.30.23とv0.30.29の間をbisectする。
   実用上はdeadlock修正を含むv0.30.29が妥当なtarget。
3. **WaitIdle挿入**: round間に`Device.WaitIdle()`を挿入したtest variantで、
   timing/completion依存かを評価する(production変更ではなくtest hook)。
4. **Zero-init切り分け**: 全matrixをzero初期化してH5を評価する。
5. **単一op family chain**: `MatMul`のみ、`Add`のみ等で絞り込む。

### Phase 3: SL-005 修正(証拠依存)

- UpgradeでGREENになる場合: `go.mod`のpinをv0.30.29へ上げ、regression testを
  残す。`Context.Release`のdrainもv0.30.23以降でupstream修正済みとなる。
  これが最小侵襲の修正である。
- UpgradeでREDのままの場合: ladder結果に基づき、module-levelの最小再現を作り、
  回避策またはupstream報告をMaintainerへ提案する。per-op強制同期、中間
  readback、暗黙CPU fallbackは修正として採用しない(D-003)。

### Phase 4: SL-006 Context.Release drain

- Pin版(v0.30.22)ではdrainがno-opであることが確認済み(前提5)。Upgrade採用なら
  upstream修正で解消する。Upgradeを採用しない場合に限り、`Context.Release`が
  `device.Release`前に`device.WaitIdle()`を呼ぶ兜底を検討する。
- いずれの場合も、in-flight workがある状態での`Context.Release`をtestで検証する。

### Phase 5: SL-007/SL-008 検証

- Failure path、idempotent release、statistics、race、coverageを既存gateで検証する。
- 反復: regression testを`-count=10`および複数回実行し、少数回のGREENを
  解消証拠にしない(findings.mdのEvidence rules)。
- 既存selectorの維持: `TestP4Software*`、`TestP4Metal*`、`make test`、
  `make lint`、`make fuzz`。

## Rollback

- Phaseごとにlocal commitし、rollbackは該当commitのrevertとする。
- Pin変更は`go.mod`/`go.sum`のrevertで完全に戻せる。
- Regression test自体はproduction contractを変えないため残してよい。

## Acceptance gates

- Regression testがv0.30.22 pinでRED、修正後に十分な反復(`-count=10`以上、
  複数session)でGREENになる。
- 中間host read/write、operationごとの強制同期、暗黙CPU fallbackを導入しない。
- Submission resourceが完了後およびContext close時にexactly onceで解放される
  ことを、注入testで検証する(production変更が入る場合)。
- 既存software/Metal contract、race、lint、coverage gateを維持する。
- 最終read-only reviewでunresolved blocking findingが無い。
- Push、tag、releaseを行わない。

## Test selectors

```sh
CGO_ENABLED=1 go test -race -gcflags=all=-d=checkptr=0 -cover -count=1 \
  -run '^TestSLSoftware' ./mat
GO_WGPU_MAT_GPU=1 CGO_ENABLED=1 go test -count=1 -parallel=1 \
  -run '^TestSLMetal' ./mat
```

`TestSLSoftware*`は`UseCPU`または注入doubleのみを使い、hardware adapterを
要求しない。`TestSLMetal*`は`UseGPU`を要求し、`GO_WGPU_MAT_GPU=1`でadapter
不在をskipではなくfailureとする(P4 selectorと同じ規約)。

## Non-goals

- `go-nn`のcheckout、dependency、統合状態への変更(SL-010はMaintainer管理)。
- gogpu/wgpu module自体への修正 fork(回避不能な場合はMaintainer判断)。
- Shader最適化、command fusion、buffer poolingなどの性能改善。
- Push、tag、release。
