# Findings

## Confirmed facts

- `go.mod`は`github.com/gogpu/wgpu v0.30.22`を固定している。
- `mat/compute.go`の`encodeAndSubmitCompute`はqueue submission後、関数return時に
  command bufferをreleaseする。
- `mat/matmul.go`はoperation return時にuniform bufferとbind groupをreleaseする。
- `mat/tensorops.go`もoperation return時にuniform bufferとbind groupをreleaseする。
- `Context.Release`はpipeline cache、device、adapter、instanceを解放するが、
  in-flight submission resourceを追跡またはdrainする仕組みを持たない。
- Downstream `go-nn`では1回目のdevice-resident backwardは正しく、readbackなしの
  2回目のgradient accumulationがMetalで非決定的に破損する。
- Downstream matrix poolを無効にしても破損は解消しなかった。
- 中間`Matrix.Read`は症状を隠すがhost transferを発生させる。

## SL-001 inventory(2026-07-31、Kimi Code CLI)

Submission pointはすべて`Context.device.Queue()`上にあり、`Context.withQueue`の
mutexがcontext単位で直列化する。Submission完了追跡(submission index、
`Device.Poll`、`OnSubmittedWorkDone`相当)はcodebaseのどこにも存在しない。

- Compute submission: `encodeAndSubmitCompute`(`mat/compute.go:82`)。encoder→
  compute pass→`Finish`→`Submit`→`recordComputeSubmission`。command bufferは
  `defer`(`mat/compute.go:147`)により、GPU完了を待たず関数return直後にrelease
  される。callerは`encodeAndSubmitAdd`(`mat/add.go:235`)、
  `encodeAndSubmitMatMul`(`mat/matmul.go:383`)、
  `dispatchTensorOperationWithDeps`(`mat/tensorops.go:439`)の3系統。
- Readback submission: `readBuffer`(`mat/mat.go:474`)。staging bufferへのcopyを
  submit後、`Map`が完了までblockingし、unmap後にdeferred release(staging bufferは
  `mat/mat.go:490`、command bufferは`mat/mat.go:503`)される。Mapによる同期がある
  ため、このpathのrelease時点ではsubmissionは完了済みである。
- Host write: `Matrix.Write`(`mat/mat.go:304`)とuniform upload
  (`mat/matmul.go:483`、`mat/tensorops.go:532`)は`Queue().WriteBuffer`を使う。
  dataがcall時にcopyされるかはSL-002でmodule側contractを確認する。
- Operation return時にreleaseされ、in-flight submissionが参照しうるresource:
  command buffer(`mat/compute.go:147`)、bind group(`mat/add.go:165`、
  `mat/matmul.go:378`、`mat/tensorops.go:437`)、uniform buffer
  (`mat/matmul.go:370`、`mat/tensorops.go:429`)。
- Bind group layoutは各dispatch内でdeferred releaseされるが、参照されるのは
  bind group/pipeline作成時だけであり、submissionは直接参照しない。
- Compute pipelineは`Context.pipes`のcacheに保持され、`Context.Release`で解放
  される。Matrix bufferはcaller所有で`Matrix.Release`(`mat/mat.go:357`)が解放する。
- `Context.Release`(`mat/context.go:400`)はpipeline cache、device、adapter、
  instanceを解放するが、in-flight submissionをdrainまたは待機しない。
- Downstreamの再現形状: `go-nn`の`nn/internal/tensorbackend/wgpu.go`はpoolされた
  matrixで`MatMul`/`Mul`/`Add`/`ReduceSumTo`/`Transp`/`BroadcastTo`/`Scale`を
  中間readbackなしで連鎖させる。go-nn `.agents/STATUS.md:228`に「Metalでは単発
  backwardと並行独立outputはGREENだが、readbackなしで2回連続するbackwardの
  gradient accumulationが不定値になる」と記録されている。

## Module contract facts(SL-002、2026-07-31、Kimi Code CLI)

すべてpin済み`gogpu/wgpu v0.30.22`のmodule source直接確認と、同CHANGELOGの
post-pin修正履歴に基づく。

- `Queue.Submit(...)`は`(uint64, error)`を返す。submission indexは
  `Queue.Poll()`(non-blocking、最終完了index)で追跡できる
  (`queue_native.go:34-40`、`:199`)。`OnSubmittedWorkDone`相当は存在しない。
- `Device.WaitIdle()`は全submission完了をblockingし、deferred破棄をtriageする
  (`device_native.go:862-877`)。
- Compute passの`SetBindGroup`はbind groupと各bound bufferの`ResourceRef`を
  Cloneし(`computepass_native.go:46-55`、`:85-92`)、Submit時に
  `TrackSubmission`でsubmissionへ関連付け、GPU完了時のTriageでDropする
  (`queue_native.go:159-167`)。`Buffer.Release`はrefcount駆動で、最後のrefが
  落ちるまでHAL bufferは破棄されない(`buffer.go:59-77`)。
- Submit後の`CommandBuffer.Release`はno-opである。`postSubmit`が
  `cb.halEncoder`と`cb.trackedRefs`をnil化し、ownershipはdeferred callbackへ
  移る(`queue_native.go:177-190`、`encoder_native.go:494-516`)。
- v0.30.22の`BindGroup.Release`は`dq.Defer(lastSubmissionIndex, ...)`を直接
  呼び、ResourceRef countingをbypassする(`bind_native.go:211-239`)。upstreamは
  これをuse-after-free(HAL resource destroyed while GPU still processing
  commands)としてv0.30.28で`ref.Drop()`方式へ修正した(CHANGELOG 0.30.28、
  ADR-056、#287)。go-wgpu-matは全compute op return時にbind groupをrelease
  しており、影響patternに合致する。
- v0.30.22の`Device.Release`は`released.Store(true)`後に`WaitIdle()`を呼ぶが、
  `WaitIdle`はreleased flagで即`ErrReleased`を返すためdrainは実行されない
  (`device_native.go:862-866`、`:912-937`)。upstreamはv0.30.23で内部waitIdleへ
  修正した(CHANGELOG 0.30.23、#264)。pin版では`Context.Release`はin-flight
  GPU workをdrainしない。
- その他のpost-pin修正: v0.30.28 DestroyQueue deadlock(Triage→onZero→Defer
  再入)、v0.30.29 `LastSubmissionIndex` deadlock(Submit→Triage→onZero→
  lastSubmissionIndex→mu)、v0.30.24 Metal checkptr under `-race`修正。
- v0.30.22はgoffi v0.6.0、v0.30.23以降はgoffi v0.6.2をpinする。
- v0.30.29でも`-race`をcheckptr有効で実行すると
  `fatal error: checkptr: pointer arithmetic result points to invalid allocation`
  が`hal/metal.newGPUCompletionBlock`のblock callback trampolineで発生する
  (2026-07-31、Kimi Code CLIが確認)。Makefileのdarwin/arm64向け
  `-gcflags=all=-d=checkptr=0`は引き続き必要であり、checkptr=0付きのraceでは
  `TestSLMetal*`を含めGREENである。

## Reproduction and isolation evidence(SL-003/004、2026-07-31、Kimi Code CLI)

- `mat/sl_contract_test.go`の`TestSLMetalChainedCompute`は、中間readbackなしで
  256 round×6 op(MatMul/Mul/Scale/ReduceSumTo/BroadcastTo/Add、accumulationを
  ping-pong bufferで更新)の依存chainを回し、最後に1回だけReadしてpure-Go参照と
  比較する。round数は`GO_WGPU_MAT_SL_ROUNDS`で変更可能。
- RED(pin v0.30.22): 3/3回、いずれも同じPCで`SIGSEGV: segmentation violation
  ... signal arrived during cgo execution`。crash siteは
  `hal/metal.(*AutoreleasePool).Drain`←`CommandEncoder.BeginEncoding`
  (`encoder.go:56`)←`Device.CreateCommandEncoder`(`device_native.go:724`)で、
  後続opのencoder作成時にFFI経由で落ちる。stackには
  `encodeAndSubmitCompute`(`mat/compute.go:92`)が見える。
- GREEN(v0.30.29): 同一testが`-count=1`、`-count=10`(2,560 round、15,360
  submission)、`GO_WGPU_MAT_SL_ROUNDS=1024 -count=3`(18,432 submission)の
  すべてでPASS。`TestSLMetalReleaseWithInflightWork`(最終Readなしでmatrix→
  contextを解放)も同時にPASS。
- 既存gate: `TestP4MetalKernels` PASS。`make test`(CGO=0およびCGO=1 race、
  checkptr=0)は両modeでGREEN(95.0% coverage)。`make lint` 0 issues。
  `make fuzz`両target 10s smoke PASS。software race selector
  `^(TestP4Software|TestSLSoftware)` PASS。
- RED-before/GREEN-after、十分な反復、既存testの成功が揃ったため、H1(pin版
  `BindGroup.Release`のuse-after-free、upstream ADR-056)を最も有力な原因候補と
  して扱う。なお最終的な原因確定には、version bisectまたはupstream issue #287の
  照合が残るが、実務上の修正(pin bump)は確定した。

## Working hypothesis

主仮説(H1): v0.30.22の`BindGroup.Release` use-after-free(upstream ADR-056)
が、operation return時のbind group releaseを通じてin-flight submissionと競合し、
勾配累積を破損させている可能性。単発では発現しにくく、連鎖・反復で顕在化する
downstream症状と整合する。→ SL-003/004のRED/GREEN証拠により有力化(上記
「Reproduction and isolation evidence」参照)。

副仮説: H2 teardown drain bug(v0.30.23で修正済みの類)はmid-run破損より終了時
crash/hangに関係する可能性。H3 upstream Metal HAL/completion trackingの別bug。
H4 go-nn側logic bug。H5 未定義初期buffer contentsの読み取り(`NewMatrix`は初期
内容未定義と明記)。H2-H5は、upgradeで再現が止まったため現時点では採用しないが、
downstreamで再発した場合に備えて記録を残す。

これらは未確認の仮説である。command bufferやuniform bufferの早期releaseは
module contract上安全と確認済みのため、切り分け対象から外す。

## Questions to resolve

1. ~~Pin済み`gogpu/wgpu v0.30.22`が各submitted resourceへ要求するownershipと
   release timingは何か。~~ 解決済み(SL-002)。Module contract factsを参照。
2. ~~Queue submission indexと`Device.Poll`を使って、同期的に毎operationを待たずに
   完了済みresourceだけを回収できるか。~~ 解決済み(SL-002)。`Queue.Poll()`の
   最終完了index比較と`Device.WaitIdle()`が使える。`Device.Poll`はmap解決のみで
   deferred破棄をtriageしない。
3. ~~`Context.Release`はoutstanding submissionをどうdrainし、error pathと二重解放を
   どう扱うべきか。~~ 方針確定(SL-002)。Pin版のdrainはno-opと確認。v0.30.23以降は
   upstreamがdrainする。Upgrade不採用時のみ`device.WaitIdle()`兜底を検討する。
4. Bind group経路のuse-after-free(H1)がRED testで再現・切り分けできるか。
   再現しない場合、H3(module Metal HAL)、H4(go-nn側)、H5(未定義初期内容)の
   どれを次に検証するか。
5. Metal以外のbackendにも同じlifecycle問題があるか。未調査。local検証は
   Metalのみ可能である。

## Evidence rules

- 少数回のGREENは非決定的問題の解消証拠にしない。
- RED-before/GREEN-after、十分な反復、resource解放確認、既存testの成功を揃える。
- 仮説は、再現testと切り分け結果が揃うまで「原因」と表現しない。
