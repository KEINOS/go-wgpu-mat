# Findings

## Confirmed facts(現行状態)

- `go.mod`は`github.com/gogpu/wgpu v0.30.29`を固定している(2026-07-31のcommit
  `ba3ea83`でv0.30.22から更新)。
- 現行pinでは、`Context.Release`→`wgpu.Device.Release`が内部waitIdleでGPU workの
  完了を待ってからdeviceを破棄する。すなわちdrainはgo-wgpu-mat自身のsubmission
  追跡機構ではなく、pinned moduleの実装によってtransitiveに提供される。
- `mat/compute.go`の`encodeAndSubmitCompute`はqueue submission後、関数return時に
  command bufferをreleaseする(現行code。現行pinのmodule contract上安全である。
  「Module contract facts」参照)。
- `mat/matmul.go`はoperation return時にuniform bufferとbind groupをreleaseする
  (現行code)。
- `mat/tensorops.go`もoperation return時にuniform bufferとbind groupをreleaseする
  (現行code)。
- Regression test(`mat/sl_contract_test.go`)は現行pinでGREENであり、中間host
  transfer無し、release idempotency、in-flight work中のcontext解放を検証している。

## Historical baseline(v0.30.22、RED-before evidence)

以下は修正前の旧pin`gogpu/wgpu v0.30.22`当時の記録であり、現行状態ではない。

- 旧pinでは`go.mod`が`github.com/gogpu/wgpu v0.30.22`を固定していた。
- 旧pinでは`Context.Release`→`wgpu.Device.Release`内部の`WaitIdle()`が
  released flagにより即`ErrReleased`を返すためno-opとなり、in-flight GPU workを
  drainしなかった(upstreamはv0.30.23で内部waitIdleへ修正。CHANGELOG 0.30.23、
  #264)。
- 旧pinの`BindGroup.Release`は`dq.Defer(lastSubmissionIndex, ...)`を直接呼び、
  ResourceRef countingをbypassしていた。upstreamはこれをuse-after-free(HAL
  resource destroyed while GPU still processing commands)としてv0.30.28で
  `ref.Drop()`方式へ修正した(CHANGELOG 0.30.28、ADR-056、#287)。
- Downstream `go-nn`では(旧pin使用時)、1回目のdevice-resident backwardは正しく、
  readbackなしの2回目のgradient accumulationがMetalで非決定的に破損した。
- Downstream matrix poolを無効にしても破損は解消しなかった(旧pin当時)。
- 中間`Matrix.Read`は症状を隠すがhost transferを発生させる(旧pin当時の観測)。

## SL-001 inventory(2026-07-31、Kimi Code CLI、v0.30.22当時のcode調査)

Submission pointはすべて`Context.device.Queue()`上にあり、`Context.withQueue`の
mutexがcontext単位で直列化する。Submission完了追跡(submission index、
`Device.Poll`、`OnSubmittedWorkDone`相当)をgo-wgpu-matのcodeは使っていない。
なお本inventoryの対象codeは現行pinでも同一である。

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
  dataがcall時にcopyされるかはSL-002でmodule側contractを確認した。
- Operation return時にreleaseされ、in-flight submissionが参照しうるresource:
  command buffer(`mat/compute.go:147`)、bind group(`mat/add.go:165`、
  `mat/matmul.go:378`、`mat/tensorops.go:437`)、uniform buffer
  (`mat/matmul.go:370`、`mat/tensorops.go:429`)。
- Bind group layoutは各dispatch内でdeferred releaseされるが、参照されるのは
  bind group/pipeline作成時だけであり、submissionは直接参照しない。
- Compute pipelineは`Context.pipes`のcacheに保持され、`Context.Release`で解放
  される。Matrix bufferはcaller所有で`Matrix.Release`(`mat/mat.go:357`)が解放する。
- `Context.Release`(`mat/context.go:400`)はpipeline cache、device、adapter、
  instanceを解放し、go-wgpu-mat自身はin-flight submissionをdrainまたは待機する
  機構を持たない。旧pinではこのためdrainが全く効かなかったが、現行pinでは
  `wgpu.Device.Release`内部のwaitIdleがdrainする(「Confirmed facts(現行状態)」
  参照)。
- Downstreamの再現形状: `go-nn`の`nn/internal/tensorbackend/wgpu.go`はpoolされた
  matrixで`MatMul`/`Mul`/`Add`/`ReduceSumTo`/`Transp`/`BroadcastTo`/`Scale`を
  中間readbackなしで連鎖させる。go-nn `.agents/STATUS.md:228`に「Metalでは単発
  backwardと並行独立outputはGREENだが、readbackなしで2回連続するbackwardの
  gradient accumulationが不定値になる」と記録されている(旧pin当時の観測)。

## Module contract facts(SL-002、2026-07-31、Kimi Code CLI)

旧pin`gogpu/wgpu v0.30.22`(historical baseline)のmodule source直接確認と、同
CHANGELOGのpost-pin修正履歴に基づく。現行pinはv0.30.29であり、下記のv0.30.22
固有の欠陥は現行pinでは修正済みである。

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
- 旧pin v0.30.22の`BindGroup.Release`は`dq.Defer(lastSubmissionIndex, ...)`を
  直接呼び、ResourceRef countingをbypassする(`bind_native.go:211-239`)。upstream
  修正は「Historical baseline」節のとおり。go-wgpu-matは全compute op return時に
  bind groupをreleaseしており、旧pinでは影響patternに合致した。
- 旧pin v0.30.22の`Device.Release`は`released.Store(true)`後に`WaitIdle()`を
  呼ぶが、`WaitIdle`はreleased flagで即`ErrReleased`を返すためdrainは実行
  されない(`device_native.go:862-866`、`:912-937`)。現行pinでは修正済み。
- その他のpost-pin修正: v0.30.28 DestroyQueue deadlock(Triage→onZero→Defer
  再入)、v0.30.29 `LastSubmissionIndex` deadlock(Submit→Triage→onZero→
  lastSubmissionIndex→mu)、v0.30.24 Metal checkptr under `-race`修正。
- 旧pin v0.30.22はgoffi v0.6.0を、現行pin(v0.30.23以降系統)はgoffi v0.6.2を
  pinする。
- 現行pin v0.30.29でも`-race`をcheckptr有効で実行すると
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
- RED(旧pin v0.30.22): 3/3回、いずれも同じPCで`SIGSEGV: segmentation violation
  ... signal arrived during cgo execution`。crash siteは
  `hal/metal.(*AutoreleasePool).Drain`←`CommandEncoder.BeginEncoding`
  (`encoder.go:56`)←`Device.CreateCommandEncoder`(`device_native.go:724`)で、
  後続opのencoder作成時にFFI経由で落ちる。stackには
  `encodeAndSubmitCompute`(`mat/compute.go:92`)が見える。
- GREEN(現行pin v0.30.29): 同一testが`-count=1`、`-count=10`(2,560 round、15,360
  submission)、`GO_WGPU_MAT_SL_ROUNDS=1024 -count=3`(18,432 submission)の
  すべてでPASS。`TestSLMetalReleaseWithInflightWork`(最終Readなしでmatrix→
  contextを解放)も同時にPASS。
- 既存gate: `TestP4MetalKernels` PASS。`make test`(CGO=0およびCGO=1 race、
  checkptr=0)は両modeでGREEN(95.0% coverage)。`make lint` 0 issues。
  `make fuzz`両target 10s smoke PASS。software race selector
  `^(TestP4Software|TestSLSoftware)` PASS。
- RED-before/GREEN-after、十分な反復、既存testの成功が揃ったため、H1(旧pinの
  `BindGroup.Release`のuse-after-free、upstream ADR-056)を最も有力な原因候補と
  して扱う。なお最終的な原因確定には、version bisectまたはupstream issue #287の
  照合が残るが、実務上の修正(pin bump)は確定した。

## Isolation ladder results(SL-004続、2026-07-31、Kimi Code CLI)

`go-wgpu-mat v0.0.3`がgo-nnのrepeated-backward破損を解消しなかったため、go-nnの
backward-2と同一op列(seed=BroadcastTo(one,1x1)、dP=BroadcastTo(seed,2x2)、
Transp×2、MatMul×2、commit Add×2、commit直後に旧gradをclose)をmat-levelで再現する
診断testを追加した(`mat/sl_gradaccum_test.go`、`mat/sl_waitidle_test.go`、
本項執筆時点で未commit)。結果:

- 再現matrix(-count=10): fresh+中間readなしは10/10 GREEN。pooled+read、
  fresh+read、pooled+readなしはすべて10/10 RED。temps-only reuseは1/10、
  grads-only reuseは2/10 RED。**buffer再利用と中間readbackは独立した2つの
  trigger**であり、go-nnがpoolを無効化しても中間readback triggerが残るため
  破損が解消しないことを説明する。
- Forensics(pooled): round-1は全buffer正しい。round-2ではBroadcastTo/Transp
  (tensorop pipeline)の出力は正しく着弾するが、MatMul/Addの出力が誤り
  (例: deltaLがdProductの内容、deltaRがrightTの内容。process内では決定的、
  run間で非決定)。
- `Device.WaitIdle()`をround-2の全op間に挟んでもRED。**完了/timing競合では
  ない**。
- `ReshapeTo`でfresh matrixへcopyしてからreadしてもdirect readと一致。
  **readbackは正直であり、kernel出力そのものが誤り**。
- go-nnのpoolを診断的に無効化(v0.0.3)しても`TestWGPUTensorMatMulBackward`
  は3/3 RED(差分はrevert済み)。
- `gogpu/wgpu v0.30.30`(2026-07-31時点の最新)でも同一の失敗matrix。
  最新upstreamでも未修正。なおgo-nnの破損はv0.30.22でも観測されており、
  0.30.28の書き換え由来のregressionではない。

結論: `gogpu/wgpu`のMetal backendは、(a)submissionをまたいで以前使われた
bufferへ書き込む場合、または(b)中間readbackの後のsubmissionで、kernel出力を
誤ることがある。fresh bufferでreadを挟まないchain(= `TestSLMetalChainedCompute`
の形状)は影響を受けない。upstream issueとして報告できる最小再現は
`mat/sl_gradaccum_test.go`である。次はMaintainer判断: upstream issue作成か、
`hal/metal`のencoder/binder/completion周辺のmodule内部調査か。

追記(2026-07-31): Maintainerのfail-fast方針によりworking pinを`v0.30.30`へ
更新した(D-007)。v0.30.30のCHANGELOGの修正はvalidation mapのmemory leak
(SetBindGroupがencoder mapへbound resourceを蓄積しBindGroupをpinし続ける問題)
であり、本破損の原因ではない。v0.30.30で全gate(`make test`両mode、P4/SL Metal
selector、software race selector、lint、vet、fuzz、`go mod verify`)がGREENで、
repro診断testは`TestRepro`prefixへquarantineし、既定の`^TestSLMetal`gateから
除外した。tag `v0.0.3`はv0.30.29 pinのまま変更しない。

追記2(2026-07-31): upstream issue調査。`gogpu/wgpu`の既存issueに本破損
(buffer再利用/中間readback後のcompute出力破損)に該当するものは**存在しない**
(`readback`、`corrupt OR garbage OR wrong results OR nondeterministic`で検索)。
隣接issueは、1つ目のbugを修正した#287(BindGroup use-after-free、v0.30.28)と、
Metal block callbackのcheckptr crashを扱う#280(不十分だった)と#293である。
issue #293はcompletion block trampolineの`uintptr→unsafe.Pointer`変換が原因で、
修正は**v0.30.31**(2026-08-01リリース)に入った。working pinをv0.30.31へ
一時的に上げてrepro suiteを実行したが、**v0.30.31でも同一signatureでRED**
(gradL[0]=6、gradR[0]=9など)であり、completion-block修正は本破損の原因では
ない。pinはv0.30.30へ戻した。

## Working hypothesis(v0.30.22 baselineに関する仮説)

主仮説(H1): 旧pin v0.30.22の`BindGroup.Release` use-after-free(upstream
ADR-056)が、operation return時のbind group releaseを通じてin-flight submissionと
競合し、勾配累積を破損させていた可能性。単発では発現しにくく、連鎖・反復で
顕在化するdownstream症状と整合する。→ SL-003/004のRED/GREEN証拠により有力化
(上記「Reproduction and isolation evidence」参照)。

副仮説: H2 teardown drain bug(旧pinの欠陥、v0.30.23でupstream修正済み)はmid-run
破損より終了時crash/hangに関係する可能性。H3 upstream Metal HAL/completion
trackingの別bug。H4 go-nn側logic bug。H5 未定義初期buffer contentsの読み取り
(`NewMatrix`は初期内容未定義と明記)。H2-H5は、upgradeで再現が止まったため現時点
では採用しないが、downstreamで再発した場合に備えて記録を残す。

これらは旧pinに関する仮説である。command bufferやuniform bufferの早期releaseは
module contract上安全と確認済みのため、切り分け対象から外した。

## Questions to resolve

1. ~~旧pin`gogpu/wgpu v0.30.22`が各submitted resourceへ要求するownershipと
   release timingは何か。~~ 解決済み(SL-002)。Module contract factsを参照。
2. ~~Queue submission indexと`Device.Poll`を使って、同期的に毎operationを待たずに
   完了済みresourceだけを回収できるか。~~ 解決済み(SL-002)。`Queue.Poll()`の
   最終完了index比較と`Device.WaitIdle()`が使える。`Device.Poll`はmap解決のみで
   deferred破棄をtriageしない。
3. ~~`Context.Release`はoutstanding submissionをどうdrainし、error pathと二重解放を
   どう扱うべきか。~~ 解決済み。旧pinのdrainはno-opと確認し、v0.30.29への
   upgradeを採用した(D-006)。現行pinでは`wgpu.Device.Release`がdrainする。
4. ~~Bind group経路のuse-after-free(H1)がRED testで再現・切り分けできるか。~~
   解決済み(SL-003/004)。旧pinで3/3 SIGSEGVを再現し、現行pinで反復GREEN。
5. Metal以外のbackendにも同じlifecycle問題があるか。未調査。local検証は
   Metalのみ可能である。

## Evidence rules

- 少数回のGREENは非決定的問題の解消証拠にしない。
- RED-before/GREEN-after、十分な反復、resource解放確認、既存testの成功を揃える。
- 仮説は、再現testと切り分け結果が揃うまで「原因」と表現しない。
