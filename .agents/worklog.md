# Work Log

## 2026-07-31 Handover initialization

Maintainerは`go-nn`のF2 WGPU統合を一旦commitし、作業場所を正規
`go-wgpu-mat` checkoutへ移した。

Handover開始時点で`main`、HEAD `9949088`、`origin/main`、tag `v0.0.2`が一致し、
working treeはcleanだった。Rootには完了済みP4計画`plan.md`があり、`AGENTS.md`、
`.agents/`、`graphify-out/`、`.codegraph/`は存在しなかった。

Kimi CLIは`kimi doctor`に成功した。Kimiへsanitized packetだけを渡してhandover構成、
task順、risk、禁止事項をadvisory reviewさせた。提案のうち、事実と仮説の分離、薄い
root routing、live status、evidence log、taskとauthorityの明記を採用した。未承認の
新phase実装やpush/tagは行わない。

現行コードをread-only確認し、compute submission後にcommand buffer、uniform
buffer、bind groupがoperation return時にreleaseされることを確認した。これと
downstream corruptionの因果関係は未確認として記録した。

`graphify update .`で`graphify-out/`を初期化し、558 nodes、1,572 edges、
27 communitiesを生成した。`.vscode`の2 JSON fileがzero-nodeとの警告はあったが、
code graph生成は成功した。

`codegraph init .`で`.codegraph/`を初期化し、32 files、575 nodes、2,419 edgesを
生成した。`codegraph status .`はindexがup to dateであると確認した。

`codegraph explore`で`encodeAndSubmitCompute`からsubmission accountingまでのcall
pathと`Context.Release`を取得できることをsmoke testした。移動前後の`plan.md`の
SHA-256が一致し、内容がbyte-identicalであることを確認した。Graphify生成reportの
空行だけをMarkdown formatterで整え、hand-written notesと生成reportのMarkdown
lintおよび`git diff --check`が成功した。

Maintainerは、Codexのquota resetまで別のAI Agentが`go-wgpu-mat`の残作業を進め、
Codex復帰時に進捗を即座に把握できるhandover環境にするよう指示した。Managed
sub-agentは存在せず、この作業で起動した外部workerも終了済みと確認した。VS Codeの
Copilot常駐processは本作業のsub-agentではないため停止対象外とした。

Fresh Agent向けにsession start/close-out規約、task ID、live担当欄、次command、
dirty tree、background worker状態を追加した。次Agentによるread-only調査、詳細計画、
test-first local実装、Metal検証、phase local commitを許可済みとし、push/tag/releaseは
引き続き禁止した。

Session close-out規約に従いGraphifyとCodeGraphを再同期した。Graphifyは561 nodes、
1,575 edges、17 communitiesとなり、community set変更により一部labelをhub名へ
自動更新した。LLM label refreshは必須でないため実行していない。CodeGraphは変更を
検出せず、32 files、575 nodes、2,419 edgesのup-to-date状態を維持した。

## 2026-07-31 SL-001/SL-002(Kimi Code CLI)

Maintainer指示によりKimi Code CLIがhandoverをintakeした。Handover一式
(`AGENTS.md`、`.agents/`、index、`plan.md`移動)をlocal commit `7502d8e`として
記録した(push禁止のため`main`は`origin/main`より1 ahead)。

SL-001としてsubmission pathをread-only inventoryした。Submission pointは
`encodeAndSubmitCompute`、readbackの`readBuffer`、`WriteBuffer`の3系統で、
完了追跡機構はcodebaseに存在しないことを確認した。Inventoryは`findings.md`へ
記録した。

SL-002として、pin済み`gogpu/wgpu v0.30.22`のmodule sourceを直接読み、ownership
contractを確認した(read-only sub-agent調査のload-bearing claimはすべてprimary
sourceで再検証)。Submit後の`CommandBuffer.Release`はno-op、buffer類は
ResourceRef refcountでGPU完了まで生存することが確認できた。一方、v0.30.22の
`BindGroup.Release`はResourceRefをbypassし、upstreamがuse-after-freeとして
v0.30.28で修正したpatternに合致すること、pin版の`Device.Release`内部drainが
released flagによりno-opであること(v0.30.23で修正)を確認した。これらを
`findings.md`の「Module contract facts」にfile/line証拠付きで記録し、詳細計画を
`.agents/plan-submission-lifetime.md`として作成した。旧SL-004の「command buffer
だけ保持する切り分け」はmodule挙動を変えないため対象外とし、SL-004の第1 rungを
v0.30.29 upgrade仮説testへ置き換えた。

変更前baselineとして`CGO_ENABLED=0 go test -timeout=60s -parallel=1 -cover ./...`
がGREEN(95.0% coverage)であることを確認した。Production codeは未変更である。

## 2026-07-31 SL-003〜SL-008(Kimi Code CLI)

SL-003として`mat/sl_contract_test.go`を追加した。`TestSLMetalChainedCompute`は
go-nnのdevice-resident backward形状を模倣し、256 round×6 opの依存chainを
中間readbackなしで回して最後に1回だけReadする。accumulationはping-pong buffer、
中間matrixはroundごとに生成・解放する。`TestSLSoftwareChainedCompute`はUseCPUの
参照controlとして常にGREENである。

Pin版(v0.30.22)でのRED確認: 3/3回、同一PCでSIGSEGVを再現した。crash siteは
`hal/metal.(*AutoreleasePool).Drain`←`CommandEncoder.BeginEncoding`←
`Device.CreateCommandEncoder`で、後続opのencoder作成がFFI経由で落ちる。

SL-004第1 rungとして`go.mod`をv0.30.29へ上げて再実行したところ、`-count=1`、
`-count=10`、`GO_WGPU_MAT_SL_ROUNDS=1024 -count=3`のすべてでGREENとなった。
RED-before/GREEN-afterと十分な反復が揃い、H1(pin版`BindGroup.Release`の
use-after-free、upstream ADR-056)を有力な原因候補として採用した。

SL-005の修正はpin bump(`github.com/gogpu/wgpu` v0.30.22→v0.30.29、伴ってgoffi
v0.6.2、gpucontext v0.23.0、naga v0.17.16、webgpu v0.5.4)とし、production
codeの変更は無い。SL-006は`TestSLMetalReleaseWithInflightWork`を追加し、
in-flight work下でのmatrix/context解放と二重Releaseのidempotencyを検証した。

SL-007/008の検証結果: `TestP4MetalKernels` PASS、`make test`(CGO=0/1、race、
checkptr=0)GREEN(95.0% coverage)、`make lint` 0 issues、`make fuzz`両target
PASS、software race selector GREEN、Metal `-count=10`と1024 round×3のstress
GREEN。lint指摘2件(G115、nonamedreturns)は`slRoundCount`を`uint64`返しへ変更し、
`slInputData`のnamed returnを除去して解消した。`golangci-lint run --fix`は
整形のみを変更した。

## 2026-07-31 SL-009 reviewとcommit(Kimi Code CLI)

Read-only reviewをcopilot、hermes、codexの3 reviewerで実施した(claude、agy、
piはpingで利用不可)。codexの初回呼び出しはCLI引数形状の不整合で失敗したため、
packet modeへ切り替えた。

Iteration 1: hermesとcodexはAGREED。copilotは2件のBLOCKINGを報告した。
検証の結果、(1)`runSLChainedRound`の中間matrixが`newP4Matrix`の`t.Cleanup`と
`defer`で二重releaseされる点は、`Matrix.Release`がCASでidempotentと確認済みの
ため実害はないが、直接`mat.NewMatrix`+`defer`へ変更してroundごとexactly onceと
した(採用)。(2)inflight testのcleanupがreleased contextへ発火するという指摘は、
LIFO順とidempotent no-opにより発火しないことを確認し却下した。あわせて、
Q4の従属論点としてmatrixの二重releaseと`Released()`を明示assertへ変更した
(部分採用)。

修正後にSL software(CGO=0)、SL Metal `-count=3`、lintを再実行しGREEN。
Iteration 2で3 reviewer全員がAGREEDし、consensusに達した。Review後のmutation
checkではworktreeにreviewer由来の変更はなかった。

追加検証: `TestSLMetal*`は`-race -gcflags=all=-d=checkptr=0`でGREEN。
checkptr有効の`-race`ではv0.30.29でも`hal/metal.newGPUCompletionBlock`の
trampolineでcheckptr fatalとなることを確認し、Makefileのcheckptr=0は維持とした
(findings.mdに記録)。`go mod verify`は全module verified。

最終gate: `make test`両mode GREEN(95.0% coverage)、P4+SL Metal selector GREEN、
software race selector GREEN、lint 0 issues、fuzz両target PASS。

Session close-out: 修正一式をcommit `ba3ea83`(`fix: bump gogpu/wgpu to v0.30.29
and add submission lifetime regression tests`)として記録した。`codegraph sync .`と
`graphify update .`を再実行し、Graphifyは591 nodes、1,639 edges、25 communities
へ更新された。Index再同期とhash記録は別のchore commitへ分けた。`main`は
`origin/main`より2 commit aheadで、権限境界によりpushしていない。
SL-010はMaintainerのpush/tag後に開始する。

## 2026-07-31 independent review(Codex)

Maintainerの依頼により、Kimi Code CLIが作成した`origin/main..70299ca`の3 commitを
独立にreviewした。Production差分はdependency pin bumpのみで、追加された
submission-lifetime test、pin済みv0.30.29のlifecycle実装、依存graphを照合した。

Validationは`go mod tidy -diff`、`go build ./...`、`make test`、Metal selector
`-count=10`、`go vet ./...`、`golangci-lint run`、Markdown lint、`make fuzz`、
`git diff --check`がすべて成功した。Graphify queryが生成した未追跡の
`graphify-out/cache/last_query_stamp`はreview副産物として削除した。

Codeまたはdependencyのblocking issueは見つからなかった。一方、live handoverの
`.agents/status.md`が`70299ca`適用前の状態を示し、`.agents/findings.md`のConfirmed
factsが旧v0.30.22 baselineを現在形で示す不整合を確認した。`RV-001`と`RV-002`として
`.agents/review.md`へ記録し、Kimi Code CLIによるremediation待ちとした。

## 2026-07-31 review remediation(Kimi Code CLI)

`RV-001`と`RV-002`に対応した。`status.md`は「Reviewed base」表現へ置き換え、
commitすると必ず偽になる`HEAD`項目を廃止して、現HEAD/ahead数はgit commandを
正とする注記を追加した。Active handoverはremediation完了・再検証待ちの状態へ
同期した。`findings.md`は「Confirmed facts(現行状態)」と「Historical
baseline(v0.30.22、RED-before evidence)」へ分割し、現行pin v0.30.29と
`wgpu.Device.Release`によるtransitive drainを正確に記録した。

`RV-003`としてacceptance checksを再実行した。`rg`による走査で`findings.md`の
全`v0.30.22`言及が明示的にhistoricalであること、`markdownlint-cli2
".agents/*.md"`が0 issues(10 files)、`git diff --check`が成功することを確認した。
結果は`.agents/review.md`の「Remediation」節へ記録した。Maintainer/Codexの
再検証待ちであり、push/tagと`SL-010`へは進まない。

## 2026-07-31 review close-out(Codex)

Kimi Code CLIのremediation commit `7ba11af`(`docs: resolve review issues RV-001 and
RV-002 in repo-local notes`)を再検証した。変更対象は`.agents/`の6 Markdown file
だけで、Go code、test、dependency、generated indexの変更はない。

`RV-001`はlive statusをreviewed-base方式へ変更し、正確な現HEAD/ahead数をGit
commandへ委ねることで解消した。`RV-002`は現行v0.30.29の事実と旧v0.30.22の
historical RED-before evidenceを分離し、現行drainが`wgpu.Device.Release`から
transitiveに提供されることを明記して解消した。

Acceptance checksとして`rg`によるpin/drain記述の確認、`markdownlint-cli2
'.agents/*.md'`、`git diff --check 70299ca..7ba11af`、CodeGraph statusを再実行し、
すべて成功した。未解決issueがないため`.agents/review.md`を削除し、完了済みreview
taskの一時sectionを`tasks.md`から除去した。Reviewの経緯は本worklogへ保存する。
