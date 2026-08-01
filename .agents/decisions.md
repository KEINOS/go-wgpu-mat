# Decisions

## D-001 Repo-local handover

- 状態: 採用
- 判断: Root `AGENTS.md`を薄い入口とし、live stateを`.agents/status.md`、証拠を
  `.agents/findings.md`、作業順を`.agents/tasks.md`へ分離する。
- 理由: 次のagentが完了済みP4計画と新しいblockerを混同せず再開できるようにする。

## D-002 Completed P4 plan location

- 状態: 採用
- 判断: Root `plan.md`を内容変更なしで`.agents/plan.md`へ移す。
- 理由: P4の履歴とclose-outを保持しつつ、rootを公開repository文書へ集中させる。

## D-003 No readback workaround

- 状態: 採用
- 判断: `Matrix.Read`、暗黙host transfer、operationごとのblocking waitを修正として
  導入しない。
- 理由: Device-resident contractと非同期submissionの目的を損なうため。

## D-004 Cause labeling

- 状態: 採用
- 判断: Early releaseはworking hypothesisとして扱い、RED testと切り分けなしに
  root causeと断定しない。
- 理由: Nondeterminismは複数のlifetime、aliasing、ordering問題と整合するため。

## D-005 Remote authority

- 状態: 採用
- 判断: Push、tag、releaseはMaintainerの明示的な許可がある場合だけ行う。

## D-006 Fix by dependency pin bump

- 状態: 採用
- 判断: Submission lifetime問題の修正は、production codeを変更せず
  `gogpu/wgpu`のpinをv0.30.22からv0.30.29へ上げることとする。
- 理由: RED testがpin版で3/3 SIGSEGVを再現し、v0.30.29で十分な反復GREENと
  なった。upstreamがuse-after-free(ADR-056、v0.30.28)とteardown drain
  (#264、v0.30.23)を修正済みであり、pin bumpが最小侵襲の修正である。
  Regression testは両versionで同じ挙動を期待するため、将来の退化を検出できる。

## D-007 Fail-fast newest upstream during investigation

- 状態: 採用
- 判断: 切り分け継続中のworking pinは最新の`gogpu/wgpu v0.30.30`とする。
- 理由: go-nn形状の破損はv0.30.29とv0.30.30で同一にREDであり、版を戻しても
  利益がない。最新版で再現確認できることがupstream issueやmodule調査の
  信頼性を高める。v0.30.30は`make test`両mode、P4/SL Metal selector、software
  race selector、lint、vet、fuzz、`go mod verify`の全gateを通過した。
  tag `v0.0.3`(= v0.30.29 pin)はそのまま維持し、go-nn側の参照は新しいtagを
  Maintainerが切るまで変えない。

## D-008 Upstream contribution boundary

- 状態: 採用
- 判断: 第三者管理のrepository(`github.com/gogpu/wgpu`を含む)へは、commit、
  push、branch、PR作成を一切行わない。commit可能なのはKEINOS管理下の
  repository(`go-wgpu-mat`、`go-nn`)だけであり、それらのpush、tag、releaseも
  D-005どおりMaintainerの明示的な許可を必要とする。
- 理由: Maintainerが2026-07-31に明示した境界。Module調査でpatch候補が出た
  場合は、diffと検証結果を報告するだけにとどめ、upstreamへの届け方(issue
  起票、patch添付、PRの要否)はMaintainerが決める。調査用のprobe copyは
  `/tmp`などrepo外に置き、`replace`は検証後に必ずrevertする。

## D-009 Fork workflow and upstream push prohibition

- 状態: 採用
- 判断: `gogpu/wgpu`の修正はKEINOS管理fork(`github.com/KEINOS/wgpu`、
  local clone `/Users/keinos/GitHub/PublicRepos/wgpu`)のbranchで行う。
  **本家`github.com/gogpu/wgpu`へのpushとPRは恒久禁止**であり、wgpu
  module関連のpushを行うときはremoteとrefspecをtriple checkする。
  ForkへのpushもMaintainerの都度明示許可とし、実行前に
  `git remote -v`で`origin`が`KEINOS/wgpu`を指すことを確認する。
  誤push防止のため、local cloneでは`upstream`のpush URLを`DISABLED`に
  設定してある(2026-08-01適用。`git remote set-url --push upstream DISABLED`。
  復旧は`git remote set-url --push upstream https://github.com/gogpu/wgpu.git`)。
- 理由: Maintainerが2026-08-01に「間違えるとやり直しが効かず、相手からの
  信頼を無くす」と明示した最優先の安全境界。警戒に頼らず構造的に防ぐ。
- 補足: D-008の「commit可能なrepository」には、Maintainerが作成した
  `KEINOS/wgpu` forkを含む(D-008の列挙をMaintainer判断で拡張)。
