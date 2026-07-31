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
