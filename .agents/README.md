# Repo Agent Notes

このディレクトリは、`go-wgpu-mat`の作業をセッション間で引き継ぐための入口で
ある。公開利用者向け文書ではなく、現在地、根拠、次の作業、権限境界を記録する。

## 読む順序

1. [`status.md`](status.md): 現在地、blocker、権限、次の一手。
2. [`findings.md`](findings.md): 確認済み事実、未確認仮説、未解決事項。
3. [`tasks.md`](tasks.md): 次の作業順と完了条件。
4. [`decisions.md`](decisions.md): 採用済み判断と禁止事項。
5. [`plan.md`](plan.md): 完了済みP4 device-resident kernel計画とclose-out。
6. [`plan-submission-lifetime.md`](plan-submission-lifetime.md): SL-002で作成した
   submission lifetimeの詳細計画。
7. [`commands.md`](commands.md): 状態確認、index更新、検証command。
8. [`worklog.md`](worklog.md): 作業経緯。

## 情報の優先順位

現在のファイル、コマンド結果、Git状態がnotesと矛盾する場合は現在の状態を優先し、
notesを同じ作業内で訂正する。Maintainerの明示的な判断は暫定判断やagentの推奨より
優先する。

## Index

- Graphify: `graphify-out/`
- CodeGraph: `.codegraph/`

コードの位置、依存、call pathを調べる場合は、`.codegraph/`が存在するため
`codegraph explore "<question>"`を`rg`や個別ファイル読み込みより先に使う。
Indexが古い場合は[`commands.md`](commands.md)の更新commandを実行する。

## Session start

1. `git status --short --branch`と`git log -5 --oneline --decorate`を確認する。
2. `status.md`のActive handover、`findings.md`、`tasks.md`を読む。
3. `.codegraph/`が利用可能なら、対象symbolとcall pathをCodeGraphで確認する。
4. 着手するtask IDと担当Agentを`status.md`へ記録してから変更を始める。

## Session close-out

作業を中断または完了する前に、同じターンで次を行う。

1. `status.md`へ現在の担当、active task、last completed、次の一手、HEAD、dirty files、
   blockerを記録する。
2. `tasks.md`のcheckboxを実際の状態へ同期する。
3. 新しい事実と仮説を`findings.md`で分離し、判断を`decisions.md`へ記録する。
4. 実行commandと結果、失敗、revertした実験を`worklog.md`へ追記する。
5. Code変更後は`graphify update .`と`codegraph sync .`を実行する。
6. `git status --short`と検証結果を再確認する。Commitした場合はhashとtitleを
   `status.md`へ記録する。

以前の作業記録を削除して現在地を表現しない。履歴は`worklog.md`へ残し、
最新状態だけを`status.md`で更新する。
