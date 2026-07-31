# Repository Agent Instructions

作業を始める前に、[`.agents/README.md`](.agents/README.md)を入口として
repo-local notesを読むこと。

ライブ状態は[`.agents/status.md`](.agents/status.md)を正とし、確認済みの事実と
未確認の仮説を混同しないこと。作業中に状態、判断、検証結果、次の手順が
変わった場合は、同じターンでrepo-local notesを更新すること。

作業終了前に`.agents/README.md`の「Session close-out」を実行し、次のAgentが
`status.md`だけで現在地を把握できる状態にすること。未commit変更、検証結果、
blocker、次の具体的commandを記録せずにセッションを終了しないこと。

Push、tag作成、releaseはMaintainerの明示的な許可なしに行わないこと。
