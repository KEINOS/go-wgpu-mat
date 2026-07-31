# Commands

Repository rootで実行する。

## Resume

```sh
git status --short --branch
git log -5 --oneline --decorate
sed -n '1,240p' .agents/status.md
sed -n '1,260p' .agents/findings.md
sed -n '1,260p' .agents/tasks.md
```

## Index

```sh
graphify update .
codegraph init .
```

初期化済みindexの状態確認は次を使う。

```sh
codegraph status .
```

Index作成後の更新は次を使う。

```sh
graphify update .
codegraph sync .
```

`.codegraph/codegraph.db`は`.codegraph/.gitignore`によりlocal生成物として扱われる。
Fresh cloneでdatabaseがない場合は`codegraph init .`を再実行する。

## Existing repository gates

```sh
make test
make lint
make fuzz
```

P4のmandatory selectorとlocal Metal gateは[`plan.md`](plan.md)を参照する。新しい
submission lifetime作業の追加selector、反復回数、coverage gateは詳細計画で確定する。
