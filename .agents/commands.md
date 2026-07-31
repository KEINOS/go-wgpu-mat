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

P4のmandatory selectorとlocal Metal gateは[`plan.md`](plan.md)を参照する。

Submission lifetime作業のselectorは次のとおり(詳細は
[`plan-submission-lifetime.md`](plan-submission-lifetime.md))。

```sh
CGO_ENABLED=1 go test -race -gcflags=all=-d=checkptr=0 -cover -count=1 \
  -run '^(TestP4Software|TestSLSoftware)' ./mat
GO_WGPU_MAT_GPU=1 CGO_ENABLED=1 go test -count=1 -parallel=1 \
  -run '^(TestP4Metal|TestSLMetal)' ./mat
```

Metal stressはround数と反復を上げて実行する。

```sh
GO_WGPU_MAT_SL_ROUNDS=1024 GO_WGPU_MAT_GPU=1 CGO_ENABLED=1 \
  go test -count=3 -parallel=1 -run '^TestSLMetal' ./mat
```

Quarantined upstream repro(既定gateから除外、upstream bugが残る間FAILが期待値):

```sh
GO_WGPU_MAT_GPU=1 CGO_ENABLED=1 go test -count=10 -parallel=1 \
  -run '^TestReproMetal' ./mat
```

`TestSLSoftware*`は`UseCPU`のみを使い、hardware adapterを要求しない。
`TestSLMetal*`は`UseGPU`を要求し、`GO_WGPU_MAT_GPU=1`でadapter不在をskipでは
なくfailureとする。反復回数の目安は`-count=10`以上およびstress実行である。
