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

## Working hypothesis

`Queue.Submit`完了前にcommand bufferまたはそのsubmissionが参照するresourceを
releaseしているため、後続submissionとlifetimeが競合している可能性がある。

これは未確認の仮説である。command bufferだけの保持で十分か、bind group、uniform
buffer、その他のresourceも完了まで保持する必要があるかは不明である。

## Questions to resolve

1. Pin済み`gogpu/wgpu v0.30.22`が各submitted resourceへ要求するownershipと
   release timingは何か。
2. Queue submission indexと`Device.Poll`を使って、同期的に毎operationを待たずに
   完了済みresourceだけを回収できるか。
3. `Context.Release`はoutstanding submissionをどうdrainし、error pathと二重解放を
   どう扱うべきか。
4. command buffer保持だけでRED testがGREENになるか。ならない場合、bind group、
   uniform buffer、matrix buffer aliasing、memory orderingのどれを次に検証するか。
5. Metal以外のbackendにも同じlifecycle問題があるか。

## Evidence rules

- 少数回のGREENは非決定的問題の解消証拠にしない。
- RED-before/GREEN-after、十分な反復、resource解放確認、既存testの成功を揃える。
- 仮説は、再現testと切り分け結果が揃うまで「原因」と表現しない。
