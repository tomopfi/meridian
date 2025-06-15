# Meridian 拡張アドストックパラメータ機能

## 概要

このドキュメントでは、Meridianライブラリに追加された拡張アドストックパラメータ機能について説明します。この機能により、従来のアドストック変換をより柔軟で高度なcarryover効果のモデリングが可能になります。

## 新機能

### 拡張アドストックパラメータ

従来のアドストック変換に加えて、以下の新しいパラメータが追加されました：

- **`peak_delay`**: 効果のピーク時点を制御するパラメータ
- **`exponent`**: アドストックカーブの形状を制御するパラメータ

### 新しいアドストック数式

```
w_i = alpha^(((i - peak_delay)^2) / exponent)
```

ここで：
- `w_i`: ラグ`i`におけるアドストック重み
- `alpha`: 従来の減衰率パラメータ
- `peak_delay`: 効果のピーク遅延（非負値）
- `exponent`: カーブ形状パラメータ（正値）

## 実装されたファイル

### 1. 定数定義 (`meridian/constants.py`)
- `PEAK_DELAY_M`, `EXPONENT_M`などの新しいパラメータ定数を追加
- `INFERENCE_DIMS`にパラメータディメンション情報を追加

### 2. 事前分布 (`meridian/model/prior_distribution.py`)
- 各メディアタイプ（media, RF, organic media, organic RF）向けの拡張パラメータ事前分布を追加
- broadcast機能で自動的に適切な形状に変換

### 3. アドストック変換 (`meridian/model/adstock_hill.py`)
- 新しいアドストック数式を実装
- 型安全性を向上（float32/float64の統一）
- `AdstockTransformer`、`carryover_adstock`関数を更新

### 4. モデル統合 (`meridian/model/model.py`)
- `adstock_hill_media`および`adstock_hill_rf`関数に新しいパラメータを統合
- カウンターファクトual分析でも新しいパラメータを使用

### 5. MCMC サンプリング
- `prior_sampler.py`: 事前分布からの新しいパラメータサンプリング
- `posterior_sampler.py`: 事後分布での新しいパラメータ処理

## 使用方法

### 基本的な使用例

```python
import tensorflow_probability as tfp
from meridian import constants
from meridian.model import prior_distribution, spec

# 拡張アドストックパラメータを含む事前分布
enhanced_prior = prior_distribution.PriorDistribution(
    # 従来のパラメータ
    roi_m=tfp.distributions.LogNormal(0.2, 0.9, name=constants.ROI_M),
    alpha_m=tfp.distributions.Uniform(0.1, 0.8, name=constants.ALPHA_M),

    # 新しい拡張アドストックパラメータ
    peak_delay_m=tfp.distributions.HalfNormal(2.0, name=constants.PEAK_DELAY_M),
    exponent_m=tfp.distributions.HalfNormal(1.0, name=constants.EXPONENT_M)
)

# モデル仕様の作成
model_spec = spec.ModelSpec(
    prior=enhanced_prior,
    max_lag=8
)
```

### チャンネル別設定例

```python
# 異なるチャンネルタイプに合わせた設定
peak_delay_m = tfp.distributions.TruncatedNormal(
    loc=[2.0, 1.0, 3.0],  # TV, Digital, Radio
    scale=[1.0, 0.5, 1.5],
    low=0.0,
    high=8.0,
    name=constants.PEAK_DELAY_M
)

exponent_m = tfp.distributions.TruncatedNormal(
    loc=[2.0, 1.5, 2.5],  # カーブ形状
    scale=[0.5, 0.3, 0.8],
    low=0.5,
    high=5.0,
    name=constants.EXPONENT_M
)
```

## デモスクリプト

### 1. 基本デモ
- `demo/meridian_carry_over_simple.py`: 拡張アドストックパラメータの基本的な使用方法

### 2. 完全デモ
- `demo/meridian_carry_over.py`: 複数のモデル設定での比較分析

### 3. デバッグテスト
- `debug_test.py`: 機能の動作確認とテスト

## 実行結果例

```
=== 拡張アドストックパラメータの分析 ===
Peak Delay パラメータ:
  チャンネル 0: 1.82 ± 0.83 週
  チャンネル 1: 0.86 ± 0.50 週
  チャンネル 2: 3.10 ± 1.10 週

Exponent パラメータ:
  チャンネル 0: 2.07 ± 0.55
  チャンネル 1: 1.47 ± 0.27
  チャンネル 2: 2.37 ± 0.79

Alpha パラメータ:
  チャンネル 0: 0.488 ± 0.159
  チャンネル 1: 0.431 ± 0.207
  チャンネル 2: 0.430 ± 0.222
```

## 利点

### 1. 柔軟性の向上
- チャンネル別に異なるcarryover効果パターンをモデル化
- TV広告（長い遅延）、デジタル広告（短い遅延）などに対応

### 2. 現実的なモデリング
- 効果のピーク時点を明示的にモデル化
- 非対称なadstockカーブを表現可能

### 3. 分析の深化
- より詳細なmedia attribution分析
- channel-specificなinsightの抽出

## 注意事項

### 1. 型の一貫性
- 事前分布定義時は`tf.float32`型を使用
- NumPy配列よりもTensorFlowテンソルを推奨

### 2. パラメータ制約
- `peak_delay`は非負値
- `exponent`は正値
- 適切な事前分布の設定が重要

### 3. 計算コスト
- 拡張パラメータにより計算時間が増加
- 小さなサンプルサイズでのテストを推奨

## 今後の拡張予定

1. **可視化機能**: アドストックカーブの可視化
2. **自動最適化**: チャンネルタイプに基づく自動パラメータ設定
3. **検証機能**: パラメータ値の妥当性チェック

## サポート

問題や質問がある場合は、以下のデモスクリプトを参考にしてください：
- `demo/meridian_carry_over_simple.py`（推奨：基本機能テスト）
- `debug_test.py`（トラブルシューティング用）

---

**更新日**: 2025年6月15日
**対応バージョン**: TensorFlow 2.18.1, TensorFlow Probability 0.24.0
