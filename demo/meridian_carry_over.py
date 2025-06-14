#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Meridian Enhanced Adstock Parameters デモ

このスクリプトでは、Meridianライブラリの新しい拡張アドストックパラメータを使用した
高度なcarryover効果のモデリング方法を説明します。

拡張アドストック機能:
- peak_delay: 効果のピーク時点を制御
- exponent: アドストックカーブの形状を制御
- alpha: 従来の減衰率パラメータ

新しいアドストック数式 (LightweightMMM互換):
w_i = alpha^(((i - peak_delay)^2) / exponent)

これにより、チャンネル別に異なるキャリーオーバーパターンをモデル化できます。
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp

# Meridianライブラリのインポート
from meridian import constants
from meridian.data import load
from meridian.model import model
from meridian.model import spec
from meridian.model import prior_distribution
from meridian.analysis import analyzer
from meridian.analysis import visualizer

def check_environment():
    """
    実行環境の確認
    """
    print("=== 実行環境の確認 ===")
    print(f"TensorFlow バージョン: {tf.__version__}")
    print(f"利用可能なGPU数: {len(tf.config.experimental.list_physical_devices('GPU'))}")
    print(f"利用可能なCPU数: {len(tf.config.experimental.list_physical_devices('CPU'))}")
    print()

def load_geo_data():
    """
    Geoレベルデータの読み込み
    """
    print("=== Geoレベルデータの読み込み ===")

    # 列名とデータタイプのマッピング
    coord_to_columns = load.CoordToColumns(
        time='time',
        geo='geo',
        controls=['sentiment_score_control', 'competitor_sales_control'],
        population='population',
        kpi='conversions',
        revenue_per_kpi='revenue_per_conversion',
        media=[
            'Channel0_impression',
            'Channel1_impression',
            'Channel2_impression',
            'Channel3_impression',
            'Channel4_impression',
        ],
        media_spend=[
            'Channel0_spend',
            'Channel1_spend',
            'Channel2_spend',
            'Channel3_spend',
            'Channel4_spend',
        ],
        organic_media=['Organic_channel0_impression'],
        non_media_treatments=['Promo'],
    )

    # メディア変数とチャネル名のマッピング
    media_to_channel = {
        'Channel0_impression': 'Channel_0',
        'Channel1_impression': 'Channel_1',
        'Channel2_impression': 'Channel_2',
        'Channel3_impression': 'Channel_3',
        'Channel4_impression': 'Channel_4',
    }

    media_spend_to_channel = {
        'Channel0_spend': 'Channel_0',
        'Channel1_spend': 'Channel_1',
        'Channel2_spend': 'Channel_2',
        'Channel3_spend': 'Channel_3',
        'Channel4_spend': 'Channel_4',
    }

    # データの読み込み
    data_path = "meridian/data/simulated_data/csv/geo_all_channels.csv"

    loader = load.CsvDataLoader(
        csv_path=data_path,
        kpi_type='non_revenue',
        coord_to_columns=coord_to_columns,
        media_to_channel=media_to_channel,
        media_spend_to_channel=media_spend_to_channel,
    )

    data = loader.load()
    print(f"データ形状: {data.kpi.shape}")
    print(f"地域数: {data.kpi.shape[0]}")
    print(f"時間期間数: {data.kpi.shape[1]}")
    print(f"メディアチャネル数: {data.media.shape[2] if len(data.media.shape) > 2 else data.media.shape[1]}")
    print()

    return data

def create_model_specs():
    """
    異なるアドストックパラメータ設定でモデル仕様を作成
    新しい拡張アドストックパラメータ（peak_delay, exponent）を含む
    """
    print("=== 拡張アドストックパラメータを使用したモデル仕様の作成 ===")

    # 1. デフォルトの拡張アドストック設定
    enhanced_prior_default = prior_distribution.PriorDistribution(
        # 従来のパラメータ
        roi_m=tfp.distributions.LogNormal(0.2, 0.9, name=constants.ROI_M),
        alpha_m=tfp.distributions.Uniform(0.1, 0.8, name=constants.ALPHA_M),

        # 新しい拡張アドストックパラメータ
        peak_delay_m=tfp.distributions.TruncatedNormal(
            loc=np.array([2.0, 1.0, 3.0, 0.5, 2.5]),  # チャンネル別のピーク遅延
            scale=np.array([1.0, 0.5, 1.5, 0.3, 1.0]),
            low=0.0,
            high=8.0,
            name=constants.PEAK_DELAY_M
        ),

        exponent_m=tfp.distributions.TruncatedNormal(
            loc=np.array([2.0, 1.5, 2.5, 1.0, 1.8]),  # チャンネル別のカーブ形状
            scale=np.array([0.5, 0.3, 0.8, 0.2, 0.4]),
            low=0.5,
            high=5.0,
            name=constants.EXPONENT_M
        )
    )

    # 2. TV特化設定（長いピーク遅延、広いカーブ）
    tv_focused_prior = prior_distribution.PriorDistribution(
        roi_m=tfp.distributions.LogNormal(0.2, 0.9, name=constants.ROI_M),
        alpha_m=tfp.distributions.Uniform(0.2, 0.7, name=constants.ALPHA_M),

        peak_delay_m=tfp.distributions.TruncatedNormal(
            loc=np.array([4.0, 1.0, 5.0, 0.5, 3.0]),  # TV系は長い遅延
            scale=np.array([1.5, 0.5, 2.0, 0.3, 1.0]),
            low=0.0,
            high=12.0,
            name=constants.PEAK_DELAY_M
        ),

        exponent_m=tfp.distributions.TruncatedNormal(
            loc=np.array([3.0, 1.5, 3.5, 1.0, 2.5]),  # TV系は広いカーブ
            scale=np.array([0.8, 0.3, 1.0, 0.2, 0.5]),
            low=1.0,
            high=6.0,
            name=constants.EXPONENT_M
        )
    )

    # 3. デジタル特化設定（短いピーク遅延、鋭いカーブ）
    digital_focused_prior = prior_distribution.PriorDistribution(
        roi_m=tfp.distributions.LogNormal(0.2, 0.9, name=constants.ROI_M),
        alpha_m=tfp.distributions.Uniform(0.1, 0.6, name=constants.ALPHA_M),

        peak_delay_m=tfp.distributions.TruncatedNormal(
            loc=np.array([1.0, 0.5, 1.5, 0.2, 1.0]),  # デジタル系は短い遅延
            scale=np.array([0.5, 0.2, 0.8, 0.1, 0.3]),
            low=0.0,
            high=4.0,
            name=constants.PEAK_DELAY_M
        ),

        exponent_m=tfp.distributions.TruncatedNormal(
            loc=np.array([1.2, 1.0, 1.5, 0.8, 1.1]),  # デジタル系は鋭いカーブ
            scale=np.array([0.3, 0.2, 0.4, 0.1, 0.2]),
            low=0.5,
            high=2.5,
            name=constants.EXPONENT_M
        )
    )

    # 4. 従来のアドストック（比較用）
    traditional_prior = prior_distribution.PriorDistribution(
        roi_m=tfp.distributions.LogNormal(0.2, 0.9, name=constants.ROI_M),
        alpha_m=tfp.distributions.Uniform(0.1, 0.8, name=constants.ALPHA_M)
        # peak_delay と exponent は使用しない（デフォルト値が適用される）
    )

    # モデル仕様の作成
    model_specs = {
        'Enhanced Default': spec.ModelSpec(
            prior=enhanced_prior_default,
            max_lag=8
        ),
        'TV Focused (Long Peak)': spec.ModelSpec(
            prior=tv_focused_prior,
            max_lag=12
        ),
        'Digital Focused (Short Peak)': spec.ModelSpec(
            prior=digital_focused_prior,
            max_lag=6
        ),
        'Traditional Adstock': spec.ModelSpec(
            prior=traditional_prior,
            max_lag=8
        ),
    }

    print(f"作成したモデル仕様数: {len(model_specs)}")
    for name in model_specs.keys():
        print(f"  - {name}")

    print("\n拡張アドストックパラメータの特徴:")
    print("  Enhanced Default: バランスの取れた設定")
    print("  TV Focused: 長いピーク遅延、広いカーブ（TV広告向け）")
    print("  Digital Focused: 短いピーク遅延、鋭いカーブ（デジタル広告向け）")
    print("  Traditional: 従来のアドストック（比較用）")
    print()

    return model_specs

def train_models(data, model_specs, quick_mode=True):
    """
    各carryover設定でモデルを訓練

    Args:
        data: 入力データ
        model_specs: モデル仕様の辞書
        quick_mode: 高速モードでの実行（デモ用に短縮）
    """
    print("=== モデルの訓練 ===")

    models = {}

    # 高速モード用のパラメータ
    if quick_mode:
        n_chains = 4
        n_adapt = 500
        n_burnin = 250
        n_keep = 500
        prior_samples = 250
    else:
        n_chains = 10
        n_adapt = 2000
        n_burnin = 500
        n_keep = 1000
        prior_samples = 500

    for name, model_spec in model_specs.items():
        print(f"\n--- {name} モデルの訓練開始 ---")

        try:
            # Meridianモデルの初期化
            mmm = model.Meridian(input_data=data, model_spec=model_spec)

            # 事前分布からのサンプリング
            print("事前分布からサンプリング中...")
            mmm.sample_prior(prior_samples)

            # 事後分布からのサンプリング
            print("事後分布からサンプリング中...")
            mmm.sample_posterior(
                n_chains=n_chains,
                n_adapt=n_adapt,
                n_burnin=n_burnin,
                n_keep=n_keep,
                seed=42
            )

            models[name] = mmm
            print(f"{name} モデルの訓練完了")

        except Exception as e:
            print(f"モデル訓練エラー: {e}")
            print(f"{name} モデルの訓練をスキップします")
            continue

    print(f"\n全{len(models)}モデルの訓練が完了しました。")
    return models

def analyze_convergence(models):
    """
    モデルの収束性分析
    """
    print("=== 収束性分析 ===")

    convergence_results = {}

    for name, mmm in models.items():
        print(f"\n--- {name} モデルの収束性 ---")

        # R-hat統計量の計算
        model_diagnostics = visualizer.ModelDiagnostics(mmm)

        # R-hat値の取得（可視化なしで統計のみ）
        try:
            # R-hat統計量の表示
            print("R-hat統計量（1.0に近いほど良好な収束）:")

            # 主要パラメータのR-hat値を確認
            rhat_summary = model_diagnostics._get_rhat_summary()
            if rhat_summary is not None:
                print(f"  平均R-hat: {rhat_summary.get('avg_rhat', 'N/A'):.3f}")
                print(f"  最大R-hat: {rhat_summary.get('max_rhat', 'N/A'):.3f}")
                print(f"  不良R-hat割合: {rhat_summary.get('percent_bad_rhat', 'N/A'):.1f}%")

                convergence_results[name] = rhat_summary
            else:
                print("  R-hat統計量を取得できませんでした")
                convergence_results[name] = None

        except Exception as e:
            print(f"  収束性分析でエラー: {e}")
            convergence_results[name] = None

    return convergence_results

def analyze_model_fit(models):
    """
    モデルの適合度分析
    """
    print("=== モデル適合度分析 ===")

    fit_results = {}

    for name, mmm in models.items():
        print(f"\n--- {name} モデルの適合度 ---")

        try:
            model_fit = visualizer.ModelFit(mmm)

            # 適合度メトリクスの取得
            fit_metrics = model_fit._get_model_fit_metrics()

            if fit_metrics is not None:
                print("適合度メトリクス:")
                for metric_name, value in fit_metrics.items():
                    if isinstance(value, (int, float)):
                        print(f"  {metric_name}: {value:.4f}")
                    else:
                        print(f"  {metric_name}: {value}")

                fit_results[name] = fit_metrics
            else:
                print("  適合度メトリクスを取得できませんでした")
                fit_results[name] = None

        except Exception as e:
            print(f"  適合度分析でエラー: {e}")
            fit_results[name] = None

    return fit_results

def analyze_media_effects(models):
    """
    メディア効果の分析（拡張アドストックパラメータを含む）
    """
    print("=== メディア効果分析 ===")

    media_effects = {}

    for name, mmm in models.items():
        print(f"\n--- {name} モデルのメディア効果 ---")

        try:
            # Analyzerを使用してメディア効果を分析
            mmm_analyzer = analyzer.Analyzer(mmm)

            # ROI分析
            roi_result = mmm_analyzer.roi()
            if roi_result is not None:
                print("ROI分析結果:")
                roi_df = roi_result.to_dataframe()
                print(roi_df.round(3))

                # 拡張アドストックパラメータの分析
                adstock_params = analyze_enhanced_adstock_parameters(mmm)

                media_effects[name] = {
                    'roi': roi_df,
                    'adstock_params': adstock_params
                }
            else:
                print("  ROI分析結果を取得できませんでした")
                media_effects[name] = None

        except Exception as e:
            print(f"  メディア効果分析でエラー: {e}")
            media_effects[name] = None

    return media_effects

def analyze_enhanced_adstock_parameters(mmm):
    """
    拡張アドストックパラメータの分析
    """
    adstock_results = {}

    try:
        # 事後分布から拡張アドストックパラメータを取得
        if hasattr(mmm.inference_data, 'posterior'):
            posterior = mmm.inference_data.posterior

            # Peak delay パラメータの分析
            if constants.PEAK_DELAY_M in posterior.data_vars:
                peak_delay_samples = posterior[constants.PEAK_DELAY_M]
                peak_delay_mean = peak_delay_samples.mean(['chain', 'draw']).values
                peak_delay_std = peak_delay_samples.std(['chain', 'draw']).values

                print("Peak Delay パラメータ:")
                for i, (mean_val, std_val) in enumerate(zip(peak_delay_mean, peak_delay_std)):
                    print(f"  チャンネル {i}: {mean_val:.2f} ± {std_val:.2f} 週")

                adstock_results['peak_delay'] = {
                    'mean': peak_delay_mean,
                    'std': peak_delay_std
                }

            # Exponent パラメータの分析
            if constants.EXPONENT_M in posterior.data_vars:
                exponent_samples = posterior[constants.EXPONENT_M]
                exponent_mean = exponent_samples.mean(['chain', 'draw']).values
                exponent_std = exponent_samples.std(['chain', 'draw']).values

                print("Exponent パラメータ:")
                for i, (mean_val, std_val) in enumerate(zip(exponent_mean, exponent_std)):
                    print(f"  チャンネル {i}: {mean_val:.2f} ± {std_val:.2f}")

                adstock_results['exponent'] = {
                    'mean': exponent_mean,
                    'std': exponent_std
                }

            # Alpha パラメータの分析
            if constants.ALPHA_M in posterior.data_vars:
                alpha_samples = posterior[constants.ALPHA_M]
                alpha_mean = alpha_samples.mean(['chain', 'draw']).values
                alpha_std = alpha_samples.std(['chain', 'draw']).values

                print("Alpha パラメータ:")
                for i, (mean_val, std_val) in enumerate(zip(alpha_mean, alpha_std)):
                    print(f"  チャンネル {i}: {mean_val:.3f} ± {std_val:.3f}")

                adstock_results['alpha'] = {
                    'mean': alpha_mean,
                    'std': alpha_std
                }

    except Exception as e:
        print(f"  拡張アドストックパラメータ分析でエラー: {e}")
        return None

    return adstock_results

def compare_carryover_effects(models, media_effects):
    """
    Carryover効果の比較
    """
    print("=== Carryover効果の比較 ===")

    if not media_effects:
        print("メディア効果データがないため、比較を実行できません。")
        return

    # ROI値の比較
    roi_comparison = {}

    for name, effects in media_effects.items():
        if effects and 'roi' in effects:
            roi_df = effects['roi']
            if 'mean' in roi_df.columns:
                roi_comparison[name] = roi_df['mean'].mean()

    if roi_comparison:
        print("\n平均ROIの比較:")
        for name, avg_roi in sorted(roi_comparison.items(), key=lambda x: x[1], reverse=True):
            print(f"  {name}: {avg_roi:.3f}")

        # 最適なcarryover設定の推奨
        best_model = max(roi_comparison.items(), key=lambda x: x[1])
        print(f"\n推奨carryover設定: {best_model[0]} (平均ROI: {best_model[1]:.3f})")
    else:
        print("ROIデータが不十分なため、比較できませんでした。")

def generate_summary_report(models, convergence_results, fit_results, media_effects):
    """
    サマリーレポートの生成
    """
    print("\n" + "="*60)
    print("             CARRYOVER効果分析 サマリーレポート")
    print("="*60)

    print(f"\n分析対象モデル数: {len(models)}")

    print("\n【モデル設定】")
    for name, mmm in models.items():
        model_spec = mmm.model_spec
        print(f"  {name}:")
        print(f"    - Carryover Type: {model_spec.carryover_transform_type}")
        if model_spec.carryover_transform_type == 'adstock':
            print(f"    - Max Lag: {model_spec.carryover_max_lag}")
        elif model_spec.carryover_transform_type == 'geometric':
            print(f"    - Decay Rate: {model_spec.carryover_decay_rate}")

    print("\n【収束性評価】")
    for name, result in convergence_results.items():
        if result:
            status = "良好" if result.get('max_rhat', 2.0) < 1.2 else "要注意"
            print(f"  {name}: {status} (最大R-hat: {result.get('max_rhat', 'N/A'):.3f})")
        else:
            print(f"  {name}: 評価不可")

    print("\n【推奨事項】")
    print("1. R-hat < 1.2 のモデルを使用してください")
    print("2. ビジネス要件に応じてcarryover期間を調整してください")
    print("3. 複数のcarryover設定でモデルを比較し、最適な設定を選択してください")

    print("\n" + "="*60)

def main():
    """
    メイン実行関数
    """
    print("Meridian Carryover効果デモを開始します...\n")

    # 1. 環境確認
    check_environment()

    # 2. データ読み込み
    try:
        data = load_geo_data()
    except Exception as e:
        print(f"データ読み込みエラー: {e}")
        print("サンプルデータの作成を試みます...")
        return

    # 3. モデル仕様作成
    model_specs = create_model_specs()

    # 4. モデル訓練（デモモードで高速実行）
    print("注意: デモモードで実行します（短時間での訓練のため精度は限定的です）")
    try:
        models = train_models(data, model_specs, quick_mode=True)
    except Exception as e:
        print(f"モデル訓練エラー: {e}")
        return

    # 5. 収束性分析
    convergence_results = analyze_convergence(models)

    # 6. 適合度分析
    fit_results = analyze_model_fit(models)

    # 7. メディア効果分析
    media_effects = analyze_media_effects(models)

    # 8. Carryover効果比較
    compare_carryover_effects(models, media_effects)

    # 9. サマリーレポート
    generate_summary_report(models, convergence_results, fit_results, media_effects)

    print("\nCarryover効果デモが完了しました。")

if __name__ == "__main__":
    main()
