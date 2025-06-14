#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Meridian Carryover効果デモ

このスクリプトでは、Meridianライブラリにおけるcarryover効果の設定と分析方法を説明します。

Carryover効果とは、広告の効果が投下された期間だけでなく、その後の期間にも持続する効果のことです。
Meridianでは以下の2つのcarryover変換タイプをサポートしています：

1. Adstock変換: より柔軟なcarryover効果をモデル化
2. Geometric変換: 幾何級数的な減衰を仮定したシンプルなcarryover効果
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
    異なるcarryover設定でモデル仕様を作成
    """
    print("=== モデル仕様の作成 ===")

    # 共通のROI prior設定
    roi_mu = 0.2
    roi_sigma = 0.9
    prior_dist = prior_distribution.PriorDistribution(
        roi_m=tfp.distributions.LogNormal(roi_mu, roi_sigma, name=constants.ROI_M)
    )

    # 1. Adstock carryover (デフォルト設定)
    model_spec_adstock = spec.ModelSpec(
        prior=prior_dist,
        carryover_transform_type='adstock',
        carryover_max_lag=8
    )

    # 2. Adstock carryover (より長いラグ)
    model_spec_adstock_long = spec.ModelSpec(
        prior=prior_dist,
        carryover_transform_type='adstock',
        carryover_max_lag=16
    )

    # 3. Geometric carryover
    model_spec_geometric = spec.ModelSpec(
        prior=prior_dist,
        carryover_transform_type='geometric',
        carryover_decay_rate=0.3
    )

    # 4. No carryover (comparison baseline)
    model_spec_no_carryover = spec.ModelSpec(
        prior=prior_dist,
        carryover_transform_type='adstock',
        carryover_max_lag=0
    )

    model_specs = {
        'Adstock (max_lag=8)': model_spec_adstock,
        'Adstock (max_lag=16)': model_spec_adstock_long,
        'Geometric (decay=0.3)': model_spec_geometric,
        'No Carryover': model_spec_no_carryover
    }

    print(f"作成したモデル仕様数: {len(model_specs)}")
    for name in model_specs.keys():
        print(f"  - {name}")
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
    メディア効果の分析
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

                media_effects[name] = {
                    'roi': roi_df,
                }
            else:
                print("  ROI分析結果を取得できませんでした")
                media_effects[name] = None

        except Exception as e:
            print(f"  メディア効果分析でエラー: {e}")
            media_effects[name] = None

    return media_effects

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
