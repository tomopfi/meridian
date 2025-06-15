#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Meridian Enhanced Adstock Parameters デモ（簡略版）

このスクリプトでは、Meridianライブラリの新しい拡張アドストックパラメータを使用した
基本的なcarryover効果のモデリング方法を説明します。
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
        ],
        media_spend=[
            'Channel0_spend',
            'Channel1_spend',
            'Channel2_spend',
        ],
    )

    # メディア変数とチャネル名のマッピング
    media_to_channel = {
        'Channel0_impression': 'Channel_0',
        'Channel1_impression': 'Channel_1',
        'Channel2_impression': 'Channel_2',
    }

    media_spend_to_channel = {
        'Channel0_spend': 'Channel_0',
        'Channel1_spend': 'Channel_1',
        'Channel2_spend': 'Channel_2',
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

def create_enhanced_model_spec():
    """
    拡張アドストックパラメータを使用したモデル仕様を作成
    """
    print("=== 拡張アドストックパラメータを使用したモデル仕様の作成 ===")

    # 拡張アドストックパラメータを含む事前分布
    enhanced_prior = prior_distribution.PriorDistribution(
        # 従来のパラメータ
        roi_m=tfp.distributions.LogNormal(
            tf.constant(0.2, dtype=tf.float32),
            tf.constant(0.9, dtype=tf.float32),
            name=constants.ROI_M
        ),
        alpha_m=tfp.distributions.Uniform(
            tf.constant(0.1, dtype=tf.float32),
            tf.constant(0.8, dtype=tf.float32),
            name=constants.ALPHA_M
        ),

        # 新しい拡張アドストックパラメータ
        peak_delay_m=tfp.distributions.TruncatedNormal(
            loc=tf.constant([2.0, 1.0, 3.0], dtype=tf.float32),  # チャンネル別のピーク遅延
            scale=tf.constant([1.0, 0.5, 1.5], dtype=tf.float32),
            low=tf.constant(0.0, dtype=tf.float32),
            high=tf.constant(8.0, dtype=tf.float32),
            name=constants.PEAK_DELAY_M
        ),

        exponent_m=tfp.distributions.TruncatedNormal(
            loc=tf.constant([2.0, 1.5, 2.5], dtype=tf.float32),  # チャンネル別のカーブ形状
            scale=tf.constant([0.5, 0.3, 0.8], dtype=tf.float32),
            low=tf.constant(0.5, dtype=tf.float32),
            high=tf.constant(5.0, dtype=tf.float32),
            name=constants.EXPONENT_M
        )
    )

    # モデル仕様の作成
    model_spec = spec.ModelSpec(
        prior=enhanced_prior,
        max_lag=8
    )

    print(f"拡張アドストックパラメータを含むモデル仕様を作成しました")
    print("新しいアドストック数式: w_i = alpha^(((i - peak_delay)^2) / exponent)")
    print()

    return model_spec

def train_model(data, model_spec, quick_mode=True):
    """
    モデルを訓練

    Args:
        data: 入力データ
        model_spec: モデル仕様
        quick_mode: 高速モードでの実行（デモ用に短縮）
    """
    print("=== モデルの訓練 ===")

    # 高速モード用のパラメータ
    if quick_mode:
        n_chains = 2
        n_adapt = 50
        n_burnin = 50
        n_keep = 50
        prior_samples = 50
    else:
        n_chains = 4
        n_adapt = 1000
        n_burnin = 500
        n_keep = 1000
        prior_samples = 500

    print("拡張アドストックモデルの訓練開始...")

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

    print("モデルの訓練が完了しました。")
    return mmm

def analyze_enhanced_adstock_parameters(mmm):
    """
    拡張アドストックパラメータの分析
    """
    print("=== 拡張アドストックパラメータの分析 ===")

    adstock_results = {}

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

    return adstock_results

def main():
    """
    メイン実行関数
    """
    print("Meridian 拡張Adstock パラメータ デモ（簡略版）を開始します...\n")

    # 1. 環境確認
    check_environment()

    # 2. データ読み込み
    try:
        data = load_geo_data()
    except Exception as e:
        print(f"データ読み込みエラー: {e}")
        return

    # 3. 拡張アドストックパラメータを含むモデル仕様作成
    try:
        model_spec = create_enhanced_model_spec()
    except Exception as e:
        print(f"モデル仕様作成エラー: {e}")
        return

    # 4. モデル訓練（デモモードで高速実行）
    print("注意: デモモードで実行します（短時間での訓練のため精度は限定的です）")
    mmm = train_model(data, model_spec, quick_mode=True)

    # 5. 拡張アドストックパラメータの分析
    try:
        adstock_results = analyze_enhanced_adstock_parameters(mmm)
        print("\n拡張アドストックパラメータの分析が完了しました。")
    except Exception as e:
        print(f"パラメータ分析エラー: {e}")

    print("\n=== デモ完了 ===")
    print("拡張アドストックパラメータ（peak_delay、exponent）を使用した")
    print("高度なcarryover効果のモデリングが正常に実行されました。")

if __name__ == "__main__":
    main()
