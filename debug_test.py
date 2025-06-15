#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
デバッグ用の簡単なテストスクリプト
"""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

# Meridianライブラリのインポート
from meridian import constants
from meridian.data import load
from meridian.model import model
from meridian.model import spec
from meridian.model import prior_distribution

def test_simple_model():
    """
    簡単なモデルをテストして、どこでエラーが発生するかを確認
    """
    print("=== 簡単なモデルのテスト ===")

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
        ],
        media_spend=[
            'Channel0_spend',
            'Channel1_spend',
        ],
    )

    # メディア変数とチャネル名のマッピング
    media_to_channel = {
        'Channel0_impression': 'Channel_0',
        'Channel1_impression': 'Channel_1',
    }

    media_spend_to_channel = {
        'Channel0_spend': 'Channel_0',
        'Channel1_spend': 'Channel_1',
    }

    # データの読み込み
    data_path = "meridian/data/simulated_data/csv/geo_all_channels.csv"

    try:
        loader = load.CsvDataLoader(
            csv_path=data_path,
            kpi_type='non_revenue',
            coord_to_columns=coord_to_columns,
            media_to_channel=media_to_channel,
            media_spend_to_channel=media_spend_to_channel,
        )

        data = loader.load()
        print(f"データ読み込み成功: {data.kpi.shape}")

        # シンプルな事前分布を作成（拡張アドストックパラメータなし）
        simple_prior = prior_distribution.PriorDistribution(
            roi_m=tfp.distributions.LogNormal(0.2, 0.9, name=constants.ROI_M),
            alpha_m=tfp.distributions.Uniform(0.1, 0.8, name=constants.ALPHA_M),
        )

        # シンプルなモデル仕様
        simple_spec = spec.ModelSpec(
            prior=simple_prior,
            max_lag=4
        )

        print("モデル初期化を開始...")
        mmm = model.Meridian(input_data=data, model_spec=simple_spec)
        print("モデル初期化成功")

        print("事前分布サンプリングを開始...")
        mmm.sample_prior(10)
        print("事前分布サンプリング成功")

        return True

    except Exception as e:
        print(f"エラー: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_posterior_sampling():
    """
    事後分布サンプリングのテスト
    """
    print("\n=== 事後分布サンプリングのテスト ===")

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
        ],
        media_spend=[
            'Channel0_spend',
            'Channel1_spend',
        ],
    )

    # メディア変数とチャネル名のマッピング
    media_to_channel = {
        'Channel0_impression': 'Channel_0',
        'Channel1_impression': 'Channel_1',
    }

    media_spend_to_channel = {
        'Channel0_spend': 'Channel_0',
        'Channel1_spend': 'Channel_1',
    }

    # データの読み込み
    data_path = "meridian/data/simulated_data/csv/geo_all_channels.csv"

    try:
        loader = load.CsvDataLoader(
            csv_path=data_path,
            kpi_type='non_revenue',
            coord_to_columns=coord_to_columns,
            media_to_channel=media_to_channel,
            media_spend_to_channel=media_spend_to_channel,
        )

        data = loader.load()
        print(f"データ読み込み成功: {data.kpi.shape}")

        # 拡張アドストックパラメータを含む事前分布
        enhanced_prior = prior_distribution.PriorDistribution(
            roi_m=tfp.distributions.LogNormal(0.2, 0.9, name=constants.ROI_M),
            alpha_m=tfp.distributions.Uniform(0.1, 0.8, name=constants.ALPHA_M),
            peak_delay_m=tfp.distributions.HalfNormal(2.0, name=constants.PEAK_DELAY_M),
            exponent_m=tfp.distributions.HalfNormal(1.0, name=constants.EXPONENT_M)
        )

        # 拡張アドストックパラメータを含むモデル仕様
        enhanced_spec = spec.ModelSpec(
            prior=enhanced_prior,
            max_lag=4
        )

        print("拡張モデル初期化を開始...")
        mmm = model.Meridian(input_data=data, model_spec=enhanced_spec)
        print("拡張モデル初期化成功")

        print("事前分布サンプリングを開始...")
        mmm.sample_prior(10)
        print("事前分布サンプリング成功")

        print("事後分布サンプリングを開始...")
        mmm.sample_posterior(
            n_chains=2,
            n_adapt=10,
            n_burnin=10,
            n_keep=10,
            seed=42
        )
        print("事後分布サンプリング成功")

        return True

    except Exception as e:
        print(f"事後分布サンプリングエラー: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_enhanced_adstock():
    """
    拡張アドストックパラメータを使用したテスト
    """
    print("\n=== 拡張アドストックパラメータのテスト ===")

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
        ],
        media_spend=[
            'Channel0_spend',
            'Channel1_spend',
        ],
    )

    # メディア変数とチャネル名のマッピング
    media_to_channel = {
        'Channel0_impression': 'Channel_0',
        'Channel1_impression': 'Channel_1',
    }

    media_spend_to_channel = {
        'Channel0_spend': 'Channel_0',
        'Channel1_spend': 'Channel_1',
    }

    # データの読み込み
    data_path = "meridian/data/simulated_data/csv/geo_all_channels.csv"

    try:
        loader = load.CsvDataLoader(
            csv_path=data_path,
            kpi_type='non_revenue',
            coord_to_columns=coord_to_columns,
            media_to_channel=media_to_channel,
            media_spend_to_channel=media_spend_to_channel,
        )

        data = loader.load()
        print(f"データ読み込み成功: {data.kpi.shape}")

        # 拡張アドストックパラメータを含む事前分布
        enhanced_prior = prior_distribution.PriorDistribution(
            roi_m=tfp.distributions.LogNormal(0.2, 0.9, name=constants.ROI_M),
            alpha_m=tfp.distributions.Uniform(0.1, 0.8, name=constants.ALPHA_M),
            peak_delay_m=tfp.distributions.HalfNormal(2.0, name=constants.PEAK_DELAY_M),
            exponent_m=tfp.distributions.HalfNormal(1.0, name=constants.EXPONENT_M)
        )

        # 拡張アドストックパラメータを含むモデル仕様
        enhanced_spec = spec.ModelSpec(
            prior=enhanced_prior,
            max_lag=4
        )

        print("拡張モデル初期化を開始...")
        mmm = model.Meridian(input_data=data, model_spec=enhanced_spec)
        print("拡張モデル初期化成功")

        print("拡張モデル事前分布サンプリングを開始...")
        mmm.sample_prior(10)
        print("拡張モデル事前分布サンプリング成功")

        return True

    except Exception as e:
        print(f"拡張モデルエラー: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("TensorFlow バージョン:", tf.__version__)

    # まずシンプルなモデルをテスト
    simple_success = test_simple_model()

    # 次に拡張アドストックパラメータをテスト
    enhanced_success = test_enhanced_adstock()

    # 事後分布サンプリングをテスト
    posterior_success = test_posterior_sampling()

    print(f"\n=== テスト結果 ===")
    print(f"シンプルモデル: {'成功' if simple_success else '失敗'}")
    print(f"拡張モデル: {'成功' if enhanced_success else '失敗'}")
    print(f"事後分布サンプリング: {'成功' if posterior_success else '失敗'}")
