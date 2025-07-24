# ClusterX Insight Engine v7.5 (Streamlit Cloud Deployed Ver)
# Base: Ver4.0
# Integrated Features: Ver5.0 (Investment Mode, Enjoyment Mode)
# New Feature: Dynamic Analyzer Panel (v6.0 Enjoyment Mode)
# Major Update: RPT1-13 Evaluation Logic (Ver.13.0 Final)
# Refactoring: v7.5 (Final UI Adjustments for Dynamic Analyzer)
# Note: Investment history feature removed for cloud deployment.

import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime
from math import comb
from pathlib import Path
from itertools import product, combinations
import re

# ===================================================================
# 1. 定数・設定エリア
# ===================================================================

# --- パス設定 ---
# スクリプト(.py)ファイルと同じフォルダにデータがあることを前提とします
APP_DIR = Path(__file__).parent
PERFORMANCE_DB_PATH = APP_DIR / "analyzer_performance_db_v1.1_20250720.csv"


# --- RPT評価ルール設定 (Ver.13.0 Final) ---
RPT_RULES_CONFIG = {
    1: {
        'C_EXPECTANCY': 0.85, 'C_FUKUSHO_RATE': 0.10, 'D_EXPECTANCY': 0.60,
        'B_EXPECTANCY_RANGE': (0.65, 0.80),
    },
    2: {
        'D_TANSHO_EXPECTANCY_RANGE': (0.5, 0.7), 'D_BB_INDEX_THRESHOLD': 55.0,
        'D_C_NINKI_VALUES': {2, 3, 4}, 'E_EXPECTANCY': 1.15, 'AA_EXPECTANCY': 0.80,
    },
    3: {
        'B_EXPECTANCY_RANGE': (0.50, 0.60), 'C_EXPECTANCY': 0.85,
        'D_EXPECTANCY_RANGE': (0.90, 1.10), 'S_EXPECTANCY': 1.10, 'AA_EXPECTANCY': 0.90,
    },
    4: {
        'C_EXPECTANCY_RANGE': (0.80, 0.90), 'E_EXPECTANCY_RANGE': (0.90, 1.00),
        'SA_EXPECTANCY': 0.90, 'B_EXPECTANCY_RANGE': (0.70, 0.80),
    },
    5: {
        'C_EXPECTANCY': 0.90, 'D_EXPECTANCY_RANGE': (0.80, 0.85),
        'B_EXPECTANCY': 0.50, 'A_EXPECTANCY': 0.85, 'B_EXPECTANCY_RANGE': (0.70, 0.85),
    },
    6: {
        'A_TANSHO_EXPECTANCY': 0.80, 'A_FUKUSHO_EXPECTANCY': 0.65, 'SSS_EXPECTANCY': 0.85,
    },
    7: {
        'AA_A_EXPECTANCY': 0.80,
    },
    8: {
        'C_EXPECTANCY_RANGE': (0.55, 0.80), 'S_EXPECTANCY': 0.75,
    },
    9: {
        'C_EXPECTANCY_RANGE': (0.50, 0.85), 'E_EXPECTANCY': 0.85, 'S_EXPECTANCY': 0.75,
    },
    10: {
        'B_EXPECTANCY': 0.80,
    },
    11: {
        'A_EXPECTANCY': 0.65, 'S_EXPECTANCY': 0.75,
    },
    12: {
        'A_EXPECTANCY': 0.70, 'S_EXPECTANCY': 0.70,
    },
    13: {
        'A_EXPECTANCY': 0.65,
    }
}

# ===================================================================
# 2. バックエンド関数群
# ===================================================================

# --- データ読み込み関数 ---

@st.cache_data
def load_performance_db():
    """過去のパフォーマンスデータベースを読み込む"""
    if not os.path.exists(PERFORMANCE_DB_PATH):
        st.error(f"エラー: {PERFORMANCE_DB_PATH.name} が見つかりません。app.pyと同じフォルダにあるか確認してください。")
        return pd.DataFrame()
    try:
        return pd.read_csv(PERFORMANCE_DB_PATH, header=0, encoding='utf_8_sig', on_bad_lines='skip')
    except Exception as e:
        st.error(f"エラー: パフォーマンスDBの読み込み中に問題が発生しました: {e}")
        return pd.DataFrame()

# --- データ解析・評価関数 ---

def parse_markdown_race_table(markdown_text: str):
    """Markdown形式の出馬表を解析し、DataFrameとレース情報を返す"""
    lines = markdown_text.strip().splitlines()
    race_info_line = next((line for line in lines if "レース情報" in line), "")
    race_info = race_info_line.replace("🏇", "").replace("レース情報：", "").strip()
    table_lines = [line for line in lines if line.strip().startswith("|")]
    if len(table_lines) < 3: raise ValueError("Markdownテーブルの形式が不正です。ヘッダーと区切り線、データ行があるか確認してください。")
    
    header_line, data_lines = table_lines[0], table_lines[2:]
    headers = [h.strip() for h in header_line.split("|")[1:-1]]
    rows = []
    for line in data_lines:
        cells = [c.strip() for c in line.split("|")[1:-1]]
        if len(cells) == len(headers):
            rows.append(dict(zip(headers, cells)))

    df = pd.DataFrame(rows)

    if '馬番' in df.columns:
        df['馬番'] = pd.to_numeric(df['馬番'], errors='coerce')
        df.dropna(subset=['馬番'], inplace=True)
        df['馬番'] = df['馬番'].astype(int)

    if 'BB順位' in df.columns:
        df['BB順位'] = df['BB順位'].str.replace('BB', '', regex=False)
    
    if 'ランク' not in df.columns:
        st.warning("入力データに'ランク'列が見つかりませんでした。評価に影響する可能性があります。")
        df['ランク'] = 'C'

    for col in ['勝率', '連対率', '複勝率']:
        if col in df.columns: df[col] = pd.to_numeric(df[col].str.replace('%', '', regex=False), errors='coerce') / 100.0
    
    for col in ['C人気', 'BB指数', '単勝オッズ', '複勝オッズ下限']:
        if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')

    if 'メンバー推奨' not in df.columns:
        df['メンバー推奨'] = ''
        
    return df, race_info

# --- RPT別評価関数群 ---
def evaluate_rpt1_rules(df_eval: pd.DataFrame) -> pd.DataFrame:
    config = RPT_RULES_CONFIG[1]
    cond_maru = ((df_eval['総合ランク'] == 'C') & (df_eval['Web_複勝期待値'] >= config['C_EXPECTANCY']) & (df_eval['複勝率'] >= config['C_FUKUSHO_RATE'])) | \
                ((df_eval['総合ランク'] == 'D') & (df_eval['Web_複勝期待値'] < config['D_EXPECTANCY']))
    cond_sankaku = (df_eval['総合ランク'] == 'B') & df_eval['Web_複勝期待値'].between(*config['B_EXPECTANCY_RANGE'])
    df_eval['評価'] = np.select([cond_maru, cond_sankaku], ['◎', '▲'], default='')
    return df_eval

def evaluate_rpt2_rules(df_eval: pd.DataFrame) -> pd.DataFrame:
    config = RPT_RULES_CONFIG[2]
    df_eval['評価'] = ''
    is_excluded = (df_eval['RPT'] == 2) & (df_eval['総合ランク'] == 'AA')
    eligible_horses = df_eval[~is_excluded].copy()
    if not eligible_horses.empty:
        cond_tankatsu = ((eligible_horses['総合ランク'] == 'D') & eligible_horses['Web_単勝期待値'].between(*config['D_TANSHO_EXPECTANCY_RANGE']) & ((eligible_horses['BB指数'] < config['D_BB_INDEX_THRESHOLD']) | eligible_horses['C人気'].isin(config['D_C_NINKI_VALUES'])))
        cond_maru = (eligible_horses['総合ランク'] == 'E') & (eligible_horses['Web_複勝期待値'] >= config['E_EXPECTANCY'])
        cond_sankaku = ((eligible_horses['総合ランク'] == 'S') | ((eligible_horses['総合ランク'] == 'AA') & (eligible_horses['Web_複勝期待値'] < config['AA_EXPECTANCY'])))
        eligible_horses['評価'] = np.select([cond_tankatsu, cond_maru, cond_sankaku], ['◉', '◎', '▲'], default='')
        df_eval.update(eligible_horses)
    return df_eval

def evaluate_rpt3_rules(df_eval: pd.DataFrame) -> pd.DataFrame:
    config = RPT_RULES_CONFIG[3]
    cond_maru = ((df_eval['総合ランク'] == 'B') & df_eval['Web_複勝期待値'].between(*config['B_EXPECTANCY_RANGE'])) | \
                ((df_eval['総合ランク'] == 'C') & (df_eval['Web_複勝期待値'] >= config['C_EXPECTANCY'])) | \
                ((df_eval['総合ランク'] == 'D') & df_eval['Web_複勝期待値'].between(*config['D_EXPECTANCY_RANGE']))
    cond_sankaku = ((df_eval['総合ランク'] == 'S') & (df_eval['Web_複勝期待値'] >= config['S_EXPECTANCY'])) | \
                   ((df_eval['総合ランク'] == 'AA') & (df_eval['Web_複勝期待値'] >= config['AA_EXPECTANCY']))
    df_eval['評価'] = np.select([cond_maru, cond_sankaku], ['◎', '▲'], default='')
    return df_eval

def evaluate_rpt4_rules(df_eval: pd.DataFrame) -> pd.DataFrame:
    config = RPT_RULES_CONFIG[4]
    cond_maru_tankatsu_c = (df_eval['総合ランク'] == 'C') & df_eval['Web_複勝期待値'].between(*config['C_EXPECTANCY_RANGE'])
    cond_maru_e = (df_eval['総合ランク'] == 'E') & df_eval['Web_複勝期待値'].between(*config['E_EXPECTANCY_RANGE'])
    cond_maru_sa = (df_eval['総合ランク'].isin(['S', 'A'])) & (df_eval['Web_複勝期待値'] >= config['SA_EXPECTANCY'])
    cond_tankatsu_b = (df_eval['総合ランク'] == 'B') & df_eval['Web_複勝期待値'].between(*config['B_EXPECTANCY_RANGE'])
    df_eval['評価'] = np.select([cond_maru_tankatsu_c, cond_maru_e, cond_maru_sa, cond_tankatsu_b], ['◎ ◉', '◎', '◎', '◉'], default='')
    return df_eval

def evaluate_rpt5_rules(df_eval: pd.DataFrame) -> pd.DataFrame:
    config = RPT_RULES_CONFIG[5]
    cond_maru_tankatsu_c = (df_eval['総合ランク'] == 'C') & (df_eval['Web_複勝期待値'] >= config['C_EXPECTANCY'])
    cond_maru_tankatsu_d = (df_eval['総合ランク'] == 'D') & df_eval['Web_複勝期待値'].between(*config['D_EXPECTANCY_RANGE'])
    cond_maru_b = (df_eval['総合ランク'] == 'B') & (df_eval['Web_複勝期待値'] < config['B_EXPECTANCY'])
    cond_sankaku = ((df_eval['総合ランク'] == 'A') & (df_eval['Web_複勝期待値'] >= config['A_EXPECTANCY'])) | \
                   ((df_eval['総合ランク'] == 'B') & df_eval['Web_複勝期待値'].between(*config['B_EXPECTANCY_RANGE']))
    df_eval['評価'] = np.select([cond_maru_tankatsu_c, cond_maru_tankatsu_d, cond_maru_b, cond_sankaku], ['◎ ◉', '◎ ◉', '◎', '▲'], default='')
    return df_eval

def evaluate_rpt6_rules(df_eval: pd.DataFrame) -> pd.DataFrame:
    config = RPT_RULES_CONFIG[6]
    cond_tankatsu_a = (df_eval['総合ランク'] == 'A') & (df_eval['Web_単勝期待値'] >= config['A_TANSHO_EXPECTANCY'])
    cond_maru_a = (df_eval['総合ランク'] == 'A') & (df_eval['Web_複勝期待値'] < config['A_FUKUSHO_EXPECTANCY'])
    cond_sankaku_sss = (df_eval['総合ランク'] == 'SSS') & (df_eval['Web_複勝期待値'] >= config['SSS_EXPECTANCY'])
    df_eval['評価'] = np.select([cond_tankatsu_a, cond_maru_a, cond_sankaku_sss], ['◉', '◎', '▲'], default='')
    return df_eval

def evaluate_rpt7_rules(df_eval: pd.DataFrame) -> pd.DataFrame:
    config = RPT_RULES_CONFIG[7]
    cond_maru_sss = (df_eval['総合ランク'] == 'SSS')
    cond_sankaku_aa_a = (df_eval['総合ランク'].isin(['A', 'AA'])) & (df_eval['Web_複勝期待値'] >= config['AA_A_EXPECTANCY'])
    df_eval['評価'] = np.select([cond_maru_sss, cond_sankaku_aa_a], ['◎', '▲'], default='')
    return df_eval

def evaluate_rpt8_rules(df_eval: pd.DataFrame) -> pd.DataFrame:
    config = RPT_RULES_CONFIG[8]
    cond_maru_tankatsu = (df_eval['総合ランク'] == 'C') & df_eval['Web_複勝期待値'].between(*config['C_EXPECTANCY_RANGE'])
    cond_sankaku = (df_eval['総合ランク'] == 'S') & (df_eval['Web_複勝期待値'] >= config['S_EXPECTANCY'])
    df_eval['評価'] = np.select([cond_maru_tankatsu, cond_sankaku], ['◎ ◉', '▲'], default='')
    return df_eval

def evaluate_rpt9_rules(df_eval: pd.DataFrame) -> pd.DataFrame:
    config = RPT_RULES_CONFIG[9]
    cond_maru_tankatsu = (df_eval['総合ランク'] == 'C') & df_eval['Web_複勝期待値'].between(*config['C_EXPECTANCY_RANGE'])
    cond_maru = (df_eval['総合ランク'] == 'E') & (df_eval['Web_複勝期待値'] >= config['E_EXPECTANCY'])
    cond_sankaku = (df_eval['総合ランク'] == 'S') & (df_eval['Web_複勝期待値'] >= config['S_EXPECTANCY'])
    df_eval['評価'] = np.select([cond_maru_tankatsu, cond_maru, cond_sankaku], ['◎ ◉', '◎', '▲'], default='')
    return df_eval

def evaluate_rpt10_rules(df_eval: pd.DataFrame) -> pd.DataFrame:
    config = RPT_RULES_CONFIG[10]
    cond_sankaku = (df_eval['総合ランク'] == 'B') & (df_eval['Web_複勝期待値'] >= config['B_EXPECTANCY'])
    df_eval['評価'] = np.select([cond_sankaku], ['▲'], default='')
    return df_eval

def evaluate_rpt11_rules(df_eval: pd.DataFrame) -> pd.DataFrame:
    config = RPT_RULES_CONFIG[11]
    cond_maru = (df_eval['総合ランク'] == 'A') & (df_eval['Web_複勝期待値'] >= config['A_EXPECTANCY'])
    cond_sankaku = (df_eval['総合ランク'] == 'S') & (df_eval['Web_複勝期待値'] >= config['S_EXPECTANCY'])
    df_eval['評価'] = np.select([cond_maru, cond_sankaku], ['◎', '▲'], default='')
    return df_eval

def evaluate_rpt12_rules(df_eval: pd.DataFrame) -> pd.DataFrame:
    config = RPT_RULES_CONFIG[12]
    cond_maru = (df_eval['総合ランク'] == 'A') & (df_eval['Web_複勝期待値'] >= config['A_EXPECTANCY'])
    cond_sankaku = (df_eval['総合ランク'] == 'S') & (df_eval['Web_複勝期待値'] >= config['S_EXPECTANCY'])
    df_eval['評価'] = np.select([cond_maru, cond_sankaku], ['◎', '▲'], default='')
    return df_eval

def evaluate_rpt13_rules(df_eval: pd.DataFrame) -> pd.DataFrame:
    config = RPT_RULES_CONFIG[13]
    cond_sankaku = (df_eval['総合ランク'] == 'A') & (df_eval['Web_複勝期待値'] >= config['A_EXPECTANCY'])
    df_eval['評価'] = np.select([cond_sankaku], ['▲'], default='')
    return df_eval

# --- 司令塔（ルーター）関数 ---
def evaluate_data(df, rpt_number):
    """
    【v7.5】RPT番号に基づき、適切な評価ロジックを呼び出す司令塔（ルーター）。
    """
    df_eval = df.copy()
    df_eval['RPT'] = rpt_number

    # --- 1. データ前処理 ---
    if 'ランク' not in df_eval.columns:
        st.warning("入力データに'ランク'列が見つかりませんでした。評価にはデフォルト値'C'を使用します。")
        df_eval['総合ランク'] = 'C'
    else:
        df_eval['総合ランク'] = df_eval['ランク'].astype(str)

    df_eval['pattern'] = list(zip(df_eval['RPT'], df_eval['総合ランク']))

    numeric_cols_for_eval = ['複勝オッズ下限', '連対率', '複勝率', '単勝オッズ', '勝率', 'BB指数', 'C人気']
    for col in numeric_cols_for_eval:
        if col in df_eval.columns:
            df_eval[col] = pd.to_numeric(df_eval[col], errors='coerce').fillna(0)
        else:
            df_eval[col] = 0
            st.warning(f"入力データに「{col}」列が見つかりません。計算には0を使用します。")

    # --- 2. 期待値計算 ---
    num_runners = len(df_eval)
    correct_hit_rate = df_eval['連対率'] if num_runners <= 7 else df_eval['複勝率']
    df_eval['Web_複勝期待値'] = df_eval['複勝オッズ下限'] * correct_hit_rate
    df_eval['Web_単勝期待値'] = df_eval['単勝オッズ'] * df_eval['勝率']

    # --- 3. 評価ロジックの実行 ---
    df_eval['評価'] = ''

    PROMISING_COMBINATIONS_V21 = set([
        (13, 'SSS'), (10, 'SSS'), (8, 'S'), (4, 'A'), (2, 'S'), (4, 'SSS'), (12, 'SS'), (7, 'A'), 
        (11, 'SSS'), (5, 'S'), (9, 'AA'), (3, 'S'), (7, 'SSS'), (11, 'SS'), (13, 'SS'), (5, 'A'), 
        (6, 'SSS'), (8, 'A'), (9, 'S'), (5, 'SS'), (1, 'C'), (13, 'AA'), (3, 'C'), (13, 'A'), 
        (10, 'SS'), (13, 'B'), (7, 'SS'), (7, 'AA'), (12, 'A')
    ])
    is_honmei_A_cond = (df_eval['pattern'].isin(PROMISING_COMBINATIONS_V21)) & \
                       (df_eval['Web_複勝期待値'] >= 0.85) & \
                       (df_eval['複勝率'] > 0.10)
    df_eval.loc[is_honmei_A_cond, '評価'] = '◎+'

    rpt_function_map = {
        1: evaluate_rpt1_rules, 2: evaluate_rpt2_rules, 3: evaluate_rpt3_rules,
        4: evaluate_rpt4_rules, 5: evaluate_rpt5_rules, 6: evaluate_rpt6_rules,
        7: evaluate_rpt7_rules, 8: evaluate_rpt8_rules, 9: evaluate_rpt9_rules,
        10: evaluate_rpt10_rules, 11: evaluate_rpt11_rules, 12: evaluate_rpt12_rules,
        13: evaluate_rpt13_rules,
    }
    
    non_plus_horses_mask = df_eval['評価'] != '◎+'
    if non_plus_horses_mask.any():
        non_plus_horses = df_eval[non_plus_horses_mask].copy()
        evaluation_function = rpt_function_map.get(rpt_number)
        if evaluation_function:
            evaluated_non_plus = evaluation_function(non_plus_horses)
            df_eval.update(evaluated_non_plus)

    # --- 4. 補足フラグの計算 ---
    df_eval['is_honmei'] = df_eval['評価'].apply(lambda x: '◎' in x or '◉' in x)
    df_eval['◎'] = df_eval['評価'].apply(lambda x: '◎' if '◎' in x else '')
    
    def assign_star_rank(row):
        if not row['is_honmei']: return ''
        cond_A = row['勝率'] >= 0.40
        cond_B = (row['総合ランク'] == 'S') and (row['RPT'] in [8, 9])
        cond_C = (row['総合ランク'] == 'SSS') and (row['BB指数'] >= 70)
        cond_D = (row['RPT'] in [3, 6, 8, 11]) and (row['単勝オッズ'] < 2.0)
        if cond_A or cond_B or cond_C or cond_D: return '☆+'
        if row['連対率'] >= 0.50: return '☆'
        return '☆-'
    df_eval['☆評価'] = df_eval.apply(assign_star_rank, axis=1)

    df_eval['RPT評価'] = ''
    return df_eval

# --- その他のバックエンド関数 ---

def generate_investment_recommendations(evaluated_race_df: pd.DataFrame):
    """投資対象馬券（◎+, ◎, ◉）を抽出・生成する"""
    investment_targets_df = evaluated_race_df[evaluated_race_df['is_honmei']].copy()
    if investment_targets_df.empty:
        return pd.DataFrame()
    
    def assign_units(evaluation):
        if '◎+' in evaluation: return 3.0
        if '◎' in evaluation: return 2.0
        if '◉' in evaluation: return 2.0
        return 0.0

    investment_targets_df['投資ユニット'] = investment_targets_df['評価'].apply(assign_units)
    
    output_columns = ['馬番', '馬名', '評価', '投資ユニット', '複勝オッズ下限']
    final_columns = [col for col in output_columns if col in investment_targets_df.columns]
    
    return investment_targets_df[final_columns].reset_index(drop=True)

def get_dynamic_analysis_data(perf_db: pd.DataFrame, rpt: int, rank: str, exp_val: float):
    """パフォーマンスDBから類似条件のデータを取得（アナライザー用）"""
    results = {'pinpoint': None, 'downward': None, 'upward': None}
    if perf_db.empty: return results

    subset_df = perf_db[(perf_db['RPT'] == rpt) & (perf_db['総合ランク'] == rank)].copy()
    if subset_df.empty: return results

    def parse_range_midpoint(range_str):
        if pd.isna(range_str): return np.nan
        try:
            clean_str = str(range_str).replace('[', '').replace(')', '')
            lower, upper = map(float, clean_str.split(', '))
            return (lower + upper) / 2
        except: return np.nan
    
    subset_df['mid_range_exp_val'] = subset_df['Web_複勝期待値_レンジ'].apply(parse_range_midpoint)
    subset_df.dropna(subset=['mid_range_exp_val'], inplace=True)
    if subset_df.empty: return results

    diff = (subset_df['mid_range_exp_val'] - exp_val).abs()
    if diff.empty: return results
    
    closest_index = diff.idxmin()
    pinpoint_data = subset_df.loc[closest_index]
    results['pinpoint'] = pinpoint_data.to_dict()

    sorted_subset = subset_df.sort_values('mid_range_exp_val').reset_index(drop=True)
    pinpoint_loc_series = sorted_subset[sorted_subset['Web_複勝期待値_レンジ'] == pinpoint_data['Web_複勝期待値_レンジ']].index
    if not pinpoint_loc_series.empty:
        pinpoint_loc = pinpoint_loc_series[0]
        if pinpoint_loc > 0:
            results['downward'] = sorted_subset.iloc[pinpoint_loc - 1].to_dict()
        if pinpoint_loc < len(sorted_subset) - 1:
            results['upward'] = sorted_subset.iloc[pinpoint_loc + 1].to_dict()
            
    return results

def calculate_combinations(bet_type, jiku_list, aite_list):
    """馬券の組み合わせ点数を計算する"""
    jiku_count, aite_count = len(jiku_list), len(aite_list)
    if jiku_count == 0: return 0
    if bet_type == "3連複":
        if jiku_count == 1: return comb(aite_count, 2) if aite_count >= 2 else 0
        if jiku_count == 2: return aite_count
        return comb(jiku_count, 3)
    elif bet_type == "3連単":
        if jiku_count == 1: return aite_count * (aite_count - 1) if aite_count >= 2 else 0
        if jiku_count == 2: return 2 * aite_count
    return 0

# ===================================================================
# 3. UI描画関数群
# ===================================================================

def render_investment_ui(investment_recs, race_info):
    """「投資競馬」タブのUIを描画する (履歴機能なし)"""
    st.subheader(f"投資競馬モード: {race_info}")
    
    if not investment_recs.empty:
        display_recs = investment_recs.copy()
        unit_value = st.session_state.current_unit_value
        display_recs['投資金額 (円)'] = (display_recs['投資ユニット'] * unit_value).astype(int)

        st.subheader("本レースの投資推奨馬券")
        st.dataframe(display_recs[['馬番', '馬名', '評価', '投資ユニット', '投資金額 (円)']], hide_index=True, use_container_width=True)

        total_units = display_recs['投資ユニット'].sum()
        total_investment_yen = display_recs['投資金額 (円)'].sum()
        
        st.metric(label="合計投資ユニット", value=f"{total_units:.1f} U")
        st.metric(label="合計投資金額", value=f"{total_investment_yen:,} 円")
    else:
        st.info("このレースには投資推奨馬券がありません。")

def _render_detailed_stats(data, bet_type_jp):
    """アナライザーの詳細な成績ブロックを描画するヘルパー関数"""
    st.metric(label=f"{bet_type_jp}回収率", value=f"{data.get(f'{bet_type_jp}回収率(%)', 0):.0f}%")
    st.metric(label=f"{bet_type_jp}的中率", value=f"{data.get(f'{bet_type_jp}的中率(%)', 0):.1f}%")
    st.markdown(f"""
    - **試行回数:** {data.get('試行回数', 0)} 回
    - **的中数:** {data.get(f'{bet_type_jp}的中数', 0)} 回
    - **投資額:** {int(data.get(f'{bet_type_jp}投資額', 0)):,} 円
    - **払戻額:** {int(data.get(f'{bet_type_jp}払戻額', 0)):,} 円
    """)

def render_dynamic_analyzer_panel(horse_data, perf_db):
    """ダイナミック・アナライザー・パネルを描画する (UI改善版)"""
    st.header(f"🐴 {horse_data['馬番']}. {horse_data['馬名']} の期待値分析")
    
    analysis_data = get_dynamic_analysis_data(perf_db, rpt=horse_data['RPT'], rank=horse_data['総合ランク'], exp_val=horse_data['Web_複勝期待値'])

    if analysis_data['pinpoint']:
        data = analysis_data['pinpoint']
        st.subheader("📊 類似条件の過去成績")
        st.info(f"現在の期待値 `{horse_data['Web_複勝期待値']:.2f}` に最も近いデータ (期待値レンジ: **{data['Web_複勝期待値_レンジ']}**)")

        with st.container(border=True):
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("複勝 (Place) 成績")
                _render_detailed_stats(data, "複勝")
            with col2:
                st.subheader("単勝 (Win) 成績")
                _render_detailed_stats(data, "単勝")
        
        st.markdown("---")
        col3, col4 = st.columns(2)
        with col3:
            with st.container(border=True):
                st.subheader("▼ 下位互換の複勝成績")
                if analysis_data['downward']:
                    down_data = analysis_data['downward']
                    st.info(f"期待値レンジ: **{down_data['Web_複勝期待値_レンジ']}**")
                    _render_detailed_stats(down_data, "複勝")
                else:
                    st.write("データなし")
        with col4:
            with st.container(border=True):
                st.subheader("▲ 上位互換の複勝成績")
                if analysis_data['upward']:
                    up_data = analysis_data['upward']
                    st.info(f"期待値レンジ: **{up_data['Web_複勝期待値_レンジ']}**")
                    _render_detailed_stats(up_data, "複勝")
                else:
                    st.write("データなし")
    else:
        st.warning("類似条件の過去データが見つかりませんでした。")

def render_enjoyment_ui(race_data, perf_db, race_info):
    """「エンジョイ競馬」タブのUIを描画する"""
    st.subheader(f"🏇 馬券検討ワークベンチ: {race_info}")
    
    st.subheader("評価結果一覧")
    display_cols = ['馬番', '馬名', 'C人気', '評価', '☆評価', 'メンバー推奨', '勝率', '連対率', '複勝率', 'Web_単勝期待値', 'Web_複勝期待値', '単勝オッズ', '複勝オッズ下限', 'BB指数', '総合ランク']
    display_cols_exist = [col for col in display_cols if col in race_data.columns]
    st.dataframe(race_data[display_cols_exist].set_index('馬番'), use_container_width=True)
    st.caption("評価: ◎+/◎/◉/▲ | ☆評価: ☆+/☆/☆- | メンバー推奨: 手動推奨")
    
    st.markdown("---")
    st.subheader("🔬 ダイナミック・アナライザー")
    horse_options = ['馬を選択してください...'] + [f"{row['馬番']}. {row['馬名']}" for index, row in race_data.iterrows()]
    selected_horse_str = st.selectbox("分析したい馬を選択してください", options=horse_options, key="analyzer_horse_select")

    if selected_horse_str and selected_horse_str != '馬を選択してください...':
        selected_horse_num = int(selected_horse_str.split('.')[0])
        filtered_df = race_data[race_data['馬番'] == selected_horse_num]
        if not filtered_df.empty:
            selected_horse_data = filtered_df.iloc[0]
            render_dynamic_analyzer_panel(selected_horse_data, perf_db)
        
            st.markdown("---")
            st.subheader("仮想馬券シミュレーション")
            bet_type = st.selectbox("券種", ["3連複", "3連単"], key="sim_bet_type")
            col_jiku, col_aite = st.columns(2)
            with col_jiku:
                jiku_horses = st.multiselect("軸馬", options=race_data['馬番'].tolist(), format_func=lambda x: f"{x}番 {race_data.loc[race_data['馬番'] == x, '馬名'].iloc[0]}", key="jiku_horses_select")
            with col_aite:
                aite_horses = st.multiselect("相手馬", options=[h for h in race_data['馬番'].tolist() if h not in jiku_horses], format_func=lambda x: f"{x}番 {race_data.loc[race_data['馬番'] == x, '馬名'].iloc[0]}", key="aite_horses_select")
            
            num_combinations = calculate_combinations(bet_type, jiku_horses, aite_horses)
            st.info(f"組み合わせ数: **{num_combinations}点**")
        else:
             st.error("選択された馬のデータが見つかりませんでした。")

# ===================================================================
# 4. メインアプリケーション実行部
# ===================================================================
def main():
    st.set_page_config(page_title="ClusterX Insight Engine", layout="wide")
    st.title("ClusterX Insight Engine")

    # --- セッション状態の初期化 ---
    if 'evaluated_data' not in st.session_state: st.session_state.evaluated_data = pd.DataFrame()
    if 'race_info' not in st.session_state: st.session_state.race_info = ""
    if 'markdown_input_prev' not in st.session_state: st.session_state.markdown_input_prev = ''
    if 'rpt_number_prev' not in st.session_state: st.session_state.rpt_number_prev = 1
    if 'current_budget' not in st.session_state: st.session_state.current_budget = 100000
    if 'current_unit_value' not in st.session_state: st.session_state.current_unit_value = 100

    # --- データのロード ---
    performance_db = load_performance_db()

    # --- UIタブ ---
    tab1, tab2, tab3 = st.tabs(["📥 データ入力", "📈 投資競馬", "🏇 エンジョイ競馬"])

    with tab1:
        st.subheader("レースデータ入力 & 評価実行")
        with st.form("data_input_form"):
            markdown_input = st.text_area("出馬表（Markdown形式）", height=250, value=st.session_state.markdown_input_prev)
            rpt_number = st.number_input("RPT番号", min_value=1, max_value=13, value=st.session_state.rpt_number_prev, step=1)
            
            st.markdown("---")
            st.subheader("資金設定")
            col_budget, col_unit = st.columns(2)
            with col_budget:
                st.session_state.current_budget = st.number_input("本日の総予算(円)", min_value=1000, step=1000, value=st.session_state.current_budget)
            with col_unit:
                st.session_state.current_unit_value = st.number_input("1ユニットあたりの金額(円)", min_value=100, step=100, value=st.session_state.current_unit_value)

            st.markdown("---")
            col_submit, col_reset = st.columns(2)
            with col_submit:
                submitted = st.form_submit_button("評価実行", use_container_width=True)
            with col_reset:
                reset_button = st.form_submit_button("入力クリア", use_container_width=True)

            if reset_button:
                st.session_state.markdown_input_prev = ''
                st.session_state.rpt_number_prev = 1
                st.session_state.evaluated_data = pd.DataFrame()
                st.session_state.race_info = ""
                st.session_state.current_budget = 100000
                st.session_state.current_unit_value = 100
                st.rerun()

            if submitted:
                if not markdown_input.strip():
                    st.warning("出馬表データを入力してください。")
                else:
                    try:
                        df_markdown, race_info = parse_markdown_race_table(markdown_input)
                        st.session_state.evaluated_data = evaluate_data(df_markdown, rpt_number)
                        st.session_state.race_info = race_info
                        st.session_state.markdown_input_prev = markdown_input
                        st.session_state.rpt_number_prev = rpt_number
                        st.success(f"✅ 評価完了｜{race_info}")

                        investment_recs = generate_investment_recommendations(st.session_state.evaluated_data)
                        if not investment_recs.empty:
                            final_recs = investment_recs[investment_recs['複勝オッズ下限'] > 1.0].copy()
                            if not final_recs.empty:
                                st.success("投資対象の推奨が見つかりました！『📈 投資競馬』タブをご確認ください。")
                            else:
                                st.info("投資推奨馬はいましたが、最低オッズ条件(1.0倍超)を満たしませんでした。")
                        else:
                            st.info("このレースに投資推奨馬券はありません。")
                    except Exception as e:
                        st.error(f"評価中にエラーが発生しました: {e}")

    with tab2:
        if not st.session_state.evaluated_data.empty:
            investment_recs_raw = generate_investment_recommendations(st.session_state.evaluated_data)
            final_investment_recs = pd.DataFrame()
            if not investment_recs_raw.empty:
                final_investment_recs = investment_recs_raw[investment_recs_raw['複勝オッズ下限'] > 1.0].copy()
            
            render_investment_ui(final_investment_recs, st.session_state.race_info)
        else:
            st.info("まず「データ入力」タブで評価を実行してください。")

    with tab3:
        if not st.session_state.evaluated_data.empty:
            render_enjoyment_ui(st.session_state.evaluated_data, performance_db, st.session_state.race_info)
        else:
            st.info("まず「データ入力」タブで評価を実行してください。")

if __name__ == '__main__':
    main()