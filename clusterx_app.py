import streamlit as st
import pandas as pd
import numpy as np
import re

# ページの基本設定
st.set_page_config(layout="wide")
st.title("ClusterX - 競馬予想アプリ Ver.10.2")

# --- UI/UX改善：カスタムCSSでフォントを変更 ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+JP:wght@400;500;700&display=swap');
html, body, [class*="st-"], [class*="css-"] {
   font-family: 'Noto Sans JP', sans-serif;
}
</style>
""", unsafe_allow_html=True)


# --- 定数定義 ---
RANK_ODDS_CSV_URL = "https://raw.githubusercontent.com/macky20bu/clusterx-app/main/%E7%B7%8F%E5%90%88%E3%83%A9%E3%83%B3%E3%82%AF%E3%82%AA%E3%83%83%E3%82%BA%E5%88%A5%E6%88%90%E7%B8%BE%E8%A1%A8_%E7%B5%B1%E5%90%88%E7%89%88.csv"
RPT_CSV_URL = "https://raw.githubusercontent.com/macky20bu/clusterx-app/main/%E5%85%A8%E7%AB%B6%E9%A6%AC%E5%A0%B4%E7%B5%B1%E5%90%88_RPT%E5%88%A5%E6%88%90%E7%B8%BE%E8%A1%A8.csv"
MARK_ORDER = ["◎", "○", "▲", "△", "✕", "🔥", "☆"]
PADDOCK_MARK_OPTIONS = ["-", "◎", "○", "▲", "△", "✕"]
RACECOURSE_OPTIONS = ["統合データ", "船橋", "浦和", "大井", "川崎", "園田"]

# --- ヘルパー関数 ---
def get_c_rank_zone(c_rank):
    if pd.isna(c_rank): return None
    if c_rank == 1: return 'C1（本命）'
    if 2 <= c_rank <= 4: return 'C2-4（中位人気）'
    if c_rank >= 5: return 'C5+（穴人気）'
    return None

def get_bb_rank_zone(bb_rank):
    if bb_rank == 1: return 'BB1（最上位）'
    if 2 <= bb_rank <= 4: return 'BB2-4（中位）'
    if bb_rank >= 5: return 'BB5+（下位）'
    return None

def get_odds_zone(odds):
    if pd.isna(odds): return None
    if 1.0 <= odds <= 2.9: return 'O1（1.0〜2.9倍）'
    if 3.0 <= odds <= 6.9: return 'O2（3.0〜6.9倍）'
    if 7.0 <= odds <= 14.9: return 'O3（7.0〜14.9倍）'
    if 15.0 <= odds <= 29.9: return 'O4（15.0〜29.9倍）'
    if odds >= 30.0: return 'O5（30.0倍〜）'
    return None

def run_step2_labeling(df):
    df_copy = df.copy()
    df_copy['印'] = ''
    # (期待値印のロジックは変更なし)
    for idx, row in df_copy.iterrows():
        if row.get('連対率_float', 0) >= 0.6 and row.get('連対率②_float', 0) >= 0.6: df_copy.at[idx, '印'] = '◎'
    for idx, row in df_copy[df_copy['印'] == ''].iterrows():
        tan_exp_ok = row.get('単勝期待値①_float', 0) > 0.9 or row.get('単勝期待値②_float', 0) > 0.9
        win_rate_ok = row.get('勝率_float', 0) > 0.15 or row.get('勝率②_float', 0) > 0.15
        if win_rate_ok and tan_exp_ok: df_copy.at[idx, '印'] = '○'
    for idx, row in df_copy[df_copy['印'] == ''].iterrows():
        fuku_exp_ok = row.get('複勝期待値①_float', 0) > 1.0 or row.get('複勝期待値②_float', 0) > 1.0
        show_rate_ok = row.get('複勝率_float', 0) > 0.10 or row.get('複勝率②_float', 0) > 0.10
        if fuku_exp_ok and show_rate_ok:
            all_expects = sorted([row.get(c, 0) for c in ['単勝期待値①_float', '単勝期待値②_float', '複勝期待値①_float', '複勝期待値②_float']], reverse=True)
            if all_expects and all_expects[0] + all_expects[1] >= 1.8 and max(row.get('複勝率_float', 0), row.get('複勝率②_float', 0)) >= 0.15:
                df_copy.at[idx, '印'] = '🔥'
    for idx, row in df_copy[df_copy['印'] == ''].iterrows():
        fuku_exp_ok = row.get('複勝期待値①_float', 0) > 1.0 or row.get('複勝期待値②_float', 0) > 1.0
        show_rate_ok = row.get('複勝率_float', 0) > 0.10 or row.get('複勝率②_float', 0) > 0.10
        if fuku_exp_ok and show_rate_ok: df_copy.at[idx, '印'] = '☆'
    for idx, row in df_copy[df_copy['印'] == ''].iterrows():
        if max(row.get('複勝率_float', 0), row.get('複勝率②_float', 0)) > 0.35: df_copy.at[idx, '印'] = '▲'
    for idx, row in df_copy[df_copy['印'] == ''].iterrows():
        if 0.8 <= row.get('複勝期待値①_float', 0) < 1.0 or 0.8 <= row.get('複勝期待値②_float', 0) < 1.0: df_copy.at[idx, '印'] = '△'
    
    labels = {mark: [] for mark in MARK_ORDER if mark != '✕'}
    for mark in labels:
        for _, row in df_copy[df_copy['印'] == mark].iterrows(): labels[mark].append((str(row["馬番"]), str(row["馬名"])))
    return labels, df_copy

def highlight_marks(row, mark_column='印'):
    style_map = {"◎": ("#e06c75", "#ffffff"),"○": ("#61afef", "#ffffff"),"🔥": ("#d19a66", "#000000"),"☆": ("#98c379", "#000000"),"▲": ("#5c6370", "#ffffff"),"△": ("#c678dd", "#ffffff"), "✕": ("#282c34", "#c8c8c8")}
    mark = row.get(mark_column, "") # .get()でエラーを回避
    bg_color, font_color = style_map.get(mark, ("", ""))
    style = [f"background-color: {bg_color}" if bg_color else "", f"color: {font_color}" if font_color else ""]
    return ["; ".join(filter(None, style))] * len(row)

@st.cache_data
def load_data():
    data = {}
    try:
        df_rank_odds = pd.read_csv(RANK_ODDS_CSV_URL)
        df_rank_odds.columns = df_rank_odds.columns.str.strip()
        df_rank_odds = df_rank_odds[df_rank_odds['単勝オッズ帯'] != '合計'].copy()
        def extract_bounds(odds_str):
            if isinstance(odds_str, str):
                parts = odds_str.replace("〜", "-").replace("～", "-").replace("−", "-").split("-")
                return pd.Series([float(parts[0]), float(parts[1]) if len(parts) > 1 and parts[1] else np.inf])
            return pd.Series([np.nan, np.nan])
        df_rank_odds[["オッズ下限", "オッズ上限"]] = df_rank_odds["単勝オッズ帯"].apply(extract_bounds)
        data["rank_odds"] = df_rank_odds
    except Exception as e:
        st.error(f"エラー: ランク・オッズ別成績表の読み込みに失敗しました - {e}")
    try:
        df_rpt = pd.read_csv(RPT_CSV_URL)
        df_rpt.columns = df_rpt.columns.str.strip()
        data["rpt"] = df_rpt
    except Exception as e:
        st.error(f"エラー: RPT成績表の読み込みに失敗しました - {e}")
    return data

def display_rpt_evaluation(df, mark_col_name, rate_type="複勝率"):
    if df.empty:
        st.write("→ 該当馬なし")
        return

    rate_map = {"勝率": {"col1": "勝率_float", "col2": "勝率②_float", "rpt_suffix": "_勝率", "avg_header": "RPT平均勝率"},"連対率": {"col1": "連対率_float", "col2": "連対率②_float", "rpt_suffix": "_連対率", "avg_header": "RPT平均連対率"},"複勝率": {"col1": "複勝率_float", "col2": "複勝率②_float", "rpt_suffix": "_複勝率", "avg_header": "RPT平均複勝率"}}
    selected_map = rate_map.get(rate_type, rate_map["複勝率"])
    df_display = df.copy()
    rpt_cols = [f'RPT_{p}{selected_map["rpt_suffix"]}' for p in ["C人気", "BB順位", "単勝オッズ帯"]]
    if all(col in df_display.columns for col in rpt_cols):
        df_display[selected_map["avg_header"]] = df_display[rpt_cols].mean(axis=1)
    else:
        df_display[selected_map["avg_header"]] = 0
    if mark_col_name not in df_display.columns: df_display[mark_col_name] = ''
    if '印_馬柱' not in df_display.columns: df_display['印_馬柱'] = ''
        
    # ★★★ 列名変更のロジックを修正 ★★★
    summary_cols = {
        mark_col_name: mark_col_name, 
        '印_馬柱': '印_馬柱', 
        '馬番': '馬番', '馬名': '馬名',
        f'{rate_type}①(出馬表)': selected_map["col1"],
        f'{rate_type}②(ランク)': selected_map["col2"],
        selected_map["avg_header"]: selected_map["avg_header"]
    }
    # 重複する列（mark_col_nameと'印_馬柱'が同じ場合）を削除
    if mark_col_name == '印_馬柱':
        summary_cols.pop('印_馬柱', None)
        
    valid_cols_inv = {v: k for k, v in summary_cols.items()}
    display_col_order = [k for k,v in summary_cols.items() if v in df_display.columns]
    
    summary_df = df_display[list(valid_cols_inv.keys())].rename(columns=valid_cols_inv)
    summary_df = summary_df[display_col_order]

    for col in summary_df.columns:
        if "率" in col: summary_df[col] = summary_df[col].map('{:.1%}'.format)
    
    st.subheader("RPT評価サマリー")
    st.dataframe(summary_df.style.apply(highlight_marks, axis=1, mark_column=mark_col_name), use_container_width=True, hide_index=True)
    
    st.markdown("##### 各馬のRPT詳細")
    for _, row in df_display.iterrows():
        expander_title = f"印：{row[mark_col_name]} (馬柱印：{row['印_馬柱']})　馬番：{row['馬番']}　馬名：{row['馬名']}"
        with st.expander(expander_title):
            detail_data = {'C人気': [row.get('RPT_C人気_勝率', 0), row.get('RPT_C人気_連対率', 0), row.get('RPT_C人気_複勝率', 0)],'BB順位': [row.get('RPT_BB順位_勝率', 0), row.get('RPT_BB順位_連対率', 0), row.get('RPT_BB順位_複勝率', 0)],'単勝オッズ帯': [row.get('RPT_単勝オッズ帯_勝率', 0), row.get('RPT_単勝オッズ帯_連対率', 0), row.get('RPT_単勝オッズ帯_複勝率', 0)]}
            detail_df = pd.DataFrame(detail_data, index=['勝率', '連対率', '複勝率'])
            st.dataframe(detail_df.style.format('{:.1%}'), use_container_width=True)

# --- アプリ本体 ---
loaded_data = load_data()
if "rank_odds" in loaded_data: st.session_state["rank_odds_stats_df"] = loaded_data["rank_odds"]
if "rpt" in loaded_data: st.session_state["rpt_stats_df"] = loaded_data["rpt"]

tab1, tab2, tab3, tab4, tab5 = st.tabs(["データ入力", "推奨馬", "期待値", "RPT評価【印】", "RPT評価【全馬】"])

with tab1:
    st.subheader("Step 1：データ入力")
    if st.button("全データをリセット"):
        keys_to_delete = list(st.session_state.keys());
        for key in keys_to_delete: del st.session_state[key]
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")
    st.markdown("##### 1. 出馬表（Markdown形式）")
    user_input = st.text_area("C人気、BB順位を含む形式の出馬表を貼り付けてください", height=250, key="user_input_area", label_visibility="collapsed")
    if 'raw_df' not in st.session_state: st.session_state.raw_df = pd.DataFrame()
    if user_input and (st.session_state.raw_df.empty or st.session_state.get("current_input") != user_input):
        st.session_state.current_input = user_input
        try:
            lines = user_input.strip().splitlines()
            st.session_state["race_info"] = lines[0].strip() if lines and lines[0].startswith("🏇") else ""
            lines = [line for line in lines if line.strip().startswith("|")]
            header = [h.strip() for h in lines[0].split("|")[1:-1]]
            data = [[cell.strip() for cell in l.split("|")[1:-1]] for l in lines[2:]]
            df = pd.DataFrame(data, columns=header)
            if 'ランク' in df.columns: df.rename(columns={'ランク': '総合ランク'}, inplace=True)
            st.session_state.raw_df = df
        except Exception as e:
            st.error(f"エラー: 出馬表の形式が正しくない可能性があります - {e}")
            st.session_state.raw_df = pd.DataFrame()

    if not st.session_state.raw_df.empty:
        st.markdown("---")
        st.markdown("##### 2. 馬柱分析印（任意）")
        st.info("各馬の印をドロップダウンから選択してください。")
        df_for_edit = st.session_state.raw_df[['馬番', '馬名']].copy()
        if '印_馬柱' not in st.session_state.raw_df.columns: df_for_edit['印_馬柱'] = '-'
        edited_df = st.data_editor(df_for_edit, column_config={"印_馬柱": st.column_config.SelectboxColumn("馬柱印",options=PADDOCK_MARK_OPTIONS,required=True)}, hide_index=True, use_container_width=True, key="paddock_editor")
        
        st.markdown("---")
        st.markdown("##### 3. 分析条件の設定と実行")
        race_info_text = st.session_state.get("race_info", "")
        detected_course = next((course for course in RACECOURSE_OPTIONS if course in race_info_text), "統合データ")
        try: detected_index = RACECOURSE_OPTIONS.index(detected_course)
        except ValueError: detected_index = 0
        st.info(f"開催競馬場を「{detected_course}」と自動判定しました。異なる場合は下記から選択してください。")
        c1, c2 = st.columns(2)
        with c1: selected_course = st.selectbox("分析に使用する競馬場データ:",RACECOURSE_OPTIONS, index=detected_index, key="course_selector")
        with c2: selected_rpt = st.selectbox("このレースのRPTパターンを選択:", list(range(1, 14)), 0, key="race_rpt_selector")

        if st.button("計算実行", type="primary", use_container_width=True):
            try:
                base_df = st.session_state.raw_df.copy()
                paddock_marks_df = edited_df[['馬番', '印_馬柱']]
                df_final = pd.merge(base_df, paddock_marks_df, on='馬番', how='left')
                df_final['印_馬柱'] = df_final['印_馬柱'].replace('-', '')
                for col in ["C人気", "単勝オッズ", "複勝オッズ下限"]: df_final[col] = pd.to_numeric(df_final[col], errors='coerce')
                df_final["BB順位_int"] = df_final["BB順位"].str.extract(r'(\d+)').astype(int)
                for col in ["勝率", "連対率", "複勝率"]: df_final[f"{col}_float"] = df_final[col].str.replace('%','').astype(float)/100
                df_final["単勝期待値①_float"] = df_final["単勝オッズ"] * df_final["勝率_float"]
                df_final["複勝期待値①_float"] = df_final["複勝オッズ下限"] * df_final["複勝率_float"]
                df_final['RPT'] = selected_rpt
                df_stats_orig, df_rpt_stats_orig = st.session_state.get("rank_odds_stats_df"), st.session_state.get("rpt_stats_df")
                if df_stats_orig is None or df_rpt_stats_orig is None: st.error("成績データが読み込まれていません。"); st.stop()
                df_stats, df_rpt_stats = df_stats_orig.copy(), df_rpt_stats_orig.copy()
                if selected_course != "統合データ":
                    if '競馬場' in df_stats.columns: df_stats = df_stats[df_stats['競馬場'] == selected_course]
                    if '競馬場' in df_rpt_stats.columns: df_rpt_stats = df_rpt_stats[df_rpt_stats['競馬場'] == selected_course]
                for col in ["勝率②_float", "連対率②_float", "複勝率②_float", "単勝期待値②_float", "複勝期待値②_float"]: df_final[col] = 0.0
                for idx, row in df_final.iterrows():
                    match = df_stats[(df_stats["総合ランク"].astype(str).str.strip().str.upper() == row["総合ランク"].strip().upper()) & (df_stats["オッズ下限"] <= row["単勝オッズ"]) & (row["単勝オッズ"] < df_stats["オッズ上限"])]
                    if not match.empty:
                        m = match.iloc[0]
                        for r in ["勝率", "連対率", "複勝率"]: df_final.at[idx, f"{r}②_float"] = float(str(m[r]).replace("%","").strip())/100
                        df_final.at[idx, "単勝期待値②_float"], df_final.at[idx, "複勝期待値②_float"] = row["単勝オッズ"] * df_final.at[idx, "勝率②_float"], row["複勝オッズ下限"] * df_final.at[idx, "複勝率②_float"]
                for p in ["C人気", "BB順位", "単勝オッズ帯"]:
                    for r in ["勝率", "連対率", "複勝率"]: df_final[f'RPT_{p}_{r}'] = 0.0
                for idx, row in df_final.iterrows():
                    zones = {'C人気': get_c_rank_zone(row['C人気']), 'BB順位': get_bb_rank_zone(row['BB順位_int']), '単勝オッズ帯': get_odds_zone(row['単勝オッズ'])}
                    for label, zone in zones.items():
                        if zone:
                            match_rpt = df_rpt_stats[(df_rpt_stats["RPT"] == row["RPT"]) & (df_rpt_stats["分類ラベル"] == label) & (df_rpt_stats["ゾーンラベル"] == zone)]
                            if not match_rpt.empty:
                                for r in ["勝率", "連対率", "複勝率"]: df_final.at[idx, f'RPT_{label}_{r}'] = float(str(match_rpt.iloc[0][r]).replace("%","").strip())/100
                st.session_state["clusterx_df_final"] = df_final
                st.success("計算が完了しました。各タブで結果を確認してください。")
            except Exception as e:
                st.error(f"計算中にエラーが発生しました: {e}")

with tab2:
    st.subheader("Step 2：推奨馬（期待値ロジック）")
    if st.button("推奨馬を分類実行", key="run_button_tab2", use_container_width=True):
        if "clusterx_df_final" in st.session_state:
            df_final = st.session_state["clusterx_df_final"].copy()
            _, st.session_state["df_with_labels"] = run_step2_labeling(df_final)
            st.success("分類が完了しました。")
        else: st.warning("タブ①でデータ入力と計算実行を先に完了してください。")
    if "df_with_labels" in st.session_state:
        st.subheader("推奨馬サマリー")
        if st.session_state.get("race_info"): st.info(st.session_state["race_info"])
        results_df = st.session_state["df_with_labels"][st.session_state["df_with_labels"]["印"] != ""].copy()
        
        display_cols = {"印": "印", "馬柱印": "印_馬柱", "馬番": "馬番", "馬名": "馬名", "単勝": "単勝オッズ", "複勝": "複勝オッズ下限", "勝率②": "勝率②_float", "連対率②": "連対率②_float", "複勝率②": "複勝率②_float", "単勝期待値②": "単勝期待値②_float", "複勝期待値②": "複勝期待値②_float"}
        if not results_df.empty:
            display_df = results_df[list(display_cols.values())].copy(); display_df.columns = list(display_cols.keys())
            for col in ["勝率②", "連対率②", "複勝率②"]: display_df[col] = display_df[col].map('{:.1%}'.format)
            for col in ["単勝期待値②", "複勝期待値②"]: display_df[col] = display_df[col].map('{:.2f}'.format)
            order = [m for m in MARK_ORDER if m != '✕']; display_df['印'] = pd.Categorical(display_df['印'], categories=order, ordered=True)
            display_df = display_df.sort_values('印')
            st.dataframe(display_df.style.apply(highlight_marks, axis=1, mark_column='印'), use_container_width=True, hide_index=True)
        else: st.write("→ 該当馬なし")
    else: st.info("タブ①で計算実行後、このボタンを押すと推奨馬が表示されます。")

with tab3:
    st.subheader("Step 3：期待値一覧")
    st.info("全出走馬の期待値（①出馬表 / ②ランクオッズ）を一覧で確認できます。")
    if "clusterx_df_final" in st.session_state:
        df_final = st.session_state["clusterx_df_final"].copy()
        if '印' not in df_final.columns: _, df_final = run_step2_labeling(df_final)
        
        display_cols = {'印': '印', '馬柱印': '印_馬柱', '馬番': '馬番', '馬名': '馬名','単勝期待値①': '単勝期待値①_float', '複勝期待値①': '複勝期待値①_float','単勝期待値②': '単勝期待値②_float', '複勝期待値②': '複勝期待値②_float'}
        display_df = df_final[list(display_cols.values())].copy()
        display_df.columns = list(display_cols.keys())
        for col in display_df.columns:
            if "期待値" in col: display_df[col] = display_df[col].map('{:.2f}'.format)
        st.dataframe(display_df.style.apply(highlight_marks, axis=1), use_container_width=True, hide_index=True)
    else:
        st.info("タブ①で「計算実行」を押してください。")

with tab4:
    st.subheader("Step 4：RPT評価【印付き馬】")
    rate_options = ["勝率", "連対率", "複勝率"]
    selected_rate_tab4 = st.radio("表示する指標を選択:", options=rate_options, index=2, horizontal=True, key="rate_selector_tab4")
    if "clusterx_df_final" in st.session_state:
        df_final = st.session_state["clusterx_df_final"]
        tab4_1, tab4_2 = st.tabs(["1. 期待値印評価", "2. 馬柱分析評価"])
        with tab4_1:
            st.markdown("##### アプリの期待値ロジックで印が付いた馬のRPT評価")
            if "df_with_labels" not in st.session_state: _, st.session_state["df_with_labels"] = run_step2_labeling(df_final.copy())
            df_to_show = st.session_state["df_with_labels"]
            display_rpt_evaluation(df_to_show[df_to_show['印'] != ''], '印', selected_rate_tab4)
        with tab4_2:
            st.markdown("##### データエディタで入力された印の馬のRPT評価")
            if '印_馬柱' in df_final.columns and df_final['印_馬柱'].str.strip().any():
                df_paddock_marks = df_final[df_final['印_馬柱'] != ''].copy()
                df_paddock_marks['印_馬柱'] = pd.Categorical(df_paddock_marks['印_馬柱'], categories=MARK_ORDER, ordered=True)
                df_paddock_marks = df_paddock_marks.sort_values('印_馬柱')
                display_rpt_evaluation(df_paddock_marks, '印_馬柱', selected_rate_tab4)
            else: st.write("→ データエディタで印が入力されていません。")
    else: st.info("タブ①で「計算実行」を押してください。")

with tab5:
    st.subheader("Step 5：RPT評価【全馬】")
    rate_options_tab5 = ["勝率", "連対率", "複勝率"]
    selected_rate_tab5 = st.radio("表示する指標を選択:", options=rate_options_tab5, index=2, horizontal=True, key="rate_selector_tab5")
    st.info("全出走馬のRPT評価をC人気順に表示します。")
    if "clusterx_df_final" in st.session_state:
        df_final = st.session_state["clusterx_df_final"].copy()
        if '印' not in df_final.columns: _, df_final = run_step2_labeling(df_final)
        df_sorted_by_c_rank = df_final.sort_values('C人気')
        display_rpt_evaluation(df_sorted_by_c_rank, '印', selected_rate_tab5)
    else: st.info("タブ①で「計算実行」を押してください。")