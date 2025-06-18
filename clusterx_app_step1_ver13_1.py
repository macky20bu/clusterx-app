import streamlit as st
import pandas as pd
import numpy as np

# ページの基本設定
st.set_page_config(layout="wide")
st.title("ClusterX｜競馬予想アプリ（UI/UX改善・最終版）")

# --- 関数の定義 ---
def run_step2_labeling(df):
    """確定した新しい印の定義に基づいて馬を分類する関数"""
    labels = {"◎": [], "○": [], "☆": [], "🔥": [], "▲": [], "△": []}
    df['印'] = ''
    
    # 上位の印から順に判定し、一度印が付いたら他の印は付けない
    # ◎の判定
    for idx, row in df.iterrows():
        if row.get('連対率_float', 0) >= 0.6 and row.get('連対率②_float', 0) >= 0.6:
            df.at[idx, '印'] = '◎'
            
    # ○の判定
    for idx, row in df[df['印'] == ''].iterrows():
        tan_exp_ok = row.get('単勝期待値①_float', 0) > 0.9 or row.get('単勝期待値②_float', 0) > 0.9
        win_rate_ok = row.get('勝率_float', 0) > 0.15 or row.get('勝率②_float', 0) > 0.15
        if win_rate_ok and tan_exp_ok:
            df.at[idx, '印'] = '○'

    # 🔥の判定 (☆の条件を包含しているため、先に判定)
    for idx, row in df[df['印'] == ''].iterrows():
        fuku_exp_ok = row.get('複勝期待値①_float', 0) > 1.0 or row.get('複勝期待値②_float', 0) > 1.0
        show_rate_ok = row.get('複勝率_float', 0) > 0.10 or row.get('複勝率②_float', 0) > 0.10
        if fuku_exp_ok and show_rate_ok:
            all_expects = sorted([
                row.get('単勝期待値①_float', 0), row.get('単勝期待値②_float', 0),
                row.get('複勝期待値①_float', 0), row.get('複勝期待値②_float', 0)
            ], reverse=True)
            score = all_expects[0] + all_expects[1]
            show_rate_max = max(row.get('複勝率_float', 0), row.get('複勝率②_float', 0))
            if score >= 1.8 and show_rate_max >= 0.15:
                df.at[idx, '印'] = '🔥'

    # ☆の判定
    for idx, row in df[df['印'] == ''].iterrows():
        fuku_exp_ok = row.get('複勝期待値①_float', 0) > 1.0 or row.get('複勝期待値②_float', 0) > 1.0
        show_rate_ok = row.get('複勝率_float', 0) > 0.10 or row.get('複勝率②_float', 0) > 0.10
        if fuku_exp_ok and show_rate_ok:
            df.at[idx, '印'] = '☆'

    # ▲の判定
    for idx, row in df[df['印'] == ''].iterrows():
        show_rate_max = max(row.get('複勝率_float', 0), row.get('複勝率②_float', 0))
        if show_rate_max > 0.35:
            df.at[idx, '印'] = '▲'
            
    # △の判定
    for idx, row in df[df['印'] == ''].iterrows():
        fuku_exp1_ok = 0.8 <= row.get('複勝期待値①_float', 0) < 1.0
        fuku_exp2_ok = 0.8 <= row.get('複勝期待値②_float', 0) < 1.0
        if fuku_exp1_ok or fuku_exp2_ok:
            df.at[idx, '印'] = '△'

    # labels辞書に結果を格納
    for mark in labels.keys():
        marked_df = df[df['印'] == mark]
        for _, row in marked_df.iterrows():
            labels[mark].append((str(row["馬番"]), str(row["馬名"])))
            
    return labels, df

def highlight_marks(row):
    """結果の表を印ごとに色付けする関数（ダークモード対応版）"""
    # (背景色, 文字色) のタプルを定義
    style_map = {
        "◎": ("#e06c75", "#ffffff"),  # 落ち着いた赤の背景、白文字
        "○": ("#61afef", "#ffffff"),  # 落ち着いた青の背景、白文字
        "🔥": ("#d19a66", "#000000"),  # 落ち着いたオレンジの背景、黒文字
        "☆": ("#98c379", "#000000"),  # 落ち着いた緑の背景、黒文字
        "▲": ("#5c6370", "#ffffff"),  # 濃いグレーの背景、白文字
        "△": ("", "")  # デフォルト（色なし）
    }
    
    bg_color, font_color = style_map.get(row['印'], ("", ""))
    
    style = []
    if bg_color:
        style.append(f"background-color: {bg_color}")
    if font_color:
        style.append(f"color: {font_color}")
        
    # セミコロンで結合して、完全なCSSスタイル文字列を作成
    full_style = "; ".join(style)
    
    # DataFrameの各セルにスタイルを適用
    return [full_style] * len(row)


# --- タブの定義 ---
tab1, tab2, tab3 = st.tabs(["① データ入力", "② 分析結果", "③ GPT連携用テキスト"])


# --- タブ①：データ入力 ---
with tab1:
    st.header("Step1：データ入力")
    if st.button("🧹 全てのデータをリセット"):
        # セッションステートのキーをループで削除
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.experimental_rerun()

    col1, col2 = st.columns(2)

    # --- 左カラム：出馬表入力 ---
    with col1:
        st.subheader("📋 出馬表（Markdown形式）")
        user_input = st.text_area("出馬表をここに貼り付けてください", height=300)

        if user_input:
            try:
                lines = user_input.strip().splitlines()
                race_info = lines[0].strip() if lines[0].startswith("🏇") else ""
                lines = lines[1:] if race_info else lines

                header_idx = next(i for i, line in enumerate(lines) if line.strip().startswith("|"))
                header = [h.strip() for h in lines[header_idx].split("|")[1:-1]]
                data = [l.strip().split("|")[1:-1] for l in lines[header_idx + 2:] if l.strip().startswith("|")]
                df = pd.DataFrame(data, columns=header)

                if '総合ランク' in df.columns:
                    pass
                elif 'ランク' in df.columns:
                    df.rename(columns={'ランク': '総合ランク'}, inplace=True)
                else:
                    st.error("エラー: 貼り付けた出馬表に「総合ランク」または「ランク」列が見つかりません。")
                    st.stop()

                df["勝率_float"] = df["勝率"].str.replace('%', '', regex=False).astype(float) / 100
                df["連対率_float"] = df["連対率"].str.replace('%', '', regex=False).astype(float) / 100
                df["複勝率_float"] = df["複勝率"].str.replace('%', '', regex=False).astype(float) / 100
                df["単勝オッズ"] = df["単勝オッズ"].astype(float)
                df["複勝オッズ下限"] = df["複勝オッズ下限"].astype(float)
                df["単勝期待値①_float"] = df["単勝オッズ"] * df["勝率_float"]
                df["複勝期待値①_float"] = df["複勝オッズ下限"] * df["複勝率_float"]

                st.session_state["clusterx_df"] = df
                st.session_state["race_info"] = race_info
                st.success("✅ 出馬表読み込み完了")

            except Exception as e:
                st.error(f"❌ エラー: 出馬表の解析に失敗しました - {e}")
    
    # --- 右カラム：CSV入力 ---
    with col2:
        st.subheader("📁 成績CSV")
        uploaded_file = st.file_uploader("ランク別オッズ帯の成績CSVを選択", type="csv")

        if uploaded_file and "clusterx_df" in st.session_state:
            df_base = st.session_state["clusterx_df"].copy()
            try:
                try:
                    df_stats = pd.read_csv(uploaded_file)
                except UnicodeDecodeError:
                    uploaded_file.seek(0)
                    df_stats = pd.read_csv(uploaded_file, encoding='cp932')

                if '総合ランク' in df_stats.columns:
                    csv_rank_col_name = '総合ランク'
                elif '馬柱ランク' in df_stats.columns:
                    csv_rank_col_name = '馬柱ランク'
                else:
                    st.error("エラー: CSVに「総合ランク」または「馬柱ランク」列が見つかりません。")
                    st.stop()

                def extract_bounds(odds_str):
                    if isinstance(odds_str, str):
                        odds_str = odds_str.replace("〜", "-").replace("～", "-").replace("−", "-")
                        parts = odds_str.split("-")
                        lower = float(parts[0])
                        upper = float(parts[1]) if len(parts) > 1 and parts[1] else np.inf
                        return pd.Series([lower, upper])
                    return pd.Series([np.nan, np.nan])

                df_stats[["下限", "上限"]] = df_stats["単勝オッズ"].apply(extract_bounds)

                def find_stats(row, csv_rank_col):
                    rank = row["総合ランク"].strip().upper()
                    odds = float(row["単勝オッズ"])
                    match = df_stats[
                        (df_stats[csv_rank_col].astype(str).str.strip().str.upper() == rank) &
                        (df_stats["下限"] <= odds) & (odds < df_stats["上限"])
                    ]
                    return match.iloc[0] if not match.empty else None

                for col in ["勝率②_float", "連対率②_float", "複勝率②_float", "単勝期待値②_float", "複勝期待値②_float"]:
                    df_base[col] = 0.0

                for idx, row in df_base.iterrows():
                    match = find_stats(row, csv_rank_col_name)
                    if match is not None:
                        win_rate2 = float(str(match["勝率"]).replace("%", "", 1).strip()) / 100
                        place_rate2 = float(str(match["連対率"]).replace("%", "", 1).strip()) / 100
                        show_rate2 = float(str(match["複勝率"]).replace("%", "", 1).strip()) / 100
                        df_base.at[idx, "勝率②_float"] = win_rate2
                        df_base.at[idx, "連対率②_float"] = place_rate2
                        df_base.at[idx, "複勝率②_float"] = show_rate2
                        df_base.at[idx, "単勝期待値②_float"] = float(row["単勝オッズ"]) * win_rate2
                        df_base.at[idx, "複勝期待値②_float"] = float(row["複勝オッズ下限"]) * show_rate2
                
                st.session_state["clusterx_df_final"] = df_base
                st.success("✅ CSV読み込み・計算完了")

            except Exception as e:
                st.error(f"❌ エラー: CSV処理に失敗しました - {e}")
    
    # 入力データの確認用エキスパンダー
    if "clusterx_df_final" in st.session_state:
        with st.expander("Step1-②計算後の全データを確認する"):
            st.dataframe(st.session_state["clusterx_df_final"])
    elif "clusterx_df" in st.session_state:
        with st.expander("Step1-①読み込み後のデータを確認する"):
            st.dataframe(st.session_state["clusterx_df"])


# --- タブ②：分析結果 ---
with tab2:
    st.header("Step2：分析結果")
    if st.button("分類実行（新ロジック）", key="run_button"):
        if "clusterx_df_final" in st.session_state:
            df_final = st.session_state["clusterx_df_final"].copy()
            labels, df_with_labels = run_step2_labeling(df_final)
            st.session_state["labels"] = labels
            st.session_state["df_with_labels"] = df_with_labels
            st.success("✅ 分類が完了しました。")
        else:
            st.warning("🚨 タブ①でデータ入力とCSVアップロードを先に完了してください。")

    if "labels" in st.session_state:
        st.subheader("📊 分類結果サマリー")
        if st.session_state.get("race_info"):
            st.info(st.session_state["race_info"])

        labels = st.session_state["labels"]
        df_labeled = st.session_state["df_with_labels"]
        display_data = []
        order = ["◎", "○", "🔥", "☆", "▲", "△"]
        
        for mark in order:
            if mark in labels and labels[mark]:
                for num, name in labels[mark]:
                    row = df_labeled[df_labeled["馬番"] == num].iloc[0]
                    display_data.append({
                        "印": mark, "馬番": num, "馬名": name,
                        "単勝": row.get('単勝オッズ', 0), "複勝": row.get('複勝オッズ下限', 0),
                        "勝率②": f"{row.get('勝率②_float', 0)*100:.1f}%",
                        "複勝率②": f"{row.get('複勝率②_float', 0)*100:.1f}%",
                        "単勝期待値②": f"{row.get('単勝期待値②_float', 0):.2f}",
                        "複勝期待値②": f"{row.get('複勝期待値②_float', 0):.2f}"
                    })
        
        if not display_data:
            st.write("→ 該当馬なし")
        else:
            results_df = pd.DataFrame(display_data)
            st.dataframe(results_df.style.apply(highlight_marks, axis=1), use_container_width=True, hide_index=True)
    else:
        st.info("ここに分析結果が表示されます。")


# --- タブ③：GPT連携用テキスト ---
with tab3:
    st.header("Step3：GPT連携用テキスト")
    if "labels" in st.session_state:
        lines = []
        if st.session_state.get("race_info"):
            lines.append(st.session_state['race_info'])
        
        order = ["◎", "○", "🔥", "☆", "▲", "△"]
        for key in order:
            if key in st.session_state["labels"] and st.session_state["labels"][key]:
                nums = [str(num) for num, _ in st.session_state["labels"][key]]
                lines.append(f"{key} " + ", ".join(nums))
        
        st.text_area("以下のテキストをコピーして使用してください", "\n".join(lines), height=250)
    else:
        st.info("分析を実行すると、ここに連携用テキストが表示されます。")