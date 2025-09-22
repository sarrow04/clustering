import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import io

# --- ページ設定 (変更なし) ---
st.set_page_config(
    page_title="K-Meansクラスタリング支援アプリ",
    page_icon="🧩",
    layout="wide"
)

# --- 関数 (変更なし) ---
def convert_df_to_csv(df):
    """DataFrameをCSV形式のバイトデータに変換する (文字化け対策済み)"""
    return df.to_csv(index=False).encode('utf-8-sig')

@st.cache_data
def calculate_clustering_scores(scaled_data, max_k):
    """エルボー法とシルエットスコアを計算する"""
    wcss = []
    silhouette_scores = []
    k_range = range(2, max_k + 1)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, init='k-means++', n_init='auto', random_state=42)
        kmeans.fit(scaled_data)
        wcss.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(scaled_data, kmeans.labels_))
        
    return k_range, wcss, silhouette_scores

# --- メイン画面 ---
st.title("🧩 K-Meansクラスタリング支援アプリ")
st.write("データの最適なクラスタ数（グループ数）を評価し、実際にデータをグループ分けします。")

# --- サイドバー ---
with st.sidebar:
    st.header("1. データ準備")
    uploaded_file = st.file_uploader("CSVファイルをアップロード", type=["csv"])
    
    df = None
    selected_cols = []

    if uploaded_file:
        # プレビュー用のdf読み込みは残しつつ、表示コードは削除
        df = pd.read_csv(uploaded_file)
        
        # <--- 変更点: ここにあったプレビュー用のexpanderを削除 ---

        st.subheader("クラスタリングに使う列を選択")
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        
        selected_cols = st.multiselect(
            "数値型の列から複数選択してください",
            options=numeric_cols,
            default=numeric_cols[:min(len(numeric_cols), 3)]
        )

# --- アプリ本体 ---
if df is not None:
    # <--- 変更点: メイン画面にプレビュー用のexpanderを移動 ---
    st.header("1. アップロードデータ確認")
    with st.expander("データの中身を表示/非表示"):
        # メイン画面なのでdf全体を表示（スクロール可能）
        st.dataframe(df)

    # クラスタリングに使う列が選択された場合のみ、分析パートに進む
    if selected_cols:
        st.header("2. 最適なクラスタ数の評価")
        
        data_to_cluster = df[selected_cols].dropna()
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data_to_cluster)
        
        max_k = st.slider("評価する最大クラスタ数(K)", min_value=5, max_value=20, value=10)
        
        if st.button("評価を実行する"):
            with st.spinner("エルボー法とシルエットスコアを計算中..."):
                k_range, wcss, silhouette_scores = calculate_clustering_scores(scaled_data, max_k)

                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("エルボー法")
                    fig, ax = plt.subplots()
                    ax.plot(k_range, wcss, marker='o')
                    ax.set_xlabel("クラスタ数 (K)")
                    ax.set_ylabel("クラスタ内誤差平方和 (WCSS)")
                    ax.set_title("Elbow Method")
                    st.pyplot(fig)
                    st.info("グラフが「肘」のように急に曲がる点が、最適なKの候補です。")

                with col2:
                    st.subheader("シルエットスコア")
                    fig, ax = plt.subplots()
                    ax.plot(k_range, silhouette_scores, marker='o')
                    ax.set_xlabel("クラスタ数 (K)")
                    ax.set_ylabel("平均シルエットスコア")
                    ax.set_title("Silhouette Score")
                    st.pyplot(fig)
                    st.info("スコアが最も高くなる点が、最適なKの候補です。")

        st.markdown("---")
        st.header("3. クラスタリングの実行と分析")
        
        final_k = st.number_input("グラフを参考に、最終的なクラスタ数を入力してください", min_value=2, max_value=max_k, value=3)
        
        if st.button("このクラスタ数で実行"):
            kmeans = KMeans(n_clusters=final_k, init='k-means++', n_init='auto', random_state=42)
            clusters = kmeans.fit_predict(scaled_data)
            
            df_clustered = df.copy()
            df_clustered.loc[data_to_cluster.index, 'cluster'] = clusters
            
            st.subheader(f"{final_k}個のクラスタに分類しました")
            st.dataframe(df_clustered.head())
            
            st.subheader("各クラスタの特徴")
            cluster_summary = df_clustered.dropna(subset=['cluster']).copy()
            cluster_summary['cluster'] = cluster_summary['cluster'].astype(int)
            st.dataframe(cluster_summary.groupby('cluster')[selected_cols].mean())
            
            st.subheader("各クラスタの所属人数")
            st.dataframe(cluster_summary['cluster'].value_counts().sort_index())
            
            st.download_button(
               label="クラスタ付きCSVをダウンロード",
               data=convert_df_to_csv(df_clustered),
               file_name='data_with_clusters.csv',
               mime='text/csv',
            )
    else:
        st.info("サイドバーで分析に使用する列を選択してください。")

else:
    st.info("サイドバーからCSVファイルをアップロードして開始してください。")
