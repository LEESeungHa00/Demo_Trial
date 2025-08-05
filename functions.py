import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
from datetime import datetime
from dateutil.relativedelta import relativedelta
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from google.oauth2.service_account import Credentials
from pandas_gbq import read_gbq

# --- 초기 설정 및 페이지 구성 ---
st.set_page_config(layout="wide", page_title="수입 경쟁력 진단 솔루션")

# --- 데이터 로딩 (BigQuery) ---
@st.cache_data(ttl=3600)
def load_company_data():
    """Google BigQuery에서 TDS 데이터를 불러오고 기본 전처리 수행"""
    try:
        creds = Credentials.from_service_account_info(st.secrets["gcp_service_account"])
        project_id = st.secrets["gcp_service_account"]["project_id"]
        table_full_id = f"{project_id}.demo_data.tds_data"
        df = read_gbq(f"SELECT * FROM `{table_full_id}`", project_id=project_id, credentials=creds)

        if df.empty:
            st.error("BigQuery에서 데이터를 불러왔지만 비어있습니다.")
            return None

        # 지능형 컬럼명 정제
        df.columns = [re.sub(r'[^a-z0-9]+', '_', col.lower().strip()) for col in df.columns]

        required_cols = ['date', 'volume', 'value', 'reported_product_name', 'export_country', 'exporter', 'importer', 'hs_code']
        if not all(col in df.columns for col in required_cols):
            st.error(f"BigQuery 테이블에 필수 컬럼이 부족합니다. (필수: {required_cols})")
            st.info(f"실제 컬럼명: {df.columns.tolist()}")
            return None

        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        for col in ['volume', 'value']:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(r'[^\d.]', '', regex=True), errors='coerce')

        df.dropna(subset=['date', 'volume', 'value', 'importer', 'exporter'], inplace=True)
        df = df[(df['volume'] > 0) & (df['value'] > 0)].copy() # 0보다 큰 값만 사용
        
        # 클리닝 후 unitPrice 계산
        df['unitPrice'] = df['value'] / df['volume']
        
        # 너무 비싸거나 싼 이상치 제거 (IQR 방식)
        Q1 = df['unitPrice'].quantile(0.25)
        Q3 = df['unitPrice'].quantile(0.75)
        IQR = Q3 - Q1
        df = df[~((df['unitPrice'] < (Q1 - 1.5 * IQR)) | (df['unitPrice'] > (Q3 + 1.5 * IQR)))]
        
        return df if not df.empty else None
    except Exception as e:
        st.error(f"데이터 로딩 중 오류가 발생했습니다: {e}")
        return None

# --- 분석 헬퍼 함수 ---
def clean_text(text):
    if not isinstance(text, str): return ''
    text = text.lower()
    text = re.sub(r'\(.*?\)|\[.*?\]', ' ', text)
    text = re.sub(r'(\d+)\s*(?:y|yo|year|years|old|년산|년)', r'\1', text)
    text = re.sub(r'[^a-z0-9\s\uac00-\ud7a3]', ' ', text)
    text = re.sub(r'\b산\b', ' ', text)
    return ' '.join(text.split())

def name_clusters(df, cluster_col='cluster'):
    """K-Means 클러스터에 동적으로 이름을 부여합니다."""
    centroids = df.groupby(cluster_col).agg({
        'Total_Volume': 'mean',
        'Avg_UnitPrice': 'mean',
        'Trade_Count': 'mean'
    }).reset_index()

    # 각 지표의 순위 계산 (높을수록 좋음: Volume, Count / 낮을수록 좋음: Price)
    centroids['volume_rank'] = centroids['Total_Volume'].rank(ascending=False)
    centroids['price_rank'] = centroids['Avg_UnitPrice'].rank(ascending=True)
    centroids['count_rank'] = centroids['Trade_Count'].rank(ascending=False)
    
    # 가중치 합산으로 종합 점수 계산
    centroids['total_score'] = centroids['volume_rank'] * 0.4 + centroids['price_rank'] * 0.4 + centroids['count_rank'] * 0.2
    
    # 점수 순으로 이름 부여
    sorted_centroids = centroids.sort_values('total_score')
    
    names = ["소규모/고가치 그룹", "중견/균형 그룹", "대규모/가성비 그룹"]
    if len(sorted_centroids) < 3:
        names = ["A 그룹", "B 그룹"] # 클러스터가 2개일 경우

    name_map = {row[cluster_col]: names[i] if i < len(names) else f"{chr(ord('A')+i)} 그룹" for i, row in sorted_centroids.iterrows()}
    return name_map

def perform_clustering(importer_stats):
    """K-Means 클러스터링을 수행하고 결과를 반환합니다."""
    # 클러스터링에 사용할 특성 선택
    features = importer_stats[['Total_Volume', 'Avg_UnitPrice', 'Trade_Count']].copy()
    
    # 데이터가 3개 미만이면 클러스터링 불가
    if len(features) < 3:
        return None, None

    # 데이터 스케일링
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # K-Means 클러스터링 수행 (k=3)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
    importer_stats['cluster'] = kmeans.fit_predict(scaled_features)
    
    # 클러스터 이름 부여
    cluster_name_map = name_clusters(importer_stats)
    importer_stats['Cluster_Name'] = importer_stats['cluster'].map(cluster_name_map)
    
    return importer_stats, cluster_name_map

# --- 메인 분석 로직 ---
def run_all_analysis(user_inputs, full_company_data, selected_products, target_importer_name):
    analysis_result = {"overview": {}, "positioning": {}, "supply_chain": {}}
    
    # 1. 데이터 필터링: 분석에 사용할 데이터만 선택
    analysis_data = full_company_data[full_company_data['reported_product_name'].isin(selected_products)].copy()
    if analysis_data.empty:
        return analysis_result

    # 2. 수입사별 통계 집계
    importer_stats = analysis_data.groupby('importer').agg(
        Total_Value=('value', 'sum'),
        Total_Volume=('volume', 'sum'),
        Trade_Count=('value', 'count'),
        Avg_UnitPrice=('unitPrice', 'mean')
    ).reset_index()

    if importer_stats.empty or importer_stats['Total_Volume'].sum() == 0:
        return analysis_result

    importer_stats = importer_stats.sort_values('Total_Value', ascending=False).reset_index(drop=True)
    
    # 3. 포지셔닝 분석 (규칙 기반 그룹 + AI 기반 클러스터링)
    # 3-1. 규칙 기반 그룹
    importer_stats['cum_share'] = importer_stats['Total_Value'].cumsum() / importer_stats['Total_Value'].sum()
    market_leaders = importer_stats[importer_stats['cum_share'] <= 0.7]
    
    direct_peers = pd.DataFrame()
    try:
        target_rank = importer_stats[importer_stats['importer'] == target_importer_name].index[0]
        rank_margin = max(1, int(len(importer_stats) * 0.1))
        direct_peers = importer_stats.iloc[max(0, target_rank - rank_margin):min(len(importer_stats), target_rank + rank_margin + 1)]
    except IndexError: pass # 타겟 업체가 데이터에 없을 경우

    price_achievers_candidates = importer_stats[importer_stats['Trade_Count'] >= 2]
    price_achievers = price_achievers_candidates[price_achievers_candidates['Avg_UnitPrice'] <= price_achievers_candidates['Avg_UnitPrice'].quantile(0.15)] if not price_achievers_candidates.empty else pd.DataFrame()
    
    # 3-2. AI 기반 클러스터링
    clustered_stats, cluster_names = perform_clustering(importer_stats.copy())

    analysis_result['positioning'] = {
        "importer_stats": importer_stats,
        "clustered_stats": clustered_stats,
        "cluster_names": cluster_names,
        "rule_based_groups": {"Market Leaders": market_leaders, "Direct Peers": direct_peers, "Price Achievers": price_achievers},
        "target_stats": importer_stats[importer_stats['importer'] == target_importer_name]
    }

    # 4. 공급망 분석 (비용 절감 및 리스크 분석)
    user_input = user_inputs[0] # 대표 사용자 입력 사용
    user_avg_price = user_input['Value'] / user_input['Volume'] if user_input['Volume'] > 0 else 0
    
    # 대안 공급처 탐색 (사용자보다 저렴한 단가로 동일 제품군을 공급하는 수출업체)
    alternative_suppliers = analysis_data[
        (analysis_data['exporter'].str.upper() != user_input['Exporter'].upper()) &
        (analysis_data['unitPrice'] < user_avg_price)
    ]
    
    if not alternative_suppliers.empty:
        supplier_analysis = alternative_suppliers.groupby('exporter').agg(
            Avg_UnitPrice=('unitPrice', 'mean'),
            Trade_Count=('value', 'count'),
            Num_Importers=('importer', 'nunique')
        ).reset_index().sort_values('Avg_UnitPrice')
        
        # 기회와 안정성 지표 추가
        supplier_analysis['Price_Saving_Pct'] = (1 - supplier_analysis['Avg_UnitPrice'] / user_avg_price) * 100
        # 안정성 점수: 거래 건수와 거래처 수에 로그를 씌워 정규화 (한쪽에 치우치지 않게)
        supplier_analysis['Stability_Score'] = np.log1p(supplier_analysis['Trade_Count']) + np.log1p(supplier_analysis['Num_Importers'])
        
        analysis_result['supply_chain'] = {
            "user_avg_price": user_avg_price,
            "user_total_volume": sum(item['Volume'] for item in user_inputs),
            "alternatives": supplier_analysis
        }
        
    return analysis_result


# --- UI Components ---
def login_screen():
    st.title("🔐 수입 경쟁력 진단 솔루션")
    st.write("솔루션 접속을 위해 비밀번호를 입력해주세요.")
    with st.form("login_form"):
        password = st.text_input("비밀번호", type="password")
        if st.form_submit_button("접속하기"):
            if password == st.secrets.get("APP_PASSWORD", "tridgeDemo_2025"):
                st.session_state['logged_in'] = True
                st.rerun()
            else:
                st.error("비밀번호가 올바르지 않습니다.")

def main_dashboard(company_data):
    st.title("📈 수입 경쟁력 진단 솔루션")
    st.markdown("트릿지 데이터를 기반으로 시장 내 경쟁력을 진단하고 비용 절감 기회를 포착하세요.")
    
    # 사용자 입력 UI
    with st.expander("STEP 1: 분석 정보 입력", expanded='analysis_result' not in st.session_state):
        importer_name = st.text_input("1. 귀사의 업체명을 입력해주세요.", key="importer_name").upper()
        if 'rows' not in st.session_state: st.session_state['rows'] = [{'id': 1}]

        for i, row in enumerate(st.session_state.rows):
            # ... (기존 입력 UI 코드와 동일, 생략) ...
            pass # Keep your original detailed input UI code here
        
        # For demonstration, a simplified input is used below.
        # Replace with your full input UI.
        if 'product_name_0' not in st.session_state:
            st.session_state.product_name_0 = "Whisky A 12YO"
            st.session_state.hscode_0 = "220830"
            st.session_state.volume_0 = 1000
            st.session_state.value_0 = 50000
            st.session_state.final_exporter_0 = "DIAGEO"
        
        st.text_input("제품 상세명", key="product_name_0")
        st.text_input("HS-CODE", key="hscode_0")
        st.number_input("수입 중량(KG)", key="volume_0")
        st.number_input("총 수입금액(USD)", key="value_0")
        st.text_input("수출업체", key="final_exporter_0")

        if st.button("분석하기", type="primary"):
            with st.spinner('데이터를 분석하고 있습니다...'):
                all_purchase_data = []
                for i in range(len(st.session_state.get('rows', [{'id':1}]))):
                     entry = {
                        'Reported Product Name': st.session_state.get(f'product_name_{i}', ''),
                        'HS-CODE': st.session_state.get(f'hscode_{i}', ''),
                        'Exporter': st.session_state.get(f'final_exporter_{i}', '').upper(),
                        'Volume': st.session_state.get(f'volume_{i}', 0),
                        'Value': st.session_state.get(f'value_{i}', 0)
                     }
                     if not all([entry['Reported Product Name'], entry['Volume'] > 0, entry['Value'] > 0]):
                        st.error(f"수입 내역 {i+1}의 필수 값을 입력해주세요."); return
                     all_purchase_data.append(entry)

                purchase_df = pd.DataFrame(all_purchase_data)
                purchase_df['cleaned_name'] = purchase_df['Reported Product Name'].apply(clean_text)
                
                # 그룹화 로직 (동일 품목 합산)
                agg_funcs = {
                    'Volume': 'sum', 'Value': 'sum', 
                    'Reported Product Name': 'first', 'HS-CODE': 'first', 'Exporter': 'first'
                }
                aggregated_purchase_df = purchase_df.groupby('cleaned_name', as_index=False).agg(agg_funcs)

                analysis_groups = []
                company_data['cleaned_name'] = company_data['reported_product_name'].apply(clean_text)
                for _, row in aggregated_purchase_df.iterrows():
                    entry = row.to_dict()
                    user_tokens = set(entry['cleaned_name'].split())
                    is_match = lambda name: user_tokens.issubset(set(str(name).split()))
                    matched_df = company_data[company_data['cleaned_name'].apply(is_match)]
                    matched_products = sorted(matched_df['reported_product_name'].unique().tolist())
                    
                    # 그룹별 분석 실행
                    result = run_all_analysis([entry], company_data, matched_products, importer_name)
                    analysis_groups.append({
                        "user_input": entry,
                        "matched_products": matched_products,
                        "selected_products": matched_products,
                        "result": result
                    })

                st.session_state['importer_name_result'] = importer_name
                st.session_state['analysis_groups'] = analysis_groups

    if 'analysis_groups' in st.session_state:
        st.header("📊 분석 결과")
        for i, group in enumerate(st.session_state.analysis_groups):
            product_name = group['user_input']['Reported Product Name']
            st.subheader(f"분석 그룹: \"{product_name}\"")
            result = group['result']
            
            # --- PART 1: 마켓 포지션 분석 ---
            st.markdown("#### PART 1. 마켓 포지션 분석")
            p_res = result.get('positioning')
            if not p_res or p_res['importer_stats'].empty:
                st.info("포지션 분석을 위한 데이터가 부족합니다.")
                continue

            # 개선 1: 버블 차트 로직
            importer_stats = p_res['importer_stats']
            target_name = st.session_state.get('importer_name_result', '')
            
            top_5 = importer_stats.head(5)
            direct_peers = p_res['rule_based_groups']['Direct Peers']
            target_row = importer_stats[importer_stats['importer'] == target_name]
            
            plot_df = pd.concat([top_5, direct_peers, target_row]).drop_duplicates().reset_index(drop=True)
            
            # 익명화 및 귀사 강조
            plot_df['Anonymized_Importer'] = [f"{chr(ord('A')+j)}사" if imp != target_name else target_name for j, imp in enumerate(plot_df['importer'])]
            plot_df['size'] = np.log1p(plot_df['Total_Value']) # 로그 스케일링으로 버블 크기 조절
            plot_df['color'] = np.where(plot_df['importer'] == target_name, '귀사', '경쟁사')
            plot_df['symbol'] = np.where(plot_df['importer'] == target_name, 'star', 'circle')

            fig = px.scatter(
                plot_df, x='Total_Volume', y='Avg_UnitPrice', size='size', 
                color='color', symbol='symbol',
                hover_name='Anonymized_Importer',
                hover_data={'Total_Volume': ':,', 'Avg_UnitPrice': ':.2f', 'Total_Value':':,', 'size':False, 'color':False, 'symbol':False},
                log_x=True,
                title="수입사 포지셔닝 맵 (Top 5, 유사 경쟁사, 및 귀사)",
                labels={'Total_Volume': '총 수입 중량 (KG, Log Scale)', 'Avg_UnitPrice': '평균 수입 단가 (USD/KG)'},
                color_discrete_map={'귀사': 'orange', '경쟁사': 'grey'},
                symbol_sequence=['star', 'circle']
            )
            fig.update_traces(marker_line_width=1, marker_line_color='black')
            st.plotly_chart(fig, use_container_width=True)

            # 개선 2: 그룹 분류 방식 선택
            st.markdown("##### 경쟁사 그룹 분석")
            grouping_method = st.radio("그룹 분류 방식 선택:", ["규칙 기반 그룹", "AI 기반 자동 그룹핑 (K-Means)"], horizontal=True, key=f"group_method_{i}")

            if grouping_method == "규칙 기반 그룹":
                groups = p_res['rule_based_groups']
                st.info("시장 선도(누적 점유율 70%), 유사 규모(순위 ±10%), 최저가 달성(단가 하위 15%) 그룹으로 분류합니다.")
                # 여기에 규칙 기반 그룹에 대한 시각화 (예: Box Plot) 추가 가능
                
            else: # AI 기반 자동 그룹핑
                if p_res.get('clustered_stats') is not None:
                    st.info("AI가 수입 규모, 단가, 빈도를 종합하여 유사한 업체끼리 3개의 그룹으로 자동 분류합니다.")
                    fig_cluster_box = px.box(p_res['clustered_stats'], x='Cluster_Name', y='Avg_UnitPrice',
                                             title="AI 기반 그룹별 단가 분포", points='all',
                                             labels={'Cluster_Name': '그룹 유형', 'Avg_UnitPrice': '평균 수입 단가'})
                    # 귀사 위치 점선으로 표시
                    if not p_res['target_stats'].empty:
                        target_price = p_res['target_stats']['Avg_UnitPrice'].iloc[0]
                        fig_cluster_box.add_hline(y=target_price, line_dash="dot", line_color="orange", annotation_text="귀사 단가")
                    st.plotly_chart(fig_cluster_box, use_container_width=True)
                else:
                    st.warning("데이터가 부족하여 AI 기반 그룹핑을 수행할 수 없습니다.")


            # --- PART 2: 공급망 분석 ---
            st.markdown("---")
            st.markdown("#### PART 2. 공급망 분석 및 비용 절감 시뮬레이션")
            s_res = result.get('supply_chain')
            if not s_res or s_res['alternatives'].empty:
                st.info("현재 거래 조건보다 더 저렴한 대안 공급처를 찾지 못했습니다.")
            else:
                # 개선 3: 기회와 리스크 동시 제시
                alts = s_res['alternatives']
                best_deal = alts.iloc[0]

                st.success(f"**비용 절감 기회 포착!** 현재 거래처보다 **최대 {best_deal['Price_Saving_Pct']:.1f}%** 저렴한 단가로 공급하는 대체 거래처가 존재합니다.")
                
                col1, col2 = st.columns(2)
                with col1:
                    target_saving_pct = st.slider(
                        "목표 단가 절감률(%)을 설정하세요:",
                        min_value=0.0, max_value=float(best_deal['Price_Saving_Pct']),
                        value=float(best_deal['Price_Saving_Pct'] / 2),
                        step=0.5, format="%.1f%%"
                    )
                
                user_total_volume = s_res['user_total_volume']
                user_avg_price = s_res['user_avg_price']
                expected_saving_amount = user_total_volume * user_avg_price * (target_saving_pct / 100)

                with col2:
                    st.metric(
                        label=f"예상 절감 금액 (연간 수입량 {user_total_volume:,.0f}KG 기준)",
                        value=f"${expected_saving_amount:,.0f}"
                    )

                st.markdown("##### **추천 대체 공급처 리스트**")
                st.info("공급처의 '가격 경쟁력'과 '공급 안정성'을 함께 고려하여 전략적으로 판단하세요.")
                
                # 추천 리스트 필터링 및 표시
                recommended_list = alts[alts['Price_Saving_Pct'] >= target_saving_pct].copy()
                recommended_list.rename(columns={
                    'exporter': '수출업체', 'Avg_UnitPrice': '평균 단가 (USD)', 
                    'Price_Saving_Pct': '가격 경쟁력 (%)', 'Trade_Count': '거래 빈도 (건)', 
                    'Num_Importers': '거래처 수', 'Stability_Score': '공급 안정성 점수'
                }, inplace=True)
                
                st.dataframe(
                    recommended_list[['수출업체', '평균 단가 (USD)', '가격 경쟁력 (%)', '거래 빈도 (건)', '거래처 수', '공급 안정성 점수']],
                    use_container_width=True,
                    column_config={
                        "평균 단가 (USD)": st.column_config.NumberColumn(format="$%.2f"),
                        "가격 경쟁력 (%)": st.column_config.ProgressColumn(
                            format="%.1f%%", min_value=0, max_value=alts['Price_Saving_Pct'].max()
                        ),
                        "공급 안정성 점수": st.column_config.BarChartColumn(y_min=0, y_max=alts['Stability_Score'].max())
                    },
                    hide_index=True
                )
            st.markdown("---")

# --- 메인 로직 실행 ---
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

if st.session_state['logged_in']:
    company_data = load_company_data()
    if company_data is not None:
        main_dashboard(company_data)
    else:
        st.error("데이터를 불러오는 데 실패했습니다. 앱 설정을 확인하거나 잠시 후 다시 시도해주세요.")
else:
    login_screen()
