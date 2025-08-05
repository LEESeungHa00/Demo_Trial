import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import re
from datetime import datetime
import gspread
from google.oauth2.service_account import Credentials
from pandas_gbq import read_gbq
import calendar
from zoneinfo import ZoneInfo

# --- 초기 설정 및 페이지 구성 ---
st.set_page_config(layout="wide", page_title="수입 경쟁력 진단 솔루션")

# --- 데이터 로딩 (BigQuery 방식에 맞춤) ---
@st.cache_data(ttl=3600)
def load_company_data():
    """Google BigQuery에서 TDS를 불러옵니다."""
    try:
        if "gcp_service_account" not in st.secrets:
            st.error("Secrets 설정 오류: [gcp_service_account] 섹션이 없습니다.")
            return None
        creds = Credentials.from_service_account_info(st.secrets["gcp_service_account"])
        project_id = st.secrets["gcp_service_account"]["project_id"]
        dataset_id = "demo_data" 
        table_id = "tds_data"   
        table_full_id = f"{project_id}.{dataset_id}.{table_id}"
        dataset_location = "asia-northeast3" 
        query = f"SELECT * FROM `{table_full_id}`"
        df = read_gbq(query, project_id=project_id, credentials=creds, location=dataset_location)
        
        if df.empty:
            st.error("BigQuery 테이블에서 데이터를 불러왔지만 비어있습니다.")
            return None

        df.columns = [re.sub(r'[^a-z0-9]+', '_', col.lower().strip()) for col in df.columns]

        required_cols = ['date', 'volume', 'value', 'reported_product_name', 'export_country', 'exporter', 'importer', 'hs_code']
        for col in required_cols:
            if col not in df.columns:
                st.error(f"BigQuery 테이블 오류: 필수 컬럼 '{col}'이 없습니다.")
                st.info(f"BigQuery 테이블의 실제 컬럼명 (수정 후): {df.columns.tolist()}")
                return None
        
        def smart_numeric_conversion(series):
            if pd.api.types.is_numeric_dtype(series): return pd.to_numeric(series, errors='coerce')
            series_str = series.astype(str)
            series_cleaned = series_str.str.replace(r'[^\d.]', '', regex=True)
            return pd.to_numeric(series_cleaned, errors='coerce')

        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['volume'] = smart_numeric_conversion(df['volume'])
        df['value'] = smart_numeric_conversion(df['value'])
        df.dropna(subset=['date', 'volume', 'value'], inplace=True)
        
        if df.empty:
            st.error("데이터 정제 후 유효한 데이터가 없습니다.")
            return None
        return df
    except Exception as e:
        st.error(f"데이터 로딩 중 심각한 오류가 발생했습니다:")
        st.exception(e)
        return None

# --- 분석 헬퍼 함수 ---
def create_monthly_frequency_chart(df, title):
    df['date'] = pd.to_datetime(df['date'])
    end_date = datetime.now()
    start_date = end_date - pd.DateOffset(years=1)
    df_filtered = df[(df['date'] >= start_date) & (df['date'] <= end_date)].copy()
    if df_filtered.empty: return None
    df_filtered['Month'] = df_filtered['date'].dt.to_period('M').astype(str)
    monthly_counts = df_filtered.groupby('Month').size().reset_index(name='counts')
    all_months = pd.date_range(start=start_date, end=end_date, freq='MS').to_period('M').astype(str)
    all_months_df = pd.DataFrame({'Month': all_months})
    monthly_counts = pd.merge(all_months_df, monthly_counts, on='Month', how='left').fillna(0)
    fig = px.bar(monthly_counts, x='Month', y='counts', title=title, labels={'Month': '월', 'counts': '수입 건수'})
    fig.update_layout(margin=dict(t=40, b=20, l=40, r=20), height=300, plot_bgcolor='white')
    return fig

# --- 새로운 범용 스마트 매칭 로직 ---
def clean_text(text):
    if not isinstance(text, str): return ''
    text = text.lower()
    text = re.sub(r'\(.*?\)|\[.*?\]', ' ', text)
    text = re.sub(r'(\d+)\s*(?:y|yo|year|years|old|년)', r'\1', text)
    text = re.sub(r'[^a-z0-9\s\uac00-\ud7a3]', ' ', text)
    return ' '.join(text.split())

# --- 메인 분석 로직 ---
def run_all_analysis(user_input, company_data, target_importer_name):
    analysis_result = {"overview": None, "positioning": None, "supply_chain": None}
    
    company_data['unitPrice'] = company_data['value'] / company_data['volume']
    
    # 0. Overview 분석
    hscode_data = company_data[company_data['hs_code'] == user_input['HS-CODE']]
    if not hscode_data.empty:
        this_year = datetime.now().year
        last_year = this_year - 1
        
        vol_this_year = hscode_data[hscode_data['date'].dt.year == this_year]['volume'].sum()
        vol_last_year = hscode_data[hscode_data['date'].dt.year == last_year]['volume'].sum()
        price_this_year = hscode_data[hscode_data['date'].dt.year == this_year]['unitPrice'].mean()
        price_last_year = hscode_data[hscode_data['date'].dt.year == last_year]['unitPrice'].mean()

        analysis_result['overview'] = {
            "vol_this_year": vol_this_year, "vol_last_year": vol_last_year,
            "price_this_year": price_this_year, "price_last_year": price_last_year,
            "freq_this_year": len(hscode_data[hscode_data['date'].dt.year == this_year]),
            "product_composition": hscode_data.groupby('reported_product_name')['value'].sum().reset_index()
        }

    # 1. 포지셔닝 분석
    importer_stats = company_data.groupby('importer').agg(
        Total_Value=('value', 'sum'), Total_Volume=('volume', 'sum'), Trade_Count=('value', 'count')
    ).reset_index()
    
    if not importer_stats.empty and importer_stats['Total_Volume'].sum() > 0:
        importer_stats['Avg_UnitPrice'] = importer_stats['Total_Value'] / importer_stats['Total_Volume']
        importer_stats = importer_stats.sort_values('Total_Value', ascending=False).reset_index(drop=True)

        total_market_value = importer_stats['Total_Value'].sum()
        importer_stats['cum_share'] = importer_stats['Total_Value'].cumsum() / total_market_value
        market_leaders = importer_stats[importer_stats['cum_share'] <= 0.7]

        try:
            target_rank = importer_stats[importer_stats['importer'] == target_importer_name].index[0]
            rank_margin = int(len(importer_stats) * 0.1)
            peer_min_rank, peer_max_rank = max(0, target_rank - rank_margin), min(len(importer_stats), target_rank + rank_margin + 1)
            direct_peers = importer_stats.iloc[peer_min_rank:peer_max_rank]
        except IndexError: direct_peers = pd.DataFrame()

        price_achievers_candidates = importer_stats[importer_stats['Trade_Count'] >= 1]
        if not price_achievers_candidates.empty:
            price_quantile = price_achievers_candidates['Avg_UnitPrice'].quantile(0.15)
            price_achievers = price_achievers_candidates[price_achievers_candidates['Avg_UnitPrice'] <= price_quantile]
        else: price_achievers = pd.DataFrame()
        
        analysis_result['positioning'] = {
            "bubble_data": importer_stats, "market_leaders": market_leaders,
            "direct_peers": direct_peers, "price_achievers": price_achievers,
            "target_stats": importer_stats[importer_stats['importer'] == target_importer_name]
        }
    
    # 2. 공급망 분석
    # ... (생략)
    
    return analysis_result

# --- UI Components ---
def login_screen():
    st.title("🔐 수입 경쟁력 진단 솔루션")
    st.write("솔루션 접속을 위해 비밀번호를 입력해주세요.")
    with st.form("login_form", clear_on_submit=True):
        password = st.text_input("비밀번호", type="password")
        submitted = st.form_submit_button("접속하기")
        if submitted:
            if password == "tridgeDemo_2025":
                st.session_state['logged_in'] = True
                st.rerun()
            else: st.error("비밀번호가 올바르지 않습니다.")

def main_dashboard(company_data):
    st.title("📈 수입 경쟁력 진단 솔루션")
    st.markdown("트릿지 데이터를 기반으로 시장 내 경쟁력을 진단하고 비용 절감 기회를 포착하세요.")

    with st.expander("STEP 1: 분석 정보 입력", expanded='analysis_groups' not in st.session_state):
        importer_name = st.text_input("1. 귀사의 업체명을 입력해주세요.", key="importer_name").upper()
        if 'rows' not in st.session_state: st.session_state['rows'] = [{'id': 1}]
        
        for i, row in enumerate(st.session_state.rows):
            cols = st.columns([2, 3, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1])
            cols[0].date_input("수입일", key=f"date_{i}")
            cols[1].text_input("제품 상세명", placeholder="예 : 엑스트라버진 올리브유", key=f"product_name_{i}")
            cols[2].text_input("HS-CODE(6자리)", max_chars=6, key=f"hscode_{i}")
            
            origin_options = [''] + ['직접 입력'] + sorted(company_data['export_country'].unique())
            selected_origin = cols[3].selectbox("원산지", origin_options, key=f"origin_{i}", format_func=lambda x: '선택 또는 직접 입력' if x == '' else x)
            if selected_origin == '직접 입력':
                cols[3].text_input("└ 원산지 직접 입력", key=f"custom_origin_{i}", placeholder="직접 입력하세요")

            exporter_options = [''] + ['직접 입력'] + sorted(company_data['exporter'].unique())
            selected_exporter = cols[4].selectbox("수출업체", exporter_options, key=f"exporter_{i}", format_func=lambda x: '선택 또는 직접 입력' if x == '' else x)
            if selected_exporter == '직접 입력':
                cols[4].text_input("└ 수출업체 직접 입력", key=f"custom_exporter_{i}", placeholder="직접 입력하세요")

            cols[5].number_input("수입 중량(KG)", min_value=0.01, format="%.2f", key=f"volume_{i}")
            cols[6].number_input("총 수입금액(USD)", min_value=0.01, format="%.2f", key=f"value_{i}")
            cols[7].selectbox("Incoterms", ["FOB", "CFR", "CIF", "EXW", "DDP", "기타"], key=f"incoterms_{i}")
            if len(st.session_state.rows) > 1 and cols[8].button("삭제", key=f"delete_{i}"):
                st.session_state.rows.pop(i)
                st.rerun()

        if st.button("➕ 내역 추가하기"):
            st.session_state.rows.append({'id': len(st.session_state.rows) + 1})
            st.rerun()
        st.markdown("---")
        consent = st.checkbox("입력하신 정보는 데이터 분석 품질 향상을 위해 저장 및 활용되는 것에 동의합니다.")
        analyze_button = st.button("분석하기", type="primary")

    if analyze_button:
        if not importer_name: st.warning("수입업체명을 입력해주세요.")
        elif not consent: st.warning("데이터 활용 동의에 체크해주세요.")
        else:
            with st.spinner('데이터를 분석하고 시트에 저장 중입니다...'):
                all_purchase_data = []
                for i in range(len(st.session_state.rows)):
                    user_product_name = st.session_state[f'product_name_{i}']
                    origin_val = st.session_state[f'origin_{i}']
                    if origin_val == '직접 입력': origin_val = st.session_state.get(f'custom_origin_{i}', "")
                    exporter_val = st.session_state[f'exporter_{i}']
                    if exporter_val == '직접 입력': exporter_val = st.session_state.get(f'custom_exporter_{i}', "")
                    entry = { 'Date': st.session_state[f'date_{i}'], 'Reported Product Name': user_product_name, 'HS-CODE': st.session_state[f'hscode_{i}'], 'Origin Country': origin_val.upper(), 'Exporter': exporter_val.upper(), 'Volume': st.session_state[f'volume_{i}'], 'Value': st.session_state[f'value_{i}'], 'Incoterms': st.session_state[f'incoterms_{i}'] }
                    if not user_product_name or not origin_val or not exporter_val:
                        st.error(f"{i+1}번째 행의 '제품 상세명', '원산지', '수출업체'는 필수 입력 항목입니다.")
                        return
                    all_purchase_data.append(entry)
                
                # 중복 제품 합산 로직
                purchase_df = pd.DataFrame(all_purchase_data)
                agg_funcs = {'Volume': 'sum', 'Value': 'sum', 'Date': 'first', 'HS-CODE': 'first', 'Origin Country': 'first', 'Exporter': 'first', 'Incoterms': 'first'}
                aggregated_purchase_df = purchase_df.groupby('Reported Product Name').agg(agg_funcs).reset_index()

                analysis_groups = []
                company_data['cleaned_name'] = company_data['reported_product_name'].apply(clean_text)
                
                for i, row in aggregated_purchase_df.iterrows():
                    entry = row.to_dict()
                    user_tokens = set(clean_text(entry['Reported Product Name']).split())
                    def is_match(cleaned_tds_name): return user_tokens.issubset(set(cleaned_tds_name.split()))
                    matched_df = company_data[company_data['cleaned_name'].apply(is_match)]
                    analysis_groups.append({ "id": i, "user_input": entry, "matched_products": sorted(matched_df['reported_product_name'].unique().tolist()), "selected_products": sorted(matched_df['reported_product_name'].unique().tolist()) })

                try:
                    # ... (Google Sheets 저장 로직) ...
                    pass
                except Exception as e:
                    st.error(f"Google Sheets 저장 중 예상치 못한 오류가 발생했습니다: {e}")

                st.session_state['importer_name_result'] = importer_name
                st.session_state['analysis_groups'] = analysis_groups
                st.rerun()

    if 'analysis_groups' in st.session_state:
        st.header("📊 분석 결과")
        
        with st.expander("STEP 2: 분석 대상 제품 필터링", expanded=True):
            for i, group in enumerate(st.session_state.analysis_groups):
                st.markdown(f"**분석 그룹: \"{group['user_input']['Reported Product Name']}\"**")
                selected = st.multiselect(
                    "이 그룹의 분석에 활용할 제품명을 선택하세요.",
                    options=group['matched_products'],
                    default=group['selected_products'],
                    key=f"filter_{group['id']}"
                )
                st.session_state.analysis_groups[i]['selected_products'] = selected
                st.markdown("---")

        for group in st.session_state.analysis_groups:
            st.subheader(f"분석 결과: \"{group['user_input']['Reported Product Name']}\"")
            
            if not group['selected_products']:
                st.warning("선택된 비교 대상 제품이 없어 분석을 건너뜁니다.")
                continue

            analysis_data = company_data[company_data['reported_product_name'].isin(group['selected_products'])]
            result = run_all_analysis(group['user_input'], analysis_data, st.session_state['importer_name_result'])

            # 0. Overview 표시
            st.markdown("### 0. Overview")
            if result.get('overview'):
                o = result['overview']
                st.markdown(f"#### HS-Code {group['user_input']['HS-CODE']}의 수입 전반 요약")
                # ... (결과 표시)
            else:
                st.info("HS-Code에 해당하는 데이터가 부족하여 Overview 분석을 생략합니다.")

            # 1. Positioning 표시
            st.markdown(f"### 1. {st.session_state['importer_name_result']}을 위한 수입 진단 및 포지셔닝 결과")
            if result.get('positioning'):
                p = result['positioning']
                st.markdown("#### PART 1. 마켓 포지션 분석")
                
                all_importers = p['bubble_data']['importer'].unique()
                anonymity_map = {name: f"{chr(65+i)}사" for i, name in enumerate(all_importers) if name != st.session_state['importer_name_result']}
                
                bubble_df = p['bubble_data'].copy()
                bubble_df['Anonymized_Importer'] = bubble_df['importer'].apply(lambda x: "귀사" if x == st.session_state['importer_name_result'] else anonymity_map.get(x, "기타"))
                
                fig_bubble = px.scatter(bubble_df, x='Total_Volume', y='Avg_UnitPrice', size='Total_Value', color='Anonymized_Importer',
                                        hover_name='Anonymized_Importer', size_max=60,
                                        labels={'Total_Volume': '수입 총 중량 (KG)', 'Avg_UnitPrice': '평균 수입 단가 (USD/KG)'})
                st.plotly_chart(fig_bubble, use_container_width=True)

                st.markdown("##### 지난 12개월간 월별 수입 빈도")
                target_df = company_data[company_data['importer'] == st.session_state['importer_name_result']]
                fig_target_freq = create_monthly_frequency_chart(target_df, "귀사")
                if fig_target_freq: st.plotly_chart(fig_target_freq, use_container_width=True)
                else: st.info("귀사의 지난 1년간 수입 데이터가 없습니다.")

            else:
                st.info("데이터가 부족하여 포지셔닝 분석을 생략합니다.")

        if st.button("🔄 새로운 분석 시작하기"):
            keys_to_keep = ['logged_in']
            for key in list(st.session_state.keys()):
                if key not in keys_to_keep: del st.session_state[key]
            st.rerun()

# --- 메인 로직 ---
if 'logged_in' not in st.session_state: 
    st.session_state['logged_in'] = False

if st.session_state['logged_in']:
    our_company_data = load_company_data()
    if our_company_data is not None:
        main_dashboard(our_company_data)
else:
    login_screen()
