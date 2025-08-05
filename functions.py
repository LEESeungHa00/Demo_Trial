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

# --- 초기 설정 및 페이지 구성 ---
st.set_page_config(layout="wide", page_title="수입 경쟁력 진단 솔루션")

# --- Google BigQuery에서 데이터 불러오기 (진단 기능 강화) ---
@st.cache_data(ttl=3600)
def load_company_data():
    """Google BigQuery에서 TDS를 불러옵니다."""
    try:
        if "gcp_service_account" not in st.secrets:
            st.error("Secrets 설정 오류: `secrets.toml` 파일에 [gcp_service_account] 섹션이 없습니다.")
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

        df.columns = [col.replace('_', ' ').title() for col in df.columns]

        required_cols = ['Date', 'Volume', 'Value', 'Reported Product Name', 'Export Country', 'Exporter']
        for col in required_cols:
            if col not in df.columns:
                st.error(f"BigQuery 테이블 오류: 필수 컬럼 '{col}'이 없습니다.")
                return None
        
        df.dropna(how="all", inplace=True)
        
        # 최종 수정: 데이터 정제 전, 어떤 데이터가 문제인지 진단하는 기능 추가
        df_original = df.copy()

        def clean_and_convert_numeric(series):
            series_str = series.astype(str)
            series_cleaned = series_str.str.replace(r'[^\d.]', '', regex=True)
            return pd.to_numeric(series_cleaned, errors='coerce')

        df['Date_converted'] = pd.to_datetime(df['Date'], errors='coerce')
        df['Volume_converted'] = clean_and_convert_numeric(df['Volume'])
        df['Value_converted'] = clean_and_convert_numeric(df['Value'])
        
        problematic_rows = df[df['Date_converted'].isnull() | df['Volume_converted'].isnull() | df['Value_converted'].isnull()]
        
        df = df_original
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df['Volume'] = clean_and_convert_numeric(df['Volume'])
        df['Value'] = clean_and_convert_numeric(df['Value'])
        df.dropna(subset=['Date', 'Volume', 'Value'], inplace=True)

        if df.empty:
            st.error("데이터 정제 후 남은 데이터가 없습니다.")
            st.info("BigQuery 테이블의 'Date', 'Volume', 'Value' 컬럼에 유효한 데이터가 있는지 확인해주세요.")
            if not problematic_rows.empty:
                st.warning("아래는 데이터 타입 변환에 실패한 행의 예시입니다. 원본 데이터(Google Sheets)의 형식을 확인해주세요:")
                st.dataframe(problematic_rows[['Date', 'Volume', 'Value']].head())
            return None
            
        return df
    except Exception as e:
        st.error(f"데이터 로딩 중 심각한 오류가 발생했습니다:")
        st.exception(e)
        return None

# --- 새로운 범용 스마트 매칭 로직 ---
def clean_text(text):
    if not isinstance(text, str): return ''
    text = text.lower()
    text = re.sub(r'\(.*?\)', ' ', text)
    text = re.sub(r'\[.*?\]', ' ', text)
    text = re.sub(r'(\d+)\s*(?:y|yo|year|years|old|년)', r'\1', text)
    text = re.sub(r'[^a-z0-9\s\uac00-\ud7a3]', ' ', text)
    return ' '.join(text.split())

# --- 데이터 처리 로직 (개별 제품 분석 지원) ---
def process_analysis_data(user_input_row, comparison_df, target_importer_name):
    target_df = pd.DataFrame([user_input_row])
    target_df['Date'] = pd.to_datetime(target_df['Date'])
    
    if comparison_df.empty or target_df.empty:
        return {}, {}, {}

    target_df['Importer'] = target_importer_name.upper()
    all_df = pd.concat([comparison_df, target_df], ignore_index=True)
    all_df['Value'] = pd.to_numeric(all_df['Value'], errors='coerce')
    all_df['Volume'] = pd.to_numeric(all_df['Volume'], errors='coerce')
    all_df.dropna(subset=['Value', 'Volume'], inplace=True)
    all_df = all_df[all_df['Volume'] > 0]

    all_df['unitPrice'] = all_df['Value'] / all_df['Volume']
    all_df['year'] = all_df['Date'].dt.year
    all_df['monthYear'] = all_df['Date'].dt.to_period('M').astype(str)

    competitor_analysis = {}
    yearly_analysis = {}
    time_series_analysis = {}

    for _, row in target_df.iterrows():
        year = row['Date'].year
        exporter = row['Exporter'].upper()
        key = (year, exporter)
        related_trades = all_df[(all_df['year'] == year) & (all_df['Exporter'].str.upper() == exporter)]
        if not related_trades.empty:
            importer_median_prices = related_trades.groupby('Importer')['unitPrice'].median().sort_values().reset_index()
            top5_importers = importer_median_prices.head(5)['Importer'].tolist()
            
            selected_importers = top5_importers
            target_importer_name_upper = target_importer_name.upper()
            if target_importer_name_upper not in selected_importers:
                if target_importer_name_upper in related_trades['Importer'].unique():
                     selected_importers.append(target_importer_name_upper)

            box_plot_data = related_trades[related_trades['Importer'].isin(selected_importers)]
            competitor_analysis[key] = box_plot_data
        
        origin = row['Origin Country'].upper()
        key_yearly = (exporter, origin)
        target_unit_price_yearly = row['Value'] / row['Volume']
        other_companies_yearly = all_df[
            (all_df['Exporter'].str.upper() == exporter) &
            (all_df['Origin Country'].str.upper() == origin) &
            (all_df['Importer'].str.upper() != target_importer_name.upper()) &
            (all_df['unitPrice'] < target_unit_price_yearly)
        ]
        saving_info_yearly = None
        if not other_companies_yearly.empty:
            avg_unit_price = other_companies_yearly['Value'].sum() / other_companies_yearly['Volume'].sum()
            potential_saving = (target_unit_price_yearly - avg_unit_price) * row['Volume']
            saving_info_yearly = {'potential_saving': potential_saving}
        yearly_data = all_df[(all_df['Exporter'].str.upper() == exporter) & (all_df['Origin Country'].str.upper() == origin)]
        summary = yearly_data.groupby('year').agg(volume=('Volume', 'sum'), value=('Value', 'sum')).reset_index()
        summary['unitPrice'] = summary['value'] / summary['volume']
        yearly_analysis[key_yearly] = {'chart_data': summary, 'saving_info': saving_info_yearly}

        key_ts = origin
        related_trades_ts = all_df[all_df['Origin Country'].str.upper() == origin]
        monthly_summary = related_trades_ts.groupby('monthYear').agg(avgPrice=('unitPrice', 'mean'), bestPrice=('unitPrice', 'min')).reset_index()
        target_trades_ts = related_trades_ts[related_trades_ts['Importer'].str.upper() == target_importer_name.upper()]
        target_monthly = target_trades_ts.groupby('monthYear').agg(targetPrice=('unitPrice', 'mean')).reset_index()
        chart_data_ts = pd.merge(monthly_summary, target_monthly, on='monthYear', how='left').sort_values('monthYear')
        target_unit_price_ts = row['Value'] / row['Volume']
        cheaper_trades_ts = all_df[(all_df['Origin Country'].str.upper() == origin) & (all_df['unitPrice'] < target_unit_price_ts)]
        saving_info_ts = None
        if not cheaper_trades_ts.empty:
            avg_unit_price_ts = cheaper_trades_ts['Value'].sum() / cheaper_trades_ts['Volume'].sum()
            potential_saving_ts = (target_unit_price_ts - avg_unit_price_ts) * row['Volume']
            saving_info_ts = {'potential_saving': potential_saving_ts}
        time_series_analysis[key_ts] = {'chart_data': chart_data_ts, 'saving_info': saving_info_ts}

    return competitor_analysis, yearly_analysis, time_series_analysis

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
            else:
                st.error("비밀번호가 올바르지 않습니다.")

def main_dashboard(company_data):
    st.title("📈 수입 경쟁력 진단 솔루션")
    st.markdown("트릿지 데이터를 기반으로 시장 내 경쟁력을 진단하고 비용 절감 기회를 포착하세요.")

    with st.expander("STEP 1: 분석 정보 입력", expanded='analysis_groups' not in st.session_state):
        importer_name = st.text_input("1. 귀사의 업체명을 입력해주세요.", key="importer_name").upper()
        st.markdown("---")
        st.markdown("2. 분석할 구매 내역을 입력해주세요. (여러 품목 입력 가능)")
        if 'rows' not in st.session_state: st.session_state['rows'] = [{'id': 1}]
        
        for i, row in enumerate(st.session_state.rows):
            cols = st.columns([2, 3, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1])
            cols[0].date_input("수입일", key=f"date_{i}")
            cols[1].text_input("제품 상세명", placeholder="예 : 엑스트라버진 올리브유", key=f"product_name_{i}")
            cols[2].text_input("HS-CODE(6자리)", max_chars=6, key=f"hscode_{i}")
            
            origin_options = ['직접 입력'] + sorted(company_data['Export Country'].unique())
            selected_origin = cols[3].selectbox("원산지", origin_options, key=f"origin_{i}")
            if selected_origin == '직접 입력':
                cols[3].text_input("└ 원산지 직접 입력", key=f"custom_origin_{i}", placeholder="직접 입력하세요")

            exporter_options = ['직접 입력'] + sorted(company_data['Exporter'].unique())
            selected_exporter = cols[4].selectbox("수출업체", exporter_options, key=f"exporter_{i}")
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
                analysis_groups = []
                all_purchase_data = []
                
                company_data['cleaned_name'] = company_data['Reported Product Name'].apply(clean_text)
                
                for i in range(len(st.session_state.rows)):
                    user_product_name = st.session_state[f'product_name_{i}']
                    
                    origin_val = st.session_state[f'origin_{i}']
                    if origin_val == '직접 입력':
                        origin_val = st.session_state.get(f'custom_origin_{i}', "")

                    exporter_val = st.session_state[f'exporter_{i}']
                    if exporter_val == '직접 입력':
                        exporter_val = st.session_state.get(f'custom_exporter_{i}', "")

                    entry = {
                        'Date': st.session_state[f'date_{i}'],
                        'Reported Product Name': user_product_name,
                        'HS-CODE': st.session_state[f'hscode_{i}'],
                        'Origin Country': origin_val.upper(),
                        'Exporter': exporter_val.upper(),
                        'Volume': st.session_st
