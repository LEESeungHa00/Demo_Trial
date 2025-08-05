import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import re
from datetime import datetime
from dateutil.relativedelta import relativedelta
from google.oauth2.service_account import Credentials
from pandas_gbq import read_gbq
import gspread
from zoneinfo import ZoneInfo

# --- 페이지 초기 설정 ---
st.set_page_config(layout="wide", page_title="수입 경쟁력 진단 솔루션")

# --- API 사용 범위(Scope) 정의 ---
# "이 서비스 계정으로 아래 API들을 사용하겠습니다"라고 명시적으로 선언합니다.
SCOPES = [
    'https://www.googleapis.com/auth/spreadsheets',
    'https://www.googleapis.com/auth/drive',
    'https://www.googleapis.com/auth/bigquery'
]

# --- 데이터 로딩 (BigQuery) ---
@st.cache_data(ttl=3600)
def load_company_data():
    """Google BigQuery에서 데이터를 불러오고 기본 전처리를 수행합니다."""
    try:
        # 인증 정보에 SCOPES를 포함하여 Credentials 객체 생성
        creds = Credentials.from_service_account_info(st.secrets["gcp_service_account"], scopes=SCOPES)
        project_id = st.secrets["gcp_service_account"]["project_id"]
        table_full_id = f"{project_id}.demo_data.tds_data"
        df = read_gbq(f"SELECT * FROM `{table_full_id}`", project_id=project_id, credentials=creds)
        
        df.columns = [re.sub(r'[^a-z0-9]+', '_', col.lower().strip()) for col in df.columns]
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        for col in ['volume', 'value']:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(r'[^\d.]', '', regex=True), errors='coerce')
        df.dropna(subset=['date', 'volume', 'value', 'importer', 'exporter'], inplace=True)
        df = df[(df['volume'] > 0) & (df['value'] > 0)].copy()
        df['unitprice'] = df['value'] / df['volume']
        Q1, Q3 = df['unitprice'].quantile(0.25), df['unitprice'].quantile(0.75)
        IQR = Q3 - Q1
        df = df[~((df['unitprice'] < (Q1 - 1.5 * IQR)) | (df['unitprice'] > (Q3 + 1.5 * IQR)))]
        return df
    except Exception as e: st.error(f"데이터 로딩 중 오류: {e}"); return None

# --- Google Sheets 저장 (사용자 제공 로직 기반으로 전면 교체) ---
def save_to_google_sheets(purchase_df, importer_name, consent):
    """사용자 입력 데이터프레임을 지정된 구글 시트에 저장합니다."""
    try:
        creds = Credentials.from_service_account_info(st.secrets["gcp_service_account"], scopes=SCOPES)
        client = gspread.authorize(creds)
        spreadsheet = client.open(st.secrets.get("google_sheets", {}).get("spreadsheet_name", "DEMO_app_DB"))
        worksheet_name = st.secrets.get("google_sheets", {}).get("worksheet_name", "Customer_input")

        try:
            worksheet = spreadsheet.worksheet(worksheet_name)
        except gspread.exceptions.WorksheetNotFound:
            worksheet = spreadsheet.add_worksheet(title=worksheet_name, rows=1, cols=20)

        save_data_df = purchase_df.copy()
        save_data_df['importer_name'] = importer_name
        save_data_df['consent'] = consent
        save_data_df['timestamp'] = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d %H:%M:%S")
        
        # 날짜 포맷팅 및 전체 문자열 변환
        save_data_df['Date'] = save_data_df['Date'].dt.strftime('%Y-%m-%d')
        save_data_df = save_data_df.astype(str)
        
        # 헤더 순서 정렬
        final_columns = ["Date", "Reported Product Name", "HS-Code", "Origin Country", "Exporter", "Volume", "Value", "Incoterms", "importer_name", "consent", "timestamp"]
        save_data_df = save_data_df[final_columns]
        
        if not worksheet.get('A1'):
            worksheet.update([save_data_df.columns.values.tolist()] + save_data_df.values.tolist(), value_input_option='USER_ENTERED')
        else:
            worksheet.append_rows(save_data_df.values.tolist(), value_input_option='USER_ENTERED')

        st.toast("입력 정보가 Google Sheet에 저장되었습니다.", icon="✅")
        return True
    except gspread.exceptions.APIError as e:
        st.error("Google Sheets API 오류로 저장에 실패했습니다. GCP에서 API가 활성화되었는지 확인하세요.")
        st.json(e.response.json())
        return False
    except Exception as e:
        st.error(f"Google Sheets 저장 중 예상치 못한 오류가 발생했습니다:")
        st.exception(e)
        return False

# --- 분석 헬퍼 함수 (이하 변경 없음) ---
def clean_text(text):
    if not isinstance(text, str): return ''
    text = text.lower(); text = re.sub(r'\(.*?\)|\[.*?\]', ' ', text); text = re.sub(r'(\d+)\s*(?:y|yo|year|years|old|년산|년)', r'\1', text); text = re.sub(r'[^a-z0-9\s\uac00-\ud7a3]', ' ', text); text = re.sub(r'\b산\b', ' ', text)
    return ' '.join(text.split())

def assign_quadrant_group(row, x_mean, y_mean):
    is_high_volume = row['total_volume'] >= x_mean; is_high_price = row['avg_unitprice'] >= y_mean
    if is_high_volume and is_high_price: return "시장 선도 그룹"
    elif not is_high_volume and is_high_price: return "니치/프리미엄 그룹"
    elif not is_high_volume and not is_high_price: return "소규모/가격 경쟁 그룹"
    else: return "대규모/가성비 그룹"

def run_all_analysis(user_inputs, full_company_data, selected_products, target_importer_name):
    analysis_result = {"positioning": {}, "supply_chain": {}}
    analysis_data = full_company_data[full_company_data['reported_product_name'].isin(selected_products)].copy()
    if analysis_data.empty: return analysis_result
    importer_stats = analysis_data.groupby('importer').agg(total_value=('value', 'sum'), total_volume=('volume', 'sum'), trade_count=('value', 'count'), avg_unitprice=('unitprice', 'mean')).reset_index().sort_values('total_value', ascending=False).reset_index(drop=True)
    if importer_stats.empty: return analysis_result
    volume_mean = importer_stats['total_volume'].mean(); price_mean = importer_stats['avg_unitprice'].mean()
    importer_stats['quadrant_group'] = importer_stats.apply(assign_quadrant_group, axis=1, args=(volume_mean, price_mean))
    analysis_result['positioning'] = {"importer_stats": importer_stats, "target_stats": importer_stats[importer_stats['importer'] == target_importer_name]}
    user_input = user_inputs[0]; user_avg_price = user_input['Value'] / user_input['Volume'] if user_input['Volume'] > 0 else 0
    alternative_suppliers = analysis_data[(analysis_data['exporter'].str.upper() != user_input['Exporter'].upper()) & (analysis_data['unitprice'] < user_avg_price)]
    if not alternative_suppliers.empty:
        supplier_analysis = alternative_suppliers.groupby('exporter').agg(avg_unitprice=('unitprice', 'mean'), trade_count=('value', 'count'), num_importers=('importer', 'nunique')).reset_index().sort_values('avg_unitprice')
        supplier_analysis['price_saving_pct'] = (1 - supplier_analysis['avg_unitprice'] / user_avg_price) * 100
        supplier_analysis['stability_score'] = np.log1p(supplier_analysis['trade_count']) + np.log1p(supplier_analysis['num_importers'])
        analysis_result['supply_chain'] = {"user_avg_price": user_avg_price, "user_total_volume": sum(item['Volume'] for item in user_inputs), "alternatives": supplier_analysis}
    return analysis_result

# --- UI 컴포넌트 (이하 변경 없음) ---
def login_screen():
    st.title("🔐 수입 경쟁력 진단 솔루션")
    with st.form("login_form"):
        password = st.text_input("비밀번호", type="password")
        if st.form_submit_button("접속하기"):
            if password == st.secrets.get("app_secrets", {}).get("password", "tridgeDemo_2025"):
                st.session_state['logged_in'] = True; st.rerun()
            else: st.error("비밀번호가 올바르지 않습니다.")

def main_dashboard(company_data):
    st.title("📈 수입 경쟁력 진단 솔루션")
    with st.expander("STEP 1: 분석 정보 입력", expanded='analysis_groups' not in st.session_state):
        importer_name = st.text_input("1. 귀사의 업체명을 입력해주세요.", key="importer_name_input").upper()
        if 'rows' not in st.session_state: st.session_state['rows'] = [{'id': 1}]
        header_cols = st.columns([1.5, 3, 1, 2, 2, 1, 1, 1, 0.5]); headers = ["수입일", "제품 상세명", "HS-CODE", "원산지", "수출업체", "수입 중량(KG)", "총 수입금액(USD)", "Incoterms", "삭제"]
        for col, header in zip(header_cols, headers): col.markdown(f"**{header}**")
        for i, row in enumerate(st.session_state.rows):
            key_suffix = f"_{row['id']}"; cols = st.columns([1.5, 3, 1, 2, 2, 1, 1, 1, 0.5])
            st.session_state[f'date{key_suffix}'] = cols[0].date_input(f"date_widget{key_suffix}", value=st.session_state.get(f'date{key_suffix}', datetime.now().date()), key=f"date_widget_k{key_suffix}", label_visibility="collapsed")
            st.session_state[f'product_name{key_suffix}'] = cols[1].text_input(f"product_name_widget{key_suffix}", value=st.session_state.get(f'product_name{key_suffix}', ''), key=f"product_name_widget_k{key_suffix}", label_visibility="collapsed")
            st.session_state[f'hscode{key_suffix}'] = cols[2].text_input(f"hscode_widget{key_suffix}", max_chars=10, value=st.session_state.get(f'hscode{key_suffix}', ''), key=f"hscode_widget_k{key_suffix}", label_visibility="collapsed")
            origin_options = [''] + ['직접 입력'] + sorted(company_data['export_country'].unique()); origin_val_selected = cols[3].selectbox(f"origin_widget{key_suffix}", origin_options, index=origin_options.index(st.session_state.get(f'origin_selected{key_suffix}', '')) if st.session_state.get(f'origin_selected{key_suffix}') in origin_options else 0, key=f"origin_widget_k{key_suffix}", label_visibility="collapsed", format_func=lambda x: '선택' if x == '' else x)
            st.session_state[f'origin_selected{key_suffix}'] = origin_val_selected
            if origin_val_selected == '직접 입력': st.session_state[f'origin{key_suffix}'] = cols[3].text_input("custom_origin", value=st.session_state.get(f'origin{key_suffix}', ''), key=f"custom_origin_k{key_suffix}", label_visibility="collapsed", placeholder="원산지 직접 입력")
            else: st.session_state[f'origin{key_suffix}'] = origin_val_selected
            exporter_options = [''] + ['직접 입력'] + sorted(company_data['exporter'].unique()); exporter_val_selected = cols[4].selectbox(f"exporter_widget{key_suffix}", exporter_options, index=exporter_options.index(st.session_state.get(f'exporter_selected{key_suffix}', '')) if st.session_state.get(f'exporter_selected{key_suffix}') in exporter_options else 0, key=f"exporter_widget_k{key_suffix}", label_visibility="collapsed", format_func=lambda x: '선택' if x == '' else x)
            st.session_state[f'exporter_selected{key_suffix}'] = exporter_val_selected
            if exporter_val_selected == '직접 입력': st.session_state[f'exporter{key_suffix}'] = cols[4].text_input("custom_exporter", value=st.session_state.get(f'exporter{key_suffix}', ''), key=f"custom_exporter_k{key_suffix}", label_visibility="collapsed", placeholder="수출업체 직접 입력")
            else: st.session_state[f'exporter{key_suffix}'] = exporter_val_selected
            st.session_state[f'volume{key_suffix}'] = cols[5].number_input(f"volume_widget{key_suffix}", min_value=0.01, format="%.2f", value=st.session_state.get(f'volume{key_suffix}', 1.0), key=f"volume_widget_k{key_suffix}", label_visibility="collapsed")
            st.session_state[f'value{key_suffix}'] = cols[6].number_input(f"value_widget{key_suffix}", min_value=0.01, format="%.2f", value=st.session_state.get(f'value{key_suffix}', 1.0), key=f"value_widget_k{key_suffix}", label_visibility="collapsed")
            st.session_state[f'incoterms{key_suffix}'] = cols[7].selectbox(f"incoterms_widget{key_suffix}", ["
