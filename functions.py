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
from io import BytesIO

# --- 페이지 초기 설정 ---
st.set_page_config(layout="wide", page_title="수입 경쟁력 진단 솔루션")

# --- API 사용 범위(Scope) 정의 ---
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
        creds = Credentials.from_service_account_info(st.secrets["gcp_service_account"], scopes=SCOPES)
        project_id = st.secrets["gcp_service_account"]["project_id"]
        table_full_id = f"{project_id}.demo_data.tds_data"
        df = read_gbq(f"SELECT * FROM `{table_full_id}`", project_id=project_id, credentials=creds)
        
        df.columns = [re.sub(r'[^a-z0-9]+', '_', col.lower().strip()) for col in df.columns]
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        for col in ['volume', 'value']:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(r'[^\d.]', '', regex=True), errors='coerce')
        df.dropna(subset=['date', 'volume', 'value', 'importer', 'exporter', 'hs_code'], inplace=True)
        df = df[(df['volume'] > 0) & (df['value'] > 0)].copy()
        df['unitprice'] = df['value'] / df['volume']
        Q1, Q3 = df['unitprice'].quantile(0.05), df['unitprice'].quantile(0.95)
        df = df[(df['unitprice'] >= Q1) & (df['unitprice'] <= Q3)]
        return df
    except Exception as e: st.error(f"데이터 로딩 중 오류: {e}"); return None

# --- Google Sheets 저장 ---
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
        save_data_df['importer_name'] = importer_name; save_data_df['consent'] = consent
        save_data_df['timestamp'] = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d %H:%M:%S")
        save_data_df['Date'] = pd.to_datetime(save_data_df['Date'])
        save_data_df['Date'] = save_data_df['Date'].dt.strftime('%Y-%m-%d')
        save_data_df = save_data_df.astype(str)
        final_columns = ["Date", "Reported Product Name", "HS-Code", "Origin Country", "Exporter", "Volume", "Value", "Incoterms", "importer_name", "consent", "timestamp"]
        save_data_df = save_data_df[final_columns]
        if not worksheet.get('A1'): worksheet.update([save_data_df.columns.values.tolist()] + save_data_df.values.tolist(), value_input_option='USER_ENTERED')
        else: worksheet.append_rows(save_data_df.values.tolist(), value_input_option='USER_ENTERED')
        st.toast("입력 정보가 정상 반영되어 분석이 진행됩니다.", icon="✅")
        return True
    except gspread.exceptions.APIError as e:
        st.error("Google Sheets API 오류. GCP에서 API 활성화 및 권한을 확인하세요.")
        st.json(e.response.json()); return False
    except Exception as e:
        st.error(f"Google Sheets 저장 중 예상치 못한 오류: {e}"); st.exception(e); return False

# --- 분석 헬퍼 함수 ---
def clean_text(text):
    if not isinstance(text, str): return ''
    text = text.lower(); text = re.sub(r'\(.*?\)|\[.*?\]', ' ', text); text = re.sub(r'(\d+)\s*(?:y|yo|year|years|old|년산|년)', r'\1', text); text = re.sub(r'[^a-z0-9\s\uac00-\ud7a3]', ' ', text); text = re.sub(r'\b산\b', ' ', text)
    return ' '.join(text.split())

def to_excel_col(n):
    name = "";
    while n >= 0:
        name = chr(ord('A') + n % 26) + name
        n = n // 26 - 1
    return name + "사"

# --- 메인 분석 로직 ---
def run_all_analysis(user_inputs, full_company_data, selected_products, target_importer_name, analysis_mode):
    analysis_result = {"overview": {}, "diagnosis": {}, "timeseries": {}, "positioning": {}, "supply_chain": {}, "performance_trend": {}}
    user_input_df = pd.DataFrame(user_inputs)
    user_input_df['Date'] = pd.to_datetime(user_input_df['Date'])
    user_input_df['unitprice'] = user_input_df['Value'] / user_input_df['Volume']
    user_avg_price = user_input_df['Value'].sum() / user_input_df['Volume'].sum() if user_input_df['Volume'].sum() > 0 else 0
    
    hscode = str(user_input_df['HS-Code'].iloc[0])
    if hscode:
        hscode_data = full_company_data[full_company_data['hs_code'].astype(str) == hscode].copy()
        if not hscode_data.empty:
            this_year = datetime.now().year; vol_this_year = hscode_data[hscode_data['date'].dt.year == this_year]['volume'].sum(); vol_last_year = hscode_data[hscode_data['date'].dt.year == this_year - 1]['volume'].sum(); price_this_year = hscode_data[hscode_data['date'].dt.year == this_year]['unitprice'].mean(); price_last_year = hscode_data[hscode_data['date'].dt.year == this_year - 1]['unitprice'].mean()
            analysis_result['overview'] = {"hscode": hscode, "this_year": this_year, "vol_this_year": vol_this_year, "vol_last_year": vol_last_year, "price_this_year": price_this_year, "price_last_year": price_last_year, "freq_this_year": len(hscode_data[hscode_data['date'].dt.year == this_year]), "product_composition": hscode_data.groupby('reported_product_name')['value'].sum().nlargest(10)}

    analysis_data = full_company_data[full_company_data['reported_product_name'].isin(selected_products)].copy()
    if analysis_data.empty: return analysis_result
    
    analysis_data['month'] = analysis_data['date'].dt.to_period('M')
    user_input_df['month'] = user_input_df['Date'].dt.to_period('M')
    monthly_benchmarks = analysis_data.groupby('month')['unitprice'].transform('mean')
    analysis_data['price_index'] = analysis_data['unitprice'] / monthly_benchmarks
    
    if analysis_mode == "이번 거래 진단":
        month_data = analysis_data[analysis_data['month'] == user_input_df['month'].iloc[0]]
        if not month_data.empty:
            month_avg_price = month_data['unitprice'].mean(); price_percentile = (month_data['unitprice'] > user_avg_price).mean() * 100
            top_10_percent_price = month_data['unitprice'].quantile(0.10); potential_savings = (user_avg_price - top_10_percent_price) * user_input_df['Volume'].sum()
            analysis_result['diagnosis'] = {"user_price": user_avg_price, "market_avg_price": month_avg_price, "percentile": price_percentile, "top_10_price": top_10_percent_price, "potential_savings": potential_savings if potential_savings > 0 else 0}
        
        monthly_avg = analysis_data.set_index('date')['unitprice'].resample('M').mean().reset_index()
        analysis_result['timeseries'] = {"all_trades": analysis_data[['date', 'unitprice', 'volume', 'importer']], "monthly_avg": monthly_avg, "current_transactions": user_input_df[['Date', 'unitprice', 'Volume']]}
            
        importer_stats = analysis_data.groupby('importer').agg(total_value=('value', 'sum'), total_volume=('volume', 'sum'), trade_count=('value', 'count'), price_index=('price_index', 'mean')).reset_index().sort_values('total_value', ascending=False).reset_index(drop=True)
        importer_stats['cum_share'] = importer_stats['total_value'].cumsum() / importer_stats['total_value'].sum()
        market_leaders = importer_stats[importer_stats['cum_share'] <= 0.7]
        try:
            target_rank = importer_stats[importer_stats['importer'] == target_importer_name].index[0]; rank_margin = max(1, int(len(importer_stats) * 0.1)); direct_peers = importer_stats.iloc[max(0, target_rank - rank_margin):min(len(importer_stats), target_rank + rank_margin + 1)]
        except IndexError: direct_peers = pd.DataFrame()
        price_achievers = importer_stats[importer_stats['price_index'] <= importer_stats['price_index'].quantile(0.15)]
        
        user_monthly_benchmarks = user_input_df['month'].map(analysis_data.drop_duplicates('month').set_index('month')['unitprice'])
        user_input_df['price_index'] = user_input_df['unitprice'] / user_monthly_benchmarks
        analysis_result['positioning'] = {"importer_stats": importer_stats, "target_stats": importer_stats[importer_stats['importer'] == target_importer_name], "rule_based_groups": {"시장 선도 그룹": market_leaders, "유사 규모 경쟁 그룹": direct_peers, "최저가 달성 그룹": price_achievers}, "current_transactions_normalized": user_input_df[['Volume', 'price_index']]}
        
        alternative_suppliers = analysis_data[(analysis_data['exporter'].str.upper() != user_input_df['Exporter'].iloc[0].upper()) & (analysis_data['unitprice'] < user_avg_price)]
        if not alternative_suppliers.empty:
            supplier_analysis = alternative_suppliers.groupby('exporter').agg(avg_unitprice=('unitprice', 'mean'), trade_count=('value', 'count'), num_importers=('importer', 'nunique')).reset_index().sort_values('avg_unitprice')
            supplier_analysis['price_saving_pct'] = (1 - supplier_analysis['avg_unitprice'] / user_avg_price) * 100
            supplier_analysis['stability_score'] = np.log1p(supplier_analysis['trade_count']) + np.log1p(supplier_analysis['num_importers'])
            if len(supplier_analysis) >= 3:
                low_q, high_q = supplier_analysis['stability_score'].quantile(0.33), supplier_analysis['stability_score'].quantile(0.67)
                conditions = [supplier_analysis['stability_score'] <= low_q, (supplier_analysis['stability_score'] > low_q) & (supplier_analysis['stability_score'] < high_q), supplier_analysis['stability_score'] >= high_q]; ratings = ['하', '중', '상']
                supplier_analysis['stability_rank'] = np.select(conditions, ratings, default='중')
            else: supplier_analysis['stability_rank'] = '중'
            analysis_result['supply_chain'] = {"user_avg_price": user_avg_price, "user_total_volume": user_input_df['Volume'].sum(), "alternatives": supplier_analysis}

    elif analysis_mode == "나의 과거 내역 분석":
        user_input_df['price_index'] = user_input_df.apply(lambda row: row['unitprice'] / analysis_data[analysis_data['month'] == row['month']]['unitprice'].mean() if not analysis_data[analysis_data['month'] == row['month']].empty else 1.0, axis=1)
        user_perf_trend = user_input_df.set_index('Date')['price_index'].resample('M').mean().reset_index()
        
        importer_stats = analysis_data.groupby('importer').agg(total_value=('value', 'sum'), price_index=('price_index', 'mean')).reset_index()
        importer_stats['cum_share'] = importer_stats['total_value'].cumsum() / importer_stats['total_value'].sum()
        market_leaders_importers = importer_stats[importer_stats['cum_share'] <= 0.7]['importer']
        price_achievers_importers = analysis_data.loc[analysis_data.groupby('month')['price_index'].idxmin()]['importer'].unique()

        market_leaders_trend = analysis_data[analysis_data['importer'].isin(market_leaders_importers)].set_index('date')['price_index'].resample('M').mean().reset_index()
        price_achievers_trend = analysis_data[analysis_data['importer'].isin(price_achievers_importers)].set_index('date')['price_index'].resample('M').mean().reset_index()
        
        user_perf_trend.rename(columns={'Date': 'date'}, inplace=True)
        analysis_result['performance_trend'] = {"user_trend": user_perf_trend, "market_leader_trend": market_leaders_trend, "price_achiever_trend": price_achievers_trend}

    return analysis_result


# --- UI 컴포넌트 ---
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


        st.markdown("---")
        st.markdown("##### **선택 1. 엑셀 파일로 업로드하기**")
        
        try:
            with open("수입내역_입력_템플릿.xlsx", "rb") as file:
                st.download_button(label="📥 엑셀 템플릿 다운로드", data=file, file_name="수입내역_입력_템플릿.xlsx", mime="application/vnd.ms-excel")
        except FileNotFoundError:
            st.warning("엑셀 템플릿 파일('수입내역_입력_템플릿.xlsx')을 찾을 수 없습니다.")
        
        uploaded_file = st.file_uploader("📂 템플렛 양식에 작성한 엑셀 파일 업로드", type=['xlsx'])
            
        st.markdown("---")
        
        # --- 툴팁(Popover) UI 개선 ---
        col1, col2 = st.columns([10, 1])
        with col1:
            st.markdown("##### **선택 2. 직접 입력하기**")
        with col2:
            with st.popover("ℹ️"):
                st.markdown("""
                **입력 요령 가이드:**
                - **수입일:** 거래가 발생한 날짜(YYYY-MM-DD)를 선택하세요.
                - **제품 상세명:** 브랜드, 연산 등 제품을 특정할 수 있는 상세명을 입력하세요. (예: Glenfiddich 12년산)
                - **HS-CODE:** 분석하고 싶은 HS-CODE 6자리를 입력하세요. (예: 220830)
                - **원산지:** 제품이 생산된 국가를 선택하거나 직접 입력하세요.
                - **수출업체:** 거래한 수출업체명을 선택하거나 직접 입력하세요.
                - **수입 중량(KG):** 수입한 총 중량을 킬로그램(KG) 단위로 입력하세요.
                - **총 수입금액(USD):** 수입에 지불한 총 금액을 미국 달러(USD) 단위로 입력하세요.
                - **Incoterms:** 거래에 적용된 인코텀즈 조건을 선택하세요.
                """)
        
        header_cols = st.columns([1.5, 3, 1, 2, 2, 1, 1, 1, 0.5]); headers = ["수입일", "제품 상세명", "HS-CODE", "원산지", "수출업체", "수입 중량(KG)", "총 수입금액(USD)", "Incoterms", "삭제"]
        for col, header in zip(header_cols, headers): col.markdown(f"**{header}**")
        
        if 'rows' not in st.session_state: st.session_state['rows'] = [{'id': 1}]
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
            st.session_state[f'volume{key_suffix}'] = cols[5].number_input(f"volume_widget{key_suffix}", min_value=0.01, format="%.2f", value=st.session_state.get(f'volume{key_suffix}', 1000.0), key=f"volume_widget_k{key_suffix}", label_visibility="collapsed")
            st.session_state[f'value{key_suffix}'] = cols[6].number_input(f"value_widget{key_suffix}", min_value=0.01, format="%.2f", value=st.session_state.get(f'value{key_suffix}', 10000.0), key=f"value_widget_k{key_suffix}", label_visibility="collapsed")
            st.session_state[f'incoterms{key_suffix}'] = cols[7].selectbox(f"incoterms_widget{key_suffix}", ["FOB", "CFR", "CIF", "EXW", "DDP", "기타"], index=["FOB", "CFR", "CIF", "EXW", "DDP", "기타"].index(st.session_state.get(f'incoterms{key_suffix}', 'CIF')), key=f"incoterms_widget_k{key_suffix}", label_visibility="collapsed")
          
            if len(st.session_state.rows) > 1:
                if cols[8].button("삭제", key=f"delete{key_suffix}"):
                    st.session_state.rows.pop(i)
                    st.rerun()

        
        if st.button("➕ 내역 추가하기"):
            new_id = max(row['id'] for row in st.session_state.rows) + 1 if st.session_state.rows else 1; st.session_state.rows.append({'id': new_id}); st.rerun()
        


        st.markdown("---")
        analysis_mode = st.radio("2. 분석 모드 선택", ["이번 거래 진단", "나의 과거 내역 분석"], key='analysis_mode', horizontal=True)
        st.info(f"**{analysis_mode} 모드:**{'입력한 거래(들)의 경쟁력을 빠르게 진단합니다.' if analysis_mode == '이번 거래 진단' else '입력한 과거 내역 전체의 성과 추이를 시장과 비교 분석합니다.'}")
        consent = st.checkbox("분석을 위해 입력하신 정보가 활용되는 것에 동의합니다.", value=st.session_state.get('consent', True), key='consent_widget'); st.session_state['consent'] = consent
        
        if st.button("분석하기", type="primary", use_container_width=True):
            all_input_data = []
            is_valid = True
            
            for i, row in enumerate(st.session_state.rows):
                key_suffix = f"_{row['id']}"; 
                if st.session_state.get(f'product_name{key_suffix}'):
                    entry = { "Date": st.session_state.get(f'date{key_suffix}'), "Reported Product Name": st.session_state.get(f'product_name{key_suffix}'), "HS-Code": st.session_state.get(f'hscode{key_suffix}'), "Origin Country": st.session_state.get(f'origin{key_suffix}'), "Exporter": st.session_state.get(f'exporter{key_suffix}'), "Volume": st.session_state.get(f'volume{key_suffix}'), "Value": st.session_state.get(f'value{key_suffix}'), "Incoterms": st.session_state.get(f'incoterms{key_suffix}')}
                    all_input_data.append(entry)

            if uploaded_file is not None:
                try:
                    excel_df = pd.read_excel(uploaded_file, header=1) # B2부터 읽기 위해 header=1 사용
                    if 'Unnamed: 0' in excel_df.columns:
                        excel_df = excel_df.drop(columns=['Unnamed: 0'])
                    excel_cols = {"수입일": "Date", "제품 상세명": "Reported Product Name", "HS-CODE": "HS-Code", "원산지": "Origin Country", "수출업체": "Exporter", "수입 중량(KG)": "Volume", "총 수입금액(USD)": "Value", "Incoterms": "Incoterms"}
                    excel_df.rename(columns=excel_cols, inplace=True)
                    all_input_data.extend(excel_df.to_dict('records'))
                except Exception as e:
                    st.error(f"엑셀 파일 처리 중 오류가 발생했습니다: {e}"); is_valid = False
            
            if not all_input_data:
                st.error("⚠️ [입력 오류] 분석할 데이터가 없습니다. 직접 입력하거나 엑셀 파일을 업로드해주세요."); is_valid = False

            if not importer_name: st.error("⚠️ [입력 오류] 귀사의 업체명을 입력해주세요."); is_valid = False
            if not consent: st.warning("⚠️ 정보 활용 동의에 체크해주세요."); is_valid = False
            
            if is_valid:
                with st.spinner('입력하신 내용을 기반으로 분석을 시작합니다...'):
                    purchase_df = pd.DataFrame(all_input_data)
                    if save_to_google_sheets(purchase_df, importer_name, consent):
                        purchase_df['cleaned_name'] = purchase_df['Reported Product Name'].apply(clean_text)
                        analysis_groups = {}
                        for name, group_df in purchase_df.groupby('cleaned_name'):
                            user_tokens = set(name.split())
                            is_match = lambda comp_name: user_tokens.issubset(set(str(comp_name).split()))
                            company_data['cleaned_name_db'] = company_data['reported_product_name'].apply(clean_text)
                            matched_df = company_data[company_data['cleaned_name_db'].apply(is_match)]
                            if matched_df.empty: continue
                            matched_products = sorted(matched_df['reported_product_name'].unique().tolist())
                            result = run_all_analysis(group_df.to_dict('records'), company_data, matched_products, importer_name, st.session_state.analysis_mode)
                            analysis_groups[name] = {"user_input_df": group_df, "result": result}
                        st.session_state['analysis_groups'] = analysis_groups
                        st.session_state['analysis_mode_result'] = st.session_state.analysis_mode
                        st.success("분석 완료!"); st.rerun()
    
    if 'analysis_groups' in st.session_state:
        if 'analysis_mode_result' not in st.session_state:
            st.warning("분석 모드를 확인할 수 없습니다. 새로운 분석을 시작해주세요.")
        else:
            st.header("📊 분석 결과")
            analysis_mode = st.session_state['analysis_mode_result']
            
            processed_hscodes = []
            for product_cleaned_name, group_info in st.session_state.analysis_groups.items():
                result = group_info.get("result", {})
                overview_res = result.get('overview')
                if overview_res and overview_res['hscode'] not in processed_hscodes:
                    st.subheader(f"📈 HS-Code {overview_res['hscode']} 시장 개요")
                    o = overview_res; cols = st.columns(3)
                    vol_yoy = (o['vol_this_year'] - o['vol_last_year']) / o['vol_last_year'] if o['vol_last_year'] > 0 else np.nan; price_yoy = (o['price_this_year'] - o['price_last_year']) / o['price_last_year'] if o['price_last_year'] > 0 else np.nan
                    cols[0].metric(f"{o['this_year']}년 수입 중량 (KG)", f"{o['vol_this_year']:,.0f}", f"{vol_yoy:.1%}" if pd.notna(vol_yoy) else "N/A", delta_color="inverse")
                    cols[1].metric(f"{o['this_year']}년 평균 단가 (USD/KG)", f"${o['price_this_year']:.2f}", f"{price_yoy:.1%}" if pd.notna(price_yoy) else "N/A", delta_color="inverse")
                    cols[2].metric(f"{o['this_year']}년 총 수입 건수", f"{o['freq_this_year']:,} 건")
                    if not o['product_composition'].empty:
                        pie_fig = px.pie(o['product_composition'], names=o['product_composition'].index, values=o['product_composition'].values, title='<b>상위 10개 제품 구성 (수입 금액 기준)</b>', hole=0.3)
                        pie_fig.update_traces(textposition='inside', textinfo='percent+label'); st.plotly_chart(pie_fig, use_container_width=True)
                    st.markdown("---"); processed_hscodes.append(overview_res['hscode'])

        for product_cleaned_name, group_info in st.session_state.analysis_groups.items():
                st.subheader(f"분석 그룹: \"{group_info['user_input_df']['Reported Product Name'].iloc[0]}\"")
                result = group_info.get("result", {})

                if analysis_mode == "이번 거래 진단":
                    diag_res, ts_res, p_res, s_res = result.get('diagnosis'), result.get('timeseries'), result.get('positioning'), result.get('supply_chain')
                    st.markdown("#### PART 1. 입력값 경쟁력 진단 요약")
                    if diag_res:
                        price_diff = (diag_res['user_price'] / diag_res['market_avg_price'] - 1) * 100 if diag_res['market_avg_price'] > 0 else 0
                        cols = st.columns(3); cols[0].metric("입력값 평균 단가", f"${diag_res['user_price']:.2f}", f"{price_diff:.1f}% vs 동월 평균", delta_color="inverse")
                        cols[1].metric("가격 경쟁력 순위", f"상위 {diag_res['percentile']:.0f}%", help="100%에 가까울수록 동월 시장에서 저렴하게 구매한 거래입니다.")
                        cols[2].metric("예상 추가 절감액", f"${diag_res['potential_savings']:,.0f}", help=f"동월 상위 10% 평균가(${diag_res['top_10_price']:.2f}) 기준")
                    else: st.info("입력값과 동일한 월의 시장 데이터가 부족하여 진단 요약을 생성할 수 없습니다.")
                    st.markdown("---")
                    st.markdown("#### PART 2. 시계열 시장 동향 및 입력값 위치")
                    if ts_res and not ts_res['all_trades'].empty:
                        fig_ts = go.Figure()
                        all_trades_df = ts_res['all_trades'].copy(); target_name = st.session_state.get('importer_name_result', '')
                        unique_importers_ts = all_trades_df['importer'].unique(); anonymity_map_ts = {name: to_excel_col(i) for i, name in enumerate(unique_importers_ts) if name != target_name}; anonymity_map_ts[target_name] = target_name
                        all_trades_df['Anonymized_Importer'] = all_trades_df['importer'].map(anonymity_map_ts)
                        log_volume = np.log1p(all_trades_df['volume']); bubble_size = 5 + ((log_volume - log_volume.min()) / (log_volume.max() - log_volume.min())) * 25 if log_volume.max() > log_volume.min() else [5]*len(log_volume)
                        fig_ts.add_trace(go.Scatter(x=all_trades_df['date'], y=all_trades_df['unitprice'], mode='markers', marker=dict(size=bubble_size, color='lightgray', opacity=0.6), name='과거 시장 거래', text=all_trades_df['Anonymized_Importer'], hovertemplate='<b>%{text}</b><br>단가: $%{y:,.2f}<extra></extra>'))
                        fig_ts.add_trace(go.Scatter(x=ts_res['monthly_avg']['date'], y=ts_res['monthly_avg']['unitprice'], mode='lines', line=dict(color='cornflowerblue', width=3), name='월별 시장 평균가'))
                        current_txs = ts_res['current_transactions']
                        log_volume_current = np.log1p(current_txs['Volume'])
                        current_bubble_sizes = [5 + ((s - log_volume.min()) / (log_volume.max() - log_volume.min())) * 25 if log_volume.max() > log_volume.min() else 15 for s in log_volume_current]
                        fig_ts.add_trace(go.Scatter(x=current_txs['Date'], y=current_txs['unitprice'], mode='markers', marker=dict(symbol='circle', color='rgba(0,0,0,0)', size=[s * 1.5 for s in current_bubble_sizes], line=dict(color='black', width=2)), name='입력값', hovertemplate='<b>입력값</b><br>단가: $%{y:,.2f}<extra></extra>'))
                        fig_ts.update_layout(title="<b>시기별 거래 동향 및 시장가 비교</b>", xaxis_title="거래 시점", yaxis_title="거래 단가 (USD/KG)", showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                        st.plotly_chart(fig_ts, use_container_width=True)
                    st.markdown("---")
                    st.markdown("#### PART 3. 경쟁 환경 및 전략 분석")
                    if not p_res or p_res['importer_stats'].empty: st.info("경쟁 환경 분석을 위한 데이터가 부족합니다."); continue
                    col1, col2 = st.columns([10,1]); col1.markdown("##### **3-1. 시장 내 전략적 위치 (시점 정규화)**")
                    with col2:
                        with st.popover("ℹ️"): st.markdown("""**가격 경쟁력 지수란?**\n계절성이나 시장 트렌드 등 시점 요인을 제거한 순수한 가격 경쟁력입니다.\n- **계산식:** `개별 거래 단가 / 해당 월의 시장 평균 단가`\n- **1.0 미만:** 시장 평균보다 저렴하게 구매\n- **1.0 초과:** 시장 평균보다 비싸게 구매""")
                    importer_stats = p_res['importer_stats']; target_name = st.session_state.get('importer_name_result', '')
                    importer_stats['Anonymized_Importer'] = [to_excel_col(j) if imp != target_name else target_name for j, imp in enumerate(importer_stats['importer'])]
                    log_values = np.log1p(importer_stats['total_volume']); min_size, max_size = 15, 80
                    if log_values.max() > log_values.min(): importer_stats['size'] = min_size + ((log_values - log_values.min()) / (log_values.max() - log_values.min())) * (max_size - min_size)
                    else: importer_stats['size'] = [min_size] * len(importer_stats)
                    x_mean = importer_stats['total_volume'].mean(); y_mean = 1.0
                    fig_pos = go.Figure()
                    competitors = importer_stats[importer_stats['importer'] != target_name]; fig_pos.add_trace(go.Scatter(x=competitors['total_volume'], y=competitors['price_index'], mode='markers', marker=dict(size=competitors['size'], color='#BDBDBD', opacity=0.5), text=competitors['Anonymized_Importer'], hovertemplate='<b>%{text}</b><br>가격 경쟁력 지수: %{y:.2f}<extra></extra>'))
                    target_df = importer_stats[importer_stats['importer'] == target_name]
                    if not target_df.empty: fig_pos.add_trace(go.Scatter(x=target_df['total_volume'], y=target_df['price_index'], mode='markers', marker=dict(size=target_df['size'], color='#FF4B4B', opacity=1.0, line=dict(width=2, color='black')), name='귀사(과거 평균)', text=target_df['Anonymized_Importer'], hovertemplate='<b>%{text} (평균)</b><br>가격 경쟁력 지수: %{y:.2f}<extra></extra>'))
                    current_txs_norm = p_res.get('current_transactions_normalized')
                    if not current_txs_norm.empty: 
                        fig_pos.add_trace(go.Scatter(x=current_txs_norm['Volume'], y=current_txs_norm['price_index'], mode='markers', marker=dict(symbol='circle', color='rgba(0,0,0,0)', size=20, line=dict(color='black', width=2)), name='입력값', hovertemplate='<b>입력값</b><br>가격 경쟁력 지수: %{y:.2f}<extra></extra>'))
                    fig_pos.add_vline(x=x_mean, line_dash="dash", line_color="gray"); fig_pos.add_hline(y=y_mean, line_dash="dash", line_color="gray")
                    fig_pos.update_layout(title="<b>수입사 포지셔닝 맵 (시기 보정)</b>", xaxis_title="총 수입 중량 (KG, Log Scale)", yaxis_title="가격 경쟁력 지수 (1.0 = 시장 평균)", showlegend=False, xaxis_type="log")
                    st.plotly_chart(fig_pos, use_container_width=True)

                    col1, col2 = st.columns([10,1]); col1.markdown("##### **3-2. 실질 경쟁 그룹과의 비교**")
                    with col2:
                        with st.popover("ℹ️"): 
                            st.markdown("""**그룹 분류 기준:**\n- **시장 선도 그룹:** 수입 금액 기준 누적 70% 차지\n- **유사 규모 경쟁 그룹:** 귀사 순위 기준 상하 ±10%\n- **최저가 달성 그룹:** 시기 보정된 '가격 경쟁력 지수' 하위 15%\n---\n**그룹이 표시되지 않는 경우:**\n데이터 특성에 따라 조건에 맞는 경쟁사가 없으면 해당 그룹은 박스 플롯에 표시되지 않을 수 있습니다.""")
                    rb_groups = p_res['rule_based_groups']; group_data = []
                    for name, df in rb_groups.items():
                        if not df.empty: 
                            df_copy = df.copy(); df_copy['group_name'] = name
                            group_data.append(df_copy[['group_name', 'price_index']])
                    
                    if not current_txs_norm.empty:
                        user_df = current_txs_norm.copy()
                        user_df['group_name'] = f"{target_name} (입력값)"
                        group_data.append(user_df.rename(columns={'price_index': 'price_index'})[['group_name', 'price_index']])

                    if group_data:
                        plot_df_box = pd.concat(group_data)
                        fig_box = px.box(plot_df_box, x='group_name', y='price_index', title="<b>주요 경쟁 그룹별 가격 경쟁력 분포</b>", labels={'group_name': '경쟁 그룹 유형', 'price_index': '가격 경쟁력 지수'})
                        if not p_res['target_stats'].empty: fig_box.add_hline(y=p_res['target_stats']['price_index'].iloc[0], line_dash="dot", line_color="orange", annotation_text="귀사 평균")
                        st.plotly_chart(fig_box, use_container_width=True)
                    st.markdown("---")
                    
                    st.markdown("#### PART 4. 공급망 분석 및 비용 절감 시뮬레이션")
                    if not s_res or s_res['alternatives'].empty: st.info("현재 거래 조건보다 더 저렴한 대안 공급처를 찾지 못했습니다.")
                    else:
                        alts, best_deal = s_res['alternatives'], s_res['alternatives'].iloc[0]
                        num_alternatives = len(alts)
                        st.success(f"**비용 절감 기회 포착!** 현재 거래처보다 **최대 {best_deal['price_saving_pct']:.1f}%** 저렴한 대체 거래처가 **{num_alternatives}개** 존재합니다.")
                        col1, col2 = st.columns(2); target_saving_pct = col1.slider("목표 단가 절감률(%)", 0.0, float(best_deal['price_saving_pct']), float(best_deal['price_saving_pct'] / 2), 0.5, "%.1f%%", key=f"slider_{i}"); expected_saving = s_res['user_total_volume'] * s_res['user_avg_price'] * (target_saving_pct / 100); col2.metric(f"예상 절감액 (수입량 {s_res['user_total_volume']:,.0f}KG 기준)", f"${expected_saving:,.0f}")
                        col1_supply, col2_supply = st.columns([10,1])
                        with col1_supply: st.markdown("##### **추천 대체 공급처 리스트**")
                        with col2_supply:
                            with st.popover("ℹ️"): st.markdown("""**공급 안정성 기준:**\n발견된 대체 공급처들의 '거래 빈도'와 '거래처 수'를 종합하여 계산된 안정성 점수를 기준으로 상대 평가됩니다.\n- **상:** 상위 33%\n- **중:** 중간 33%\n- **하:** 하위 33%""")
                        recommended_list = alts[alts['price_saving_pct'] >= target_saving_pct].copy()
                        recommended_list.reset_index(drop=True, inplace=True); recommended_list['순번'] = recommended_list.index + 1
                        recommended_list.rename(columns={'avg_unitprice': '평균 단가', 'price_saving_pct': '가격 경쟁력(%)', 'trade_count': '거래 빈도', 'num_importers': '거래처 수', 'stability_rank': '공급 안정성'}, inplace=True)
                        st.dataframe(recommended_list[['순번', '평균 단가', '가격 경쟁력(%)', '거래 빈도', '거래처 수', '공급 안정성']], use_container_width=True, 
                                     column_config={"평균 단가": st.column_config.NumberColumn(format="$%.2f"), "가격 경쟁력(%)": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=alts['price_saving_pct'].max())}, hide_index=True)
                    st.markdown("---")

                elif analysis_mode == "나의 과거 내역 분석":
                    perf_res = result.get('performance_trend')
                    st.markdown("#### 나의 구매 성과 대시보드")
                    if perf_res and not perf_res['user_trend'].empty:
                        fig_perf = go.Figure()
                        fig_perf.add_hline(y=1.0, line_dash="dash", line_color="gray", annotation_text="시장 평균")
                        fig_perf.add_trace(go.Scatter(x=perf_res['user_trend']['date'], y=perf_res['user_trend']['price_index'], name='나의 성과', mode='lines', line=dict(color='black', width=4)))
                        if not perf_res['market_leader_trend'].empty: fig_perf.add_trace(go.Scatter(x=perf_res['market_leader_trend']['date'], y=perf_res['market_leader_trend']['price_index'], name='시장 선도 그룹', mode='lines', line=dict(color='blue', width=2)))
                        if not perf_res['price_achiever_trend'].empty: fig_perf.add_trace(go.Scatter(x=perf_res['price_achiever_trend']['date'], y=perf_res['price_achiever_trend']['price_index'], name='최저가 달성 그룹', mode='lines', line=dict(color='green', width=2)))
                        fig_perf.update_layout(title="<b>경쟁 그룹별 가격 경쟁력 지수 추이</b>", yaxis_title="가격 경쟁력 지수 (낮을수록 좋음)")
                        st.plotly_chart(fig_perf, use_container_width=True)
                    else:
                        st.info("성과 추이 분석을 위한 데이터가 부족합니다.")

# --- 메인 실행 로직 ---
if 'logged_in' not in st.session_state: st.session_state['logged_in'] = False
if not st.session_state['logged_in']:
    login_screen()
else:
    company_data = load_company_data()
    if company_data is not None:
        main_dashboard(company_data)
    else:
        st.error("데이터 로딩에 실패했습니다. 페이지를 새로고침하거나 앱 설정을 확인해주세요.")
