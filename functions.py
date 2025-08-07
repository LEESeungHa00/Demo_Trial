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
        #st.toast("입력 정보가 Google Sheet에 저장되었습니다.", icon="✅")
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

def assign_quadrant_group(row, x_mean, y_mean):
    is_high_volume = row['total_volume'] >= x_mean; is_high_price = row['price_index'] >= y_mean
    if is_high_volume and is_high_price: return "마켓 리더"
    elif not is_high_volume and is_high_price: return "프리미엄 전략 그룹"
    elif not is_high_volume and not is_high_price: return "효율적 소싱 그룹"
    else: return "원가 우위 그룹"

def to_excel_col(n): # 0부터 시작하는 숫자를 받아 A사, B사... Z사, AA사... 등으로 변환
    name = ""
    while n >= 0:
        name = chr(ord('A') + n % 26) + name
        n = n // 26 - 1
    return name + "사"
    
# --- 메인 분석 로직 ---
def run_all_analysis(user_inputs, full_company_data, selected_products, target_importer_name):
    analysis_result = {"overview": {}, "diagnosis": {}, "timeseries": {}, "positioning": {}, "supply_chain": {}}
    user_total_volume = sum(item['Volume'] for item in user_inputs); user_total_value = sum(item['Value'] for item in user_inputs)
    user_avg_price = user_total_value / user_total_volume if user_total_volume > 0 else 0
    user_input = user_inputs[0]; user_date = pd.to_datetime(user_input['Date'])
    
    hscode = str(user_input.get('HS-Code', ''))
    if hscode:
        hscode_data = full_company_data[full_company_data['hs_code'].astype(str) == hscode].copy()
        if not hscode_data.empty:
            this_year = datetime.now().year; vol_this_year = hscode_data[hscode_data['date'].dt.year == this_year]['volume'].sum(); vol_last_year = hscode_data[hscode_data['date'].dt.year == this_year - 1]['volume'].sum(); price_this_year = hscode_data[hscode_data['date'].dt.year == this_year]['unitprice'].mean(); price_last_year = hscode_data[hscode_data['date'].dt.year == this_year - 1]['unitprice'].mean()
            analysis_result['overview'] = {"hscode": hscode, "this_year": this_year, "vol_this_year": vol_this_year, "vol_last_year": vol_last_year, "price_this_year": price_this_year, "price_last_year": price_last_year, "freq_this_year": len(hscode_data[hscode_data['date'].dt.year == this_year]), "product_composition": hscode_data.groupby('reported_product_name')['value'].sum().nlargest(10)}

    analysis_data = full_company_data[full_company_data['reported_product_name'].isin(selected_products)].copy()
    if analysis_data.empty: return analysis_result
    
    month_data = analysis_data[analysis_data['date'].dt.to_period('M') == user_date.to_period('M')]
    month_avg_price = 0
    if not month_data.empty:
        month_avg_price = month_data['unitprice'].mean(); price_percentile = (month_data['unitprice'] < user_avg_price).mean() * 100
        top_10_percent_price = month_data['unitprice'].quantile(0.10); potential_savings = (user_avg_price - top_10_percent_price) * user_total_volume
        analysis_result['diagnosis'] = {"user_price": user_avg_price, "market_avg_price": month_avg_price, "percentile": price_percentile, "top_10_price": top_10_percent_price, "potential_savings": potential_savings if potential_savings > 0 else 0}
        
    monthly_avg = analysis_data.set_index('date')['unitprice'].resample('M').mean().reset_index()
    analysis_result['timeseries'] = {"all_trades": analysis_data[['date', 'unitprice', 'volume']], "monthly_avg": monthly_avg, "current_transaction": {'date': user_date, 'unitprice': user_avg_price, 'volume': user_total_volume}}
        
    analysis_data['month'] = analysis_data['date'].dt.to_period('M')
    monthly_benchmarks = analysis_data.groupby('month')['unitprice'].transform('mean')
    analysis_data['price_index'] = analysis_data['unitprice'] / monthly_benchmarks
    importer_stats = analysis_data.groupby('importer').agg(total_value=('value', 'sum'), total_volume=('volume', 'sum'), trade_count=('value', 'count'), price_index=('price_index', 'mean')).reset_index().sort_values('total_value', ascending=False).reset_index(drop=True)
    
    volume_mean = importer_stats['total_volume'].mean(); price_index_mean = 1.0
    importer_stats['quadrant_group'] = importer_stats.apply(assign_quadrant_group, axis=1, args=(volume_mean, price_index_mean))
    importer_stats['cum_share'] = importer_stats['total_value'].cumsum() / importer_stats['total_value'].sum()
    market_leaders = importer_stats[importer_stats['cum_share'] <= 0.7]
    try:
        target_rank = importer_stats[importer_stats['importer'] == target_importer_name].index[0]; rank_margin = max(1, int(len(importer_stats) * 0.1)); direct_peers = importer_stats.iloc[max(0, target_rank - rank_margin):min(len(importer_stats), target_rank + rank_margin + 1)]
    except IndexError: direct_peers = pd.DataFrame()
    price_achievers = importer_stats[importer_stats['price_index'] <= importer_stats['price_index'].quantile(0.15)]
    current_transaction_price_index = user_avg_price / month_avg_price if month_avg_price > 0 else 1.0
    current_tx_normalized = {'total_volume': user_total_volume, 'price_index': current_transaction_price_index}
    analysis_result['positioning'] = {"importer_stats": importer_stats, "target_stats": importer_stats[importer_stats['importer'] == target_importer_name], "rule_based_groups": {"시장 선도 그룹": market_leaders, "유사 규모 경쟁 그룹": direct_peers, "최저가 달성 그룹": price_achievers}, "current_transaction_normalized": current_tx_normalized}
    
    alternative_suppliers = analysis_data[(analysis_data['exporter'].str.upper() != user_input['Exporter'].upper()) & (analysis_data['unitprice'] < user_avg_price)]
    if not alternative_suppliers.empty:
        supplier_analysis = alternative_suppliers.groupby('exporter').agg(avg_unitprice=('unitprice', 'mean'), trade_count=('value', 'count'), num_importers=('importer', 'nunique')).reset_index().sort_values('avg_unitprice')
        supplier_analysis['price_saving_pct'] = (1 - supplier_analysis['avg_unitprice'] / user_avg_price) * 100
        supplier_analysis['stability_score'] = np.log1p(supplier_analysis['trade_count']) + np.log1p(supplier_analysis['num_importers'])
        if len(supplier_analysis) >= 3:
            low_q, high_q = supplier_analysis['stability_score'].quantile(0.33), supplier_analysis['stability_score'].quantile(0.67)
            conditions = [supplier_analysis['stability_score'] <= low_q, (supplier_analysis['stability_score'] > low_q) & (supplier_analysis['stability_score'] < high_q), supplier_analysis['stability_score'] >= high_q]; ratings = ['하', '중', '상']
            supplier_analysis['stability_rank'] = np.select(conditions, ratings, default='중')
        else: supplier_analysis['stability_rank'] = '중'
        analysis_result['supply_chain'] = {"user_avg_price": user_avg_price, "user_total_volume": user_total_volume, "alternatives": supplier_analysis}
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
            st.session_state[f'volume{key_suffix}'] = cols[5].number_input(f"volume_widget{key_suffix}", min_value=0.01, format="%.2f", value=st.session_state.get(f'volume{key_suffix}', 1000.0), key=f"volume_widget_k{key_suffix}", label_visibility="collapsed")
            st.session_state[f'value{key_suffix}'] = cols[6].number_input(f"value_widget{key_suffix}", min_value=0.01, format="%.2f", value=st.session_state.get(f'value{key_suffix}', 10000.0), key=f"value_widget_k{key_suffix}", label_visibility="collapsed")
            st.session_state[f'incoterms{key_suffix}'] = cols[7].selectbox(f"incoterms_widget{key_suffix}", ["FOB", "CFR", "CIF", "EXW", "DDP", "기타"], index=["FOB", "CFR", "CIF", "EXW", "DDP", "기타"].index(st.session_state.get(f'incoterms{key_suffix}', 'FOB')), key=f"incoterms_widget_k{key_suffix}", label_visibility="collapsed")
            if len(st.session_state.rows) > 1 and cols[8].button("삭제", key=f"delete{key_suffix}"): st.session_state.rows.pop(i); st.rerun()
        if st.button("➕ 내역 추가하기"):
            new_id = max(row['id'] for row in st.session_state.rows) + 1 if st.session_state.rows else 1; st.session_state.rows.append({'id': new_id}); st.rerun()
        st.markdown("---")
        consent = st.checkbox("정확한 분석을 위해 입력한 데이터가 활용되는 것에 동의합니다.", value=st.session_state.get('consent', True), key='consent_widget'); st.session_state['consent'] = consent
        if st.button("분석하기", type="primary", use_container_width=True):
            all_input_data = []; is_valid = True
            if not importer_name: st.error("⚠️ [입력 오류] 귀사의 업체명을 입력해주세요."); is_valid = False
            if not consent: st.warning("⚠️ 정보 활용 동의에 체크해주세요."); is_valid = False
            for i, row in enumerate(st.session_state.rows):
                key_suffix = f"_{row['id']}"; entry = { "Date": st.session_state.get(f'date{key_suffix}'), "Reported Product Name": st.session_state.get(f'product_name{key_suffix}'), "HS-Code": st.session_state.get(f'hscode{key_suffix}'), "Origin Country": st.session_state.get(f'origin{key_suffix}'), "Exporter": st.session_state.get(f'exporter{key_suffix}'), "Volume": st.session_state.get(f'volume{key_suffix}'), "Value": st.session_state.get(f'value{key_suffix}'), "Incoterms": st.session_state.get(f'incoterms{key_suffix}')}
                all_input_data.append(entry)
                if not all([entry['Reported Product Name'], entry['HS-Code'], entry['Origin Country'], entry['Exporter']]): st.error(f"⚠️ [입력 오류] {i+1}번째 줄의 필수 항목을 모두 입력해주세요."); is_valid = False
            if is_valid:
                with st.spinner('입력 데이터를 DB에 반영해 분석을 시작합니다...'):
                    purchase_df = pd.DataFrame(all_input_data)
                    if save_to_google_sheets(purchase_df, importer_name, consent):
                        purchase_df['cleaned_name'] = purchase_df['Reported Product Name'].apply(clean_text)
                        agg_funcs = {'Volume': 'sum', 'Value': 'sum', 'Reported Product Name': 'first', 'HS-Code': 'first', 'Exporter': 'first', 'Date':'first', 'Origin Country':'first', 'Incoterms':'first'}
                        aggregated_purchase_df = purchase_df.groupby('cleaned_name', as_index=False).agg(agg_funcs)
                        analysis_groups = []
                        company_data['cleaned_name'] = company_data['reported_product_name'].apply(clean_text)
                        for _, row in aggregated_purchase_df.iterrows():
                            entry = row.to_dict(); user_tokens = set(entry['cleaned_name'].split()); is_match = lambda name: user_tokens.issubset(set(str(name).split())); matched_df = company_data[company_data['cleaned_name'].apply(is_match)]; matched_products = sorted(matched_df['reported_product_name'].unique().tolist()); result = run_all_analysis([entry], company_data, matched_products, importer_name)
                            analysis_groups.append({"user_input": entry, "matched_products": matched_products, "selected_products": matched_products, "result": result})
                        st.session_state['importer_name_result'] = importer_name; st.session_state['analysis_groups'] = analysis_groups
                        st.success("분석 완료!"); st.rerun()
    
    if 'analysis_groups' in st.session_state:
        st.header("📊 분석 결과")
        processed_hscodes = []
        for group in st.session_state.analysis_groups:
            overview_res = group['result'].get('overview')
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

        for i, group in enumerate(st.session_state.analysis_groups):
            product_name = group['user_input']['Reported Product Name']; st.subheader(f"분석 그룹: \"{product_name}\"")
            result, diag_res, ts_res, p_res, s_res = group['result'], group['result'].get('diagnosis'), group['result'].get('timeseries'), group['result'].get('positioning'), group['result'].get('supply_chain')
            
            st.markdown("#### PART 1. 이번 거래 경쟁력 진단 요약")
            if diag_res:
                price_diff = (diag_res['user_price'] / diag_res['market_avg_price'] - 1) * 100 if diag_res['market_avg_price'] > 0 else 0
                cols = st.columns(3); cols[0].metric("이번 거래 단가", f"${diag_res['user_price']:.2f}", f"{price_diff:.1f}% (동월 평균 대비)", delta_color="inverse")
                cols[1].metric("동월 내 가격 백분위", f"상위 {100-diag_res['percentile']:.0f}%", help="100%에 가까울수록 동월 시장에서 저렴하게 구매한 거래입니다.")
                cols[2].metric("예상 추가 절감 가능 금액", f"${diag_res['potential_savings']:,.0f} 내외", help=f"동월 상위 10% 평균가(${diag_res['top_10_price']:.2f}) 기준")
            else: st.info("이번 거래와 동일한 월의 시장 데이터가 부족하여 진단 요약을 생성할 수 없습니다.")
            st.markdown("---")

            st.markdown("#### PART 2. 시계열 시장 동향 및 거래 위치")
            if ts_res and not ts_res['all_trades'].empty:
                fig_ts = go.Figure()
                all_trades_df = ts_res['all_trades'].copy()
                target_name = st.session_state.get('importer_name_result', '')
            
                # --- 익명화 로직 시작 ---
                # 1. 시계열 데이터에 있는 모든 고유 수입사 목록을 가져옵니다.
                unique_importers_ts = all_trades_df['importer'].unique()
                
                # 2. 각 수입사 이름에 익명(A사, B사...)을 짝지어주는 딕셔너리(지도)를 만듭니다.
                #    (단, 귀사의 이름은 바꾸지 않습니다.)
                anonymity_map_ts = {name: to_excel_col(i) for i, name in enumerate(unique_importers_ts) if name != target_name}
                anonymity_map_ts[target_name] = target_name
                
                # 3. 위에서 만든 지도를 사용하여 'Anonymized_Importer'라는 새 열을 추가합니다.
                all_trades_df['Anonymized_Importer'] = all_trades_df['importer'].map(anonymity_map_ts)
                # --- 익명화 로직 끝 ---
            
                # 차트의 회색 버블을 그릴 때, 익명화된 이름(Anonymized_Importer)을 hover 정보로 사용합니다.
                log_volume = np.log1p(all_trades_df['volume'])
                # ... (버블 사이즈 계산) ...
                fig_ts.add_trace(go.Scatter(
                    x=all_trades_df['date'], 
                    y=all_trades_df['unitprice'], 
                    mode='markers', 
                    marker=dict(size=bubble_size, color='lightgray', opacity=0.6), 
                    name='과거 시장 거래', 
                    text=all_trades_df['Anonymized_Importer'], # 익명화된 이름 사용
                    hovertemplate='<b>%{text}</b><br>단가: $%{y:,.2f}<extra></extra>' # hover 시 익명 이름 표시
                ))
            st.markdown("---")

            # --- main_dashboard 함수 내 PART 3 시각화 부분 ---
            
            st.markdown("#### PART 3. 경쟁 환경 및 전략 분석")
            if not p_res or p_res['importer_stats'].empty: st.info("경쟁 환경 분석을 위한 데이터가 부족합니다."); continue
            
            # 3-1. 시장 내 전략적 위치 (시점 정규화)
            col1, col2 = st.columns([10,1])
            col1.markdown("##### **3-1. 시장 내 전략적 위치 (시점 정규화)**")
            with col2:
                with st.popover("ℹ️"): st.markdown("""**가격 경쟁력 지수란?**\n계절성이나 시장 트렌드 등 시점 요인을 제거한 순수한 가격 경쟁력입니다.\n- **계산식:** `개별 거래 단가 / 해당 월의 시장 평균 단가`\n- **1.0 미만:** 시장 평균보다 저렴하게 구매\n- **1.0 초과:** 시장 평균보다 비싸게 구매""")
            
            importer_stats = p_res['importer_stats']
            target_name = st.session_state.get('importer_name_result', '')
            
            # 차트에 표시할 회사 목록(plot_df) 생성
            try:
                target_rank = importer_stats[importer_stats['importer'] == target_name].index[0]
                rank_margin = max(1, int(len(importer_stats) * 0.1))
                direct_peers = importer_stats.iloc[max(0, target_rank - rank_margin):min(len(importer_stats), target_rank + rank_margin + 1)]
            except IndexError:
                direct_peers = pd.DataFrame()
            plot_df = pd.concat([importer_stats.head(5), direct_peers, p_res['target_stats']]).drop_duplicates().reset_index(drop=True)
            
            # 익명화 이름 생성
            plot_df['Anonymized_Importer'] = [to_excel_col(j) if imp != target_name else target_name for j, imp in enumerate(plot_df['importer'])]
            
            # (수정) 버블 크기 계산 기준을 plot_df로 변경하여 정확도 향상
            log_values = np.log1p(plot_df['total_value'])
            min_size, max_size = 15, 80
            if log_values.max() > log_values.min():
                plot_df['size'] = min_size + ((log_values - log_values.min()) / (log_values.max() - log_values.min())) * (max_size - min_size)
            else:
                plot_df['size'] = [min_size] * len(plot_df)
            
            # 차트 생성
            x_mean = importer_stats['total_volume'].mean() # 시장 평균은 전체 데이터 기준이므로 importer_stats 사용 (정상)
            y_mean = 1.0
            fig_pos = go.Figure()
            
            # (수정) 데이터 참조를 plot_df로 변경하고, hover 정보에 익명화된 이름 사용
            competitors = plot_df[plot_df['importer'] != target_name]
            fig_pos.add_trace(go.Scatter(
                x=competitors['total_volume'], y=competitors['price_index'], 
                mode='markers', marker=dict(size=competitors['size'], color='#BDBDBD', opacity=0.5), 
                text=competitors['Anonymized_Importer'], # 익명 이름 사용
                hovertemplate='<b>%{text}</b><br>가격 경쟁력 지수: %{y:.2f}<extra></extra>'
            ))
            
            # (수정) 데이터 참조를 plot_df로 변경하고, hover 정보에 익명화된 이름 사용
            target_df = plot_df[plot_df['importer'] == target_name]
            if not target_df.empty:
                fig_pos.add_trace(go.Scatter(
                    x=target_df['total_volume'], y=target_df['price_index'], 
                    mode='markers', marker=dict(size=target_df['size'], color='#FF4B4B', opacity=1.0, line=dict(width=2, color='black')), 
                    name='귀사(과거 평균)', 
                    text=target_df['Anonymized_Importer'], # 익명 이름 사용 (실제로는 귀사 이름)
                    hovertemplate='<b>%{text} (평균)</b><br>가격 경쟁력 지수: %{y:.2f}<extra></extra>'
                ))
            
            current_tx_norm = p_res.get('current_transaction_normalized')
            if current_tx_norm:
                fig_pos.add_trace(go.Scatter(
                    x=[current_tx_norm['total_volume']], y=[current_tx_norm['price_index']], 
                    mode='markers', marker=dict(symbol='star', color='black', size=20, line=dict(color='white', width=2)), 
                    name='이번 거래', 
                    hovertemplate='<b>이번 거래</b><br>가격 경쟁력 지수: %{y:.2f}<extra></extra>'
                ))
            
            fig_pos.add_vline(x=x_mean, line_dash="dash", line_color="gray")
            fig_pos.add_hline(y=y_mean, line_dash="dash", line_color="gray")
            fig_pos.update_layout(title="<b>수입사 포지셔닝 맵 (시기 보정)</b>", xaxis_title="총 수입 중량 (KG, Log Scale)", yaxis_title="가격 경쟁력 지수 (1.0 = 시장 평균)", showlegend=False, xaxis_type="log")
            st.plotly_chart(fig_pos, use_container_width=True)

            col1, col2 = st.columns([10,1]); col1.markdown("##### **3-2. 실질 경쟁 그룹과의 비교**")
            with col2:
                with st.popover("ℹ️"): st.markdown("""**그룹 분류 기준:**\n- **시장 선도 그룹:** 수입 금액 기준 누적 70% 차지\n- **유사 규모 경쟁 그룹:** 귀사 순위 기준 상하 ±10%\n- **최저가 달성 그룹:** 시기 보정된 '가격 경쟁력 지수' 하위 15%\n *데이터 특성에 따라 조건에 맞는 경쟁사가 없으면 해당 그룹은 박스 플롯에 표시되지 않을 수 있습니다.""")
            rb_groups = p_res['rule_based_groups']; group_data = []
            for name, df in rb_groups.items():
                if not df.empty: 
                    df_copy = df.copy()
                    df_copy['group_name'] = name
                    group_data.append(df_copy[['group_name', 'price_index']])
            if group_data:
                plot_df_box = pd.concat(group_data)
                fig_box = px.box(plot_df_box, x='group_name', y='price_index', title="<b>주요 경쟁 그룹별 가격 경쟁력 분포</b>", points='all', labels={'group_name': '경쟁 그룹 유형', 'price_index': '가격 경쟁력 지수'})
                if not p_res['target_stats'].empty: fig_box.add_hline(y=p_res['target_stats']['price_index'].iloc[0], line_dash="dot", line_color="orange", annotation_text="귀사 평균")
                if current_tx_norm: fig_box.add_hline(y=current_tx_norm['price_index'], line_dash="dash", line_color="blue", annotation_text="이번 거래")
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
                    with st.popover("ℹ️"):
                        st.markdown("""**공급 안정성 기준:**\n발견된 대체 공급처들의 '거래 빈도'와 '거래처 수'를 종합하여 계산된 안정성 점수를 기준으로 상대 평가됩니다.\n- **상:** 상위 33%\n- **중:** 중간 33%\n- **하:** 하위 33%""")
                
                recommended_list = alts[alts['price_saving_pct'] >= target_saving_pct].copy()
                recommended_list.reset_index(drop=True, inplace=True); recommended_list['순번'] = recommended_list.index + 1
                recommended_list.rename(columns={'avg_unitprice': '평균 단가', 'price_saving_pct': '가격 경쟁력(%)', 'trade_count': '거래 빈도', 'num_importers': '거래 수입사 개수', 'stability_rank': '공급 안정성'}, inplace=True)
                st.dataframe(recommended_list[['순번', '평균 단가', '가격 경쟁력(%)', '거래 빈도', '거래 수입사 개수', '공급 안정성']], use_container_width=True, 
                             column_config={"평균 단가": st.column_config.NumberColumn(format="$%.2f"), "가격 경쟁력(%)": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=alts['price_saving_pct'].max())}, hide_index=True)
            st.markdown("---")

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
