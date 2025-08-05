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
        Q1, Q3 = df['unitprice'].quantile(0.25), df['unitprice'].quantile(0.75)
        IQR = Q3 - Q1
        df = df[~((df['unitprice'] < (Q1 - 1.5 * IQR)) | (df['unitprice'] > (Q3 + 1.5 * IQR)))]
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
        st.toast("입력 정보가 Google Sheet에 저장되었습니다.", icon="✅")
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
    is_high_volume = row['total_volume'] >= x_mean; is_high_price = row['avg_unitprice'] >= y_mean
    if is_high_volume and is_high_price: return "마켓 리더"
    elif not is_high_volume and is_high_price: return "프리미엄 전략 그룹"
    elif not is_high_volume and not is_high_price: return "효율적 소싱 그룹"
    else: return "원가 우위 그룹"

# --- 메인 분석 로직 (Overview 기능 복원) ---
def run_all_analysis(user_inputs, full_company_data, selected_products, target_importer_name):
    analysis_result = {"overview": {}, "positioning": {}, "supply_chain": {}}
    user_input = user_inputs[0]
    hscode = str(user_input.get('HS-Code', ''))

    # 1. Overview 분석 (HS-CODE 기준)
    if hscode:
        hscode_data = full_company_data[full_company_data['hs_code'].astype(str) == hscode].copy()
        if not hscode_data.empty:
            this_year = datetime.now().year
            vol_this_year = hscode_data[hscode_data['date'].dt.year == this_year]['volume'].sum()
            vol_last_year = hscode_data[hscode_data['date'].dt.year == this_year - 1]['volume'].sum()
            price_this_year = hscode_data[hscode_data['date'].dt.year == this_year]['unitprice'].mean()
            price_last_year = hscode_data[hscode_data['date'].dt.year == this_year - 1]['unitprice'].mean()
            
            analysis_result['overview'] = {
                "hscode": hscode, "this_year": this_year,
                "vol_this_year": vol_this_year, "vol_last_year": vol_last_year,
                "price_this_year": price_this_year, "price_last_year": price_last_year,
                "freq_this_year": len(hscode_data[hscode_data['date'].dt.year == this_year]),
                "product_composition": hscode_data.groupby('reported_product_name')['value'].sum().nlargest(10)
            }

    # 2. Positioning 및 Supply Chain 분석 (선택된 제품 기준)
    analysis_data = full_company_data[full_company_data['reported_product_name'].isin(selected_products)].copy()
    if not analysis_data.empty:
        importer_stats = analysis_data.groupby('importer').agg(total_value=('value', 'sum'), total_volume=('volume', 'sum'), trade_count=('value', 'count'), avg_unitprice=('unitprice', 'mean')).reset_index().sort_values('total_value', ascending=False).reset_index(drop=True)
        if not importer_stats.empty:
            volume_mean = importer_stats['total_volume'].mean(); price_mean = importer_stats['avg_unitprice'].mean()
            importer_stats['quadrant_group'] = importer_stats.apply(assign_quadrant_group, axis=1, args=(volume_mean, price_mean))
            analysis_result['positioning'] = {"importer_stats": importer_stats, "target_stats": importer_stats[importer_stats['importer'] == target_importer_name]}

        user_avg_price = user_input['Value'] / user_input['Volume'] if user_input['Volume'] > 0 else 0
        alternative_suppliers = analysis_data[(analysis_data['exporter'].str.upper() != user_input['Exporter'].upper()) & (analysis_data['unitprice'] < user_avg_price)]
        if not alternative_suppliers.empty:
            supplier_analysis = alternative_suppliers.groupby('exporter').agg(avg_unitprice=('unitprice', 'mean'), trade_count=('value', 'count'), num_importers=('importer', 'nunique')).reset_index().sort_values('avg_unitprice')
            supplier_analysis['price_saving_pct'] = (1 - supplier_analysis['avg_unitprice'] / user_avg_price) * 100
            supplier_analysis['stability_score'] = np.log1p(supplier_analysis['trade_count']) + np.log1p(supplier_analysis['num_importers'])
            analysis_result['supply_chain'] = {"user_avg_price": user_avg_price, "user_total_volume": sum(item['Volume'] for item in user_inputs), "alternatives": supplier_analysis}
    
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
            st.session_state[f'volume{key_suffix}'] = cols[5].number_input(f"volume_widget{key_suffix}", min_value=0.01, format="%.2f", value=st.session_state.get(f'volume{key_suffix}', 1.0), key=f"volume_widget_k{key_suffix}", label_visibility="collapsed")
            st.session_state[f'value{key_suffix}'] = cols[6].number_input(f"value_widget{key_suffix}", min_value=0.01, format="%.2f", value=st.session_state.get(f'value{key_suffix}', 1.0), key=f"value_widget_k{key_suffix}", label_visibility="collapsed")
            st.session_state[f'incoterms{key_suffix}'] = cols[7].selectbox(f"incoterms_widget{key_suffix}", ["FOB", "CFR", "CIF", "EXW", "DDP", "기타"], index=["FOB", "CFR", "CIF", "EXW", "DDP", "기타"].index(st.session_state.get(f'incoterms{key_suffix}', 'FOB')), key=f"incoterms_widget_k{key_suffix}", label_visibility="collapsed")
            if len(st.session_state.rows) > 1 and cols[8].button("삭제", key=f"delete{key_suffix}"): st.session_state.rows.pop(i); st.rerun()
        if st.button("➕ 내역 추가하기"):
            new_id = max(row['id'] for row in st.session_state.rows) + 1 if st.session_state.rows else 1; st.session_state.rows.append({'id': new_id}); st.rerun()
        st.markdown("---")
        consent = st.checkbox("정보 활용 동의", value=st.session_state.get('consent', True), key='consent_widget'); st.session_state['consent'] = consent
        if st.button("분석하기", type="primary", use_container_width=True):
            all_input_data = []; is_valid = True
            if not importer_name: st.error("⚠️ [입력 오류] 귀사의 업체명을 입력해주세요."); is_valid = False
            if not consent: st.warning("⚠️ 정보 활용 동의에 체크해주세요."); is_valid = False
            for i, row in enumerate(st.session_state.rows):
                key_suffix = f"_{row['id']}"; entry = { "Date": st.session_state.get(f'date{key_suffix}'), "Reported Product Name": st.session_state.get(f'product_name{key_suffix}'), "HS-Code": st.session_state.get(f'hscode{key_suffix}'), "Origin Country": st.session_state.get(f'origin{key_suffix}'), "Exporter": st.session_state.get(f'exporter{key_suffix}'), "Volume": st.session_state.get(f'volume{key_suffix}'), "Value": st.session_state.get(f'value{key_suffix}'), "Incoterms": st.session_state.get(f'incoterms{key_suffix}')}
                all_input_data.append(entry)
                if not all([entry['Reported Product Name'], entry['HS-Code'], entry['Origin Country'], entry['Exporter']]): st.error(f"⚠️ [입력 오류] {i+1}번째 줄의 필수 항목을 모두 입력해주세요."); is_valid = False
            if is_valid:
                with st.spinner('입력 데이터를 저장하고 분석을 시작합니다...'):
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
        
        # --- Overview UI (복원 및 개선) ---
        processed_hscodes = []
        for group in st.session_state.analysis_groups:
            overview_res = group['result'].get('overview')
            if overview_res and overview_res['hscode'] not in processed_hscodes:
                st.subheader(f"📈 HS-Code {overview_res['hscode']} 시장 개요")
                o = overview_res
                cols = st.columns(3)
                vol_yoy = (o['vol_this_year'] - o['vol_last_year']) / o['vol_last_year'] if o['vol_last_year'] > 0 else np.nan
                price_yoy = (o['price_this_year'] - o['price_last_year']) / o['price_last_year'] if o['price_last_year'] > 0 else np.nan
                
                cols[0].metric(f"{o['this_year']}년 수입 중량 (KG)", f"{o['vol_this_year']:,.0f}", f"{vol_yoy:.1%}" if pd.notna(vol_yoy) else "N/A", delta_color="inverse")
                cols[1].metric(f"{o['this_year']}년 평균 단가 (USD/KG)", f"${o['price_this_year']:.2f}", f"{price_yoy:.1%}" if pd.notna(price_yoy) else "N/A", delta_color="inverse")
                cols[2].metric(f"{o['this_year']}년 총 수입 건수", f"{o['freq_this_year']:,} 건")

                if not o['product_composition'].empty:
                    pie_fig = px.pie(o['product_composition'], names=o['product_composition'].index, values='value', title='<b>상위 10개 제품 구성 (수입 금액 기준)</b>', hole=0.3)
                    pie_fig.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(pie_fig, use_container_width=True)
                
                st.markdown("---")
                processed_hscodes.append(overview_res['hscode'])


        for i, group in enumerate(st.session_state.analysis_groups):
            product_name = group['user_input']['Reported Product Name']; st.subheader(f"분석 그룹: \"{product_name}\"")
            result, p_res, s_res = group['result'], group['result'].get('positioning'), group['result'].get('supply_chain')
            st.markdown("#### PART 1. 시장 포지션 분석")
            if not p_res or p_res['importer_stats'].empty: st.info("포지션 분석을 위한 데이터가 부족합니다."); continue
            importer_stats = p_res['importer_stats']; target_name = st.session_state.get('importer_name_result', '')
            try:
                target_rank = importer_stats[importer_stats['importer'] == target_name].index[0]; rank_margin = max(1, int(len(importer_stats) * 0.1)); direct_peers = importer_stats.iloc[max(0, target_rank - rank_margin):min(len(importer_stats), target_rank + rank_margin + 1)]
            except IndexError: direct_peers = pd.DataFrame()
            plot_df = pd.concat([importer_stats.head(5), direct_peers, p_res['target_stats']]).drop_duplicates().reset_index(drop=True)
            plot_df['Anonymized_Importer'] = [f"{chr(ord('A')+j)}사" if imp != target_name else target_name for j, imp in enumerate(plot_df['importer'])]
            log_values = np.log1p(plot_df['total_value']); min_size, max_size = 15, 80
            if log_values.max() > log_values.min(): plot_df['size'] = min_size + ((log_values - log_values.min()) / (log_values.max() - log_values.min())) * (max_size - min_size)
            else: plot_df['size'] = [min_size] * len(plot_df)
            x_mean = importer_stats['total_volume'].mean(); y_mean = importer_stats['avg_unitprice'].mean()
            fig = go.Figure()
            competitors = plot_df[plot_df['importer'] != target_name]; fig.add_trace(go.Scatter(x=competitors['total_volume'], y=competitors['avg_unitprice'], mode='markers', marker=dict(size=competitors['size'], color='#BDBDBD', opacity=0.5), text=competitors['Anonymized_Importer'], hovertemplate='<b>%{text}</b><br>수입량: %{x:,.0f} KG<br>평균단가: $%{y:,.2f}<extra></extra>'))
            target_df = plot_df[plot_df['importer'] == target_name]
            if not target_df.empty: fig.add_trace(go.Scatter(x=target_df['total_volume'], y=target_df['avg_unitprice'], mode='markers', marker=dict(size=target_df['size'], color='#FF4B4B', opacity=1.0, line=dict(width=2, color='black')), text=target_df['Anonymized_Importer'], hovertemplate='<b>%{text}</b><br>수입량: %{x:,.0f} KG<br>평균단가: $%{y:,.2f}<extra></extra>'))
            fig.add_vline(x=x_mean, line_dash="dash", line_color="gray", annotation_text="평균 수입량"); fig.add_hline(y=y_mean, line_dash="dash", line_color="gray", annotation_text="평균 단가")
            x_range = np.log10(importer_stats['total_volume'].max()) - np.log10(importer_stats['total_volume'].min()); y_range = importer_stats['avg_unitprice'].max() - importer_stats['avg_unitprice'].min()
            fig.add_annotation(x=np.log10(x_mean) + x_range*0.4, y=y_mean+y_range*0.4, text="<b>마켓 리더</b>", showarrow=False, font=dict(color="grey")); fig.add_annotation(x=np.log10(x_mean) - x_range*0.4, y=y_mean+y_range*0.4, text="<b>프리미엄 전략 그룹</b>", showarrow=False, font=dict(color="grey")); fig.add_annotation(x=np.log10(x_mean) - x_range*0.4, y=y_mean-y_range*0.4, text="<b>효율적 소싱 그룹</b>", showarrow=False, font=dict(color="grey")); fig.add_annotation(x=np.log10(x_mean) + x_range*0.4, y=y_mean-y_range*0.4, text="<b>원가 우위 그룹</b>", showarrow=False, font=dict(color="grey"))
            if not target_df.empty: target = target_df.iloc[0]; fig.add_annotation(x=np.log10(target['total_volume']), y=target['avg_unitprice'], text="<b>귀사 위치</b>", showarrow=True, arrowhead=2, arrowcolor="#FF4B4B", ax=-40, ay=-40, bordercolor="#FF4B4B", borderwidth=2, bgcolor="white")
            fig.update_layout(title="<b>수입사 포지셔닝 맵 (시장 전략 분석)</b>", xaxis_title="총 수입 중량 (KG, Log Scale)", yaxis_title="평균 수입 단가 (USD/KG)", showlegend=False, xaxis_type="log")
            st.plotly_chart(fig, use_container_width=True)
            col1, col2 = st.columns([10,1]); col1.markdown("##### **시장 전략 그룹별 상세 분석**")
            with col2:
                with st.popover("ℹ️"):
                    st.markdown("""**그룹 분류 기준:** 포지셔닝 맵의 4개 영역은 시장 평균 수입량과 평균 단가를 기준으로 나뉩니다.
- **마켓 리더:** 품질과 물량 모두를 장악하는 시장의 가장 강력한 경쟁자 그룹.
- **프리미엄 전략 그룹:** 특정 고부가가치 제품에 집중하여 수익을 극대화하는 그룹.
- **효율적 소싱 그룹:** 가격 경쟁력을 바탕으로 민첩하게 구매 기회를 포착하는 그룹.
- **원가 우위 그룹:** 압도적인 물량을 통해 원가 경쟁력을 확보하는 그룹.""")
            st.info("포지셔닝 맵의 4개 그룹에 속한 기업들의 상세 데이터를 비교하여 각 그룹의 특징을 파악합니다.")
            category_orders={"quadrant_group": ["효율적 소싱 그룹", "원가 우위 그룹", "프리미엄 전략 그룹", "마켓 리더"]}
            fig_box = px.box(importer_stats, x='quadrant_group', y='avg_unitprice', title="<b>전략 그룹별 단가 분포</b>", points='all', labels={'quadrant_group': '전략 그룹 유형', 'avg_unitprice': '평균 수입 단가'}, category_orders=category_orders)
            if not p_res['target_stats'].empty: fig_box.add_hline(y=p_res['target_stats']['avg_unitprice'].iloc[0], line_dash="dot", line_color="orange", annotation_text="귀사 단가")
            st.plotly_chart(fig_box, use_container_width=True)
            st.markdown("---"); st.markdown("#### PART 2. 공급망 분석 및 비용 절감 시뮬레이션")
            if not s_res or s_res['alternatives'].empty: st.info("현재 거래 조건보다 더 저렴한 대안 공급처를 찾지 못했습니다.")
            else:
                alts, best_deal = s_res['alternatives'], s_res['alternatives'].iloc[0]; st.success(f"**비용 절감 기회 포착!** 현재 거래처보다 **최대 {best_deal['price_saving_pct']:.1f}%** 저렴한 대체 거래처가 존재합니다.")
                col1, col2 = st.columns(2); target_saving_pct = col1.slider("목표 단가 절감률(%)", 0.0, float(best_deal['price_saving_pct']), float(best_deal['price_saving_pct'] / 2), 0.5, "%.1f%%", key=f"slider_{i}"); expected_saving = s_res['user_total_volume'] * s_res['user_avg_price'] * (target_saving_pct / 100); col2.metric(f"예상 절감액 (수입량 {s_res['user_total_volume']:,.0f}KG 기준)", f"${expected_saving:,.0f}")
                st.markdown("##### **추천 대체 공급처 리스트** (안정성 함께 고려)"); recommended_list = alts[alts['price_saving_pct'] >= target_saving_pct].copy(); recommended_list.rename(columns={'exporter': '수출업체', 'avg_unitprice': '평균 단가', 'price_saving_pct': '가격 경쟁력(%)', 'trade_count': '거래 빈도', 'num_importers': '거래처 수', 'stability_score': '공급 안정성'}, inplace=True)
                st.dataframe(recommended_list[['수출업체', '평균 단가', '가격 경쟁력(%)', '거래 빈도', '거래처 수', '공급 안정성']], use_container_width=True, column_config={"평균 단가": st.column_config.NumberColumn(format="$%.2f"), "가격 경쟁력(%)": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=alts['price_saving_pct'].max()), "공급 안정성": st.column_config.BarChartColumn(y_min=0, y_max=alts['stability_score'].max())}, hide_index=True)
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
