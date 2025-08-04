import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import re
from datetime import datetime
import gspread
from google.oauth2.service_account import Credentials

# --- 초기 설정 및 페이지 구성 ---
st.set_page_config(layout="wide", page_title="수입 경쟁력 진단 솔루션")

# --- Google Sheets에서 데이터 불러오기 ---
@st.cache_data(ttl=600)
def load_company_data():
    """Google Sheets에서 TDS를 불러옵니다."""
    try:
        if "gcp_service_account" not in st.secrets:
            st.error("Secrets 설정 오류: [gcp_service_account] 섹션을 찾을 수 없습니다.")
            return pd.DataFrame()
        scopes = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
        creds = Credentials.from_service_account_info(st.secrets["gcp_service_account"], scopes=scopes)
        client = gspread.authorize(creds)
        spreadsheet = client.open("DEMO_app_DB")
        worksheet = spreadsheet.worksheet("TDS")
        data = worksheet.get_all_records()
        df = pd.DataFrame(data)
        df.dropna(how="all", inplace=True)
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
        df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
        df.dropna(subset=['Date', 'Volume', 'Value'], inplace=True)
        return df
    except Exception as e:
        st.error(f"Google Sheets 연결 중 오류가 발생했습니다: {e}")
        return pd.DataFrame()

OUR_COMPANY_DATA = load_company_data()

# --- 새로운 범용 스마트 매칭 로직 ---
def clean_text(text):
    """어떤 제품명이든 통용되는 텍스트 정제 함수"""
    if not isinstance(text, str): return ''
    text = text.lower()
    text = re.sub(r'\(.*?\)', ' ', text)
    text = re.sub(r'\[.*?\]', ' ', text)
    text = re.sub(r'(\d+)\s*(?:y|yo|year|years|old|년)', r'\1', text)
    text = re.sub(r'[^a-z0-9\s\uac00-\ud7a3]', ' ', text)
    return ' '.join(text.split())

# --- 데이터 처리 로직 (개별 제품 분석 지원) ---
def process_analysis_data(user_input_row, comparison_df, target_importer_name):
    """하나의 제품 그룹에 대한 분석을 수행합니다."""
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

def main_dashboard():
    st.title("📈 수입 경쟁력 진단 솔루션")
    st.markdown("트릿지 데이터를 기반으로 시장 내 경쟁력을 진단하고 비용 절감 기회를 포착하세요.")

    if OUR_COMPANY_DATA.empty: return

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
            cols[3].selectbox("원산지", [''] + sorted(OUR_COMPANY_DATA['Export Country'].unique()), key=f"origin_{i}")
            cols[4].selectbox("수출업체", [''] + sorted(OUR_COMPANY_DATA['Exporter'].unique()), key=f"exporter_{i}")
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
                
                OUR_COMPANY_DATA['cleaned_name'] = OUR_COMPANY_DATA['Reported Product Name'].apply(clean_text)
                
                for i in range(len(st.session_state.rows)):
                    user_product_name = st.session_state[f'product_name_{i}']
                    entry = {
                        'Date': st.session_state[f'date_{i}'],
                        'Reported Product Name': user_product_name,
                        'HS-CODE': st.session_state[f'hscode_{i}'],
                        'Origin Country': st.session_state[f'origin_{i}'].upper(),
                        'Exporter': st.session_state[f'exporter_{i}'].upper(),
                        'Volume': st.session_state[f'volume_{i}'],
                        'Value': st.session_state[f'value_{i}'],
                        'Incoterms': st.session_state[f'incoterms_{i}'],
                    }
                    if not user_product_name:
                        st.error(f"{i+1}번째 행의 제품 상세명을 입력해주세요.")
                        return
                    
                    all_purchase_data.append(entry)
                    user_tokens = set(clean_text(user_product_name).split())
                    
                    def is_match(cleaned_tds_name):
                        return user_tokens.issubset(set(cleaned_tds_name.split()))
                    
                    matched_df = OUR_COMPANY_DATA[OUR_COMPANY_DATA['cleaned_name'].apply(is_match)]
                    
                    analysis_groups.append({
                        "id": i,
                        "user_input": entry,
                        "matched_products": sorted(matched_df['Reported Product Name'].unique().tolist()),
                        "selected_products": sorted(matched_df['Reported Product Name'].unique().tolist())
                    })

                # 수정: Google Sheets 저장 로직 안정성 강화 및 상세 오류 출력
                try:
                    scopes = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
                    creds = Credentials.from_service_account_info(st.secrets["gcp_service_account"], scopes=scopes)
                    client = gspread.authorize(creds)
                    spreadsheet = client.open("DEMO_app_DB")
                    
                    try:
                        worksheet = spreadsheet.worksheet("Customer_input")
                    except gspread.exceptions.WorksheetNotFound:
                        worksheet = spreadsheet.add_worksheet(title="Customer_input", rows=1, cols=20)

                    save_data_df = pd.DataFrame(all_purchase_data)
                    save_data_df['importer_name'] = importer_name
                    save_data_df['consent'] = consent
                    save_data_df['timestamp'] = datetime.now().strftime("%Y-%m-%d")
                    save_data_df['Date'] = save_data_df['Date'].dt.strftime('%Y-%m-%d')
                    
                    if not worksheet.get('A1'):
                        worksheet.update([save_data_df.columns.values.tolist()] + save_data_df.values.tolist(), value_input_option='USER_ENTERED')
                    else:
                        worksheet.append_rows(save_data_df.values.tolist(), value_input_option='USER_ENTERED')

                    st.toast("입력 정보가 Google Sheet에 저장되었습니다.", icon="✅")
                except gspread.exceptions.APIError as e:
                    st.error("Google Sheets API 오류로 저장에 실패했습니다.")
                    st.json(e.response.json()) # Google이 보낸 실제 오류 메시지를 출력
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

            comparison_df = OUR_COMPANY_DATA[OUR_COMPANY_DATA['Reported Product Name'].isin(group['selected_products'])]
            
            competitor_res, yearly_res, timeseries_res = process_analysis_data(
                group['user_input'], 
                comparison_df, 
                st.session_state['importer_name_result']
            )
            
            st.markdown("#### 1. 경쟁사 Unit Price 비교 분석")
            if not competitor_res:
                st.write("비교할 경쟁사 데이터가 없습니다.")
            else:
                for (year, exporter), data in competitor_res.items():
                    with st.container(border=True):
                        st.markdown(f"**{year}년 / 수출업체: {exporter}**")
                        data['구분'] = np.where(data['Importer'] == st.session_state['importer_name_result'].upper(), '귀사', '경쟁사')
                        fig = px.box(data, x='Importer', y='unitPrice', title=f"경쟁사 Unit Price 분포 비교",
                                     color='구분',
                                     color_discrete_map={'귀사': '#ef4444', '경쟁사': '#3b82f6'},
                                     points='all')
                        fig.update_layout(legend_title_text=None, xaxis_title="수입사", yaxis_title="Unit Price (USD/KG)")
                        st.plotly_chart(fig, use_container_width=True)

            st.markdown("#### 2. 연도별 수입 중량 및 Unit Price 트렌드")
            if not yearly_res:
                st.write("분석할 연도별 데이터가 없습니다.")
            else:
                for (exporter, origin), data in yearly_res.items():
                    with st.container(border=True):
                        st.markdown(f"**{exporter} 로부터의 {origin}산 품목 수입 트렌드**")
                        fig = go.Figure()
                        fig.add_trace(go.Bar(x=data['chart_data']['year'], y=data['chart_data']['volume'], name='수입 중량 (KG)', yaxis='y1'))
                        fig.add_trace(go.Line(x=data['chart_data']['year'], y=data['chart_data']['unitPrice'], name='Unit Price (USD/KG)', yaxis='y2', mode='lines+markers'))
                        fig.update_layout(yaxis=dict(title="수입 중량 (KG)"), yaxis2=dict(title="Unit Price (USD/KG)", overlaying='y', side='right'), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                        st.plotly_chart(fig, use_container_width=True)
                        if data['saving_info']: st.success(f"💰 데이터 기반 예상 절감 가능 금액: 약 ${data['saving_info']['potential_saving']:,.0f}")

            st.markdown(f"#### 3. \"{group['user_input']['Reported Product Name']}\" 수입 추이")
            if not timeseries_res:
                st.write("분석할 시계열 데이터가 없습니다.")
            else:
                for origin, data in timeseries_res.items():
                    with st.container(border=True):
                        st.markdown(f"**{origin} 원산지 품목 Unit Price 트렌드**")
                        fig = px.line(data['chart_data'], x='monthYear', y=['avgPrice', 'targetPrice', 'bestPrice'], markers=True, labels={'monthYear': '월', 'value': 'Unit Price (USD/KG)'})
                        new_names = {'avgPrice':'시장 평균가', 'targetPrice':'귀사 평균가', 'bestPrice':'시장 최저가'}
                        fig.for_each_trace(lambda t: t.update(name = new_names[t.name]))
                        st.plotly_chart(fig, use_container_width=True)
                        if data['saving_info']: st.success(f"💰 데이터 기반 예상 절감 가능 금액: 약 ${data['saving_info']['potential_saving']:,.0f}")

        if st.button("🔄 새로운 분석 시작하기"):
            keys_to_keep = ['logged_in']
            for key in list(st.session_state.keys()):
                if key not in keys_to_keep: del st.session_state[key]
            st.rerun()

# --- 메인 로직 ---
if 'logged_in' not in st.session_state: st.session_state['logged_in'] = False
if st.session_state['logged_in']: main_dashboard()
else: login_screen()
