import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import re
from datetime import datetime
from streamlit_gsheets import GSheetsConnection

# --- 초기 설정 및 페이지 구성 ---
st.set_page_config(layout="wide", page_title="수입 경쟁력 진단 솔루션")

# --- Google Sheets에서 데이터 불러오기 ---
@st.cache_data(ttl=600)
def load_company_data():
    """Google Sheets에서 회사 데이터를 불러옵니다."""
    try:
        conn = st.connection("gsheets", type=GSheetsConnection)
        # 'TDS'는 회사 데이터가 있는 실제 시트 이름입니다.
        df = conn.read(worksheet="TDS") 
        df.dropna(how="all", inplace=True)
        # 데이터 타입 변환 (오류 방지)
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
        df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
        df.dropna(subset=['Date', 'Volume', 'Value'], inplace=True) # 필수 컬럼 누락 데이터 제거
        return df
    except Exception as e:
        st.error(f"Google Sheets에서 데이터를 불러오는 중 오류가 발생했습니다: {e}")
        st.info("Streamlit Secrets 설정 및 Google Sheet('TDS' 탭) 공유 설정을 확인해주세요.")
        return pd.DataFrame()

OUR_COMPANY_DATA = load_company_data()

# --- 스마트 매칭 로직 (하이브리드 방식) ---
def clean_text(text):
    if not isinstance(text, str): return ''
    text = text.lower()
    text = re.sub(r'\[.*?\]', ' ', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return ' '.join(text.split())

def smart_match_products(user_product_name, company_product_list):
    matched_products = set()
    user_tokens = set(clean_text(user_product_name).split())
    if not user_tokens: return []
    user_no_space = re.sub(r'\s+', '', clean_text(user_product_name))
    for db_product_name in company_product_list:
        db_tokens = set(clean_text(db_product_name).split())
        if user_tokens.issubset(db_tokens):
            matched_products.add(db_product_name)
            continue
        db_no_space = re.sub(r'\s+', '', clean_text(db_product_name))
        if user_no_space in db_no_space:
            matched_products.add(db_product_name)
    return sorted(list(matched_products))

# --- 데이터 처리 로직 (전체 복원) ---
def process_analysis_data(target_df, company_df, target_importer_name):
    if company_df.empty or target_df.empty:
        return {}, {}, {}
        
    target_df['Importer'] = target_importer_name.upper()
    all_df = pd.concat([company_df, target_df], ignore_index=True)
    all_df['unitPrice'] = all_df['Value'] / all_df['Volume']
    all_df['year'] = all_df['Date'].dt.year
    all_df['monthYear'] = all_df['Date'].dt.to_period('M').astype(str)

    # 1. 경쟁사 단가 비교 분석
    competitor_analysis = {}
    for _, row in target_df.iterrows():
        year = row['Date'].year
        exporter = row['Exporter'].upper()
        
        key = (year, exporter)
        if key not in competitor_analysis:
            related_trades = all_df[(all_df['year'] == year) & (all_df['Exporter'].str.upper() == exporter)]
            if related_trades.empty: continue

            importer_prices = related_trades.groupby('Importer').apply(
                lambda x: x['Value'].sum() / x['Volume'].sum() if x['Volume'].sum() > 0 else 0
            ).reset_index(name='unitPrice').sort_values('unitPrice')

            top5 = importer_prices.head(5)
            target_up = row['Value'] / row['Volume'] if row['Volume'] > 0 else 0
            is_target_in_top5 = target_importer_name.upper() in top5['Importer'].values
            
            if not is_target_in_top5 and target_up > 0:
                target_price_df = pd.DataFrame([{'Importer': target_importer_name.upper(), 'unitPrice': target_up}])
                top5 = pd.concat([top5, target_price_df]).sort_values('unitPrice').head(6)
            
            competitor_analysis[key] = top5

    # 2. 연도별 수입 중량 및 단가 분석
    yearly_analysis = {}
    for _, row in target_df.iterrows():
        exporter = row['Exporter'].upper()
        origin = row['Origin Country'].upper()
        
        key = (exporter, origin)
        if key not in yearly_analysis:
            target_unit_price = row['Value'] / row['Volume']
            
            other_companies = all_df[
                (all_df['Exporter'].str.upper() == exporter) & 
                (all_df['Origin Country'].str.upper() == origin) &
                (all_df['Importer'].str.upper() != target_importer_name.upper()) &
                (all_df['unitPrice'] < target_unit_price)
            ]
            
            saving_info = None
            if not other_companies.empty:
                avg_unit_price = other_companies['Value'].sum() / other_companies['Volume'].sum()
                potential_saving = (target_unit_price - avg_unit_price) * row['Volume']
                saving_info = {'potential_saving': potential_saving}

            yearly_data = all_df[(all_df['Exporter'].str.upper() == exporter) & (all_df['Origin Country'].str.upper() == origin)]
            summary = yearly_data.groupby('year').agg(
                volume=('Volume', 'sum'),
                value=('Value', 'sum')
            ).reset_index()
            summary['unitPrice'] = summary['value'] / summary['volume']
            
            yearly_analysis[key] = {'chart_data': summary, 'saving_info': saving_info}

    # 3. 시계열 단가 비교 분석
    time_series_analysis = {}
    for _, row in target_df.iterrows():
        origin = row['Origin Country'].upper()
        if origin not in time_series_analysis:
            related_trades = all_df[all_df['Origin Country'].str.upper() == origin]
            
            monthly_summary = related_trades.groupby('monthYear').agg(
                avgPrice=('unitPrice', 'mean'),
                bestPrice=('unitPrice', 'min')
            ).reset_index()

            target_trades = related_trades[related_trades['Importer'].str.upper() == target_importer_name.upper()]
            target_monthly = target_trades.groupby('monthYear').agg(
                targetPrice=('unitPrice', 'mean')
            ).reset_index()

            chart_data = pd.merge(monthly_summary, target_monthly, on='monthYear', how='left').sort_values('monthYear')
            
            target_unit_price = row['Value'] / row['Volume']
            cheaper_trades = all_df[(all_df['Origin Country'].str.upper() == origin) & (all_df['unitPrice'] < target_unit_price)]
            saving_info = None
            if not cheaper_trades.empty:
                avg_unit_price = cheaper_trades['Value'].sum() / cheaper_trades['Volume'].sum()
                potential_saving = (target_unit_price - avg_unit_price) * row['Volume']
                saving_info = {'potential_saving': potential_saving}

            time_series_analysis[origin] = {'chart_data': chart_data, 'saving_info': saving_info}
            
    return competitor_analysis, yearly_analysis, time_series_analysis

# --- UI Components ---
def login_screen():
    st.title("🔐 수입 경쟁력 진단 솔루션")
    st.write("솔루션 접속을 위해 비밀번호를 입력해주세요.")
    with st.form("login_form"):
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
    st.markdown("귀사의 수입 데이터를 기반으로 시장 내 경쟁력을 진단하고 비용 절감 기회를 포착하세요.")

    if OUR_COMPANY_DATA.empty: return

    with st.expander("STEP 1: 분석 정보 입력", expanded='analysis_results' not in st.session_state):
        importer_name = st.text_input("1. 귀사의 수입업체명을 입력해주세요.", key="importer_name").upper()
        st.markdown("---")
        st.markdown("2. 분석할 구매 내역을 입력해주세요.")
        if 'rows' not in st.session_state: st.session_state['rows'] = [{'id': 1}]
        
        for i, row in enumerate(st.session_state.rows):
            cols = st.columns([2, 3, 2, 2, 2, 2, 2, 1])
            cols[0].date_input("수입일", key=f"date_{i}")
            cols[1].text_input("제품 상세명", placeholder="예: 발렌타인 17년", key=f"product_name_{i}")
            cols[2].text_input("HS-CODE(6자리)", max_chars=6, key=f"hscode_{i}")
            cols[3].selectbox("원산지", [''] + sorted(OUR_COMPANY_DATA['Export Country'].unique()), key=f"origin_{i}")
            cols[4].selectbox("수출업체", [''] + sorted(OUR_COMPANY_DATA['Exporter'].unique()), key=f"exporter_{i}")
            cols[5].number_input("수입 중량(KG)", min_value=0.01, format="%.2f", key=f"volume_{i}")
            cols[6].number_input("총 수입금액(USD)", min_value=0.01, format="%.2f", key=f"value_{i}")
            if len(st.session_state.rows) > 1 and cols[7].button("삭제", key=f"delete_{i}"):
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
                purchase_data = []
                all_matched_products = set()
                company_product_list = OUR_COMPANY_DATA['Reported Product Name'].unique()
                for i in range(len(st.session_state.rows)):
                    entry = {
                        'Date': st.session_state[f'date_{i}'],
                        'Reported Product Name': st.session_state[f'product_name_{i}'],
                        'HS-CODE': st.session_state[f'hscode_{i}'],
                        'Origin Country': st.session_state[f'origin_{i}'].upper(),
                        'Exporter': st.session_state[f'exporter_{i}'].upper(),
                        'Volume': st.session_state[f'volume_{i}'],
                        'Value': st.session_state[f'value_{i}'],
                    }
                    if not all(entry.values()):
                        st.error(f"{i+1}번째 행의 모든 필드를 입력해주세요.")
                        return
                    matched = smart_match_products(entry['Reported Product Name'], company_product_list)
                    all_matched_products.update(matched)
                    purchase_data.append(entry)
                
                # Google Sheets 저장 로직 (Append-Only)
                try:
                    conn = st.connection("gsheets", type=GSheetsConnection)
                    save_data_df = pd.DataFrame(purchase_data)
                    save_data_df['importer_name'] = importer_name
                    save_data_df['consent'] = consent
                    save_data_df['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    
                    # 날짜 형식을 문자열로 변환
                    save_data_df['Date'] = save_data_df['Date'].dt.strftime('%Y-%m-%d')

                    worksheet = conn.get_worksheet(worksheet="Customer_input")
                    if worksheet is None: # 시트가 없는 경우
                        conn.create(worksheet="Customer_input", data=save_data_df)
                    else: # 시트가 있는 경우
                        # 헤더가 없으면 헤더를 추가하고, 있으면 데이터만 추가
                        header = worksheet.row_values(1)
                        if not header:
                             worksheet.update([save_data_df.columns.values.tolist()] + save_data_df.values.tolist())
                        else:
                             worksheet.append_rows(save_data_df.values.tolist(), value_input_option='USER_ENTERED')
                    st.toast("입력 정보가 Google Sheet에 저장되었습니다.", icon="✅")
                except Exception as e:
                    st.error(f"Google Sheets 저장 실패: {e}")

                st.session_state['user_input_df'] = pd.DataFrame(purchase_data)
                st.session_state['matched_products'] = sorted(list(all_matched_products))
                st.session_state['selected_products'] = st.session_state['matched_products']
                st.session_state['importer_name_result'] = importer_name
                st.session_state['analysis_results'] = True
                st.rerun()

    if 'analysis_results' in st.session_state:
        st.header("📊 분석 결과")
        with st.expander("STEP 2: 분석 대상 제품 필터링", expanded=True):
            selected = st.multiselect(
                "분석에 활용할 제품명을 선택하세요.",
                options=st.session_state['matched_products'],
                default=st.session_state['selected_products'],
                key="product_filter"
            )
            st.session_state['selected_products'] = selected

        if not st.session_state['selected_products']:
            st.warning("분석할 제품을 하나 이상 선택해주세요.")
        else:
            filtered_company_df = OUR_COMPANY_DATA[OUR_COMPANY_DATA['Reported Product Name'].isin(st.session_state['selected_products'])]
            target_df = pd.DataFrame(st.session_state['user_input_df'])
            target_df['Date'] = pd.to_datetime(target_df['Date'])
            target_df_filtered = target_df[target_df['Reported Product Name'].apply(lambda x: bool(smart_match_products(x, st.session_state['selected_products'])))]
            
            if target_df_filtered.empty:
                st.warning("선택된 제품과 매칭되는 사용자 입력이 없습니다.")
            else:
                competitor_res, yearly_res, timeseries_res = process_analysis_data(target_df_filtered, filtered_company_df, st.session_state['importer_name_result'])
                
                st.subheader("1. 경쟁사 Unit Price 비교 분석")
                if not competitor_res: st.write("비교할 경쟁사 데이터가 없습니다.")
                for (year, exporter), data in competitor_res.items():
                    with st.container(border=True):
                        st.markdown(f"**{year}년 / 수출업체: {exporter}**")
                        data['color'] = np.where(data['Importer'] == st.session_state['importer_name_result'].upper(), '#ef4444', '#3b82f6')
                        fig = px.bar(data, x='Importer', y='unitPrice', title=f"{year}년 {exporter} 수입사별 Unit Price", color='color', color_discrete_map={'#ef4444':'귀사', '#3b82f6':'경쟁사'})
                        fig.update_layout(showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)

                st.subheader("2. 연도별 수입 중량 및 Unit Price 트렌드")
                if not yearly_res: st.write("분석할 연도별 데이터가 없습니다.")
                for (exporter, origin), data in yearly_res.items():
                     with st.container(border=True):
                        st.markdown(f"**{exporter} 로부터의 {origin}산 품목 수입 트렌드**")
                        fig = go.Figure()
                        fig.add_trace(go.Bar(x=data['chart_data']['year'], y=data['chart_data']['volume'], name='수입 중량 (KG)', yaxis='y1'))
                        fig.add_trace(go.Line(x=data['chart_data']['year'], y=data['chart_data']['unitPrice'], name='Unit Price (USD/KG)', yaxis='y2', mode='lines+markers'))
                        fig.update_layout(yaxis=dict(title="수입 중량 (KG)"), yaxis2=dict(title="Unit Price (USD/KG)", overlaying='y', side='right'))
                        st.plotly_chart(fig, use_container_width=True)
                        if data['saving_info']: st.success(f"💰 데이터 기반 예상 절감 가능 금액: 약 ${data['saving_info']['potential_saving']:,.0f}")

                st.subheader("3. 시계열 Unit Price 비교")
                if not timeseries_res: st.write("분석할 시계열 데이터가 없습니다.")
                for origin, data in timeseries_res.items():
                    with st.container(border=True):
                        st.markdown(f"**{origin} 원산지 품목 Unit Price 트렌드**")
                        fig = px.line(data['chart_data'], x='monthYear', y=['avgPrice', 'targetPrice', 'bestPrice'], markers=True)
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
