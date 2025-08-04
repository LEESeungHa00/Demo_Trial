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
        # 'Data'는 회사 데이터가 있는 실제 시트 이름으로 변경해야 합니다.
        df = conn.read(worksheet="TDS") 
        df.dropna(how="all", inplace=True)
        df['Date'] = pd.to_datetime(df['Date'])
        df['Volume'] = pd.to_numeric(df['Volume'])
        df['Value'] = pd.to_numeric(df['Value'])
        return df
    except Exception as e:
        st.error(f"Google Sheets에서 데이터를 불러오는 중 오류가 발생했습니다: {e}")
        st.info("Streamlit Secrets 설정 및 Google Sheet 공유 설정을 확인해주세요. 자세한 내용은 가이드를 참고하세요.")
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

# --- 데이터 처리 로직 (이전과 동일) ---
def process_analysis_data(target_df, company_df, target_importer_name):
    if company_df.empty: return {}, {}, {}
    target_df['Importer'] = target_importer_name.upper()
    all_df = pd.concat([company_df, target_df], ignore_index=True)
    all_df['unitPrice'] = all_df['Value'] / all_df['Volume']
    all_df['year'] = all_df['Date'].dt.year
    all_df['monthYear'] = all_df['Date'].dt.to_period('M').astype(str)
    # 분석 로직은 생략 (이전과 동일)
    competitor_analysis, yearly_analysis, time_series_analysis = {}, {}, {} # Placeholder
    # ... (이전 버전의 전체 분석 로직이 여기에 들어갑니다) ...
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
            cols = st.columns([2, 3, 2, 2, 2, 2, 1])
            cols[0].date_input("수입일", key=f"date_{i}")
            cols[1].text_input("제품 상세명", placeholder="예: 발렌타인 17년", key=f"product_name_{i}")
            cols[2].text_input("HS-CODE(6자리)", max_chars=6, key=f"hscode_{i}")
            cols[3].selectbox("원산지", [''] + sorted(OUR_COMPANY_DATA['Export Country'].unique()), key=f"origin_{i}")
            cols[4].selectbox("수출업체", [''] + sorted(OUR_COMPANY_DATA['Exporter'].unique()), key=f"exporter_{i}")
            cols[5].number_input("수입 중량(KG)", min_value=0.01, format="%.2f", key=f"volume_{i}")
            if len(st.session_state.rows) > 1 and cols[6].button("삭제", key=f"delete_{i}"):
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
                # 1. 사용자 입력 데이터 수집
                purchase_data = []
                all_matched_products = set()
                company_product_list = OUR_COMPANY_DATA['Reported Product Name'].unique()
                for i in range(len(st.session_state.rows)):
                    user_product_name = st.session_state[f'product_name_{i}']
                    if not user_product_name:
                        st.error(f"{i+1}번째 행의 '제품 상세명'을 입력해주세요.")
                        return
                    matched = smart_match_products(user_product_name, company_product_list)
                    all_matched_products.update(matched)
                    purchase_data.append({
                        'Date': st.session_state[f'date_{i}'],
                        'Reported Product Name': user_product_name,
                        'HS-CODE': st.session_state[f'hscode_{i}'],
                        'Origin Country': st.session_state[f'origin_{i}'].upper(),
                        'Exporter': st.session_state[f'exporter_{i}'].upper(),
                        'Volume': st.session_state[f'volume_{i}'],
                    })
                
                # 2. Google Sheets에 저장
                try:
                    conn = st.connection("gsheets", type=GSheetsConnection)
                    worksheet_name = "Customer_input"
                    save_data_df = pd.DataFrame(purchase_data)
                    save_data_df['importer_name'] = importer_name
                    save_data_df['consent'] = consent
                    save_data_df['timestamp'] = datetime.now()
                    
                    existing_df = conn.read(worksheet=worksheet_name, usecols=list(range(save_data_df.shape[1])))
                    existing_df.dropna(how='all', inplace=True)
                    updated_df = pd.concat([existing_df, save_data_df], ignore_index=True)
                    conn.update(worksheet=worksheet_name, data=updated_df)
                    st.toast("입력 정보가 Google Sheet에 저장되었습니다.", icon="✅")
                except Exception as e:
                    st.error(f"Google Sheets 저장 실패: {e}")
                    st.info("서비스 계정에 '편집자' 권한이 있는지, 'secrets.toml' 설정과 탭 이름을 확인하세요.")

                # 3. 분석을 위한 세션 상태 저장
                st.session_state['user_input_df'] = pd.DataFrame(purchase_data)
                st.session_state['matched_products'] = sorted(list(all_matched_products))
                st.session_state['selected_products'] = st.session_state['matched_products']
                st.session_state['importer_name_result'] = importer_name
                st.session_state['analysis_results'] = True

    if 'analysis_results' in st.session_state:
        st.header("📊 분석 결과")
        with st.expander("STEP 2: 분석 대상 제품 필터링", expanded=True):
            st.info("스마트 매칭된 제품 목록입니다. 원치 않는 제품은 체크 해제하여 제외할 수 있습니다.")
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
            target_df = st.session_state['user_input_df']
            target_df_filtered = target_df[target_df['Reported Product Name'].apply(lambda x: bool(smart_match_products(x, st.session_state['selected_products'])))]
            
            if target_df_filtered.empty:
                st.warning("선택된 제품과 매칭되는 사용자 입력이 없습니다.")
            else:
                # 여기에 전체 분석 및 차트 표시 로직이 들어갑니다.
                st.success("분석 데이터가 준비되었습니다. (차트 표시 로직은 생략됨)")

        if st.button("🔄 새로운 분석 시작하기"):
            keys_to_keep = ['logged_in']
            for key in list(st.session_state.keys()):
                if key not in keys_to_keep:
                    del st.session_state[key]
            st.rerun()

# --- 메인 로직 ---
if 'logged_in' not in st.session_state: st.session_state['logged_in'] = False
if st.session_state['logged_in']: main_dashboard()
else: login_screen()

