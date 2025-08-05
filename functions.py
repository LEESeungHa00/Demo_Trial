import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
from datetime import datetime, timedelta
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
        df = df[df['volume'] > 0] # 0으로 나누는 오류 방지

        if df.empty:
            st.error("데이터 정제 후 유효한 데이터가 없습니다.")
            return None
        return df
    except Exception as e:
        st.error(f"데이터 로딩 중 심각한 오류가 발생했습니다:")
        st.exception(e)
        return None

# --- 분석 헬퍼 함수 ---
def clean_text(text):
    """분석을 위한 텍스트 정제 함수"""
    if not isinstance(text, str): return ''
    text = text.lower()
    text = re.sub(r'\(.*?\)|\[.*?\]', ' ', text)
    text = re.sub(r'(\d+)\s*(?:y|yo|year|years|old|년산|년)', r'\1', text)
    text = re.sub(r'[^a-z0-9\s\uac00-\ud7a3]', ' ', text)
    text = re.sub(r'\b산\b', ' ', text)
    return ' '.join(text.split())

def get_excel_col_name(n):
    """0-based index를 Excel 열 이름 (A, B, ..., Z, AA)으로 변환"""
    name = ""
    while n >= 0:
        name = chr(ord('A') + n % 26) + name
        n = n // 26 - 1
    return name

def create_calendar_heatmap(df, title):
    """Plotly를 사용하여 캘린더 히트맵 생성"""
    if df.empty:
        return None

    # 최근 1년 데이터만 사용
    end_date = datetime.now()
    start_date = end_date - pd.DateOffset(years=1)
    df_filtered = df[(df['date'] >= start_date) & (df['date'] <= end_date)].copy()
    
    if df_filtered.empty:
        return None

    daily_counts = df_filtered.set_index('date').resample('D').size().reset_index(name='counts')
    
    # 1년치 모든 날짜 생성
    all_days = pd.date_range(start_date, end_date, freq='D')
    daily_counts = daily_counts.set_index('date').reindex(all_days, fill_value=0).reset_index().rename(columns={'index':'date'})
    
    daily_counts['day_of_week'] = daily_counts['date'].dt.day_name()
    daily_counts['week_of_year'] = daily_counts['date'].dt.isocalendar().week
    daily_counts['month_abbr'] = daily_counts['date'].dt.strftime('%b')
    
    # 월-주 텍스트 생성
    daily_counts['text'] = daily_counts.apply(lambda row: f"<b>{row['date'].strftime('%Y-%m-%d')}</b><br>Count: {row['counts']}", axis=1)

    # Plotly 히트맵 생성
    fig = go.Figure(data=go.Heatmap(
        z=daily_counts['counts'],
        x=daily_counts['week_of_year'],
        y=daily_counts['day_of_week'],
        hovertext=daily_counts['text'],
        hoverinfo='text',
        colorscale='Greens',
        showscale=False
    ))

    fig.update_layout(
        title=title,
        yaxis=dict(
            categoryorder='array', 
            categoryarray=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        ),
        margin=dict(t=50, b=20, l=40, r=20),
        height=250,
        plot_bgcolor='white',
    )
    return fig

# --- 메인 분석 로직 ---
def run_all_analysis(user_input, company_data, target_importer_name):
    """모든 분석을 수행하고 결과를 딕셔너리로 반환"""
    analysis_result = {"overview": {}, "positioning": {}, "supply_chain": {}}

    if company_data.empty:
        st.warning("선택된 제품군에 해당하는 데이터가 없습니다.")
        return analysis_result

    company_data['unitPrice'] = company_data['value'] / company_data['volume']

    # --- 0. Overview 분석 ---
    hscode_data = company_data[company_data['hs_code'].astype(str) == str(user_input['HS-CODE'])]
    if not hscode_data.empty:
        this_year = datetime.now().year
        last_year = this_year - 1

        vol_this_year = hscode_data[hscode_data['date'].dt.year == this_year]['volume'].sum()
        vol_last_year = hscode_data[hscode_data['date'].dt.year == last_year]['volume'].sum()
        price_this_year = hscode_data[hscode_data['date'].dt.year == this_year]['unitPrice'].mean()
        price_last_year = hscode_data[hscode_data['date'].dt.year == last_year]['unitPrice'].mean()

        vol_yoy = (vol_this_year - vol_last_year) / vol_last_year if vol_last_year > 0 else np.nan
        price_yoy = (price_this_year - price_last_year) / price_last_year if price_last_year > 0 else np.nan

        importer_cycles = {}
        for importer in hscode_data['importer'].unique():
            importer_df = hscode_data[hscode_data['importer'] == importer].sort_values('date')
            if len(importer_df) > 1:
                avg_cycle = importer_df['date'].diff().mean().days
                importer_cycles[importer] = avg_cycle
        
        analysis_result['overview'] = {
            "vol_this_year": vol_this_year, "vol_last_year": vol_last_year, "vol_yoy": vol_yoy,
            "price_this_year": price_this_year, "price_last_year": price_last_year, "price_yoy": price_yoy,
            "freq_this_year": len(hscode_data[hscode_data['date'].dt.year == this_year]),
            "importer_cycles": importer_cycles,
            "product_composition": company_data.groupby('reported_product_name')['value'].sum().nlargest(10).reset_index()
        }

    # --- 1. 포지셔닝 분석 ---
    importer_stats = company_data.groupby('importer').agg(
        Total_Value=('value', 'sum'),
        Total_Volume=('volume', 'sum'),
        Trade_Count=('value', 'count'),
        Avg_UnitPrice=('unitPrice', 'mean')
    ).reset_index()

    if not importer_stats.empty and importer_stats['Total_Volume'].sum() > 0:
        importer_stats = importer_stats.sort_values('Total_Value', ascending=False).reset_index(drop=True)

        # 그룹 분류
        total_market_value = importer_stats['Total_Value'].sum()
        importer_stats['cum_share'] = importer_stats['Total_Value'].cumsum() / total_market_value
        market_leaders_df = importer_stats[importer_stats['cum_share'] <= 0.7]

        try:
            target_rank = importer_stats[importer_stats['importer'] == target_importer_name].index[0]
            rank_margin = max(1, int(len(importer_stats) * 0.1)) # 최소 1개는 포함
            peer_min_rank, peer_max_rank = max(0, target_rank - rank_margin), min(len(importer_stats), target_rank + rank_margin + 1)
            direct_peers_df = importer_stats.iloc[peer_min_rank:peer_max_rank]
        except IndexError:
            target_rank = -1
            direct_peers_df = pd.DataFrame()

        price_achievers_candidates = importer_stats[importer_stats['Trade_Count'] >= 2]
        if not price_achievers_candidates.empty:
            price_quantile = price_achievers_candidates['Avg_UnitPrice'].quantile(0.15)
            price_achievers_df = price_achievers_candidates[price_achievers_candidates['Avg_UnitPrice'] <= price_quantile]
        else:
            price_achievers_df = pd.DataFrame()

        analysis_result['positioning'] = {
            "bubble_data": importer_stats,
            "groups": {
                "Market Leaders": market_leaders_df,
                "Direct Peers": direct_peers_df,
                "Price Achievers": price_achievers_df
            },
            "target_stats": importer_stats[importer_stats['importer'] == target_importer_name]
        }

    # --- 2. 공급망 분석 ---
    target_exporter = user_input.get('Exporter', '').upper()
    target_country = user_input.get('Origin Country', '').upper()

    # 동일 수출업체 거래 경쟁사 분석
    if target_exporter:
        same_exporter_df = company_data[company_data['exporter'] == target_exporter]
        same_exporter_stats = same_exporter_df.groupby('importer').agg(
            Total_Volume=('volume', 'sum'), Avg_UnitPrice=('unitPrice', 'mean')
        ).reset_index()
        analysis_result['supply_chain']['same_exporter_stats'] = same_exporter_stats
    
    # 더 저렴한 수출업체 분석
    if target_exporter:
        target_exporter_price = company_data[company_data['exporter'] == target_exporter]['unitPrice'].mean()
        if not np.isnan(target_exporter_price):
            cheaper_exporters = company_data[
                (company_data['exporter'] != target_exporter) &
                (company_data['unitPrice'] < target_exporter_price)
            ].groupby('exporter').agg(Avg_UnitPrice=('unitPrice', 'mean')).reset_index()
            
            if not cheaper_exporters.empty:
                best_exporter = cheaper_exporters.sort_values('Avg_UnitPrice').iloc[0]
                analysis_result['supply_chain']['best_exporter'] = {
                    'name': best_exporter['exporter'],
                    'price': best_exporter['Avg_UnitPrice'],
                    'saving_rate': (target_exporter_price - best_exporter['Avg_UnitPrice']) / target_exporter_price
                }

    # 더 저렴한 원산지 분석
    if target_country:
        target_country_price = company_data[company_data['export_country'] == target_country]['unitPrice'].mean()
        if not np.isnan(target_country_price):
            cheaper_countries = company_data[
                (company_data['export_country'] != target_country) &
                (company_data['unitPrice'] < target_country_price)
            ].groupby('export_country').agg(Avg_UnitPrice=('unitPrice', 'mean')).reset_index()

            if not cheaper_countries.empty:
                best_country = cheaper_countries.sort_values('Avg_UnitPrice').iloc[0]
                analysis_result['supply_chain']['best_country'] = {
                    'name': best_country['export_country'],
                    'price': best_country['Avg_UnitPrice'],
                    'saving_rate': (target_country_price - best_country['Avg_UnitPrice']) / target_country_price
                }

    return analysis_result


# --- UI Components ---
def login_screen():
    st.title("🔐 수입 경쟁력 진단 솔루션")
    st.write("솔루션 접속을 위해 비밀번호를 입력해주세요.")
    with st.form("login_form", clear_on_submit=True):
        password = st.text_input("비밀번호", type="password")
        submitted = st.form_submit_button("접속하기")
        if submitted:
            if password == st.secrets.get("APP_PASSWORD", "tridgeDemo_2025"): # Secrets 활용 권장
                st.session_state['logged_in'] = True
                st.rerun()
            else:
                st.error("비밀번호가 올바르지 않습니다.")

def main_dashboard(company_data):
    st.title("📈 수입 경쟁력 진단 솔루션")
    st.markdown("트릿지 데이터를 기반으로 시장 내 경쟁력을 진단하고 비용 절감 기회를 포착하세요.")

    # STEP 1: 입력 폼
    with st.expander("STEP 1: 분석 정보 입력", expanded='analysis_groups' not in st.session_state):
        importer_name = st.text_input("1. 귀사의 업체명을 입력해주세요.", key="importer_name").upper()
        if 'rows' not in st.session_state: st.session_state['rows'] = [{'id': 1}]
        
        for i, row in enumerate(st.session_state.rows):
            st.markdown(f"**수입 내역 {i+1}**")
            cols = st.columns([3, 1, 2, 2, 1, 1, 1])
            cols[0].text_input("제품 상세명", placeholder="예 : 엑스트라버진 올리브유", key=f"product_name_{i}")
            cols[1].text_input("HS-CODE(6자리)", max_chars=6, key=f"hscode_{i}")
            
            origin_options = [''] + ['직접 입력'] + sorted(company_data['export_country'].unique())
            selected_origin = cols[2].selectbox("원산지", origin_options, key=f"origin_{i}", format_func=lambda x: '선택 또는 직접 입력' if x == '' else x)
            if selected_origin == '직접 입력':
                st.session_state[f'final_origin_{i}'] = cols[2].text_input("└ 원산지 직접 입력", key=f"custom_origin_{i}", placeholder="직접 입력하세요")
            else:
                st.session_state[f'final_origin_{i}'] = selected_origin

            exporter_options = [''] + ['직접 입력'] + sorted(company_data['exporter'].unique())
            selected_exporter = cols[3].selectbox("수출업체", exporter_options, key=f"exporter_{i}", format_func=lambda x: '선택 또는 직접 입력' if x == '' else x)
            if selected_exporter == '직접 입력':
                st.session_state[f'final_exporter_{i}'] = cols[3].text_input("└ 수출업체 직접 입력", key=f"custom_exporter_{i}", placeholder="직접 입력하세요")
            else:
                st.session_state[f'final_exporter_{i}'] = selected_exporter

            cols[4].number_input("수입 중량(KG)", min_value=0.01, format="%.2f", key=f"volume_{i}")
            cols[5].number_input("총 수입금액(USD)", min_value=0.01, format="%.2f", key=f"value_{i}")
            if len(st.session_state.rows) > 1 and cols[6].button("삭제", key=f"delete_{i}"):
                st.session_state.rows.pop(i)
                st.rerun()

        col1, col2 = st.columns([1, 6])
        if col1.button("➕ 내역 추가하기"):
            st.session_state.rows.append({'id': len(st.session_state.rows) + 1})
            st.rerun()
            
        st.markdown("---")
        consent = st.checkbox("입력하신 정보는 데이터 분석 품질 향상을 위해 저장 및 활용되는 것에 동의합니다.", value=True)
        analyze_button = st.button("분석하기", type="primary", use_container_width=True)

    if analyze_button:
        # 입력값 검증 및 데이터 처리
        if not importer_name: st.warning("수입업체명을 입력해주세요.")
        elif not consent: st.warning("데이터 활용 동의에 체크해주세요.")
        else:
            with st.spinner('데이터를 분석하고 있습니다...'):
                all_purchase_data = []
                for i in range(len(st.session_state.rows)):
                    user_product_name = st.session_state[f'product_name_{i}']
                    origin_val = st.session_state[f'final_origin_{i}']
                    exporter_val = st.session_state[f'final_exporter_{i}']
                    entry = { 'Reported Product Name': user_product_name, 'HS-CODE': st.session_state[f'hscode_{i}'], 'Origin Country': origin_val.upper(), 'Exporter': exporter_val.upper(), 'Volume': st.session_state[f'volume_{i}'], 'Value': st.session_state[f'value_{i}'] }
                    if not all([user_product_name, origin_val, exporter_val, entry['HS-CODE']]):
                        st.error(f"{i+1}번째 행의 '제품 상세명', 'HS-CODE', '원산지', '수출업체'는 필수 입력 항목입니다.")
                        return
                    all_purchase_data.append(entry)
                
                # 제품명 기준으로 데이터 집계
                purchase_df = pd.DataFrame(all_purchase_data)
                agg_funcs = {'Volume': 'sum', 'Value': 'sum', 'HS-CODE': 'first', 'Origin Country': 'first', 'Exporter': 'first'}
                aggregated_purchase_df = purchase_df.groupby('Reported Product Name').agg(agg_funcs).reset_index()

                # 분석 그룹 생성
                analysis_groups = []
                company_data['cleaned_name'] = company_data['reported_product_name'].apply(clean_text)
                for i, row in aggregated_purchase_df.iterrows():
                    entry = row.to_dict()
                    user_tokens = set(clean_text(entry['Reported Product Name']).split())
                    def is_match(cleaned_tds_name): return user_tokens.issubset(set(cleaned_tds_name.split()))
                    matched_df = company_data[company_data['cleaned_name'].apply(is_match)]
                    analysis_groups.append({ "id": i, "user_input": entry, "matched_products": sorted(matched_df['reported_product_name'].unique().tolist()), "selected_products": sorted(matched_df['reported_product_name'].unique().tolist()) })
                
                # 세션 상태에 저장 및 재실행
                st.session_state['importer_name_result'] = importer_name
                st.session_state['analysis_groups'] = analysis_groups
                st.rerun()

    if 'analysis_groups' in st.session_state:
        st.header("📊 분석 결과")
        
        # STEP 2: 필터링
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

        # 각 그룹별 분석 결과 출력
        for group in st.session_state.analysis_groups:
            st.subheader(f"분석 결과: \"{group['user_input']['Reported Product Name']}\"")
            
            if not group['selected_products']:
                st.warning("선택된 비교 대상 제품이 없어 분석을 건너뜁니다.")
                continue

            analysis_data = company_data[company_data['reported_product_name'].isin(group['selected_products'])]
            result = run_all_analysis(group['user_input'], analysis_data, st.session_state['importer_name_result'])
            target_importer_name = st.session_state['importer_name_result']

            # --- 0. Overview 표시 ---
            st.markdown("### 0. Overview")
            if result.get('overview'):
                o = result['overview']
                hscode = group['user_input']['HS-CODE']
                st.markdown(f"#### HS-Code {hscode}의 수입 전반 요약")
                
                cols = st.columns(3)
                cols[0].metric("금년 수입 중량 (KG)", f"{o['vol_this_year']:,.0f}", f"{o['vol_yoy']:.1%}" if not np.isnan(o['vol_yoy']) else "N/A", delta_color="normal")
                cols[1].metric("금년 평균 단가 (USD/KG)", f"${o['price_this_year']:.2f}", f"{o['price_yoy']:.1%}" if not np.isnan(o['price_yoy']) else "N/A", delta_color="inverse")
                cols[2].metric("금년 수입 빈도", f"{o['freq_this_year']} 건")

                with st.expander("상세 분석 보기"):
                    c1, c2 = st.columns(2)
                    # 제품 구성 파이 차트
                    if not o['product_composition'].empty:
                        fig_pie = px.pie(o['product_composition'], values='value', names='reported_product_name', title=f'HS-Code {hscode} 주요 제품 구성 (수입금액 기준)')
                        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                        fig_pie.update_layout(height=400)
                        c1.plotly_chart(fig_pie, use_container_width=True)
                    else:
                        c1.info("제품 구성 데이터를 표시할 수 없습니다.")
                    
                    # 수입 주기
                    cycle_df = pd.DataFrame(o['importer_cycles'].items(), columns=['Importer', 'Avg Cycle (days)']).sort_values('Avg Cycle (days)').reset_index(drop=True)
                    c2.markdown("**주요 수입사별 평균 수입 주기**")
                    c2.dataframe(cycle_df, use_container_width=True, hide_index=True)

            else:
                st.info("HS-Code에 해당하는 데이터가 부족하여 Overview 분석을 생략합니다.")

            # --- 1. Positioning 표시 ---
            st.markdown(f"### 1. {target_importer_name}을 위한 수입 진단 및 포지셔닝 결과")
            if result.get('positioning') and not result['positioning']['bubble_data'].empty:
                p = result['positioning']
                
                # 익명화 맵 생성
                all_importers = p['bubble_data']['importer'].unique()
                anonymity_map = {name: f"{get_excel_col_name(i)}사" for i, name in enumerate(all_importers) if name != target_importer_name}
                anonymity_map[target_importer_name] = "귀사"

                st.markdown("#### PART 1. 마켓 포지션 분석")
                
                # 버블 차트
                bubble_df = p['bubble_data'].copy()
                bubble_df['Anonymized_Importer'] = bubble_df['importer'].apply(lambda x: anonymity_map.get(x, "기타"))
                fig_bubble = px.scatter(bubble_df, x='Total_Volume', y='Avg_UnitPrice', size='Total_Value', color='Anonymized_Importer',
                                        hover_name='Anonymized_Importer', size_max=60, log_x=True,
                                        labels={'Total_Volume': '수입 총 중량 (KG, log scale)', 'Avg_UnitPrice': '평균 수입 단가 (USD/KG)', 'Total_Value': '총 수입 금액 (USD)'},
                                        title="수입사 포지셔닝 맵")
                st.plotly_chart(fig_bubble, use_container_width=True)

                # 요약 메트릭
                target_stats = p['target_stats']
                if not target_stats.empty:
                    market_avg_price = p['bubble_data']['Avg_UnitPrice'].mean()
                    target_price = target_stats['Avg_UnitPrice'].iloc[0]
                    top_10_percent_price = p['bubble_data']['Avg_UnitPrice'].quantile(0.1)
                    
                    delta_val = (target_price - market_avg_price) / market_avg_price if market_avg_price > 0 else 0
                    
                    m_cols = st.columns(3)
                    m_cols[0].metric("귀사 평균단가", f"{target_price:.2f} USD/KG")
                    m_cols[1].metric("시장 평균 단가 대비", f"{market_avg_price:.2f} USD/KG", f"{delta_val:.1%}", delta_color="inverse")
                    m_cols[2].metric("가격 선도그룹(상위10%) 평균단가", f"{top_10_percent_price:.2f} USD/KG")
                
                st.markdown("---")
                # 캘린더 히트맵
                cal_cols = st.columns([1, 20])
                with cal_cols[0]:
                    with st.popover("ℹ️"):
                        st.markdown("""
                        **그룹 분류 기준:**
                        - **시장 선도 그룹**: 수입금액 기준 누적 70%를 차지하는 상위 기업
                        - **유사 규모 경쟁 그룹**: 귀사 순위 기준 상하 ±10% 범위의 기업
                        - **최저가 달성 그룹**: 평균 수입 단가 하위 15% 이내 기업 (최소 2회 이상 수입)
                        """)
                cal_cols[1].markdown("##### 그룹별 수입 활동 꾸준함 분석 (지난 1년)")

                target_df = analysis_data[analysis_data['importer'] == target_importer_name]
                groups_data = {name: analysis_data[analysis_data['importer'].isin(df['importer'])] for name, df in p['groups'].items()}

                # 2x2 그리드 출력
                c1, c2 = st.columns(2)
                with c1:
                    fig_target = create_calendar_heatmap(target_df, f"귀사 ({len(target_df)} 건)")
                    if fig_target: st.plotly_chart(fig_target, use_container_width=True)
                    else: st.info("귀사의 지난 1년간 데이터가 없습니다.")

                    fig_peers = create_calendar_heatmap(groups_data['Direct Peers'], f"유사 규모 경쟁 그룹 ({len(p['groups']['Direct Peers'])}개사)")
                    if fig_peers: st.plotly_chart(fig_peers, use_container_width=True)
                    else: st.info("유사 규모 경쟁 그룹의 데이터가 없습니다.")
                with c2:
                    fig_leaders = create_calendar_heatmap(groups_data['Market Leaders'], f"시장 선도 그룹 ({len(p['groups']['Market Leaders'])}개사)")
                    if fig_leaders: st.plotly_chart(fig_leaders, use_container_width=True)
                    else: st.info("시장 선도 그룹의 데이터가 없습니다.")

                    fig_achievers = create_calendar_heatmap(groups_data['Price Achievers'], f"최저가 달성 그룹 ({len(p['groups']['Price Achievers'])}개사)")
                    if fig_achievers: st.plotly_chart(fig_achievers, use_container_width=True)
                    else: st.info("최저가 달성 그룹의 데이터가 없습니다.")

                # 박스플롯
                st.markdown("##### 그룹별 수입 단가 분포 비교")
                fig_box = go.Figure()
                for name, df in groups_data.items():
                    if not df.empty:
                        fig_box.add_trace(go.Box(y=df['unitPrice'], name=name))
                
                if not target_stats.empty:
                    fig_box.add_hline(y=target_stats['Avg_UnitPrice'].iloc[0], line_dash="dot",
                                      annotation_text="귀사 평균단가", annotation_position="bottom right")

                fig_box.update_layout(yaxis_title="수입 단가 (USD/KG)", title="그룹별 단가 분포와 귀사 위치", plot_bgcolor='white')
                st.plotly_chart(fig_box, use_container_width=True)

                # --- 2. 공급망 분석 표시 ---
                st.markdown(f"#### PART 2. {target_importer_name}의 공급망 분석")
                s = result.get('supply_chain', {})

                # 동일 수출업체 거래 경쟁사 분석
                if 'same_exporter_stats' in s and not s['same_exporter_stats'].empty:
                    st.markdown(f"##### **{group['user_input']['Exporter']}** 거래 경쟁사 비교")
                    df_plot = s['same_exporter_stats']
                    
                    target_row = df_plot[df_plot['importer'] == target_importer_name]
                    others_avg_price = df_plot[df_plot['importer'] != target_importer_name]['Avg_UnitPrice'].mean()

                    if not target_row.empty:
                        sc_cols = st.columns(3)
                        target_price = target_row['Avg_UnitPrice'].iloc[0]
                        price_diff = (target_price - others_avg_price) / others_avg_price if others_avg_price > 0 else 0
                        sc_cols[0].metric("귀사 평균단가", f"${target_price:.2f}")
                        sc_cols[1].metric("타사 평균단가", f"${others_avg_price:.2f}")
                        sc_cols[2].metric("타사 대비", f"{price_diff:.1%}", delta_color="inverse")

                    fig_bar_line = make_subplots(specs=[[{"secondary_y": True}]])
                    fig_bar_line.add_trace(go.Bar(x=df_plot['importer'], y=df_plot['Total_Volume'], name='총 수입 중량'), secondary_y=False)
                    fig_bar_line.add_trace(go.Scatter(x=df_plot['importer'], y=df_plot['Avg_UnitPrice'], name='평균 단가'), secondary_y=True)
                    fig_bar_line.update_layout(title_text='동일 수출업체 거래사별 수입량 및 단가', plot_bgcolor='white')
                    fig_bar_line.update_yaxes(title_text="총 수입 중량 (KG)", secondary_y=False)
                    fig_bar_line.update_yaxes(title_text="평균 단가 (USD/KG)", secondary_y=True)
                    st.plotly_chart(fig_bar_line, use_container_width=True)

                # 비용 절감 기회
                st.markdown("##### 비용 절감 기회 분석")
                if 'best_exporter' in s:
                    be = s['best_exporter']
                    st.success(f"**수출업체 변경**: 현재 거래처보다 저렴한 **{be['name']}**와(과) 거래 시, 평균 단가를 최대 **{be['saving_rate']:.1%}**까지 절감할 수 있습니다.")
                    max_saving_pct = be['saving_rate'] * 100
                    
                    selected_pct = st.slider("만약 단가를 이만큼 절감한다면?", 0.0, max_saving_pct, float(min(10.0, max_saving_pct)), format="%.1f%%", key=f"slider_exporter_{group['id']}")
                    user_total_value = group['user_input']['Value']
                    potential_saving = user_total_value * (selected_pct / 100)
                    st.info(f"👉 **{selected_pct:.1f}%** 절감 시, **${potential_saving:,.2f} USD**의 비용을 아낄 수 있습니다. (귀사 수입금액 기준)")


                if 'best_country' in s:
                    bc = s['best_country']
                    st.success(f"**원산지 변경**: **{bc['name']}**에서 수입할 경우, 평균 단가를 최대 **{bc['saving_rate']:.1%}**까지 절감할 수 있는 기회가 있습니다.")
                    max_saving_pct_ct = bc['saving_rate'] * 100
                    
                    selected_pct_ct = st.slider("만약 원산지 변경으로 단가를 이만큼 절감한다면?", 0.0, max_saving_pct_ct, float(min(10.0, max_saving_pct_ct)), format="%.1f%%", key=f"slider_country_{group['id']}")
                    user_total_value = group['user_input']['Value']
                    potential_saving_ct = user_total_value * (selected_pct_ct / 100)
                    st.info(f"👉 **{selected_pct_ct:.1f}%** 절감 시, **${potential_saving_ct:,.2f} USD**의 비용을 아낄 수 있습니다. (귀사 수입금액 기준)")

                if 'best_exporter' not in s and 'best_country' not in s:
                    st.info("현재 분석된 데이터 내에서는 더 저렴한 공급망(수출업체/원산지)을 찾지 못했습니다.")

            else:
                st.info("데이터가 부족하여 포지셔닝 및 공급망 분석을 생략합니다.")
            st.markdown("---") # 그룹별 구분선

        if st.button("🔄 새로운 분석 시작하기", use_container_width=True):
            keys_to_keep = ['logged_in']
            for key in list(st.session_state.keys()):
                if key not in keys_to_keep:
                    del st.session_state[key]
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
