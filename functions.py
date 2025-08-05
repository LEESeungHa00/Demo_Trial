import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
from datetime import datetime
from dateutil.relativedelta import relativedelta
import gspread
from google.oauth2.service_account import Credentials
from pandas_gbq import read_gbq
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
        table_full_id = f"{project_id}.demo_data.tds_data"
        df = read_gbq(f"SELECT * FROM `{table_full_id}`", project_id=project_id, credentials=creds, location="asia-northeast3")
        
        if df.empty:
            st.error("BigQuery 테이블에서 데이터를 불러왔지만 비어있습니다.")
            return None
            
        df.columns = [re.sub(r'[^a-z0-9]+', '_', col.lower().strip()) for col in df.columns]
        
        required_cols = ['date', 'volume', 'value', 'reported_product_name', 'export_country', 'exporter', 'importer', 'hs_code']
        if not all(col in df.columns for col in required_cols):
            st.error(f"BigQuery 테이블에 필수 컬럼이 부족합니다. (필수: {required_cols})")
            return None
            
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        for col in ['volume', 'value']:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(r'[^\d.]', '', regex=True), errors='coerce')
            
        df.dropna(subset=['date', 'volume', 'value'], inplace=True)
        df = df[df['volume'] > 0]
        
        return df if not df.empty else None
    except Exception as e:
        st.error(f"데이터 로딩 중 심각한 오류가 발생했습니다: {e}")
        return None

# --- 분석 헬퍼 함수 ---
def clean_text(text):
    if not isinstance(text, str): return ''
    return ' '.join(re.sub(r'[^a-z0-9\s\uac00-\ud7a3]', ' ', text.lower()).split())

def get_excel_col_name(n):
    name = ""
    while n >= 0:
        name = chr(ord('A') + n % 26) + name
        n = n // 26 - 1
    return name

def create_monthly_frequency_bar_chart(df, title):
    if df is None or df.empty: return None
    end_date = datetime.now()
    start_date = end_date - relativedelta(years=1)
    df_filtered = df[(df['date'] >= start_date) & (df['date'] <= end_date)].copy()
    if df_filtered.empty: return None
    df_filtered['YearMonth'] = df_filtered['date'].dt.strftime('%Y-%m')
    monthly_counts = df_filtered.groupby('YearMonth').size().reset_index(name='counts')
    all_months_range = pd.date_range(start=start_date.replace(day=1), end=end_date, freq='MS')
    all_months_df = pd.DataFrame({'YearMonth': all_months_range.strftime('%Y-%m')})
    monthly_counts = pd.merge(all_months_df, monthly_counts, on='YearMonth', how='left').fillna(0)
    fig = px.bar(monthly_counts, x='YearMonth', y='counts', title=title, labels={'YearMonth': '월', 'counts': '수입 건수'})
    fig.update_layout(margin=dict(t=50, b=20, l=40, r=20), height=300, plot_bgcolor='white')
    return fig

# --- 메인 분석 로직 ---
def run_all_analysis(user_input, full_company_data, selected_products, target_importer_name):
    analysis_result = {"overview": {}, "positioning": {}, "supply_chain": {}}
    hscode_data = full_company_data[full_company_data['hs_code'].astype(str) == str(user_input['HS-CODE'])].copy()
    if not hscode_data.empty:
        this_year = datetime.now().year
        hscode_data.loc[:, 'unitPrice'] = hscode_data['value'] / hscode_data['volume']
        vol_this_year = hscode_data[hscode_data['date'].dt.year == this_year]['volume'].sum()
        vol_last_year = hscode_data[hscode_data['date'].dt.year == this_year - 1]['volume'].sum()
        price_this_year = hscode_data[hscode_data['date'].dt.year == this_year]['unitPrice'].mean()
        price_last_year = hscode_data[hscode_data['date'].dt.year == this_year - 1]['unitPrice'].mean()
        vol_yoy = (vol_this_year - vol_last_year) / vol_last_year if vol_last_year > 0 else np.nan
        price_yoy = (price_this_year - price_last_year) / price_last_year if price_last_year > 0 else np.nan
        all_cycles = [cycle.days for importer in hscode_data['importer'].unique() if len(df := hscode_data[hscode_data['importer'] == importer]) > 1 for cycle in df.sort_values('date')['date'].diff().dropna()]
        analysis_result['overview'] = {
            "this_year": this_year, "vol_this_year": vol_this_year, "vol_yoy": vol_yoy,
            "price_this_year": price_this_year, "price_yoy": price_yoy,
            "avg_total_cycle": np.mean(all_cycles) if all_cycles else np.nan,
            "product_composition": hscode_data.groupby('reported_product_name')['value'].sum().nlargest(10).reset_index()
        }
    analysis_data = full_company_data[full_company_data['reported_product_name'].isin(selected_products)].copy()
    if not analysis_data.empty:
        analysis_data.loc[:, 'unitPrice'] = analysis_data['value'] / analysis_data['volume']
        importer_stats = analysis_data.groupby('importer').agg(Total_Value=('value', 'sum'), Total_Volume=('volume', 'sum'), Trade_Count=('value', 'count'), Avg_UnitPrice=('unitPrice', 'mean')).reset_index()
        if not importer_stats.empty and importer_stats['Total_Volume'].sum() > 0:
            importer_stats = importer_stats.sort_values('Total_Value', ascending=False).reset_index(drop=True)
            importer_stats['cum_share'] = importer_stats['Total_Value'].cumsum() / importer_stats['Total_Value'].sum()
            market_leaders_df = importer_stats[importer_stats['cum_share'] <= 0.7]
            try:
                target_rank = importer_stats[importer_stats['importer'] == target_importer_name].index[0]
                rank_margin = max(1, int(len(importer_stats) * 0.1))
                direct_peers_df = importer_stats.iloc[max(0, target_rank - rank_margin):min(len(importer_stats), target_rank + rank_margin + 1)]
            except IndexError: direct_peers_df = pd.DataFrame()
            price_achievers_candidates = importer_stats[importer_stats['Trade_Count'] >= 2]
            price_achievers_df = price_achievers_candidates[price_achievers_candidates['Avg_UnitPrice'] <= price_achievers_candidates['Avg_UnitPrice'].quantile(0.15)] if not price_achievers_candidates.empty else pd.DataFrame()
            analysis_result['positioning'] = {"bubble_data": importer_stats, "groups": {"Market Leaders": market_leaders_df, "Direct Peers": direct_peers_df, "Price Achievers": price_achievers_df}, "target_stats": importer_stats[importer_stats['importer'] == target_importer_name]}
        target_exporter, target_country = user_input.get('Exporter', '').upper(), user_input.get('Origin Country', '').upper()
        if target_exporter:
            same_exporter_df = analysis_data[analysis_data['exporter'] == target_exporter]
            analysis_result['supply_chain']['same_exporter_stats'] = same_exporter_df.groupby('importer').agg(Total_Volume=('volume', 'sum'), Avg_UnitPrice=('unitPrice', 'mean')).reset_index()
            target_price = same_exporter_df['unitPrice'].mean()
            if not np.isnan(target_price):
                cheaper = analysis_data[(analysis_data['exporter'] != target_exporter) & (analysis_data['unitPrice'] < target_price)].groupby('exporter').agg(Avg_UnitPrice=('unitPrice', 'mean')).nsmallest(1, 'Avg_UnitPrice')
                if not cheaper.empty: analysis_result['supply_chain']['best_exporter'] = {'name': cheaper.index[0], 'saving_rate': (target_price - cheaper['Avg_UnitPrice'].iloc[0]) / target_price}
    return analysis_result

# --- UI Components ---
def login_screen():
    st.title("🔐 수입 경쟁력 진단 솔루션")
    st.write("솔루션 접속을 위해 비밀번호를 입력해주세요.")
    with st.form("login_form", clear_on_submit=True):
        password = st.text_input("비밀번호", type="password")
        if st.form_submit_button("접속하기"):
            if password == st.secrets.get("APP_PASSWORD", "tridgeDemo_2025"):
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
            st.markdown(f"**수입 내역 {i+1}**")
            cols = st.columns([1.5, 3, 1, 2, 2, 1, 1, 1])
            cols[0].date_input(f"수입일_{i+1}", key=f"date_{i}", label_visibility="collapsed", value=datetime(2025, 8, 5))
            cols[1].text_input(f"제품상세명_{i+1}", placeholder="제품 상세명", key=f"product_name_{i}", label_visibility="collapsed")
            cols[2].text_input(f"HSCODE_{i+1}", max_chars=6, key=f"hscode_{i}", placeholder="HS-CODE", label_visibility="collapsed")
            origin_options = [''] + ['직접 입력'] + sorted(company_data['export_country'].unique())
            selected_origin = cols[3].selectbox(f"원산지_{i+1}", origin_options, key=f"origin_{i}", label_visibility="collapsed", format_func=lambda x: '원산지 선택' if x == '' else x)
            if selected_origin == '직접 입력': st.session_state[f'final_origin_{i}'] = cols[3].text_input(f"원산지직접_{i+1}", key=f"custom_origin_{i}", label_visibility="collapsed", placeholder="원산지 직접 입력")
            else: st.session_state[f'final_origin_{i}'] = selected_origin
            exporter_options = [''] + ['직접 입력'] + sorted(company_data['exporter'].unique())
            selected_exporter = cols[4].selectbox(f"수출업체_{i+1}", exporter_options, key=f"exporter_{i}", label_visibility="collapsed", format_func=lambda x: '수출업체 선택' if x == '' else x)
            if selected_exporter == '직접 입력': st.session_state[f'final_exporter_{i}'] = cols[4].text_input(f"수출업체직접_{i+1}", key=f"custom_exporter_{i}", label_visibility="collapsed", placeholder="수출업체 직접 입력")
            else: st.session_state[f'final_exporter_{i}'] = selected_exporter
            cols[5].number_input(f"수입중량_{i+1}", min_value=0.01, format="%.2f", key=f"volume_{i}", label_visibility="collapsed", placeholder="수입 중량(KG)")
            cols[6].number_input(f"수입금액_{i+1}", min_value=0.01, format="%.2f", key=f"value_{i}", label_visibility="collapsed", placeholder="총 수입금액(USD)")
            if len(st.session_state.rows) > 1 and cols[7].button("삭제", key=f"delete_{i}"):
                st.session_state.rows.pop(i); st.rerun()
        if st.button("➕ 내역 추가하기"):
            st.session_state.rows.append({'id': len(st.session_state.rows) + 1}); st.rerun()
        st.markdown("---")
        if st.button("분석하기", type="primary", use_container_width=True):
            with st.spinner('데이터를 분석하고 있습니다...'):
                # ... (입력값 검증 및 그룹 생성 로직) ...
                st.rerun()

    if 'analysis_groups' in st.session_state:
        st.header("1. HS-Code 시장 개요")
        for group in st.session_state.analysis_groups:
            result_overview = run_all_analysis(group['user_input'], company_data, [], st.session_state.get('importer_name_result', ''))
            st.markdown(f"#### HS-Code: {group['user_input']['HS-CODE']}")
            if 'overview' in result_overview and result_overview['overview']:
                o = result_overview['overview']
                cols = st.columns(3)
                cols[0].metric(f"{o['this_year']}년 수입 중량 (KG)", f"{o['vol_this_year']:,.0f}", f"{o['vol_yoy']:.1%}" if pd.notna(o['vol_yoy']) else "N/A", delta_color="inverse")
                cols[1].metric(f"{o['this_year']}년 평균 단가 (USD/KG)", f"${o['price_this_year']:.2f}", f"{o['price_yoy']:.1%}" if pd.notna(o['price_yoy']) else "N/A", delta_color="inverse")
                cols[2].metric("평균 수입 주기", f"{o['avg_total_cycle']:.1f} 일" if pd.notna(o['avg_total_cycle']) else "N/A", help="해당 HS-Code를 수입하는 모든 업체의 평균적인 거래 간격입니다.")
            else: st.info("해당 HS-Code에 대한 데이터가 부족하여 Overview 분석을 생략합니다.")
            st.markdown("---")

        with st.expander("STEP 2: 상세 분석을 위한 제품 필터링", expanded=True):
            for i, group in enumerate(st.session_state.analysis_groups):
                st.markdown(f"**분석 그룹: \"{group['user_input']['Reported Product Name']}\"**")
                selected = st.multiselect("분석에 활용할 제품명 선택:", options=group['matched_products'], default=group['selected_products'], key=f"filter_{group['id']}")
                st.session_state.analysis_groups[i]['selected_products'] = selected

        st.header("2. 제품별 상세 경쟁 분석")
        for group in st.session_state.analysis_groups:
            st.subheader(f"분석 결과: \"{group['user_input']['Reported Product Name']}\"")
            if not group['selected_products']: st.warning("선택된 제품이 없어 상세 분석을 건너뜁니다."); continue
            
            result = run_all_analysis(group['user_input'], company_data, group['selected_products'], st.session_state.get('importer_name_result', ''))
            if not result.get('positioning'): st.info("선택된 제품군에 대한 데이터가 부족하여 상세 분석을 진행할 수 없습니다."); continue
            
            p = result['positioning']
            analysis_data_pos = company_data[company_data['reported_product_name'].isin(group['selected_products'])]
            
            st.markdown("#### PART 1. 마켓 포지션 분석")
            if p['bubble_data'].empty: st.info("포지션 맵을 그리기 위한 데이터가 충분하지 않습니다.")
            else:
                bubble_df = p['bubble_data'].copy()
                all_importers = bubble_df['importer'].unique()
                target_name = st.session_state.get('importer_name_result', '')
                anonymity_map = {name: f"{get_excel_col_name(i)}사" for i, name in enumerate(all_importers) if name != target_name}
                anonymity_map[target_name] = "귀사"
                bubble_df['Anonymized_Importer'] = bubble_df['importer'].apply(lambda x: anonymity_map.get(x, "기타"))
                st.plotly_chart(px.scatter(bubble_df, x='Total_Volume', y='Avg_UnitPrice', size='Total_Value', color='Anonymized_Importer', log_x=True, hover_name='Anonymized_Importer', title="수입사 포지셔닝 맵"), use_container_width=True)

            col1, col2 = st.columns([10, 1])
            with col1: st.markdown("##### 수입 업체 그룹별 수입 빈도 분석(최근 1년)")
            with col2:
                with st.popover("ℹ️"):
                    st.markdown("""**그룹 분류 기준:**\n- **시장 선도 그룹**: 수입금액 기준 누적 70% 차지 상위 기업\n- **유사 규모 경쟁 그룹**: 귀사 순위 기준 상하 ±10% 범위 기업\n- **최저가 달성 그룹**: 평균 단가 하위 15% 기업 (최소 2회 이상 수입)""")
            
            groups_data = {name: analysis_data_pos[analysis_data_pos['importer'].isin(df['importer'])] for name, df in p['groups'].items() if not df.empty}
            group_names_map = {"Market Leaders": "시장 선도 그룹", "Direct Peers": "유사 규모 경쟁 그룹", "Price Achievers": "최저가 달성 그룹"}
            cols = st.columns(2)
            col_map = {0: cols[0], 1: cols[1], 2: cols[0], 3: cols[1]}
            for i, (key, name) in enumerate(group_names_map.items()):
                with col_map[i]:
                    if key in groups_data:
                        fig = create_monthly_frequency_bar_chart(groups_data[key], name)
                        if fig: st.plotly_chart(fig, use_container_width=True)
                        else: st.info(f"'{name}' 그룹은 존재하나, 최근 1년간의 수입 기록이 없어 빈도 분석을 생략합니다.")
                    else: st.info(f"조건에 맞는 '{name}'이(가) 없어 해당 그룹의 수입 빈도 분석을 생략합니다.")

            st.markdown("##### 그룹별 수입 단가 분포 비교")
            if not groups_data: st.info("그룹으로 분류할 수 있는 업체가 부족하여, 단가 분포 비교 분석을 진행할 수 없습니다.")
            else:
                fig_box = go.Figure()
                for name, df in groups_data.items(): fig_box.add_trace(go.Box(y=df['unitPrice'], name=group_names_map.get(name, name)))
                if not p['target_stats'].empty: fig_box.add_hline(y=p['target_stats']['Avg_UnitPrice'].iloc[0], line_dash="dot", annotation_text="귀사 평균단가")
                st.plotly_chart(fig_box, use_container_width=True)

            st.markdown("#### PART 2. 공급망 분석")
            s = result.get('supply_chain', {})
            if not s: st.info("거래 데이터를 비교하여 추가적인 비용 절감 기회를 분석하기에는 데이터가 부족합니다.")
            else:
                if 'same_exporter_stats' in s and len(s['same_exporter_stats']) > 1:
                    st.markdown(f"##### **{group['user_input']['Exporter']}** 거래 경쟁사 비교")
                    df_plot = s['same_exporter_stats']
                    fig = make_subplots(specs=[[{"secondary_y": True}]])
                    fig.add_trace(go.Bar(x=df_plot['importer'], y=df_plot['Total_Volume'], name='총 수입 중량'), secondary_y=False)
                    fig.add_trace(go.Scatter(x=df_plot['importer'], y=df_plot['Avg_UnitPrice'], name='평균 단가'), secondary_y=True)
                    st.plotly_chart(fig, use_container_width=True)
                if 'best_exporter' in s:
                    be = s['best_exporter']
                    st.success(f"**수출업체 변경** 시 평균 단가 **{be['saving_rate']:.1%}** 절감 가능성이 있습니다.")
                    pct = st.slider(f"'{be['name']}' 고려 시 절감률 설정", 0.0, be['saving_rate']*100, 5.0, format="%.1f%%", key=f"exp_{group['id']}")
                    st.info(f"👉 **${group['user_input']['Value'] * (pct / 100):,.2f} USD** 상당의 비용을 절약할 수 있습니다.")
            st.markdown("---")
        if st.button("🔄 새로운 분석 시작하기", use_container_width=True):
            for key in list(st.session_state.keys()):
                if key != 'logged_in': del st.session_state[key]
            st.rerun()

# --- 메인 로직 ---
if 'logged_in' not in st.session_state: st.session_state['logged_in'] = False
if st.session_state['logged_in']:
    company_data = load_company_data()
    if company_data is not None: main_dashboard(company_data)
    else: st.error("데이터를 불러오는 데 실패했습니다. 잠시 후 다시 시도해주세요.")
else:
    login_screen()
