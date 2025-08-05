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
        df = df[df['volume'] > 0]

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
    if not isinstance(text, str): return ''
    text = text.lower()
    text = re.sub(r'\(.*?\)|\[.*?\]', ' ', text)
    text = re.sub(r'(\d+)\s*(?:y|yo|year|years|old|년산|년)', r'\1', text)
    text = re.sub(r'[^a-z0-9\s\uac00-\ud7a3]', ' ', text)
    text = re.sub(r'\b산\b', ' ', text)
    return ' '.join(text.split())

def get_excel_col_name(n):
    name = ""
    while n >= 0:
        name = chr(ord('A') + n % 26) + name
        n = n // 26 - 1
    return name

def create_calendar_heatmap(df, title):
    if df.empty:
        return None
    end_date = datetime.now()
    start_date = end_date - pd.DateOffset(years=1)
    df_filtered = df[(df['date'] >= start_date) & (df['date'] <= end_date)].copy()
    if df_filtered.empty:
        return None
    daily_counts = df_filtered.set_index('date').resample('D').size().reset_index(name='counts')
    all_days = pd.date_range(start_date, end_date, freq='D')
    daily_counts = daily_counts.set_index('date').reindex(all_days, fill_value=0).reset_index().rename(columns={'index':'date'})
    daily_counts['day_of_week'] = daily_counts['date'].dt.day_name()
    daily_counts['week_of_year'] = daily_counts['date'].dt.isocalendar().week
    daily_counts['text'] = daily_counts.apply(lambda row: f"<b>{row['date'].strftime('%Y-%m-%d')}</b><br>Count: {row['counts']}", axis=1)
    fig = go.Figure(data=go.Heatmap(
        z=daily_counts['counts'], x=daily_counts['week_of_year'], y=daily_counts['day_of_week'],
        hovertext=daily_counts['text'], hoverinfo='text', colorscale='Greens', showscale=False))
    fig.update_layout(title=title, yaxis=dict(categoryorder='array', categoryarray=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']),
                      margin=dict(t=50, b=20, l=40, r=20), height=250, plot_bgcolor='white')
    return fig

# --- 메인 분석 로직 ---
def run_all_analysis(user_input, full_company_data, selected_products, target_importer_name):
    analysis_result = {"overview": {}, "positioning": {}, "supply_chain": {}}
    
    # --- 0. Overview 분석 (HS-Code 기준) ---
    hscode_data = full_company_data[full_company_data['hs_code'].astype(str) == str(user_input['HS-CODE'])].copy()
    if not hscode_data.empty:
        this_year = datetime.now().year
        last_year = this_year - 1
        hscode_data.loc[:, 'unitPrice'] = hscode_data['value'] / hscode_data['volume']

        vol_this_year = hscode_data[hscode_data['date'].dt.year == this_year]['volume'].sum()
        vol_last_year = hscode_data[hscode_data['date'].dt.year == last_year]['volume'].sum()
        price_this_year = hscode_data[hscode_data['date'].dt.year == this_year]['unitPrice'].mean()
        price_last_year = hscode_data[hscode_data['date'].dt.year == last_year]['unitPrice'].mean()

        vol_yoy = (vol_this_year - vol_last_year) / vol_last_year if vol_last_year > 0 else np.nan
        price_yoy = (price_this_year - price_last_year) / price_last_year if price_last_year > 0 else np.nan

        all_cycles = []
        for importer in hscode_data['importer'].unique():
            importer_df = hscode_data[hscode_data['importer'] == importer].sort_values('date')
            if len(importer_df) > 1:
                cycles = importer_df['date'].diff().dt.days.dropna()
                all_cycles.extend(cycles)
        avg_total_cycle = np.mean(all_cycles) if all_cycles else np.nan
        
        analysis_result['overview'] = {
            "this_year": this_year,
            "vol_this_year": vol_this_year, "vol_yoy": vol_yoy,
            "price_this_year": price_this_year, "price_yoy": price_yoy,
            "avg_total_cycle": avg_total_cycle,
            "product_composition": hscode_data.groupby('reported_product_name')['value'].sum().nlargest(10).reset_index()
        }

    # --- 1. & 2. 포지셔닝 및 공급망 분석 (선택된 제품군 기준) ---
    analysis_data = full_company_data[full_company_data['reported_product_name'].isin(selected_products)].copy()
    if analysis_data.empty:
        return analysis_result
        
    analysis_data.loc[:, 'unitPrice'] = analysis_data['value'] / analysis_data['volume']
    
    importer_stats = analysis_data.groupby('importer').agg(
        Total_Value=('value', 'sum'), Total_Volume=('volume', 'sum'),
        Trade_Count=('value', 'count'), Avg_UnitPrice=('unitPrice', 'mean')
    ).reset_index()

    if not importer_stats.empty and importer_stats['Total_Volume'].sum() > 0:
        importer_stats = importer_stats.sort_values('Total_Value', ascending=False).reset_index(drop=True)
        total_market_value = importer_stats['Total_Value'].sum()
        importer_stats['cum_share'] = importer_stats['Total_Value'].cumsum() / total_market_value
        market_leaders_df = importer_stats[importer_stats['cum_share'] <= 0.7]

        try:
            target_rank = importer_stats[importer_stats['importer'] == target_importer_name].index[0]
            rank_margin = max(1, int(len(importer_stats) * 0.1))
            peer_min_rank, peer_max_rank = max(0, target_rank - rank_margin), min(len(importer_stats), target_rank + rank_margin + 1)
            direct_peers_df = importer_stats.iloc[peer_min_rank:peer_max_rank]
        except IndexError:
            direct_peers_df = pd.DataFrame()

        price_achievers_candidates = importer_stats[importer_stats['Trade_Count'] >= 2]
        price_achievers_df = price_achievers_candidates[price_achievers_candidates['Avg_UnitPrice'] <= price_achievers_candidates['Avg_UnitPrice'].quantile(0.15)] if not price_achievers_candidates.empty else pd.DataFrame()

        analysis_result['positioning'] = {
            "bubble_data": importer_stats,
            "groups": {"Market Leaders": market_leaders_df, "Direct Peers": direct_peers_df, "Price Achievers": price_achievers_df},
            "target_stats": importer_stats[importer_stats['importer'] == target_importer_name]
        }

    target_exporter = user_input.get('Exporter', '').upper()
    target_country = user_input.get('Origin Country', '').upper()
    
    if target_exporter:
        same_exporter_df = analysis_data[analysis_data['exporter'] == target_exporter]
        analysis_result['supply_chain']['same_exporter_stats'] = same_exporter_df.groupby('importer').agg(Total_Volume=('volume', 'sum'), Avg_UnitPrice=('unitPrice', 'mean')).reset_index()
        target_exporter_price = analysis_data[analysis_data['exporter'] == target_exporter]['unitPrice'].mean()
        if not np.isnan(target_exporter_price):
            cheaper_exporters = analysis_data[(analysis_data['exporter'] != target_exporter) & (analysis_data['unitPrice'] < target_exporter_price)].groupby('exporter').agg(Avg_UnitPrice=('unitPrice', 'mean')).reset_index()
            if not cheaper_exporters.empty:
                best_exporter = cheaper_exporters.sort_values('Avg_UnitPrice').iloc[0]
                analysis_result['supply_chain']['best_exporter'] = {'name': best_exporter['exporter'], 'saving_rate': (target_exporter_price - best_exporter['Avg_UnitPrice']) / target_exporter_price}

    if target_country:
        target_country_price = analysis_data[analysis_data['export_country'] == target_country]['unitPrice'].mean()
        if not np.isnan(target_country_price):
            cheaper_countries = analysis_data[(analysis_data['export_country'] != target_country) & (analysis_data['unitPrice'] < target_country_price)].groupby('export_country').agg(Avg_UnitPrice=('unitPrice', 'mean')).reset_index()
            if not cheaper_countries.empty:
                best_country = cheaper_countries.sort_values('Avg_UnitPrice').iloc[0]
                analysis_result['supply_chain']['best_country'] = {'name': best_country['export_country'], 'saving_rate': (target_country_price - best_country['Avg_UnitPrice']) / target_country_price}

    return analysis_result

# --- UI Components ---
def login_screen():
    st.title("🔐 수입 경쟁력 진단 솔루션")
    st.write("솔루션 접속을 위해 비밀번호를 입력해주세요.")
    with st.form("login_form", clear_on_submit=True):
        password = st.text_input("비밀번호", type="password")
        submitted = st.form_submit_button("접속하기")
        if submitted:
            if password == st.secrets.get("APP_PASSWORD", "tridgeDemo_2025"):
                st.session_state['logged_in'] = True
                st.rerun()
            else:
                st.error("비밀번호가 올바르지 않습니다.")

def main_dashboard(company_data):
    st.title("📈 수입 경쟁력 진단 솔루션")
    st.markdown("트릿지 데이터를 기반으로 시장 내 경쟁력을 진단하고 비용 절감 기회를 포착하세요.")

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
            st.session_state[f'final_origin_{i}'] = cols[2].text_input("└ 원산지 직접 입력", key=f"custom_origin_{i}", placeholder="직접 입력하세요") if selected_origin == '직접 입력' else selected_origin
            exporter_options = [''] + ['직접 입력'] + sorted(company_data['exporter'].unique())
            selected_exporter = cols[3].selectbox("수출업체", exporter_options, key=f"exporter_{i}", format_func=lambda x: '선택 또는 직접 입력' if x == '' else x)
            st.session_state[f'final_exporter_{i}'] = cols[3].text_input("└ 수출업체 직접 입력", key=f"custom_exporter_{i}", placeholder="직접 입력하세요") if selected_exporter == '직접 입력' else selected_exporter
            cols[4].number_input("수입 중량(KG)", min_value=0.01, format="%.2f", key=f"volume_{i}")
            cols[5].number_input("총 수입금액(USD)", min_value=0.01, format="%.2f", key=f"value_{i}")
            if len(st.session_state.rows) > 1 and cols[6].button("삭제", key=f"delete_{i}"):
                st.session_state.rows.pop(i)
                st.rerun()
        if st.button("➕ 내역 추가하기"):
            st.session_state.rows.append({'id': len(st.session_state.rows) + 1})
            st.rerun()
        st.markdown("---")
        consent = st.checkbox("정보 활용 동의", value=True)
        if st.button("분석하기", type="primary", use_container_width=True):
            if not importer_name: st.warning("수입업체명을 입력해주세요.")
            elif not consent: st.warning("데이터 활용 동의에 체크해주세요.")
            else:
                with st.spinner('데이터를 분석하고 있습니다...'):
                    all_purchase_data = []
                    for i in range(len(st.session_state.rows)):
                        entry = {
                            'Reported Product Name': st.session_state.get(f'product_name_{i}', ''),
                            'HS-CODE': st.session_state.get(f'hscode_{i}', ''),
                            'Origin Country': st.session_state.get(f'final_origin_{i}', '').upper(),
                            'Exporter': st.session_state.get(f'final_exporter_{i}', '').upper(),
                            'Volume': st.session_state.get(f'volume_{i}', 0),
                            'Value': st.session_state.get(f'value_{i}', 0)
                        }
                        if not all(entry.values()):
                            st.error(f"{i+1}번째 행의 모든 값을 입력해주세요.")
                            return
                        all_purchase_data.append(entry)
                    
                    purchase_df = pd.DataFrame(all_purchase_data)
                    agg_funcs = {'Volume': 'sum', 'Value': 'sum', 'HS-CODE': 'first', 'Origin Country': 'first', 'Exporter': 'first'}
                    aggregated_purchase_df = purchase_df.groupby('Reported Product Name').agg(agg_funcs).reset_index()

                    analysis_groups = []
                    company_data['cleaned_name'] = company_data['reported_product_name'].apply(clean_text)
                    for i, row in aggregated_purchase_df.iterrows():
                        entry = row.to_dict()
                        user_tokens = set(clean_text(entry['Reported Product Name']).split())
                        is_match = lambda name: user_tokens.issubset(set(name))
                        matched_df = company_data[company_data['cleaned_name'].apply(is_match)]
                        analysis_groups.append({ "id": i, "user_input": entry, "matched_products": sorted(matched_df['reported_product_name'].unique().tolist()), "selected_products": sorted(matched_df['reported_product_name'].unique().tolist()) })
                    
                    st.session_state['importer_name_result'] = importer_name
                    st.session_state['analysis_groups'] = analysis_groups
                    st.rerun()

    if 'analysis_groups' in st.session_state:
        st.header("📊 분석 결과")
        with st.expander("STEP 2: 분석 대상 제품 필터링", expanded=True):
            for i, group in enumerate(st.session_state.analysis_groups):
                st.markdown(f"**분석 그룹: \"{group['user_input']['Reported Product Name']}\"**")
                selected = st.multiselect("분석에 활용할 제품명 선택:", options=group['matched_products'], default=group['selected_products'], key=f"filter_{group['id']}")
                st.session_state.analysis_groups[i]['selected_products'] = selected
                st.markdown("---")

        for group in st.session_state.analysis_groups:
            st.subheader(f"분석 결과: \"{group['user_input']['Reported Product Name']}\"")
            
            analysis_data_for_pos = company_data[company_data['reported_product_name'].isin(group['selected_products'])]
            result = run_all_analysis(group['user_input'], company_data, group['selected_products'], st.session_state['importer_name_result'])
            
            st.markdown("### 0. Overview (HS-Code 기준)")
            # ... (Overview 표시 코드는 이전과 동일) ...
            if result.get('overview'):
                o = result['overview']
                hscode = group['user_input']['HS-CODE']
                cols = st.columns(3)
                cols[0].metric(f"{o['this_year']}년 수입 중량 (KG)", f"{o['vol_this_year']:,.0f}", f"{o['vol_yoy']:.1%}" if not np.isnan(o['vol_yoy']) else "N/A", delta_color="inverse")
                cols[1].metric(f"{o['this_year']}년 평균 단가 (USD/KG)", f"${o['price_this_year']:.2f}", f"{o['price_yoy']:.1%}" if not np.isnan(o['price_yoy']) else "N/A", delta_color="inverse")
                if not np.isnan(o['avg_total_cycle']):
                    cols[2].metric("평균 수입 주기", f"{o['avg_total_cycle']:.1f} 일")
                    with cols[2]: st.caption("※ 해당 HS-Code 전체 수입사 평균")
                else:
                    cols[2].metric("평균 수입 주기", "N/A")
            else:
                st.info("해당 HS-Code에 대한 Overview 데이터가 부족합니다.")

            if not group['selected_products']:
                st.warning("선택된 비교 대상 제품이 없어 포지셔닝 및 공급망 분석을 건너뜁니다.")
                st.markdown("---")
                continue
                
            st.markdown(f"### 1. 포지셔닝 및 공급망 분석 ({st.session_state['importer_name_result']})")
            if result.get('positioning') and not result['positioning']['bubble_data'].empty:
                p = result['positioning']
                
                # ... (버블차트, 익명화 등) ...
                
                # *** BUG FIX & UX IMPROVEMENT STARTS HERE ***
                groups_data = {}
                for name, df in p['groups'].items():
                    if not df.empty:
                        groups_data[name] = analysis_data_for_pos[analysis_data_for_pos['importer'].isin(df['importer'])]
                    else:
                        groups_data[name] = pd.DataFrame()

                st.markdown("##### 그룹별 수입 활동 꾸준함 분석 (지난 1년)")
                c1, c2 = st.columns(2)
                with c1:
                    target_df = analysis_data_for_pos[analysis_data_for_pos['importer'] == st.session_state['importer_name_result']]
                    fig_target = create_calendar_heatmap(target_df, f"귀사 ({len(target_df)} 건)")
                    if fig_target: st.plotly_chart(fig_target, use_container_width=True)
                    else: st.info("귀사의 지난 1년간 데이터가 없습니다.")

                    peers_df_group = p['groups'].get('Direct Peers', pd.DataFrame())
                    if not peers_df_group.empty:
                        fig_peers = create_calendar_heatmap(groups_data.get('Direct Peers'), f"유사 규모 경쟁 그룹 ({len(peers_df_group)}개사)")
                        if fig_peers: st.plotly_chart(fig_peers, use_container_width=True)
                        else: st.info(f"유사 규모 경쟁 그룹({len(peers_df_group)}개사)은 있으나, 지난 1년간 수입 기록이 없습니다.")
                    else:
                        st.info("조건에 해당하는 '유사 규모 경쟁 그룹' 업체가 없어 해당 분석을 진행할 수 없습니다.")
                
                with c2:
                    leaders_df_group = p['groups'].get('Market Leaders', pd.DataFrame())
                    if not leaders_df_group.empty:
                        fig_leaders = create_calendar_heatmap(groups_data.get('Market Leaders'), f"시장 선도 그룹 ({len(leaders_df_group)}개사)")
                        if fig_leaders: st.plotly_chart(fig_leaders, use_container_width=True)
                        else: st.info(f"시장 선도 그룹({len(leaders_df_group)}개사)은 있으나, 지난 1년간 수입 기록이 없습니다.")
                    else:
                        st.info("조건에 해당하는 '시장 선도 그룹' 업체가 없어 해당 분석을 진행할 수 없습니다.")

                    achievers_df_group = p['groups'].get('Price Achievers', pd.DataFrame())
                    if not achievers_df_group.empty:
                        fig_achievers = create_calendar_heatmap(groups_data.get('Price Achievers'), f"최저가 달성 그룹 ({len(achievers_df_group)}개사)")
                        if fig_achievers: st.plotly_chart(fig_achievers, use_container_width=True)
                        else: st.info(f"최저가 달성 그룹({len(achievers_df_group)}개사)은 있으나, 지난 1년간 수입 기록이 없습니다.")
                    else:
                        st.info("조건에 해당하는 '최저가 달성 그룹' 업체가 없어 해당 분석을 진행할 수 없습니다.")
                
                # ... (나머지 공급망 분석 등 UI 코드) ...
                
            else:
                st.info("선택된 제품군에 대한 데이터가 부족하여 포지셔닝 및 공급망 분석을 생략합니다.")
            st.markdown("---")

        if st.button("🔄 새로운 분석 시작하기", use_container_width=True):
            keys_to_keep = ['logged_in']
            for key in list(st.session_state.keys()):
                if key not in keys_to_keep: del st.session_state[key]
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
