import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import re
from datetime import datetime
from dateutil.relativedelta import relativedelta
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from google.oauth2.service_account import Credentials
from pandas_gbq import read_gbq
import gspread
from zoneinfo import ZoneInfo

# --- 페이지 초기 설정 ---
st.set_page_config(layout="wide", page_title="수입 경쟁력 진단 솔루션")

# --- 데이터 로딩 (BigQuery) ---
@st.cache_data(ttl=3600)
def load_company_data():
    """Google BigQuery에서 데이터를 불러오고 기본 전처리를 수행합니다."""
    try:
        creds = Credentials.from_service_account_info(st.secrets["gcp_service_account"])
        project_id = st.secrets["gcp_service_account"]["project_id"]
        # 실제 테이블명으로 변경해주세요. 예: "your_dataset.your_table"
        table_full_id = f"{project_id}.demo_data.tds_data"
        df = read_gbq(f"SELECT * FROM `{table_full_id}`", project_id=project_id, credentials=creds)

        if df.empty:
            st.error("BigQuery에서 데이터를 불러왔지만 비어있습니다.")
            return None

        # 지능형 컬럼명 정제
        df.columns = [re.sub(r'[^a-z0-9]+', '_', col.lower().strip()) for col in df.columns]

        # 필수 컬럼 확인
        required_cols = ['date', 'volume', 'value', 'reported_product_name', 'export_country', 'exporter', 'importer', 'hs_code']
        if not all(col in df.columns for col in required_cols):
            st.error(f"BigQuery 테이블에 필수 컬럼이 부족합니다. (필수: {required_cols})")
            st.info(f"실제 컬럼명: {df.columns.tolist()}")
            return None

        # 데이터 타입 변환 및 정제
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        for col in ['volume', 'value']:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(r'[^\d.]', '', regex=True), errors='coerce')

        df.dropna(subset=['date', 'volume', 'value', 'importer', 'exporter'], inplace=True)
        df = df[(df['volume'] > 0) & (df['value'] > 0)].copy()

        # 단가 계산 및 이상치 제거 (IQR 방식)
        df['unitprice'] = df['value'] / df['volume']
        Q1 = df['unitprice'].quantile(0.25)
        Q3 = df['unitprice'].quantile(0.75)
        IQR = Q3 - Q1
        df = df[~((df['unitprice'] < (Q1 - 1.5 * IQR)) | (df['unitprice'] > (Q3 + 1.5 * IQR)))]

        return df if not df.empty else None
    except Exception as e:
        st.error(f"데이터 로딩 중 심각한 오류가 발생했습니다: {e}")
        st.info("Streamlit Secrets의 'gcp_service_account' 설정을 다시 확인해주세요.")
        return None

# --- Google Sheets 저장 ---
def save_to_google_sheets(data_to_save):
    """사용자 입력 데이터를 지정된 구글 시트에 저장합니다."""
    try:
        creds = Credentials.from_service_account_info(st.secrets["gcp_service_account"])
        client = gspread.authorize(creds)
        
        spreadsheet_name = st.secrets.get("google_sheets", {}).get("spreadsheet_name", "DEMO_app_DB")
        worksheet_name = st.secrets.get("google_sheets", {}).get("worksheet_name", "Customer_input")

        sheet = client.open(spreadsheet_name).worksheet(worksheet_name)
        
        # 헤더가 비어있을 경우, 헤더 추가
        if not sheet.get_all_values():
            header = ["Date", "Reported Product Name", "HS-Code", "Export Country", "Exporter",
                      "Volume(KG)", "Value(USD)", "Incoterms", "Importer", "IS_Agreed", "Input_time"]
            sheet.append_row(header)
            
        sheet.append_row(data_to_save, value_input_option='USER_ENTERED')
        return True
    except Exception as e:
        st.error(f"Google Sheets 저장 중 오류가 발생했습니다: {e}")
        st.warning("Google Sheets API가 활성화되어 있는지, 서비스 계정에 편집자 권한이 있는지 확인해주세요.")
        return False

# --- 분석 헬퍼 함수 ---
def clean_text(text):
    """제품명 텍스트를 정제합니다."""
    if not isinstance(text, str): return ''
    text = text.lower()
    text = re.sub(r'\(.*?\)|\[.*?\]', ' ', text)
    text = re.sub(r'(\d+)\s*(?:y|yo|year|years|old|년산|년)', r'\1', text)
    text = re.sub(r'[^a-z0-9\s\uac00-\ud7a3]', ' ', text)
    text = re.sub(r'\b산\b', ' ', text)
    return ' '.join(text.split())

def name_clusters(df, cluster_col='cluster'):
    """K-Means 클러스터에 동적으로 이름을 부여합니다."""
    centroids = df.groupby(cluster_col).agg(
        Total_Volume=('total_volume', 'mean'),
        Avg_UnitPrice=('avg_unitprice', 'mean'),
        Trade_Count=('trade_count', 'mean')
    ).reset_index()
    centroids['volume_rank'] = centroids['Total_Volume'].rank(ascending=False)
    centroids['price_rank'] = centroids['Avg_UnitPrice'].rank(ascending=True)
    centroids['count_rank'] = centroids['Trade_Count'].rank(ascending=False)
    centroids['total_score'] = centroids['volume_rank'] * 0.4 + centroids['price_rank'] * 0.4 + centroids['count_rank'] * 0.2
    sorted_centroids = centroids.sort_values('total_score')
    names = ["소규모/고가치 그룹", "중견/균형 그룹", "대규모/가성비 그룹"]
    name_map = {row[cluster_col]: names[i] if i < len(names) else f"{chr(ord('A')+i)} 그룹" for i, row in sorted_centroids.iterrows()}
    return name_map

def perform_clustering(importer_stats):
    """K-Means 클러스터링을 수행합니다."""
    features = importer_stats[['total_volume', 'avg_unitprice', 'trade_count']].copy()
    if len(features) < 3: return None, None
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
    importer_stats['cluster'] = kmeans.fit_predict(scaled_features)
    cluster_name_map = name_clusters(importer_stats)
    importer_stats['Cluster_Name'] = importer_stats['cluster'].map(cluster_name_map)
    return importer_stats, cluster_name_map

# --- 메인 분석 로직 ---
def run_all_analysis(user_inputs, full_company_data, selected_products, target_importer_name):
    """핵심 분석 로직을 수행합니다."""
    analysis_result = {"positioning": {}, "supply_chain": {}}
    analysis_data = full_company_data[full_company_data['reported_product_name'].isin(selected_products)].copy()
    if analysis_data.empty: return analysis_result

    # 1. 수입사별 통계 집계
    importer_stats = analysis_data.groupby('importer').agg(
        total_value=('value', 'sum'),
        total_volume=('volume', 'sum'),
        trade_count=('value', 'count'),
        avg_unitprice=('unitprice', 'mean')
    ).reset_index()
    if importer_stats.empty: return analysis_result
    importer_stats = importer_stats.sort_values('total_value', ascending=False).reset_index(drop=True)

    # 2. 포지셔닝 분석 (규칙 기반 + AI 기반)
    importer_stats['cum_share'] = importer_stats['total_value'].cumsum() / importer_stats['total_value'].sum()
    market_leaders = importer_stats[importer_stats['cum_share'] <= 0.7]
    try:
        target_rank = importer_stats[importer_stats['importer'] == target_importer_name].index[0]
        rank_margin = max(1, int(len(importer_stats) * 0.1))
        direct_peers = importer_stats.iloc[max(0, target_rank - rank_margin):min(len(importer_stats), target_rank + rank_margin + 1)]
    except IndexError: direct_peers = pd.DataFrame()
    price_achievers_candidates = importer_stats[importer_stats['trade_count'] >= 2]
    price_achievers = price_achievers_candidates[price_achievers_candidates['avg_unitprice'] <= price_achievers_candidates['avg_unitprice'].quantile(0.15)] if not price_achievers_candidates.empty else pd.DataFrame()
    clustered_stats, cluster_names = perform_clustering(importer_stats.copy())
    analysis_result['positioning'] = {
        "importer_stats": importer_stats, "clustered_stats": clustered_stats, "cluster_names": cluster_names,
        "rule_based_groups": {"Market Leaders": market_leaders, "Direct Peers": direct_peers, "Price Achievers": price_achievers},
        "target_stats": importer_stats[importer_stats['importer'] == target_importer_name]
    }

    # 3. 공급망 분석
    user_input = user_inputs[0]
    user_avg_price = user_input['Value'] / user_input['Volume'] if user_input['Volume'] > 0 else 0
    alternative_suppliers = analysis_data[(analysis_data['exporter'].str.upper() != user_input['Exporter'].upper()) & (analysis_data['unitprice'] < user_avg_price)]
    if not alternative_suppliers.empty:
        supplier_analysis = alternative_suppliers.groupby('exporter').agg(
            avg_unitprice=('unitprice', 'mean'), trade_count=('value', 'count'), num_importers=('importer', 'nunique')
        ).reset_index().sort_values('avg_unitprice')
        supplier_analysis['price_saving_pct'] = (1 - supplier_analysis['avg_unitprice'] / user_avg_price) * 100
        supplier_analysis['stability_score'] = np.log1p(supplier_analysis['trade_count']) + np.log1p(supplier_analysis['num_importers'])
        analysis_result['supply_chain'] = {
            "user_avg_price": user_avg_price, "user_total_volume": sum(item['Volume'] for item in user_inputs), "alternatives": supplier_analysis
        }
    return analysis_result


# --- UI 컴포넌트 ---
def login_screen():
    """로그인 화면을 표시합니다."""
    st.title("🔐 수입 경쟁력 진단 솔루션")
    st.write("솔루션 접속을 위해 비밀번호를 입력해주세요.")
    with st.form("login_form"):
        password = st.text_input("비밀번호", type="password")
        if st.form_submit_button("접속하기"):
            if password == st.secrets.get("app_secrets", {}).get("password", "tridgeDemo_2025"):
                st.session_state['logged_in'] = True; st.rerun()
            else: st.error("비밀번호가 올바르지 않습니다.")

def main_dashboard(company_data):
    """메인 대시보드 UI를 구성합니다."""
    st.title("📈 수입 경쟁력 진단 솔루션")
    st.markdown("트릿지 데이터를 기반으로 시장 내 경쟁력을 진단하고 비용 절감 기회를 포착하세요.")

    with st.expander("STEP 1: 분석 정보 입력", expanded='analysis_groups' not in st.session_state):
        importer_name = st.text_input("1. 귀사의 업체명을 입력해주세요.", key="importer_name_input").upper()
        if 'rows' not in st.session_state: st.session_state['rows'] = [{'id': 1}]

        # --- 수평 입력 UI ---
        header_cols = st.columns([1.5, 3, 1, 2, 2, 1, 1, 1, 0.5])
        headers = ["수입일", "제품 상세명", "HS-CODE", "원산지", "수출업체", "수입 중량(KG)", "총 수입금액(USD)", "Incoterms", "삭제"]
        for col, header in zip(header_cols, headers): col.markdown(f"**{header}**")

        all_input_data = []
        for i, row in enumerate(st.session_state.rows):
            key_suffix = f"_{row['id']}"
            cols = st.columns([1.5, 3, 1, 2, 2, 1, 1, 1, 0.5])
            
            date_val = cols[0].date_input(f"date{key_suffix}", key=f"date{key_suffix}", label_visibility="collapsed", value=datetime.now())
            product_name_val = cols[1].text_input(f"product_name{key_suffix}", key=f"product_name{key_suffix}", label_visibility="collapsed")
            hscode_val = cols[2].text_input(f"hscode{key_suffix}", max_chars=10, key=f"hscode{key_suffix}", label_visibility="collapsed")
            
            origin_options = [''] + ['직접 입력'] + sorted(company_data['export_country'].unique())
            origin_val = cols[3].selectbox(f"origin{key_suffix}", origin_options, key=f"origin{key_suffix}", label_visibility="collapsed", format_func=lambda x: '선택' if x == '' else x)
            if origin_val == '직접 입력': origin_val = cols[3].text_input(f"custom_origin{key_suffix}", label_visibility="collapsed", placeholder="원산지 직접 입력")

            exporter_options = [''] + ['직접 입력'] + sorted(company_data['exporter'].unique())
            exporter_val = cols[4].selectbox(f"exporter{key_suffix}", exporter_options, key=f"exporter{key_suffix}", label_visibility="collapsed", format_func=lambda x: '선택' if x == '' else x)
            if exporter_val == '직접 입력': exporter_val = cols[4].text_input(f"custom_exporter{key_suffix}", label_visibility="collapsed", placeholder="수출업체 직접 입력")

            volume_val = cols[5].number_input(f"volume{key_suffix}", min_value=0.01, format="%.2f", key=f"volume{key_suffix}", label_visibility="collapsed")
            value_val = cols[6].number_input(f"value{key_suffix}", min_value=0.01, format="%.2f", key=f"value{key_suffix}", label_visibility="collapsed")
            incoterms_val = cols[7].selectbox(f"incoterms{key_suffix}", ["FOB", "CFR", "CIF", "EXW", "DDP", "기타"], key=f"incoterms{key_suffix}", label_visibility="collapsed")

            if len(st.session_state.rows) > 1 and cols[8].button("삭제", key=f"delete{key_suffix}"):
                st.session_state.rows.pop(i); st.rerun()

            all_input_data.append({
                "Date": date_val, "Reported Product Name": product_name_val, "HS-Code": hscode_val,
                "Origin Country": origin_val, "Exporter": exporter_val, "Volume": volume_val,
                "Value": value_val, "Incoterms": incoterms_val
            })
            
        if st.button("➕ 내역 추가하기"):
            new_id = max(row['id'] for row in st.session_state.rows) + 1 if st.session_state.rows else 1
            st.session_state.rows.append({'id': new_id}); st.rerun()
        
        st.markdown("---")
        consent = st.checkbox("입력하신 정보는 데이터 분석 품질 향상을 위해 저장 및 활용되는 것에 동의합니다.", value=True)
        
        if st.button("분석하기", type="primary", use_container_width=True):
            is_valid = True
            for i, entry in enumerate(all_input_data):
                if not all([entry['Reported Product Name'], entry['HS-Code'], entry['Origin Country'], entry['Exporter'], entry['Volume'] > 0, entry['Value'] > 0]):
                    st.error(f"{i+1}번째 입력 줄의 모든 값을 정확히 입력해주세요."); is_valid = False
            if not importer_name: st.error("귀사의 업체명을 입력해주세요."); is_valid = False
            if not consent: st.warning("정보 활용 동의에 체크해주세요."); is_valid = False
            
            if is_valid:
                with st.spinner('입력 데이터를 저장하고 분석을 시작합니다...'):
                    # Google Sheets 저장
                    for entry in all_input_data:
                        row_to_save = [
                            entry["Date"].strftime('%Y-%m-%d'), entry["Reported Product Name"], entry["HS-Code"],
                            entry["Origin Country"], entry["Exporter"].upper(), entry["Volume"], entry["Value"],
                            entry["Incoterms"], importer_name, consent,
                            datetime.now(ZoneInfo("Asia/Seoul")).strftime('%Y-%m-%d %H:%M:%S')
                        ]
                        save_to_google_sheets(row_to_save)
                    
                    # 분석 로직 실행
                    purchase_df = pd.DataFrame(all_input_data)
                    purchase_df['cleaned_name'] = purchase_df['Reported Product Name'].apply(clean_text)
                    agg_funcs = {'Volume': 'sum', 'Value': 'sum', 'Reported Product Name': 'first', 'HS-Code': 'first', 'Exporter': 'first', 'Date':'first', 'Origin Country':'first', 'Incoterms':'first'}
                    aggregated_purchase_df = purchase_df.groupby('cleaned_name', as_index=False).agg(agg_funcs)

                    analysis_groups = []
                    company_data['cleaned_name'] = company_data['reported_product_name'].apply(clean_text)
                    for _, row in aggregated_purchase_df.iterrows():
                        entry = row.to_dict()
                        user_tokens = set(entry['cleaned_name'].split())
                        is_match = lambda name: user_tokens.issubset(set(str(name).split()))
                        matched_df = company_data[company_data['cleaned_name'].apply(is_match)]
                        matched_products = sorted(matched_df['reported_product_name'].unique().tolist())
                        result = run_all_analysis([entry], company_data, matched_products, importer_name)
                        analysis_groups.append({
                            "user_input": entry, "matched_products": matched_products,
                            "selected_products": matched_products, "result": result
                        })
                    st.session_state['importer_name_result'] = importer_name
                    st.session_state['analysis_groups'] = analysis_groups
                    st.success("분석 완료!")
                    st.rerun()
    
    # --- 분석 결과 표시 ---
    if 'analysis_groups' in st.session_state:
        st.header("📊 분석 결과")
        for i, group in enumerate(st.session_state.analysis_groups):
            product_name = group['user_input']['Reported Product Name']
            st.subheader(f"분석 그룹: \"{product_name}\"")
            
            # 분석 결과 데이터 추출
            result = group['result']
            p_res = result.get('positioning')
            s_res = result.get('supply_chain')
            
            st.markdown("#### PART 1. 마켓 포지션 분석")
            if not p_res or p_res['importer_stats'].empty:
                st.info("포지션 분석을 위한 데이터가 부족합니다.")
                continue
    
            # --- 전문가 제안: 사분면 + 강조 버블 차트 ---
            importer_stats = p_res['importer_stats']
            target_name = st.session_state.get('importer_name_result', '')
            
            # 시각화할 데이터 준비 (Top 5 + 유사그룹 + 귀사)
            plot_df = pd.concat([
                importer_stats.head(5), 
                p_res['rule_based_groups']['Direct Peers'], 
                p_res['target_stats']
            ]).drop_duplicates().reset_index(drop=True)
            
            # 익명화 및 사이즈/색상 설정
            plot_df['Anonymized_Importer'] = [f"{chr(ord('A')+j)}사" if imp != target_name else target_name for j, imp in enumerate(plot_df['importer'])]
            plot_df['size'] = np.log1p(plot_df['total_value']) # 로그 스케일링으로 버블 크기 조절
            
            # 귀사 강조를 위한 색상 및 투명도 설정
            colors = ['#FF4B4B' if imp == target_name else '#BDBDBD' for imp in plot_df['importer']]
            opacities = [1.0 if imp == target_name else 0.5 for imp in plot_df['importer']]
            plot_df['color'] = colors
            plot_df['opacity'] = opacities
            
            # 사분면 기준선 (시장 평균) 계산
            x_mean = importer_stats['total_volume'].mean()
            y_mean = importer_stats['avg_unitprice'].mean()
    
            # 차트 생성
            fig = px.scatter(
                plot_df, x='total_volume', y='avg_unitprice', size='size',
                color='color', # 개별 색상 적용
                opacity=0.8, # 기본 투명도
                hover_name='Anonymized_Importer',
                hover_data={'total_volume': ':,', 'avg_unitprice': ':.2f', 'total_value':':,', 'size':False, 'color':False, 'opacity':False},
                log_x=True, title="수입사 포지셔닝 맵 (시장 전략 분석)"
            )
            
            # 개별 점에 대한 투명도 직접 설정 (px.scatter에서 직접 지원 안하므로 생성 후 변경)
            for i, o in enumerate(plot_df['opacity']):
                 fig.data[0].marker.color[i] = fig.data[0].marker.color[i].replace('1)', f'{o})').replace('rgb', 'rgba')
    
    
            # 평균선 추가
            fig.add_vline(x=x_mean, line_dash="dash", line_color="gray", annotation_text="평균 수입량")
            fig.add_hline(y=y_mean, line_dash="dash", line_color="gray", annotation_text="평균 단가")
            
            # 사분면 텍스트 추가
            chart_max_x = plot_df['total_volume'].max() * 1.5 # 로그 스케일 감안
            chart_max_y = plot_df['avg_unitprice'].max() * 1.1
            
            fig.add_annotation(x=np.log10(chart_max_x), y=chart_max_y, text="<b>니치/프리미엄 그룹</b>", showarrow=False, xanchor='right', yanchor='top', font=dict(color="grey", size=12))
            fig.add_annotation(x=np.log10(x_mean*0.95), y=chart_max_y, text="<b>시장 선도 그룹</b>", showarrow=False, xanchor='right', yanchor='top', font=dict(color="grey", size=12))
            fig.add_annotation(x=np.log10(chart_max_x), y=plot_df['avg_unitprice'].min(), text="<b>소규모/가격 경쟁 그룹</b>", showarrow=False, xanchor='right', yanchor='bottom', font=dict(color="grey", size=12))
            fig.add_annotation(x=np.log10(x_mean*0.95), y=plot_df['avg_unitprice'].min(), text="<b>대규모/가성비 그룹</b>", showarrow=False, xanchor='right', yanchor='bottom', font=dict(color="grey", size=12))
    
            # 귀사 위치에 화살표 추가
            target_row = plot_df[plot_df['importer'] == target_name]
            if not target_row.empty:
                target = target_row.iloc[0]
                fig.add_annotation(
                    x=np.log10(target['total_volume']), y=target['avg_unitprice'],
                    text="<b>귀사 위치</b>", showarrow=True, arrowhead=2, arrowcolor="#FF4B4B",
                    ax=-40, ay=-40, bordercolor="#FF4B4B", borderwidth=2, bgcolor="white"
                )
    
            fig.update_layout(
                xaxis_title="총 수입 중량 (KG, Log Scale)", yaxis_title="평균 수입 단가 (USD/KG)",
                showlegend=False # 범례 숨기기
            )
        st.plotly_chart(fig, use_container_width=True)
            # 그룹 분류 방식 선택
            st.markdown("##### 경쟁사 그룹 분석")
            grouping_method = st.radio("그룹 분류 방식 선택:", ["규칙 기반 그룹", "AI 기반 자동 그룹핑 (K-Means)"], horizontal=True, key=f"group_method_{i}")
            if grouping_method == "규칙 기반 그룹": st.info("시장 선도(누적 점유율 70%), 유사 규모(순위 ±10%), 최저가 달성(단가 하위 15%) 그룹으로 분류합니다.")
            else:
                if p_res.get('clustered_stats') is not None:
                    st.info("AI가 수입 규모, 단가, 빈도를 종합하여 유사한 업체끼리 3개의 그룹으로 자동 분류합니다.")
                    fig_box = px.box(p_res['clustered_stats'], x='Cluster_Name', y='avg_unitprice', title="AI 기반 그룹별 단가 분포", points='all', labels={'Cluster_Name': '그룹 유형', 'avg_unitprice': '평균 수입 단가'})
                    if not p_res['target_stats'].empty:
                        fig_box.add_hline(y=p_res['target_stats']['avg_unitprice'].iloc[0], line_dash="dot", line_color="orange", annotation_text="귀사 단가")
                    st.plotly_chart(fig_box, use_container_width=True)
                else: st.warning("데이터가 부족하여 AI 기반 그룹핑을 수행할 수 없습니다.")
            
            # 공급망 분석
            st.markdown("---")
            st.markdown("#### PART 2. 공급망 분석 및 비용 절감 시뮬레이션")
            if not s_res or s_res['alternatives'].empty: st.info("현재 거래 조건보다 더 저렴한 대안 공급처를 찾지 못했습니다.")
            else:
                alts, best_deal = s_res['alternatives'], s_res['alternatives'].iloc[0]
                st.success(f"**비용 절감 기회 포착!** 현재 거래처보다 **최대 {best_deal['price_saving_pct']:.1f}%** 저렴한 대체 거래처가 존재합니다.")
                col1, col2 = st.columns(2)
                target_saving_pct = col1.slider("목표 단가 절감률(%)", 0.0, float(best_deal['price_saving_pct']), float(best_deal['price_saving_pct'] / 2), 0.5, "%.1f%%")
                expected_saving = s_res['user_total_volume'] * s_res['user_avg_price'] * (target_saving_pct / 100)
                col2.metric(f"예상 절감액 (수입량 {s_res['user_total_volume']:,.0f}KG 기준)", f"${expected_saving:,.0f}")
                
                st.markdown("##### **추천 대체 공급처 리스트** (안정성 함께 고려)")
                recommended_list = alts[alts['price_saving_pct'] >= target_saving_pct].copy()
                recommended_list.rename(columns={'exporter': '수출업체', 'avg_unitprice': '평균 단가', 'price_saving_pct': '가격 경쟁력(%)', 'trade_count': '거래 빈도', 'num_importers': '거래처 수', 'stability_score': '공급 안정성'}, inplace=True)
                st.dataframe(
                    recommended_list[['수출업체', '평균 단가', '가격 경쟁력(%)', '거래 빈도', '거래처 수', '공급 안정성']], use_container_width=True,
                    column_config={
                        "평균 단가": st.column_config.NumberColumn(format="$%.2f"),
                        "가격 경쟁력(%)": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=alts['price_saving_pct'].max()),
                        "공급 안정성": st.column_config.BarChartColumn(y_min=0, y_max=alts['stability_score'].max())
                    }, hide_index=True
                )
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
