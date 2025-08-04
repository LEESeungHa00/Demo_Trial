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
            importer_median_prices = rela
