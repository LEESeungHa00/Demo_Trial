# 📈 수입 경쟁력 진단 솔루션

> **Google BigQuery의 방대한 수입 데이터를 기반으로, 개별 거래의 경쟁력을 심층 분석하고 데이터 기반의 구매 전략을 제시하는 솔루션입니다.**
> 시장 동향, 경쟁 환경, 대체 공급망 분석을 통해 잠재적인 비용 절감 기회를 발견하세요.

<br>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python Badge"/>
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit Badge"/>
  <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas Badge"/>
  <img src="https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white" alt="Plotly Badge"/>
  <img src="https://img.shields.io/badge/Google%20BigQuery-4285F4?style=for-the-badge&logo=google-bigquery&logoColor=white" alt="Google BigQuery Badge"/>
  <img src="https://img.shields.io/badge/Google%20Sheets-34A853?style=for-the-badge&logo=google-sheets&logoColor=white" alt="Google Sheets Badge"/>
</p>

---

## ✨ 주요 기능

이 솔루션은 사용자가 입력한 수입 내역을 바탕으로 다각적인 분석 리포트를 제공합니다.

### 1. 시장 개요 및 트렌드 분석
*입력한 품목(HS-Code)의 전체 시장 동향을 파악하여 거시적인 관점을 제공합니다.*
- **시장 규모 및 성장률**: 전년 대비 수입 중량 및 평균 단가의 변화를 시각적으로 보여줍니다.
- **제품 구성 분석**: 해당 HS-Code 내에서 가장 많이 수입되는 상위 10개 제품을 파이 차트로 분석합니다.
- **시계열 동향**: 과거 모든 시장 거래를 버블 차트로 시각화하여, 시장의 가격 변동성과 월별 평균가를 한눈에 파악할 수 있습니다.

### 2. 구매 경쟁력 심층 진단
*입력한 거래가 시장 평균 대비 얼마나 경쟁력 있는지 정량적으로 진단합니다.*
- **가격 경쟁력 순위**: 동월에 발생한 모든 거래와 비교하여 입력값의 가격 순위를 백분위로 제시합니다.
- **잠재적 비용 절감액**: 시장 상위 10% 수준의 단가로 구매했을 경우 절감할 수 있는 잠재적 비용을 계산합니다.

### 3. 경쟁 환경 및 포지셔닝 분석
*시장 내 다른 수입사들과의 경쟁 구도를 분석하여 귀사의 전략적 위치를 파악합니다.*
- **수입사 포지셔닝 맵**: '총 수입 중량'과 시점 요인을 보정한 '가격 경쟁력 지수'를 기준으로 시장 내 모든 경쟁사와 귀사의 위치를 시각화합니다.
- **실질 경쟁 그룹 비교**: `시장 선도 그룹`, `유사 규모 경쟁 그룹`, `최저가 달성 그룹` 등 의미 있는 경쟁 그룹을 자동으로 분류하고, 그룹별 가격 경쟁력 분포를 비교 분석합니다.

### 4. 공급망 최적화 및 성과 추적
*더 나은 거래 조건을 가진 대체 공급처를 발굴하고, 과거 구매 성과를 추적합니다.*
- **대체 공급처 추천**: 현재 거래보다 저렴한 단가를 제공하는 대체 공급처 리스트를 제시하고, '공급 안정성'을 함께 분석하여 합리적인 의사결정을 돕습니다.
- **비용 절감 시뮬레이션**: 목표 단가 절감률에 따른 예상 절감액을 실시간으로 시뮬레이션합니다.
- **과거 성과 대시보드**: 입력한 과거 거래 내역 전체의 '가격 경쟁력 지수' 추이를 경쟁 그룹과 비교하여 장기적인 구매 성과를 평가합니다.

---

## 🚀 시작하기

### **사전 준비**
- Python 3.8 이상 설치
- Google Cloud Platform(GCP) 프로젝트 설정
  - 서비스 계정(Service Account) 생성 및 **JSON 키 파일** 다운로드
  - **BigQuery API**, **Google Drive API**, **Google Sheets API** 활성화
- 분석 결과를 저장할 Google Sheets 파일 생성 및 서비스 계정에 **편집자(Editor)** 권한 공유

### **설치 및 실행**
1.  **저장소 복제 및 이동**:
    ```bash
    git clone [Your_Repository_URL]
    cd [Your_Project_Directory]
    ```

2.  **필요한 라이브러리 설치**:
    *프로젝트 폴더에 `requirements.txt` 파일을 생성하고 아래 내용을 추가한 후, 터미널에서 설치 명령어를 실행하세요.*

    **requirements.txt**:
    ```
    streamlit
    pandas
    numpy
    plotly
    gspread
    google-auth-oauthlib
    google-api-python-client
    pandas-gbq
    openpyxl
    ```

    **터미널**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **인증 정보 및 설정**:
    프로젝트 폴더 내에 `.streamlit/secrets.toml` 파일을 생성하고 아래 내용을 채워넣습니다.
    ```toml
    # .streamlit/secrets.toml

    # GCP 서비스 계정 JSON 키 파일 내용 전체 복사
    [gcp_service_account]
    type = "service_account"
    project_id = "your-gcp-project-id"
    private_key_id = "..."
    private_key = "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n"
    client_email = "your-service-account@your-project.iam.gserviceaccount.com"
    client_id = "..."
    auth_uri = "[https://accounts.google.com/o/oauth2/auth](https://accounts.google.com/o/oauth2/auth)"
    token_uri = "[https://oauth2.googleapis.com/token](https://oauth2.googleapis.com/token)"
    auth_provider_x509_cert_url = "[https://www.googleapis.com/oauth2/v1/certs](https://www.googleapis.com/oauth2/v1/certs)"
    client_x509_cert_url = "..."

    # 앱 접속 비밀번호 및 Google Sheets 정보
    [app_secrets]
    password = "your_strong_password"

    [google_sheets]
    spreadsheet_name = "Your_Google_Sheet_Name"
    worksheet_name = "Your_Target_Worksheet_Name"
    ```

4.  **애플리케이션 실행**:
    터미널에서 아래 명령어를 실행합니다.
    ```bash
    streamlit run your_app_name.py
    ```
