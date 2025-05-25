# 📘 SARIMA Backend for Solar Forecast

이 프로젝트는 사전 학습된 SARIMA 모델을 사용하여 무안군 태양광 발전소의 6일 누적 발전량(중기 예측)을 수행하고, 결과를 MySQL 데이터베이스에 저장하는 Flask 기반 백엔드 서버입니다.

> ⚠️ **주의:** `sarima_model.pkl` 파일은 GitHub의 파일 크기 제한(100MB)을 초과하므로 저장소에 포함되어 있지 않습니다.
> 모델은 별도로 제공되는 `capstone_ml` 프로젝트 내 학습 코드에서 직접 생성하여 사용해야 합니다.

---

## 🚀 주요 기능

* `sarima_model.pkl` 로드 후 예측 수행 (6일치, 시간 단위)
* MySQL DB에서 실측 데이터를 불러와 예측에 활용
* `forecast_sarima` 테이블에 누적 발전량 저장
* `/` 접속 시 웹에서 HTML로 예측 결과 확인
* APScheduler를 통해 **매일 오전 7시 30분 자동 예측 실행**

---

## 📁 프로젝트 구조

```bash
sarima_backend/
├── sarima_backend.py        # Flask 웹 서버 및 예측 로직
├── sarima_model.pkl         # 학습된 SARIMA 모델 파일 (직접 생성 필요)
├── requirements.txt         # 필요한 Python 패키지 목록
```

---

## ⚙️ 환경 설정

### 1. 패키지 설치

```bash
pip install -r requirements.txt
```

### 2. MySQL 설정 예시

```sql
CREATE DATABASE IF NOT EXISTS solar_forecast_muan DEFAULT CHARACTER SET utf8mb4;

CREATE USER IF NOT EXISTS 'solar_user'@'localhost' IDENTIFIED BY 'solar_pass_2025';
GRANT ALL PRIVILEGES ON solar_forecast_muan.* TO 'solar_user'@'localhost';
FLUSH PRIVILEGES;
```

### 3. 테이블 구조 예시

```sql
CREATE TABLE IF NOT EXISTS measurement (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    measured_at DATETIME NOT NULL,
    cumulative_kwh FLOAT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS forecast_sarima (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    forecast_start DATE NOT NULL,
    forecast_end DATE NOT NULL,
    predicted_kwh FLOAT NOT NULL,
    actual_kwh FLOAT,
    rmse FLOAT,
    mae FLOAT,
    mape FLOAT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

---

## 🖥️ 실행 방법

### 1. 서버 실행

```bash
python sarima_backend.py
```

### 2. 웹 접속

```url
http://localhost:5000
```

웹 브라우저에서 위 주소로 접속하면 최신 예측 결과를 확인할 수 있습니다.

### 3. 자동 실행 스케줄 확인

* APScheduler가 매일 오전 7:30에 자동으로 예측 수행
* 콘솔에 결과 출력 및 DB 저장

---

## 🧪 예시 출력 (웹)

```
SARIMA 예측 결과
예측 시작일: 2024-12-12
예측 종료일: 2024-12-17
예측 누적 발전량 (kWh): 864.12
```

---

## 📬 문의

* 담당자: 국립목포대학교 공과대학 융합소프트웨어학과 스포트라이트팀 팀장
* 버그 리포트 또는 기능 요청은 GitHub Issue 또는 이메일로 전달해주세요.
