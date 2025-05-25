import pymysql
import pandas as pd
from joblib import load
from datetime import datetime, timedelta
import pytz
from flask import Flask
from apscheduler.schedulers.background import BackgroundScheduler

# Flask 애플리케이션 초기화
app = Flask(__name__)
KST = pytz.timezone("Asia/Seoul")  # 한국 표준시 (KST) 설정

# DB 접속 정보 설정
DB_CONFIG = {
    'host': 'localhost',
    'user': 'solar_user',
    'password': 'solar_pass_2025',
    'db': 'solar_forecast_muan',
    'charset': 'utf8mb4',
    'cursorclass': pymysql.cursors.DictCursor  # SELECT 결과를 딕셔너리 형태로 반환
}

# 1. 실측 누적 발전량 데이터를 DB에서 시간순으로 불러오는 함수
def load_measurements():
    conn = pymysql.connect(**DB_CONFIG)
    query = """
        SELECT measured_at, cumulative_kwh
        FROM measurement
        WHERE cumulative_kwh IS NOT NULL
        ORDER BY measured_at
    """
    df = pd.read_sql(query, conn, parse_dates=['measured_at'])  # measured_at 컬럼을 datetime으로 파싱
    conn.close()
    return df

# 2. 예측 결과를 forecast_sarima 테이블에 저장하는 함수
def save_forecast_to_db(start_date, end_date, predicted_kwh):
    conn = pymysql.connect(**DB_CONFIG)
    with conn.cursor() as cursor:
        cursor.execute("""
            INSERT INTO forecast_sarima (forecast_start, forecast_end, predicted_kwh, created_at)
            VALUES (%s, %s, %s, NOW())
        """, (start_date, end_date, predicted_kwh))
    conn.commit()
    conn.close()

# 3. 저장된 SARIMA 모델을 불러와서 6일간의 누적 발전량을 예측하는 함수
def run_sarima_forecast():
    df = load_measurements()  # 실측 데이터를 불러옴
    if df.empty:
        return None, None, "❌ 실측 데이터가 없습니다."

    try:
        model = load("sarima_model.pkl")  # joblib으로 학습된 SARIMA 모델 불러오기
        forecast = model.predict(n_periods=6*24)  # 하루 24시간 기준으로 6일치 예측 수행
        predicted_sum = float(forecast.sum())  # 예측값을 누적하여 총 발전량 계산

        # 예측 시작일은 마지막 실측일 기준 다음날부터, 종료일은 6일 후까지
        forecast_start = df["measured_at"].iloc[-1].date() + timedelta(days=1)
        forecast_end = forecast_start + timedelta(days=5)

        save_forecast_to_db(forecast_start, forecast_end, predicted_sum)  # DB에 결과 저장
        return forecast_start, forecast_end, predicted_sum

    except Exception as e:
        # 모델 로드 또는 예측 중 오류가 발생한 경우
        return None, None, f"❌ 예측 오류 발생: {e}"

# 4. 루트 라우트: 웹 브라우저에서 접속 시 예측을 수행하고 결과를 HTML로 반환
@app.route("/")
def index():
    try:
        start, end, result = run_sarima_forecast()
        if start is None:
            return f"<h1>예측 실패</h1><p>{result}</p>", 400  # 오류 메시지 출력
        return f"""
            <h1>SARIMA 예측 결과</h1>
            <p><strong>예측 시작일:</strong> {start}</p>
            <p><strong>예측 종료일:</strong> {end}</p>
            <p><strong>예측 누적 발전량 (kWh):</strong> {result:.2f}</p>
        """
    except Exception as e:
        # 예외 발생 시 예외 내용을 HTML로 출력하여 디버깅 가능
        return f"<h1>500 내부 오류 발생</h1><pre>{e}</pre>", 500

# 5. 스케줄러: 매일 오전 7시 30분에 SARIMA 예측 자동 실행
def start_scheduler():
    scheduler = BackgroundScheduler(timezone=KST)
    scheduler.add_job(run_sarima_forecast, 'cron', hour=7, minute=30)
    scheduler.start()

# 6. 직접 실행 시 콘솔에 예측 결과 출력 및 스케줄러 시작
if __name__ == "__main__":
    print("✅ SARIMA 백엔드 시작 및 스케줄러 등록")
    start_scheduler()  # 07:30 자동 실행 스케줄러 시작
    app.run(host="0.0.0.0", port=5000)
