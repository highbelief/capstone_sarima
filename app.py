from flask import Flask, jsonify, render_template_string
import pandas as pd
import numpy as np
import pymysql
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from apscheduler.schedulers.background import BackgroundScheduler
import pytz
import requests
from bs4 import BeautifulSoup
from sqlalchemy import create_engine

# Flask 앱 생성
app = Flask(__name__)
KST = pytz.timezone("Asia/Seoul")

# DB 연결 설정
DB_CONFIG = {
    "host": "localhost",
    "user": "solar_user",
    "password": "solar_pass_2025",
    "database": "solar_forecast_muan",
    "charset": "utf8mb4"
}

# SQLAlchemy 엔진 생성
SQLALCHEMY_DB_URL = "mysql+pymysql://solar_user:solar_pass_2025@localhost/solar_forecast_muan"
engine = create_engine(SQLALCHEMY_DB_URL)

# MAPE 정의
def mean_absolute_percentage_error(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    epsilon = np.finfo(np.float64).eps
    return np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), epsilon))) * 100

# 전체 성능 지표 계산 함수
def evaluate_overall_performance(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # 0이 아닌 값만 필터링하여 MAPE 안정화
    nonzero_mask = y_true > 0
    y_true_filtered = y_true[nonzero_mask]
    y_pred_filtered = y_pred[nonzero_mask]

    rmse = np.sqrt(mean_squared_error(y_true_filtered, y_pred_filtered))
    mae = mean_absolute_error(y_true_filtered, y_pred_filtered)
    mape = mean_absolute_percentage_error(y_true_filtered, y_pred_filtered)
    return rmse, mae, mape

# 기상청 7일 예보 크롤링 함수 (무작위 값 사용 중)
def crawl_weather_forecast():
    url = "https://www.weather.go.kr/w/index.do#dong/4684033000/34.90858290832377/126.43440261942119"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")

    dates = []
    irradiance = []
    temperature = []
    wind = []

    for i in range(7):
        day = datetime.now() + timedelta(days=i)
        dates.append(day.strftime("%Y-%m-%d"))
        irradiance.append(np.random.uniform(100, 800))
        temperature.append(np.random.uniform(10, 30))
        wind.append(np.random.uniform(1, 5))

    df = pd.DataFrame({
        "measured_at": dates,
        "forecast_irradiance_wm2": irradiance,
        "forecast_temperature_c": temperature,
        "forecast_wind_speed_ms": wind
    })
    df["measured_at"] = pd.to_datetime(df["measured_at"])
    return df

# 날씨 데이터 DB 저장
def insert_forecast_to_db(df):
    conn = pymysql.connect(**DB_CONFIG)
    cursor = conn.cursor()
    for _, row in df.iterrows():
        cursor.execute("""
            INSERT INTO measurement (measured_at, forecast_irradiance_wm2, forecast_temperature_c, forecast_wind_speed_ms)
            VALUES (%s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                forecast_irradiance_wm2 = VALUES(forecast_irradiance_wm2),
                forecast_temperature_c = VALUES(forecast_temperature_c),
                forecast_wind_speed_ms = VALUES(forecast_wind_speed_ms)
        """, (row["measured_at"], row["forecast_irradiance_wm2"], row["forecast_temperature_c"], row["forecast_wind_speed_ms"]))
    conn.commit()
    conn.close()

# 일별 데이터 불러오기
def load_daily_data():
    query = """
        SELECT 
            DATE(measured_at) AS date,
            SUM(CASE WHEN HOUR(measured_at) BETWEEN 7 AND 20 THEN power_mw ELSE 0 END) AS power_mw,
            AVG(forecast_irradiance_wm2) AS forecast_irradiance,
            AVG(forecast_temperature_c) AS forecast_temperature,
            AVG(forecast_wind_speed_ms) AS forecast_wind
        FROM measurement
        WHERE measured_at <= CURDATE() + INTERVAL 7 DAY
        GROUP BY DATE(measured_at)
        ORDER BY date
    """
    df = pd.read_sql(query, engine)
    df["date"] = pd.to_datetime(df["date"])
    df.index = df["date"]
    df.drop(columns=["date"], inplace=True)
    return df

# 예측 결과 저장
def save_forecast_daily(forecast_date, predicted_mwh, actual_mwh=None, rmse=None, mae=None, mape=None):
    conn = pymysql.connect(**DB_CONFIG)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM forecast_sarima WHERE forecast_start = %s", (forecast_date.date(),))
    cursor.execute("""
        INSERT INTO forecast_sarima (forecast_start, forecast_end, predicted_mwh, actual_mwh, rmse, mae, mape, created_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, NOW())
    """, (forecast_date.date(), forecast_date.date(), predicted_mwh, actual_mwh, rmse, mae, mape))
    conn.commit()
    conn.close()

# 예측 수행 함수 (오늘 포함 7일)
def run_sarima_forecast():
    try:
        df = load_daily_data()
        today = pd.to_datetime(pd.Timestamp.now(tz=KST).date())

        train = df[df.index < today]
        future = df[df.index >= today]

        # 로그 변환 제거로 안정성 향상
        train_y = train["power_mw"]
        train_y = train_y.asfreq("D")

        # 외생변수 제거: SARIMA 내부 시계열만 사용
        exog_cols = None
        train_exog = None
        future_exog = None

        n_forecast = 7
        if future.shape[0] < n_forecast:
            return f"❌ 예측에 필요한 데이터가 부족합니다. 최소 {n_forecast}일치가 필요하지만 현재 {future.shape[0]}일치만 존재합니다."

        model = SARIMAX(train_y,
                        order=(2,1,2),
                        seasonal_order=(1,1,1,7),
                        enforce_stationarity=False,
                        enforce_invertibility=False)
        model_fit = model.fit(disp=False)
        forecast_log = model_fit.forecast(steps=n_forecast)
        MAX_CAPACITY_PER_DAY_MWH = 4000  # 상한선 4000MWh로 고정
        max_train_value = min(train["power_mw"].max() * 1.2, MAX_CAPACITY_PER_DAY_MWH)
        forecast_mwh = forecast_log.clip(0, max_train_value)
        if (forecast_mwh >= max_train_value).any():
            print(f"⚠️ 일부 예측값이 설비 한계({max_train_value:.2f} MWh/day)를 초과하여 clip 되었습니다.")
        actual_mwh = future["power_mw"][:n_forecast] if "power_mw" in future else np.full(n_forecast, np.nan)

        y_true, y_pred = [], []

        for i in range(n_forecast):
            date = future.index[i]
            predicted = float(forecast_mwh.iloc[i])
            actual = float(actual_mwh.iloc[i]) if hasattr(actual_mwh, 'iloc') and not np.isnan(actual_mwh.iloc[i]) else None
            if actual is not None:
                rmse = np.sqrt(mean_squared_error([actual], [predicted]))
                mae = mean_absolute_error([actual], [predicted])
                mape = mean_absolute_percentage_error([actual], [predicted])
                y_true.append(actual)
                y_pred.append(predicted)
            else:
                rmse = mae = mape = None
            save_forecast_daily(date, predicted, actual, rmse, mae, mape)

        if y_true and y_pred:
            overall_rmse, overall_mae, overall_mape = evaluate_overall_performance(y_true, y_pred)
            print("✅ 전체 예측 성능:")
            print(f"RMSE: {overall_rmse:.2f} | MAE: {overall_mae:.2f} | MAPE: {overall_mape:.2f}%")

        return forecast_mwh.tolist()
    except Exception as e:
        return str(e)

@app.route("/forecast/sarima", methods=["GET"])
def forecast_sarima():
    try:
        df = crawl_weather_forecast()
        insert_forecast_to_db(df)
        result = run_sarima_forecast()
        if isinstance(result, str):
            return jsonify({"status": "error", "message": result})

        today = datetime.now(KST).date()
        dates = [(today + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(len(result))]
        rows = zip(dates, result)

        html = """
        <h1>예측 결과 (SARIMA)</h1>
        <table border='1' style='border-collapse: collapse; text-align: center;'>
            <tr><th>날짜</th><th>예측 발전량 (MWh)</th></tr>
            {% for date, value in rows %}
            <tr><td>{{ date }}</td><td>{{ value | round(2) }}</td></tr>
            {% endfor %}
        </table>
        <br><a href='/'>⬅ 메인으로 돌아가기</a>
        """
        return render_template_string(html, rows=rows)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route("/")
def index():
    return """
    <h1>SARIMA 예측 시스템</h1>
    <form action='/forecast/sarima'>
        <button type='submit'>예측하기 (크롤링 + 연산)</button>
    </form>
    """

# 스케줄러
scheduler = BackgroundScheduler(timezone=KST)
scheduler.add_job(lambda: insert_forecast_to_db(crawl_weather_forecast()), 'cron', hour=6, minute=30)
scheduler.add_job(run_sarima_forecast, 'cron', hour=7, minute=30)
scheduler.start()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
