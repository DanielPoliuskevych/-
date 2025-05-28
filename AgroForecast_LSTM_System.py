import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datetime import datetime
from meteostat import Point, Daily, Hourly
import os
import pickle
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter.messagebox as msgbox

# Общие функции для работы с данными
# Переменная-флаг: использовать кэш или нет
use_cached_data = None

def ask_update_permission():
    return msgbox.askyesno(
        "Select Data Source",
        "Do you want saved data to be used?\n\n"
        "Press 'Yes' — use old data\n"
        "Press 'No' — download new data"
    )

def fetch_hourly_data(data_type, location, start, end, city_name="unknown"):
    import os
    import pandas as pd
    from meteostat import Hourly

    global use_cached_data
    filename = f"data_cache/{city_name}_hourly.csv"
    os.makedirs("data_cache", exist_ok=True)

    if use_cached_data is None:
        use_cached_data = ask_update_permission()

    if use_cached_data and os.path.exists(filename):
        data = pd.read_csv(filename, index_col=0, parse_dates=True)
    else:
        data = Hourly(location, start, end).fetch()
        data.to_csv(filename)

    if data_type in data.columns:
        daily_max = data[data_type].groupby(data.index.date).mean()
        return daily_max.bfill().values
    else:
        raise ValueError(f"Column '{data_type}' not found in the dataset.")

def fetch_daily_data(data_type, location, start, end, city_name="unknown"):
    import os
    import pandas as pd
    from meteostat import Daily

    global use_cached_data
    filename = f"data_cache/{city_name}_daily.csv"
    os.makedirs("data_cache", exist_ok=True)

    if use_cached_data is None:
        use_cached_data = ask_update_permission()

    if use_cached_data and os.path.exists(filename):
        data = pd.read_csv(filename, index_col=0, parse_dates=True)
    else:
        data = Daily(location, start, end).fetch()
        data.to_csv(filename)

    if data_type in data.columns:
        return data[data_type].bfill().values
    else:
        raise ValueError(f"Column '{data_type}' not found in the dataset.")




def decompose_and_prepare(data, period=365):
    data_series = pd.Series(data).interpolate().bfill().ffill()
    result = seasonal_decompose(data_series, period=period, model='additive', extrapolate_trend='freq')
    trend = pd.Series(result.trend).interpolate().values
    seasonality = result.seasonal
    data_min = trend.min()
    data_max = trend.max()
    data_scaled = (trend - data_min) / (data_max - data_min)
    return data_scaled, seasonality, data_min, data_max


def create_sequences(data, seq_length):
    sequences, targets = [], []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i + seq_length])
        targets.append(data[i + seq_length])
    return np.array(sequences), np.array(targets)


def prepare_tensors(X, y):
    X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    return X_tensor, y_tensor


# Определение модели LSTM
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


def train_model(X_train, y_train, input_dim=1, hidden_dim=64, output_dim=1, num_layers=2, num_epochs=50, lr=0.001):
    model = LSTMModel(input_dim, hidden_dim, output_dim, num_layers)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs.squeeze(), y_train)
        loss.backward()
        optimizer.step()
    return model


def forecast(model, data_scaled, seq_length, forecast_days, data_min, data_max, seasonality):
    input_seq = data_scaled[-seq_length:]
    forecast = []
    model.eval()
    with torch.no_grad():
        for _ in range(forecast_days):
            input_tensor = torch.tensor(input_seq.reshape(1, seq_length, 1), dtype=torch.float32)
            pred = model(input_tensor).item()
            forecast.append(pred)
            input_seq = np.append(input_seq[1:], pred)
    forecast = np.array(forecast) * (data_max - data_min) + data_min
    return forecast + seasonality[-forecast_days:]


# Функция для получения рекомендаций по сельскохозяйственным действиям
def show_recommendations():
    # Получаем средние значения прогноза
    avg_temp = global_forecast_temp[0]
    avg_prcp = global_forecast_prcp[0]
    avg_rhum = global_forecast_rhum[0]
    avg_wind = global_forecast_wind[0]

    # Создаем новое окно для рекомендаций
    recommendation_window = tk.Toplevel(root)
    recommendation_window.title("Sowing Recommendations")
    recommendation_window.geometry("700x600")

    # Выводим средние значения прогноза
    avg_values_label = tk.Label(
        recommendation_window,
        text=f"Forecast for the next day:\n"
             f"Temperature: {avg_temp:.1f}°C\n"
             f"Humidity: {avg_rhum:.1f}%\n"
             f"Precipitation: {avg_prcp:.1f} mm/day\n"
             f"Wind Speed: {avg_wind:.1f} m/s",
        font=("Arial", 12),
        justify=tk.LEFT,
    )
    avg_values_label.pack(pady=10)

    # Список культур с их оптимальными условиями для выращивания
    crops = [
        {
            "name": "Пшениця",
            "temp_range": (15, 25),
            "humidity_range": (50, 70),
            "precipitation_range": (1, 5),
            "wind_range": (0, 20),
        },
        {
            "name": "Кукурудза",
            "temp_range": (18, 32),
            "humidity_range": (50, 70),
            "precipitation_range": (0.5, 7),
            "wind_range": (0, 20),
        },
        {
            "name": "Соняшник",
            "temp_range": (20, 30),
            "humidity_range": (40, 60),
            "precipitation_range": (0.5, 5),
            "wind_range": (0, 20),
        },
        {
            "name": "Ячмінь",
            "temp_range": (12, 25),
            "humidity_range": (50, 70),
            "precipitation_range": (1, 5),
            "wind_range": (0, 20),
        },
        {
            "name": "Соя",
            "temp_range": (20, 30),
            "humidity_range": (50, 70),
            "precipitation_range": (2, 7),
            "wind_range": (0, 20),
        },
        {
            "name": "Картопля",
            "temp_range": (15, 20),
            "humidity_range": (60, 80),
            "precipitation_range": (2, 7),
            "wind_range": (0, 20),
        },
        {
            "name": "Цукровий буряк",
            "temp_range": (15, 25),
            "humidity_range": (60, 80),
            "precipitation_range": (2, 7),
            "wind_range": (0, 20),
        },
        {
            "name": "Гречка",
            "temp_range": (15, 25),
            "humidity_range": (50, 70),
            "precipitation_range": (1, 5),
            "wind_range": (0, 20),
        },
        {
            "name": "Овес",
            "temp_range": (10, 20),
            "humidity_range": (50, 70),
            "precipitation_range": (1, 5),
            "wind_range": (0, 20),
        },
        {
            "name": "Жито",
            "temp_range": (12, 22),
            "humidity_range": (50, 70),
            "precipitation_range": (1, 5),
            "wind_range": (0, 20),
        },
        {
            "name": "Ріпак",
            "temp_range": (15, 25),
            "humidity_range": (50, 70),
            "precipitation_range": (1, 5),
            "wind_range": (0, 20),
        },
        {
            "name": "Горох",
            "temp_range": (15, 25),
            "humidity_range": (50, 70),
            "precipitation_range": (1, 5),
            "wind_range": (0, 20),
        },
        {
            "name": "Льон",
            "temp_range": (15, 25),
            "humidity_range": (50, 70),
            "precipitation_range": (1, 5),
            "wind_range": (0, 20),
        },
        {
            "name": "Рижик",
            "temp_range": (10, 20),
            "humidity_range": (50, 70),
            "precipitation_range": (1, 5),
            "wind_range": (0, 20),
        },
    ]

    # Создаем текстовое поле для вывода параметров культур
    crops_text = tk.Text(recommendation_window, wrap=tk.WORD, height=15, width=80)
    crops_text.pack(pady=10, padx=10)

    # Выводим параметры каждой культуры
    crops_text.insert(tk.END, "Crop parameters:\n\n")
    for crop in crops:
        crops_text.insert(
            tk.END,
            f"{crop['name']}:\n"
            f"  Temperature: {crop['temp_range'][0]}–{crop['temp_range'][1]}°C\n"
            f"  Humidity: {crop['humidity_range'][0]}–{crop['humidity_range'][1]}%\n"
            f"  Precipitation: {crop['precipitation_range'][0]}–{crop['precipitation_range'][1]} mm/day\n"
            f"  Wind Speed: {crop['wind_range'][0]}–{crop['wind_range'][1]} m/s\n\n",
        )
    crops_text.config(state=tk.DISABLED)  # Запрещаем редактирование

    # Sowing Recommendations
    recommendations = []
    for crop in crops:
        # Расчёт допустимых диапазонов с отклонением 10%
        temp_min = crop["temp_range"][0] * 0.9
        temp_max = crop["temp_range"][1] * 1.1
        humidity_min = crop["humidity_range"][0] * 0.9
        humidity_max = crop["humidity_range"][1] * 1.1
        precipitation_min = crop["precipitation_range"][0] * 0.9
        precipitation_max = crop["precipitation_range"][1] * 1.1
        wind_min = crop["wind_range"][0] * 0.9
        wind_max = crop["wind_range"][1] * 1.1

        # Проверка условий
        if (
                temp_min <= avg_temp <= temp_max
                and humidity_min <= avg_rhum <= humidity_max
                and precipitation_min <= avg_prcp <= precipitation_max
                and wind_min <= avg_wind <= wind_max
        ):
            recommendations.append(
                f"✅ {crop['name']} — Suitable conditions for sowing"
            )

    # Вывод рекомендаций
    recommendation_label = tk.Label(
        recommendation_window,
        text="Sowing recommendations:",
        font=("Arial", 14, "bold"),
    )
    recommendation_label.pack(pady=10)

    if recommendations:
        for rec in recommendations:
            rec_label = tk.Label(
                recommendation_window,
                text=rec,
                font=("Arial", 12),
                anchor='w',
                justify=tk.LEFT
            )
            rec_label.pack(padx=20, anchor='w')
    else:
        no_rec_label = tk.Label(
            recommendation_window,
            text="🚫 According to current weather conditions, it is not recommended to sow any of the suggested crops.",
            font=("Arial", 12),
            fg="red",
            wraplength=600,
            justify=tk.LEFT
        )
        no_rec_label.pack(padx=20, pady=10)

def show_historical_data():
    city_name = city_combobox.get()
    location = cities[city_name]

    # Параметры
    today = datetime.today()
    end = today - pd.Timedelta(days=1)
    start = end - pd.Timedelta(days=30)

    # Исторические данные
    rhum_data = fetch_hourly_data('rhum', location, start, end, city_name)
    temp_data = fetch_daily_data('tavg', location, start, end, city_name)
    prcp_data = fetch_daily_data('prcp', location, start, end, city_name)
    wdsp_data = fetch_daily_data('wspd', location, start, end, city_name)

    # Подготовка данных для графиков
    forecast_dates = [start + pd.Timedelta(days=i) for i in range(0, 30)]

    # Убедимся, что количество данных совпадает
    if len(rhum_data) != len(forecast_dates):
        rhum_data = rhum_data[:len(forecast_dates)]
    if len(temp_data) != len(forecast_dates):
        temp_data = temp_data[:len(forecast_dates)]
    if len(prcp_data) != len(forecast_dates):
        prcp_data = prcp_data[:len(forecast_dates)]
    if len(wdsp_data) != len(forecast_dates):
        wdsp_data = wdsp_data[:len(forecast_dates)]

    # Создание нового окна для графиков
    new_window = tk.Toplevel(root)
    new_window.title(f"Historical Data for City {city_name}")

    # Создание фигуры для графиков
    figure_hist = plt.Figure(figsize=(8, 6), dpi=100)
    canvas_hist = FigureCanvasTkAgg(figure_hist, master=new_window)
    canvas_hist.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # Создаем подграфики (subplots)
    ax1 = figure_hist.add_subplot(221)  # График влажности
    ax2 = figure_hist.add_subplot(223)  # График температуры
    ax3 = figure_hist.add_subplot(222)  # График осадков
    ax4 = figure_hist.add_subplot(224)  # График скорости ветра

    # Humidity
    ax1.plot(forecast_dates, rhum_data, color="blue")
    ax1.set_title("Humidity (30 days)")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Humidity (%)")
    ax1.set_xticks(forecast_dates[::7])  # Даты раз в 7 days
    ax1.set_xticklabels([date.strftime('%d-%m') for date in forecast_dates[::7]], rotation=45, fontsize=8)
    ax1.legend()
    ax1.grid()

    # Temperature
    ax2.plot(forecast_dates, temp_data, color="red")
    ax2.set_title("Temperature (30 days)")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Temperature (°C)")
    ax2.set_xticks(forecast_dates[::7])  # Даты раз в 7 days
    ax2.set_xticklabels([date.strftime('%d-%m') for date in forecast_dates[::7]], rotation=45, fontsize=8)
    ax2.legend()
    ax2.grid()

    # Precipitation
    ax3.plot(forecast_dates, prcp_data, color="green")
    ax3.set_title("Precipitation (30 days)")
    ax3.set_xlabel("Date")
    ax3.set_ylabel("Precipitation (мм)")
    ax3.set_xticks(forecast_dates[::7])  # Даты раз в 7 days
    ax3.set_xticklabels([date.strftime('%d-%m') for date in forecast_dates[::7]], rotation=45, fontsize=8)
    ax3.legend()
    ax3.grid()

    # Wind Speed
    ax4.plot(forecast_dates, wdsp_data, color="purple")
    ax4.set_title("Wind Speed (30 days)")
    ax4.set_xlabel("Date")
    ax4.set_ylabel("Wind Speed (m/s)")
    ax4.set_xticks(forecast_dates[::7])  # Даты раз в 7 days
    ax4.set_xticklabels([date.strftime('%d-%m') for date in forecast_dates[::7]], rotation=45, fontsize=8)
    ax4.legend()
    ax4.grid()

    figure_hist.tight_layout(pad=3.0)
    canvas_hist.draw()

def run_forecast():
    global global_forecast_temp, global_forecast_prcp, global_forecast_rhum, global_forecast_wind

    city_name = city_combobox.get()
    location = cities[city_name]

    # Параметры
    today = datetime.today()
    end = today - pd.Timedelta(days=1)  # Вчера
    start = end - pd.DateOffset(years=2)  # За 2 года до вчерашнего дня
    seq_length = 30
    forecast_days = 14

    # Прогноз влажности
    rhum_data = fetch_hourly_data('rhum', location, start, end, city_name)
    rhum_scaled, rhum_seasonality, rhum_min, rhum_max = decompose_and_prepare(rhum_data)
    X_rhum, y_rhum = create_sequences(rhum_scaled, seq_length)
    X_rhum_train, y_rhum_train = prepare_tensors(X_rhum[:int(len(X_rhum) * 0.8)], y_rhum[:int(len(y_rhum) * 0.8)])
    model_rhum = train_model(X_rhum_train, y_rhum_train)
    forecast_rhum = forecast(model_rhum, rhum_scaled, seq_length, forecast_days, rhum_min, rhum_max, rhum_seasonality)

    # Прогноз температуры
    temp_data = fetch_daily_data('tavg', location, start, end, city_name)
    temp_scaled, temp_seasonality, temp_min, temp_max = decompose_and_prepare(temp_data)
    X_temp, y_temp = create_sequences(temp_scaled, seq_length)
    X_temp_train, y_temp_train = prepare_tensors(X_temp[:int(len(X_temp) * 0.8)], y_temp[:int(len(y_temp) * 0.8)])
    model_temp = train_model(X_temp_train, y_temp_train)
    forecast_temp = forecast(model_temp, temp_scaled, seq_length, forecast_days, temp_min, temp_max, temp_seasonality)

    # Прогноз осадков
    prcp_data = fetch_daily_data('prcp', location, start, end, city_name)
    prcp_scaled, prcp_seasonality, prcp_min, prcp_max = decompose_and_prepare(prcp_data)
    X_prcp, y_prcp = create_sequences(prcp_scaled, seq_length)
    X_prcp_train, y_prcp_train = prepare_tensors(X_prcp[:int(len(X_prcp) * 0.8)], y_prcp[:int(len(y_prcp) * 0.8)])
    model_prcp = train_model(X_prcp_train, y_prcp_train)

    forecast_prcp = forecast(model_prcp, prcp_scaled, seq_length, forecast_days, prcp_min, prcp_max, prcp_seasonality)
    forecast_prcp = np.where(forecast_prcp < 0.2, 0, forecast_prcp)  # убираем "шум"
    global_forecast_prcp = np.array(forecast_prcp)

    # Прогноз скорости ветра
    wdsp_data = fetch_daily_data('wspd', location, start, end, city_name)
    wdsp_scaled, wdsp_seasonality, wdsp_min, wdsp_max = decompose_and_prepare(wdsp_data)
    X_wdsp, y_wdsp = create_sequences(wdsp_scaled, seq_length)
    X_wdsp_train, y_wdsp_train = prepare_tensors(X_wdsp[:int(len(X_wdsp) * 0.8)], y_wdsp[:int(len(y_wdsp) * 0.8)])
    model_wdsp = train_model(X_wdsp_train, y_wdsp_train)
    forecast_wdsp = forecast(model_wdsp, wdsp_scaled, seq_length, forecast_days, wdsp_min, wdsp_max, wdsp_seasonality)

    global_forecast_temp = np.array(forecast_temp)
    global_forecast_prcp = np.array(forecast_prcp)
    global_forecast_rhum = np.array(forecast_rhum)
    global_forecast_wind = np.array(forecast_wdsp)

    #Даты для прогноза
    forecast_dates = [end + pd.Timedelta(days=i) for i in range(1, forecast_days + 1)]

    # Построение графика
    figure.clear()

    # Создаём два подграфика (subplots)
    ax1 = figure.add_subplot(221)  # Первый подграфик для влажности
    ax2 = figure.add_subplot(223)  # Второй подграфик для температуры
    ax3 = figure.add_subplot(222)  # Третий подграфик для осадков
    ax4 = figure.add_subplot(224)  # Четвертый подграфик для скорости ветра

    # Forecast of humidity on the first graph
    ax1.plot(forecast_dates, forecast_rhum, color="blue", linestyle="--")
    ax1.set_title("Humidity Forecast")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Humidity (%)")
    ax1.set_xticks(forecast_dates)
    ax1.set_xticklabels([date.strftime('%d-%m') for date in forecast_dates], rotation=45, fontsize=8)
    ax1.legend()
    ax1.grid()

    # Forecast of temperature on the second graph
    ax2.plot(forecast_dates, forecast_temp, color="red", linestyle="--")
    ax2.set_title("Temperature Forecast")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Temperature (°C)")
    ax2.set_xticks(forecast_dates)
    ax2.set_xticklabels([date.strftime('%d-%m') for date in forecast_dates], rotation=45, fontsize=8)
    ax2.legend()
    ax2.grid()

    # Forecast of precipitation on the third graph
    ax3.plot(forecast_dates, forecast_prcp, color="green", linestyle="--")
    ax3.set_title("Precipitation Forecast")
    ax3.set_xlabel("Date")
    ax3.set_ylabel("Precipitation (mm)")
    ax3.set_xticks(forecast_dates)
    ax3.set_xticklabels([date.strftime('%d-%m') for date in forecast_dates], rotation=45, fontsize=8)
    ax3.legend()
    ax3.grid()

    # Forecast of wind speed on the fourth graph
    ax4.plot(forecast_dates, forecast_wdsp, color="purple", linestyle="--")
    ax4.set_title("Wind Speed Forecast")
    ax4.set_xlabel("Date")
    ax4.set_ylabel("Speed (m/s)")
    ax4.set_xticks(forecast_dates)
    ax4.set_xticklabels([date.strftime('%d-%m') for date in forecast_dates], rotation=45, fontsize=8)
    ax4.legend()
    ax4.grid()

    figure.tight_layout(pad=3.0)
    canvas.draw()

    # Сохраняем прогнозы в глобальные переменные для использования в рекомендациях


    # Активируем кнопку для рекомендаций
    recommendation_button.config(state=tk.NORMAL)


# Функция для вызова рекомендаций

# Данные городов
cities = {
    "Харків": Point(49.9935, 36.2304, 100),
    "Київ": Point(50.4501, 30.5234, 100),
    "Одеса": Point(46.4825, 30.7233, 50),
    "Дніпро": Point(48.4647, 35.0462, 155),
    "Запоріжжя": Point(47.8388, 35.1396, 50),
    "Львів": Point(49.8397, 24.0297, 296),
    "Кривий Ріг": Point(47.9105, 33.3918, 125),
}

# Создание окна
root = tk.Tk()
root.title("Weather Forecast")

# Выбор города
tk.Label(root, text="Select a city:").pack(pady=5)
city_combobox = ttk.Combobox(root, values=list(cities.keys()), state="readonly")
city_combobox.pack(pady=5)
city_combobox.set("Харків")

# Кнопка для запуска прогноза
tk.Button(root, text="Generate Forecast", command=run_forecast).pack(pady=10)
tk.Button(root, text="Show Historical Data", command=show_historical_data).pack(pady=10)

# Кнопка для рекомендаций (изначально неактивна)
recommendation_button = tk.Button(root, text="Get Recommendations", command=show_recommendations, state=tk.DISABLED)
recommendation_button.pack(pady=10)

# Создание области для отображения графиков
figure = plt.Figure(figsize=(8, 6), dpi=100)
canvas = FigureCanvasTkAgg(figure, master=root)
canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

# Label для отображения рекомендаций
recommendation_label = tk.Label(root, text="", justify=tk.LEFT)
recommendation_label.pack(pady=10)

root.mainloop()