ОПИС:
Цей програмний продукт реалізує модель прогнозування погодних параметрів з використанням LSTM-нейромережі, побудованої на фреймворку PyTorch. 
Рішення орієнтоване на аграрний сектор і дозволяє формувати рекомендації щодо доцільності вирощування певних культур у заданому регіоні на основі короткострокового метеорологічного прогнозу.

ОСНОВНІ ФУНКЦІЇ:
- Завантаження погодних даних (Meteostat API)
- Декомпозиція часових рядів
- Масштабування та формування послідовностей
- Навчання моделі LSTM
- Генерація прогнозу температури, вологості, опадів і вітру
- Побудова графіків
- Автоматичне формування рекомендацій щодо посіву культур

ВИМОГИ:
Python 3.9 або сумісна версія
(В коді немає жорсткої прив'язки до версії 3.9, але через специфіку використаних бібліотек та Tkinter-інтерфейсу рекомендується Python 3.9 для уникнення конфліктів)

ОБОВ'ЯЗКОВІ БІБЛІОТЕКИ
- torch
- numpy
- pandas
- matplotlib
- statsmodels
- meteostat
- tkinter

Для запуску програми необхідна наявність папки data_cache у кореневій директорії. 
У ній зберігаються локальні копії завантажених погодних даних.

СТРУКТУРА ЗАПУСКУ
- Запустіть файл AgroForecast_LSTM_System.py.
- Оберіть місто з випадного списку.
- Натисніть Generate Forecast для створення прогнозу.
- Натисніть Get Recommendations, щоб побачити перелік культур, рекомендованих до вирощування.
- Можна також переглянути історичні дані (Show Historical Data).

