import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
import asyncio
import aiohttp
import concurrent.futures
from datetime import datetime
from io import StringIO, BytesIO
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Temperature Analysis App", layout="wide")

st.title("Анализ и мониторинг температурных данных")
st.markdown("""
Это приложение смотрит, какая температура была раньше, и сравнивает её с текущей погодой из сервиса OpenWeatherMap. 
Так оно помогает заметить необычные изменения — например, если сейчас намного жарче или холоднее, чем обычно в это время года.

**Функционал:**
- Изучение, как менялась температура раньше
- Поиск повторяющихся закономерностей по сезонам
- Поиск «странностей» в температуре
- Сравнение «как сейчас» с «как было»
- Проверка, какой способ работы быстрее
""")



def calc_roll_stats(data, window=30):
    """Анализ и расчет данных по ттемпературе"""
    data = data.copy()
    data['rolling_avg'] = data['temperature'].rolling(window=window).mean()
    data['rolling_std'] = data['temperature'].rolling(window=window).std()
    data['up_gran'] = data['rolling_avg'] + 2 * data['rolling_std']
    data['low_gran'] = data['rolling_avg'] - 2 * data['rolling_std']
    data['anomaly'] = ((data['temperature'] > data['up_gran']) |
                       (data['temperature'] < data['low_gran']))
    return data


def analyze_city_data(city_df, window=30):
    """Анализ температуры для конкретного города"""
    if city_df.empty:
        return None
    city_df = calc_roll_stats(city_df, window)

    seasonal_stats = city_df.groupby('season')['temperature'].agg(['mean', 'std', 'count']).reset_index()
    seasonal_stats['up_gran'] = seasonal_stats['mean'] + 2 * seasonal_stats['std']
    seasonal_stats['low_gran'] = seasonal_stats['mean'] - 2 * seasonal_stats['std']

    seasonal_anomalies = {}
    for season in city_df['season'].unique():
        season_data = city_df[city_df['season'] == season]
        mean_temp = season_data['temperature'].mean()
        std_temp = season_data['temperature'].std()
        up_gran = mean_temp + 2 * std_temp
        low_gran = mean_temp - 2 * std_temp
        anomalies = season_data[(season_data['temperature'] > up_gran) |
                                (season_data['temperature'] < low_gran)]
        seasonal_anomalies[season] = {
            'mean': mean_temp,
            'std': std_temp,
            'up_gran': up_gran,
            'low_gran': low_gran,
            'anomaly_cnt': len(anomalies),
            'total_cnt': len(season_data),
            'anomaly_percent': (len(anomalies) / len(season_data) * 100) if len(season_data) > 0 else 0
        }

    if len(city_df) > 0:
        city_df['year'] = city_df['timestamp'].dt.year
        year_temper = city_df.groupby('year')['temperature'].agg(['mean', 'min', 'max']).reset_index()

        if len(year_temper) > 1:
            x = year_temper['year'].values
            y = year_temper['mean'].values
            z = np.polyfit(x, y, 1)
            trend = z[0]
        else:
            trend = 0
    else:
        year_temper = pd.DataFrame()
        trend = 0

    return {
        'clear_data': city_df,
        'seasonal_stats': seasonal_stats,
        'seasonal_anomalies': seasonal_anomalies,
        'year_temper': year_temper,
        'trend': trend
    }


def analyze_temper_data_parallell(df, cities, window=30):
    """Параллел анализ для нескольких городов"""
    start_time = time.time()

    res = {}
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {executor.submit(analyze_city_data, df[df['city'] == city].copy(), window): city for city in cities}

        for future in concurrent.futures.as_completed(futures):
            city = futures[future]
            try:
                city_data = future.result()
                res[city] = city_data
            except Exception as e:
                st.error(f"Ошибка обработки {city}: {str(e)}")

    parallel_time = time.time() - start_time
    return res, parallel_time


def analyze_temper_data_posled(df, cities, window=30):
    start_time = time.time()

    res = {}
    for city in cities:
        res[city] = analyze_city_data(df[df['city'] == city].copy(), window)

    posled_time = time.time() - start_time
    return res, posled_time


def curr_weather_sync(city, api_key):
    """Синхрон запрос текущей погоды"""
    start_time = time.time()
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"

    try:
        response = requests.get(url)
        needed_time = time.time() - start_time

        if response.status_code == 200:
            data = response.json()
            return {
                'temper': data['main']['temp'],
                'feels_like': data['main']['feels_like'],
                'humidity': data['main']['humidity'],
                'wind_speed': data['wind']['speed'],
                'description': data['weather'][0]['description'],
                'icon': data['weather'][0]['icon'],
                'needed_time': needed_time
            }
        else:
            error_data = response.json()
            return {
                'error': f"Ошибка {response.status_code}: {error_data.get('message', 'Неизвестная ошибка')}",
                'needed_time': needed_time
            }
    except Exception as e:
        return {
            'error': f"Ошибка запроса: {str(e)}",
            'needed_time': time.time() - start_time
        }


async def curr_weather_sync(city, api_key):
    """Асинхрон запрос текущей погоды"""
    start_time = time.time()
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                needed_time = time.time() - start_time

                if response.status == 200:
                    data = await response.json()
                    return {
                        'temper': data['main']['temp'],
                        'feels_like': data['main']['feels_like'],
                        'humidity': data['main']['humidity'],
                        'wind_speed': data['wind']['speed'],
                        'description': data['weather'][0]['description'],
                        'icon': data['weather'][0]['icon'],
                        'needed_time': needed_time
                    }
                else:
                    error_data = await response.json()
                    return {
                        'error': f"Ошибка {response.status}: {error_data.get('message', 'Неизвестная ошибка')}",
                        'needed_time': needed_time
                    }
    except Exception as e:
        return {
            'error': f"Ошибка запроса: {str(e)}",
            'needed_time': time.time() - start_time
        }


def check_temper_anomaly(curr_temp, seasonal_anomalies, curr_season):
    """Проверяем что-то необычное для текущей температуры смотрим на historic данные"""
    if curr_season not in seasonal_anomalies:
        return "Неизвестно (нет данных для текущего сезона)"

    season_stats = seasonal_anomalies[curr_season]
    up_gran = season_stats['up_gran']
    low_gran = season_stats['low_gran']

    if curr_temp > up_gran:
        return f"Аномально высока (>{up_gran:.2f}°C)"
    elif curr_temp < low_gran:
        return f"Аномально низкая (<{low_gran:.2f}°C)"
    else:
        return f"Нормальная (между {low_gran:.2f}°C и {up_gran:.2f}°C)"


def get_current_season():
    """Определение текущего сезона"""
    month = datetime.now().month
    month_to_season = {
        12: "winter", 1: "winter", 2: "winter",
        3: "spring", 4: "spring", 5: "spring",
        6: "summer", 7: "summer", 8: "summer",
        9: "autumn", 10: "autumn", 11: "autumn"
    }
    return month_to_season[month]


@st.cache_data
def load_data(file):
    """Подгрузка csv таблы"""
    data = pd.read_csv(file)
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    return data


def temper_time_graf(clear_data, city):
    """Создаём график изменений температуры за период и смотрим необычные отклонения"""
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=clear_data['timestamp'],
            y=clear_data['temperature'],
            mode='lines',
            name='Температура',
            line=dict(color='royalblue')
        )
    )

    fig.add_trace(
        go.Scatter(
            x=clear_data['timestamp'],
            y=clear_data['rolling_avg'],
            mode='lines',
            name='Скользящее среднее',
            line=dict(color='green')
        )
    )

    fig.add_trace(
        go.Scatter(
            x=clear_data['timestamp'],
            y=clear_data['up_gran'],
            mode='lines',
            name='Верхняя граница',
            line=dict(color='red', dash='dash')
        )
    )

    fig.add_trace(
        go.Scatter(
            x=clear_data['timestamp'],
            y=clear_data['low_gran'],
            mode='lines',
            name='Нижняя граница',
            line=dict(color='red', dash='dash')
        )
    )

    anomalies = clear_data[clear_data['anomaly'] == True]
    fig.add_trace(
        go.Scatter(
            x=anomalies['timestamp'],
            y=anomalies['temperature'],
            mode='markers',
            name='Аномалии',
            marker=dict(color='red', size=8, symbol='circle-open')
        )
    )

    fig.update_layout(
        title=f'График, который показывает, как температура менялась со временем {city}',
        xaxis_title='Дата',
        yaxis_title='Температура (°C)',
        hovermode='closest',
        height=500,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    return fig


def yearly_trends(year_temper, city, trend):
    """Создаем график показывающий годовые тренды по температуре"""

    if len(year_temper) <= 1:
        return None

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=year_temper['year'],
            y=year_temper['mean'],
            mode='lines+markers',
            name='Среднегодовая',
            line=dict(color='green')
        )
    )

    fig.add_trace(
        go.Scatter(
            x=year_temper['year'],
            y=year_temper['min'],
            mode='lines',
            name='Минимум',
            line=dict(color='blue', dash='dot')
        )
    )

    fig.add_trace(
        go.Scatter(
            x=year_temper['year'],
            y=year_temper['max'],
            mode='lines',
            name='Максимум',
            line=dict(color='red', dash='dot')
        )
    )

    x = year_temper['year'].values
    y = year_temper['mean'].values
    trend_line = np.poly1d(np.polyfit(x, y, 1))(x)

    fig.add_trace(
        go.Scatter(
            x=year_temper['year'],
            y=trend_line,
            mode='lines',
            name=f'Тренд  ({trend:.3f}°C/год)',
            line=dict(color='black', dash='dash')
        )
    )

    fig.update_layout(
        title=f'Как температура меняется из года в год в {city}',
        xaxis_title='Год',
        yaxis_title='Температура (°C)',
        hovermode='closest',
        height=400
    )

    return fig


st.sidebar.header("Ввод данных")
uploaded_file = st.sidebar.file_uploader("Загрузите данные (CSV)", type=["csv"])

if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'analysis_res' not in st.session_state:
    st.session_state.analysis_res = None
if 'selected_city' not in st.session_state:
    st.session_state.selected_city = None
if 'parallel_time' not in st.session_state:
    st.session_state.parallel_time = None
if 'posled_time' not in st.session_state:
    st.session_state.posled_time = None

if uploaded_file is not None:
    with st.spinner('Загрузка данных...'):
        data = load_data(uploaded_file)
        cities = sorted(data['city'].unique())
        st.session_state.data_loaded = True

    st.sidebar.header("Настройки анализа")
    window_size = st.sidebar.slider("За сколько дней считать среднее", min_value=7, max_value=90, value=30, step=1)
    selected_city = st.sidebar.selectbox("Выберите город", cities)
    st.session_state.selected_city = selected_city

    st.sidebar.header("OpenWeatherMap API")
    api_key = st.sidebar.text_input("Введите API-ключ", type="password")
    api_method = st.sidebar.radio("Метод запроса", ["Синхронный", "Асинхронный"])


    if st.sidebar.button("Запустить анализ"):
        with st.spinner('Анализ...'):
            res_parallel, parallel_time = analyze_temper_data_parallell(data, cities, window_size)
            res_sequential, posled_time = analyze_temper_data_posled(data, [selected_city],
                                                                                      window_size)

            st.session_state.analysis_res = res_parallel
            st.session_state.parallel_time = parallel_time
            st.session_state.posled_time = posled_time

        st.sidebar.success("Анализ завершен!")
        st.sidebar.metric("Время параллельной обработки", f"{parallel_time:.4f}с")
        st.sidebar.metric("Время последовательной обработки", f"{posled_time:.4f}с")

        st.sidebar.info("""
        **Примечание:**
        - Параллельная обработка эффективнее для нескольких городов
        - Для одного города последовательный метод может быть быстрее
        """)

    if st.session_state.analysis_res is not None and st.session_state.selected_city in st.session_state.analysis_res:
        city_data = st.session_state.analysis_res[st.session_state.selected_city]

        if city_data is not None:
            clear_data = city_data['clear_data']

            tab1, tab2, tab3, tab4 = st.tabs(
                ["Обзор", "Временные ряды", "Сезонный анализ", "Текущая погода"])

            with tab1:
                st.header(f"Обзор температуры для {selected_city}")

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Средняя температура", f"{clear_data['temperature'].mean():.2f}°C")
                with col2:
                    st.metric("Минимум", f"{clear_data['temperature'].min():.2f}°C")
                with col3:
                    st.metric("Максимум", f"{clear_data['temperature'].max():.2f}°C")
                with col4:
                    anomaly_cnt = clear_data['anomaly'].sum()
                    anomaly_percent = (anomaly_cnt / len(clear_data)) * 100
                    st.metric("Аномалии", f"{anomaly_cnt} ({anomaly_percent:.1f}%)")

                st.subheader("Распределение температуры")
                hist_graf = px.histogram(
                    clear_data,
                    x="temperature",
                    nbins=50,
                    color_discrete_sequence=['royalblue'],
                    title=f"Распределение температуры в {selected_city}"
                )
                hist_graf.update_layout(
                    xaxis_title="Температура (°C)",
                    yaxis_title="Частота"
                )
                st.plotly_chart(hist_graf, use_container_width=True)

                st.subheader("Изменения температуры за много лет")
                if 'year_temper' in city_data and len(city_data['year_temper']) > 1:
                    yearly_graf = yearly_trends(city_data['year_temper'], selected_city, city_data['trend'])
                    if yearly_graf:
                        st.plotly_chart(yearly_graf, use_container_width=True)

                        trend = city_data['trend']
                        if abs(trend) < 0.01:
                            st.info("**Анализ изменений**: Значительных изменений температуры не обнаружено")
                        elif trend > 0:
                            st.warning(f"**Анализ изменений**: Рост температуры ~{trend:.3f}°C/год.")
                        else:
                            st.info(f"**Анализ изменений**: Снижение температуры ~{abs(trend):.3f}°C/год.")
                else:
                    st.info("Недостаточно данных для анализа")

            with tab2:
                st.header("Изучение данных, которые меняются со временем")

                time_series_graf = temper_time_graf(clear_data, selected_city)
                st.plotly_chart(time_series_graf, use_container_width=True)

                st.subheader("Метод обнаружения аномалий")
                st.markdown("""
                Выявляем аномалии так:

                1. **Среднее значение за 30 дней**: Берём температуру за последний месяц и вычисляем «усреднённое» 
                значение — это помогает увидеть общий тренд, игнорируя случайные скачки
                2. **Разброс данных**: Считаем, насколько сильно температура обычно отклоняется от среднего за те же 30 дней.
                3. **Границы нормы**: Вычисляем диапазон, где температура считается обычной.
                4. **Обнаружение**: Выход за границы считается аномалией
                """)

                if clear_data['anomaly'].sum() > 0:
                    st.subheader("Основные аномалии")
                    anomalies = clear_data[clear_data['anomaly']].copy()
                    anomalies['otklon'] = abs(anomalies['temperature'] - anomalies['rolling_avg'])
                    anomalies = anomalies.sort_values('otklon', ascending=False).head(10)
                    anomalies_vivod = anomalies[
                        ['timestamp', 'temperature', 'rolling_avg', 'otklon', 'season']].copy()
                    anomalies_vivod.columns = ['Дата', 'Температура (°C)', 'Ожидаемая (°C)', 'Отклонение (°C)', 'Сезон']
                    anomalies_vivod = anomalies_vivod.reset_index(drop=True)
                    st.dataframe(anomalies_vivod.round(2), use_container_width=True)

            with tab3:
                st.header("Анализ по сезонам")
                seasonal_graf = px.box(
                    clear_data,
                    x='season',
                    y='temperature',
                    color='season',
                    title=f'Распределение температуры по сезонам в {selected_city}',
                    category_orders={"season": ["winter", "spring", "summer", "autumn"]},
                    points="all"
                )
                seasonal_graf.update_layout(
                    xaxis_title="Season",
                    yaxis_title="Temperature (°C)"
                )
                st.plotly_chart(seasonal_graf, use_container_width=True)

                st.subheader("Статистика по сезонам")
                seasonal_stats = city_data['seasonal_stats'].copy()

                vivod_stats = seasonal_stats.copy()
                vivod_stats = vivod_stats.sort_values(by='season', key=lambda x: pd.Categorical(
                    x, categories=['winter', 'spring', 'summer', 'autumn'], ordered=True))
                vivod_stats.columns = ['Сезон', 'Средняя (°C)', 'Ст. отклонение (°C)', 'Количество', 'Верхняя граница', 'Нижняя граница']

                st.dataframe(vivod_stats.round(2), use_container_width=True)

                st.subheader("Анализ аномалий по сезонам")

                seasonal_anomaly_data = []
                for season, stats in city_data['seasonal_anomalies'].items():
                    seasonal_anomaly_data.append({
                        'season': season,
                        'anomaly_percent': stats['anomaly_percent'],
                        'anomaly_cnt': stats['anomaly_cnt'],
                        'total_cnt': stats['total_cnt']
                    })

                seasonal_anomaly_df = pd.DataFrame(seasonal_anomaly_data)
                if not seasonal_anomaly_df.empty:
                    seasonal_anomaly_df = seasonal_anomaly_df.sort_values(by='season', key=lambda x: pd.Categorical(
                        x, categories=['winter', 'spring', 'summer', 'autumn'], ordered=True))

                    anomaly_graf = px.bar(
                        seasonal_anomaly_df,
                        x='season',
                        y='anomaly_percent',
                        color='season',
                        text='anomaly_cnt',
                        title=f'Как часто в каждом сезоне встречаются необычные температуры {selected_city}'
                    )
                    anomaly_graf.update_layout(
                        xaxis_title="Сезон",
                        yaxis_title="Процент аномалий (%)",
                        yaxis=dict(range=[0, max(seasonal_anomaly_df['anomaly_percent']) * 1.1])
                    )
                    st.plotly_chart(anomaly_graf, use_container_width=True)

            with tab4:
                st.header("Текущая погода")

                if api_key:
                    with st.spinner("Получение данных..."):
                        if api_method == "Синхронный":
                            curr_weather = asyncio.run(curr_weather_sync(selected_city, api_key))
                        else:
                            try:
                                curr_weather = asyncio.run(curr_weather_sync(selected_city, api_key))
                            except Exception as e:
                                curr_weather = {'error': f"Ошибка асинхронного запроса: {str(e)}"}

                    if "error" in curr_weather:
                        st.error(curr_weather['error'])
                    else:
                        col1, col2 = st.columns([1, 2])

                        with col1:
                            st.markdown(f"### {selected_city}")
                            st.markdown(f"**{curr_weather['description'].title()}**")
                            if 'icon' in curr_weather:
                                icon_url = f"http://openweathermap.org/img/wn/{curr_weather['icon']}@2x.png"
                                st.image(icon_url, width=100)

                        with col2:
                            st.write(curr_weather)
                            weather_stata = {
                                "Температура": f"{curr_weather['temperature']:.1f} °C",
                                "Ощущается как": f"{curr_weather['feels_like']:.1f} °C",
                                "Влажность": f"{curr_weather['humidity']} %",
                                "Скорость ветра": f"{curr_weather['wind_speed']} m/s"
                            }

                            for label, value in weather_stata.items():
                                st.metric(label, value)

                        curr_season = get_current_season()
                        anomaly_status = check_temper_anomaly(
                            curr_weather['temperature'],
                            city_data['seasonal_anomalies'],
                            curr_season
                        )

                        if "Нормальная" in anomaly_status:
                            st.success(f"**Статус для {curr_season.capitalize()}:** {anomaly_status}")
                        elif "высокая" in anomaly_status:
                            st.warning(f"**Статус для {curr_season.capitalize()}:** {anomaly_status}")
                        elif "низкая" in anomaly_status:
                            st.error(f"**Статус для {curr_season.capitalize()}:** {anomaly_status}")
                        else:
                            st.info(f"**Статус для {curr_season.capitalize()}:** {anomaly_status}")

                        st.subheader("Производительность запроса")
                        st.metric(f"{api_method} Время запроса", f"{curr_weather['needed_time']:.4f} сек")

                        curr_temp = curr_weather['temperature']
                        seasonal_data = city_data['seasonal_stats'].copy()

                        fig = go.Figure()

                        fig.add_trace(go.Bar(
                            x=seasonal_data['season'],
                            y=seasonal_data['mean'],
                            name='Среднее по сезонам',
                            marker_color='lightblue'
                        ))

                        for idx, row in seasonal_data.iterrows():
                            fig.add_trace(go.Scatter(
                                x=[row['season'], row['season']],
                                y=[row['low_gran'], row['up_gran']],
                                mode='lines',
                                name=f"{row['season']} Нормальный диапазон" if idx == 0 else None,
                                showlegend=idx == 0,
                                line=dict(color='gray', width=2)
                            ))

                        fig.add_trace(go.Scatter(
                            x=[curr_season],
                            y=[curr_temp],
                            mode='markers',
                            name='Текущая температура',
                            marker=dict(
                                color='red',
                                size=12,
                                symbol='star'
                            )
                        ))

                        fig.update_layout(
                            title=f'Текущая температура vs сезонные нормы {selected_city}',
                            xaxis_title='Сезон',
                            yaxis_title='Температура (°C)',
                            xaxis=dict(
                                categoryorder='array',
                                categoryarray=['winter', 'spring', 'summer', 'autumn']
                            )
                        )

                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Введите API-ключ в боковой панели, чтобы добавить текующую погоду")
        else:
            st.error(f"Нет данных для {selected_city}.")
else:
    st.info("Загрузите CSV файл через боковую панель для начала")
    st.subheader("Нужен формат CSV")
    sample_data = {
        'city': ['Москва', 'Москва', 'Санкт-Петербург', 'Санкт-Петербург'],
        'timestamp': ['2023-01-01', '2023-01-02', '2023-01-01', '2023-01-02'],
        'temperature': [-5.2, -6.1, -7.3, -8.2],
        'season': ['winter', 'winter', 'winter', 'winter']
    }
    sample_df = pd.DataFrame(sample_data)
    st.dataframe(sample_df, use_container_width=True)

    st.subheader("Возможности приложения")
    st.markdown("""
    - **Анализ трендов**: Скользящие средние и стандартные отклонения
    - **Обнаружение аномалий**: Статистические методы
    - **Сезонный анализ**: Анализ по сезонам
    - **Текущая погода**: Интеграция с OpenWeatherMap API""")
