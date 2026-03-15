import marimo

__generated_with = "0.17.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    from io import StringIO
    import lets_plot as lp
    from lets_plot import ggplot, geom_line, labs, ggsize, aes, scale_color_manual, geom_hline, geom_vline
    import numpy as np
    from scipy.signal import savgol_filter
    from scipy.signal import find_peaks
    from scipy.optimize import curve_fit
    from lets_plot.mapping import as_discrete
    return (
        StringIO,
        aes,
        curve_fit,
        find_peaks,
        geom_line,
        ggplot,
        ggsize,
        labs,
        lp,
        mo,
        np,
        pl,
        savgol_filter,
    )


@app.cell
def _(StringIO, pl):
    def process_coordinates_file(file_content):
        try:
            # Валидация формата: каждая непустая строка должна содержать минимум два ЧИСЛА (x y).
            # Если встречаются нечисловые значения — останавливаем обработку с понятной ошибкой.
            import re

            xs: list[float] = []
            ys: list[float] = []

            for line_no, raw in enumerate(file_content.splitlines(), 1):
                s = raw.strip()
                if not s:
                    continue

                # Разбиваем по любому whitespace (пробелы/таб/несколько пробелов)
                parts = re.split(r"\s+", s)
                if len(parts) < 2:
                    return None, None, None, f"Ошибка формата в строке {line_no}: ожидаются две колонки чисел (x y)."

                # Берём только первые две колонки, остальное игнорируем (как и раньше)
                x_raw, y_raw = parts[0], parts[1]

                # Разрешаем десятичную запятую (частый формат), но в остальном требуем float
                x_raw = x_raw.replace(",", ".")
                y_raw = y_raw.replace(",", ".")

                try:
                    x_val = float(x_raw)
                    y_val = float(y_raw)
                except Exception:
                    return (
                        None,
                        None,
                        None,
                        f"Ошибка формата в строке {line_no}: значения должны быть числами (x y). Получено: {parts[0]} {parts[1]}",
                    )

                xs.append(x_val)
                ys.append(y_val)

            if len(xs) == 0:
                return None, None, None, "Файл не содержит корректных пар координат (x y)."

            df = pl.DataFrame({"x_coord": xs, "y_coord": ys})
            return df, xs, ys, None

        except Exception as e:
            return None, None, None, f"Ошибка обработки файла: {str(e)}"
    return (process_coordinates_file,)


@app.cell
def _(mo):
    # Создаем элемент загрузки файла
    file_upload = mo.ui.file(
        filetypes=[".txt"],
        multiple=False,
        label="Загрузите файл с координатами (.txt)"
    )

    # Объединяем все элементы в один вывод
    mo.vstack([
        mo.md("# 📁 Загрузка файла с координатами"),
        mo.md("## 📤 Загрузка файла"),
        mo.md("Загрузите текстовый файл с координатами в формате: x1 y1 x2 y2, x3 y3..."),
        mo.md("x1 y1"),
        mo.md("x2 y2"),
        mo.md("..."),
        mo.md("x(n) y(n)"),
        file_upload
    ])
    return (file_upload,)


@app.cell
def _(file_upload, mo, process_coordinates_file):
    # Объявляем переменные в глобальной области видимости
    df = None
    x_coords = None
    y_coords = None
    error = None
    file_info = None
    y_done = []
    # Проверяем, загружен ли файл
    if file_upload.value:
        # Получаем содержимое файла
        file_content_bytes = file_upload.value[0].contents
        file_content = file_content_bytes.decode('utf-8')
        file_info = f"Загружен файл: {file_upload.value[0].name}"

        # Обрабатываем данные
        df, x_coords, y_coords, error = process_coordinates_file(file_content)

    # Создаем результат для отображения
    if file_upload.value:
        if error:
            result_display = mo.md(f"**❌ Ошибка:** {error}")
        elif df is not None:
            # Упрощенный вариант без колонок
            result_display = mo.vstack([
                mo.md("## 📊 Результаты обработки"),
                mo.md(file_info),
                mo.md("### Таблица данных"),
                mo.ui.table(df),
                mo.md(f"**Всего обработано точек:** {len(df)}")
            ])
        else:
            result_display = mo.md("**⚠️ Не удалось обработать файл**")
    else:
        result_display = mo.md("**⏳ Ожидание загрузки файла...**")

    # Отображаем результат
    result_display
    return df, x_coords, y_coords


@app.cell
def _(aes, df, geom_line, ggplot, ggsize, labs, lp, mo, x_coords, y_coords):
    # Проверяем, что данные загружены
    if df is not None:
        try:
            lp.LetsPlot.setup_html()

            # Создаем данные для Lets-Plot
            x_coords_list = x_coords if isinstance(x_coords, list) else x_coords.tolist()
            y_coords_list = y_coords if isinstance(y_coords, list) else y_coords.tolist()

            plot_data2 = {
                'x_coord': x_coords_list,
                'y_coord': y_coords_list
            }

            # Строим график
            plot2 = ggplot(plot_data2) + \
                   geom_line(aes(x='x_coord', y='y_coord'), color='blue', size=1) + \
                    labs(x='2θ', y='Интенсивность') + \
                   ggsize(1000, 500)

            # ВСЕ в одном vstack
            result_display1 = mo.vstack([
                mo.md("## 📈 График данных"),
                plot2
            ])

        except ImportError:
            result_display1 = mo.vstack([
                mo.md("## 📦 Требуется установка Lets-Plot"),
                mo.md("**Установите:** `pip install lets-plot`"),
                mo.md("После установки перезапустите ячейку")
            ])
        except Exception as e:
            result_display1 = mo.vstack([
                mo.md("## ❌ Ошибка построения графика"),
                mo.md(str(e))
            ])
            print(e)
    else:
        result_display1 = mo.md("**⏳ Данные еще не загружены**")

    result_display1
    return


@app.cell
def _(mo):
    iterations = mo.ui.slider(
            start=5, stop=150, step=5, value=80,
            label="Количество итераций SNIP"
        )
    iterations
    return (iterations,)


@app.cell
def _(
    aes,
    df,
    geom_line,
    ggplot,
    ggsize,
    iterations,
    labs,
    lp,
    mo,
    np,
    x_coords,
    y_coords,
):

    # --- Реализация SNIP ---
    def snip_baseline(y, iterations=30):
        """Реализация SNIP (Statistics-sensitive Nonlinear Iterative Peak-clipping)"""
        y = np.array(y, dtype=float)
        L = len(y)
        baseline = y.copy()

        for k in range(1, iterations + 1):
            temp = baseline.copy()
            for i in range(k, L - k):
                avg = 0.5 * (temp[i - k] + temp[i + k])
                if baseline[i] > avg:
                    baseline[i] = avg
        return baseline

    if df is not None and 'iterations' in locals():
        lp.LetsPlot.setup_html()
        # Вычисляем baseline
        baseline = snip_baseline(y_coords, iterations=iterations.value)
        y_corrected = np.array(y_coords) - baseline

        # Данные для графиков
        x_list = x_coords if isinstance(x_coords, list) else x_coords.tolist()
        y_list = y_coords if isinstance(y_coords, list) else y_coords.tolist()
        plot_dataa = {
            "x": x_list * 3,
            "y": y_list + baseline.tolist() + y_corrected.tolist(),
            "type": (["Raw data"] * len(x_list)) +
                    (["Фон (SNIP)"] * len(x_list)) +
                    (["Вычтенный фон"] * len(x_list))
        }

        # График
        plot_1 = (
            ggplot(plot_dataa)
            + geom_line(aes(x="x", y="y", color="type"), size=1)
            + labs(x="2θ", y="Интенсивность", title=f"SNIP (итераций: {iterations.value})", color="Тип")
            + ggsize(1000, 500)
        )

        result2 = mo.vstack([
            mo.md("## 🧮 SNIP определение фона"),
            plot_1
        ])
    else:
        result2 = mo.md("**⏳ Ожидание сглаженных данных...**")
        y_corrected = np.array([1,2])
    y_corrected
    result2
    return (y_corrected,)


@app.cell
def _(mo):
    # Слайдеры для настройки адаптивного сглаживания
    min_window_slider = mo.ui.slider(
        start=5, stop=21, step=2, value=11,
        label="Минимальное окно (для пиков)"
    )
    max_window_slider = mo.ui.slider(
        start=31, stop=101, step=2, value=51,
        label="Максимальное окно (для фона)"
    )

    polyorder_slider = mo.ui.slider(
        start=1, stop=5, step=1, value=2,
        label="Порядок полинома"
    )

    mo.vstack([
        mo.md("## ⚙️ Настройки адаптивного сглаживания Савицкого-Голея"),
        mo.md("**Минимальное окно** используется для высоких пиков (сохраняет детали)"),
        min_window_slider,
        mo.md("**Максимальное окно** используется для фоновых областей (сильное сглаживание)"),
        max_window_slider,
        mo.md("**Порядок полинома** определяет степень аппроксимирующего полинома"),
        polyorder_slider
    ])
    return max_window_slider, min_window_slider, polyorder_slider


@app.cell
def _(
    aes,
    df,
    geom_line,
    ggplot,
    ggsize,
    labs,
    lp,
    max_window_slider,
    min_window_slider,
    mo,
    np,
    polyorder_slider,
    savgol_filter,
    x_coords,
    y_corrected,
):
    def adaptive_savgol_smooth(y_data, min_window, max_window, polyorder):
        """
        Адаптивное сглаживание Савицкого-Голея с переменным размером окна.

        Параметры:
        - y_data: массив данных для сглаживания
        - min_window: минимальный размер окна (для пиков)
        - max_window: максимальный размер окна (для фоновых областей)
        - polyorder: порядок полинома для фильтра

        Возвращает:
        - Сглаженный массив данных
        """
        y = np.array(y_data, dtype=float)
        N = len(y)
        if N == 0:
            return y

        y_smoothed = np.zeros_like(y)

        # --- КЛЮЧЕВОЕ ИЗМЕНЕНИЕ ---
        # Пики могут иметь очень разную длину (50..1000 точек),
        # поэтому абсолютные окна из слайдеров надо ограничивать относительно N,
        # иначе узкие пики будут размываться.
        #
        # Эффективные границы окна:
        # - w_min: минимум из slider_min и доли от N
        # - w_max: минимум из slider_max и доли от N
        # Далее ранговая адаптация работает ТОЛЬКО в диапазоне [w_min, w_max].
        def _odd(k: int) -> int:
            k = int(k)
            return k if (k % 2 == 1) else (k + 1)

        N_odd_max = N if (N % 2 == 1) else (N - 1)
        if N_odd_max < 5:
            # Слишком короткий пик: сглаживать нечем — возвращаем как есть
            return y

        # Доли от N (под сохранение формы; ограничиваем размывание узких пиков)
        alpha_min = 0.12  # ~6 точек на N=50
        alpha_max = 0.25  # ~12-13 точек на N=50

        w_min_cap = max(5, int(np.floor(alpha_min * N)))
        w_max_cap = max(5, int(np.floor(alpha_max * N)))

        # Базовые окна, учёт slider + ограничение сверху по N
        w_min_eff = _odd(int(np.clip(min(int(min_window), w_min_cap), 5, N_odd_max)))

        # Окно должно быть > polyorder
        min_required = _odd(int(polyorder) + 2)
        if w_min_eff < min_required:
            w_min_eff = min(min_required, N_odd_max)
            w_min_eff = _odd(w_min_eff)
            if w_min_eff > N_odd_max:
                w_min_eff = N_odd_max

        w_max_eff = _odd(int(np.clip(min(int(max_window), w_max_cap), w_min_eff, N_odd_max)))
        if w_max_eff < w_min_eff:
            w_max_eff = w_min_eff

        # --- Ранг точки (быстро O(N log N) вместо O(N^2)) ---
        # rank_greater = сколько точек имеют значение строго больше y[i]
        order = np.argsort(y)  # по возрастанию
        rank_small = np.empty(N, dtype=int)
        rank_small[order] = np.arange(N)  # 0..N-1: сколько значений меньше
        rank_greater = (N - 1) - rank_small

        normalized_ranks = rank_greater / (N - 1) if N > 1 else np.zeros(N)

        # Вычисляем размер окна для каждой точки
        window_sizes = w_min_eff + (w_max_eff - w_min_eff) * normalized_ranks

        # Округляем до нечетных целых чисел и ограничиваем
        window_sizes = np.round(window_sizes).astype(int)
        window_sizes = np.where(window_sizes % 2 == 0, window_sizes + 1, window_sizes)
        window_sizes = np.clip(window_sizes, w_min_eff, w_max_eff)

        # Применяем сглаживание для каждой точки с индивидуальным окном
        for i in range(N):
            window = int(window_sizes[i])

            # Определяем границы окна вокруг точки i
            half_window = window // 2
            start_idx = max(0, i - half_window)
            end_idx = min(N, i + half_window + 1)

            # Извлекаем подмассив для сглаживания
            window_data = y[start_idx:end_idx]

            # Убедимся, что размер окна не больше доступных данных
            actual_window = len(window_data)
            if actual_window < polyorder + 1:
                # Если данных недостаточно, используем исходное значение
                y_smoothed[i] = y[i]
            else:
                # Если окно четное, делаем нечетным
                if actual_window % 2 == 0:
                    actual_window -= 1
                    window_data = window_data[:actual_window]

                # Проверяем, что порядок полинома меньше размера окна
                effective_polyorder = min(polyorder, actual_window - 1)

                try:
                    # Применяем фильтр Савицкого-Голея к подмассиву
                    smoothed_window = savgol_filter(window_data, actual_window, effective_polyorder)

                    # Берем среднюю точку сглаженного окна
                    center_idx = len(smoothed_window) // 2
                    y_smoothed[i] = smoothed_window[center_idx]
                except:
                    # В случае ошибки используем исходное значение
                    y_smoothed[i] = y[i]

        return y_smoothed

    if df is not None and len(y_corrected) > 10:
        try:
            lp.LetsPlot.setup_html()

            # Применяем адаптивное сглаживание
            y_smoothed = adaptive_savgol_smooth(
                y_corrected,
                min_window=min_window_slider.value,
                max_window=max_window_slider.value,
                polyorder=polyorder_slider.value
            )

            # Данные для графика сравнения
            x_coords_list2 = x_coords if isinstance(x_coords, list) else x_coords.tolist()

            plot_data_smooth = {
                "x": x_coords_list2 * 2,
                "y": y_corrected.tolist() + y_smoothed.tolist(),
                "type": (["После вычитания фона (SNIP)"] * len(x_coords_list2)) +
                        (["Адаптивное сглаживание"] * len(x_coords_list2))
            }

            # График сравнения
            plot_smooth = (
                ggplot(plot_data_smooth)
                + geom_line(aes(x="x", y="y", color="type"), size=1)
                + labs(
                    x="2θ", 
                    y="Интенсивность", 
                    title=f"Адаптивное сглаживание Савицкого-Голея (окно: {min_window_slider.value}-{max_window_slider.value}, порядок: {polyorder_slider.value})",
                    color="Тип данных"
                )
                + ggsize(1000, 450)
            )

            result_smooth = mo.vstack([
                mo.md("## 🎯 Адаптивное сглаживание"),
                mo.md(f"**Параметры:** мин. окно = {min_window_slider.value}, макс. окно = {max_window_slider.value}, порядок полинома = {polyorder_slider.value}"),
                mo.md("*Размер окна автоматически адаптируется: малое окно для пиков (сохранение деталей), большое окно для фона (сильное сглаживание)*"),
                plot_smooth
            ])
        except Exception as e:
            result_smooth = mo.vstack([
                mo.md("## ❌ Ошибка адаптивного сглаживания"),
                mo.md(f"**Ошибка:** {str(e)}")
            ])
    else:
        result_smooth = mo.md("**⏳ Ожидание данных после SNIP...**")
        y_smoothed = np.array([])

    result_smooth
    return (y_smoothed,)


@app.cell
def _(mo):

    min_distance_slider = 100

    prominence_slider = mo.ui.slider(
        start=10, stop=100, step=5, value=30,
        label="Минимальная значимость пика (prominence)"
    )

    threshold_ratio_slider = mo.ui.slider(
        start=1, stop=30, step=1, value=6,
        label="Порог для границ"
    )

    min_peak_width_slider = mo.ui.slider(
        start=0.1, stop=1.5, step=0.1, value=0.5,
        label="Минимальная ширина пика (ΔX)"
    )

    mo.vstack([
        mo.md("## 🔍 Настройки определения пиков"),
        #mo.md("**Минимальная высота** - пики ниже этого значения игнорируются"),
        #min_height_slider,
        mo.md("**Минимальная значимость** - насколько пик должен выделяться относительно окружения"),
        prominence_slider,
        mo.md("**Значение для фона** - граница пика определяется там, где интенсивность падает до этого значения"),
        threshold_ratio_slider,
        mo.md("**Минимальная ширина пика** - пики уже вычислены с границами; далее отбрасываем слишком узкие (по ΔX)"),
        min_peak_width_slider
    ])
    return (
        min_distance_slider,
        min_peak_width_slider,
        prominence_slider,
        threshold_ratio_slider,
    )


@app.cell
def _(
    aes,
    df,
    find_peaks,
    geom_line,
    ggplot,
    ggsize,
    labs,
    lp,
    min_distance_slider,
    min_peak_width_slider,
    mo,
    np,
    pl,
    prominence_slider,
    threshold_ratio_slider,
    x_coords,
    y_smoothed,
):
    def find_peaks_with_bounds(y_data, x_data, min_height, min_distance, prominence, threshold_ratio):
        """
        Определение пиков с вычислением их границ.

        Параметры:
        - y_data: массив интенсивностей (сглаженные данные)
        - x_data: массив x-координат
        - min_height: минимальная высота пика
        - min_distance: минимальное расстояние между пиками (в точках)
        - prominence: минимальная значимость пика
        - threshold_ratio: порог для определения границ (доля от высоты пика)

        Возвращает:
        - Список словарей с информацией о каждом пике
        """
        y = np.array(y_data, dtype=float)
        x = np.array(x_data, dtype=float)
        N = len(y)

        # Первичное обнаружение пиков
        peaks_indices, properties = find_peaks(
            y,
            height=min_height,
            distance=min_distance,
            prominence=prominence
        )

        peaks_info = []

        for idx in peaks_indices:
            peak_y = y[idx]
            peak_x = x[idx]

            # Вычисляем порог для границ, 10 - как значение для начала ФОНА и конца пика
            threshold = threshold_ratio

            # === Поиск левой границы ===
            left_bound_idx = idx
            for ind in range(idx - 1, -1, -1):
                # Проверка 1: приближение к порогу
                if y[ind] <= threshold:
                    left_bound_idx = ind
                    break

                #Проверка 2: анализ производной (стабилизация)
                # if ind >= 40:
                #     # Вычисляем производную в окне из 5 точек
                #     window = y[ind-40:ind+1]
                #     derivatives = np.diff(window)
                #     # Если производная близка к нулю или меняет знак
                #     if np.median(derivatives) < 0:
                #         left_bound_idx = ind
                #         break

                # Проверка 3: достигли начала массива
                if ind == 0:
                    left_bound_idx = 0
                    break

            # === Поиск правой границы ===
            right_bound_idx = idx
            for ind in range(idx + 1, N):
                # Проверка 1: приближение к порогу
                if y[ind] <= threshold:
                    right_bound_idx = ind
                    break

                # Проверка 2: анализ производной (стабилизация)
                # if ind < N - 40:
                #     # Вычисляем производную в окне из 5 точек
                #     window = y[ind:ind+41]
                #     derivatives = np.diff(window)
                #     # Если производная близка к нулю или меняет знак
                #     if np.median(derivatives) > 0:
                #         right_bound_idx = ind
                #         break

                # Проверка 3: достигли конца массива
                if ind == N - 1:
                    right_bound_idx = N - 1
                    break

            # Получаем x-координаты границ
            left_bound_x = x[left_bound_idx]
            right_bound_x = x[right_bound_idx]
            width = right_bound_x - left_bound_x

            # Получаем prominence из свойств
            peak_prominence = properties['prominences'][len(peaks_info)] if len(peaks_info) < len(properties['prominences']) else 0

            peaks_info.append({
                'peak_index': int(idx),
                'peak_x': float(peak_x),
                'peak_y': float(peak_y),
                'left_bound': float(left_bound_x),
                'right_bound': float(right_bound_x),
                'left_bound_idx': int(left_bound_idx),
                'right_bound_idx': int(right_bound_idx),
                'width': float(width),
                'prominence': float(peak_prominence)
            })

        return peaks_info

    if df is not None and len(y_smoothed) > 10:
        try:
            lp.LetsPlot.setup_html()

            # Определяем пики с границами
            peaks_data = find_peaks_with_bounds(
                y_smoothed,
                x_coords,
                min_height=0,
                min_distance=min_distance_slider,
                prominence=prominence_slider.value,
                threshold_ratio= threshold_ratio_slider.value
            )

            # Фильтр по минимальной ширине (ΔX между границами)
            min_width = float(min_peak_width_slider.value)
            peaks_data = [p for p in peaks_data if float(p.get('width', 0.0)) >= min_width]

            if len(peaks_data) > 0:
                # Создаем данные для графика
                plot_data_peaks = {
                    "x": x_coords if isinstance(x_coords, list) else x_coords.tolist(),
                    "y": y_smoothed.tolist()
                }

                # Собираем координаты вертикальных линий
                bound_xintercepts = []
                bound_types = []
                for peak in peaks_data:
                    bound_xintercepts.append(peak['left_bound'])
                    bound_types.append('Левая граница')
                    bound_xintercepts.append(peak['peak_x'])
                    bound_types.append('Пик')
                    bound_xintercepts.append(peak['right_bound'])
                    bound_types.append('Правая граница')

                bounds_data2 = {
                    'xintercept': bound_xintercepts,
                    'type': bound_types
                }

                # График с пиками и границами
                plot_peaks = (
                    ggplot(plot_data_peaks)
                    + geom_line(aes(x="x", y="y"), color='blue', size=1)
                    + labs(
                        x="2θ",
                        y="Интенсивность",
                        title=f"Определенные пики ({len(peaks_data)} шт.) с границами",
                        color="Элементы"
                    )
                    + ggsize(1000, 450)
                )

                # Создаем таблицу с информацией о пиках
                peaks_table_data = []
                for it, peak in enumerate(peaks_data, 1):
                    peaks_table_data.append({
                        'Номер': it,
                        'X пика': round(peak['peak_x'], 2),
                        'Интенсивность': round(peak['peak_y'], 2),
                        'Левая граница': round(peak['left_bound'], 2),
                        'Правая граница': round(peak['right_bound'], 2),
                        'Ширина': round(peak['width'], 2),
                        'Значимость': round(peak['prominence'], 2)
                    })

                df_peaks = pl.DataFrame(peaks_table_data)

                result_peaks = mo.vstack([
                    mo.md("## 📊 Определение пиков"),
                    mo.md(f"**Найдено пиков:** {len(peaks_data)}"),
                    mo.md(f"**Параметры:** значимость ≥ {prominence_slider.value}, ширина ≥ {min_peak_width_slider.value}"),
                    mo.md("### 📋 Таблица пиков"),
                    mo.ui.table(df_peaks)
                ])
            else:
                result_peaks = mo.vstack([
                    mo.md("## 📊 Определение пиков"),
                    mo.md("**⚠️ Пики не найдены.** Попробуйте изменить параметры (уменьшить значимость/расстояние или минимальную ширину).")
                ])

        except Exception as e:
            result_peaks = mo.vstack([
                mo.md("## ❌ Ошибка определения пиков"),
                mo.md(f"**Ошибка:** {str(e)}")
            ])
    else:
        result_peaks = mo.md("**⏳ Ожидание сглаженных данных...**")
        peaks_data = []

    result_peaks
    return (peaks_data,)


@app.cell
def _():
    return


@app.cell
def _(mo):
    # Ручной ввод порога tail-weighted R²
    # Ограничение: 0.80 ≤ threshold ≤ 0.999
    r2_threshold_input = mo.ui.text(
        value="0.993",
        label="Порог качества (tail-weighted R²) — ввод вручную"
    )

    mo.vstack([
        mo.md("## ⚙️ Настройка аппроксимации"),
        mo.md("**Порог качества** - минимальное качество аппроксимации для остановки добавления компонент."),
        mo.md("*Метрика: tail-weighted R² (хвосты имеют больший вес). Если score ≥ порога, добавление компонент прекращается.*"),
        mo.md("**Ограничения ввода:** значение должно быть числом в диапазоне **[0.80; 0.999]**. Если ввод невалиден — будет использовано значение по умолчанию **0.993**."),
        r2_threshold_input,
    ])
    return (r2_threshold_input,)


@app.cell
def _(
    aes,
    curve_fit,
    df,
    geom_line,
    ggplot,
    ggsize,
    labs,
    lp,
    mo,
    np,
    peaks_data,
    pl,
    r2_threshold_input,
    savgol_filter,
    x_coords,
    y_coords,
):
    # ==============================================================================
    def _effective_r2_threshold() -> float:
        """
        Порог качества: ввод вручную.
        Если ввод невалиден — используем значение по умолчанию 0.993.
        """
        default_v = 0.993
        try:
            v = float(str(r2_threshold_input.value).strip())
        except Exception:
            return float(default_v)
        if not (0.80 <= v <= 0.999):
            return float(default_v)
        return float(v)

    # === СЕКЦИЯ 1: МАТЕМАТИКА ===
    # Базовые математические функции для аппроксимации
    # ==============================================================================

    def pseudo_voigt(x, A, mu, sigma, eta):
        """
        Функция псевдо-Войгта.

        f(x) = A * [η * L(x) + (1 - η) * G(x)]

        Параметры:
        - A: амплитуда
        - mu: центр
        - sigma: ширина (FWHM, единая параметризация для Лоренца и Гаусса)
        - eta: доля лоренцевой компоненты [0, 1]
        """
        x = np.asarray(x, dtype=float)
        # sigma трактуем как FWHM (как в распространённых реализациях псевдо‑Воигта)
        # L(x): 1 / (1 + 4 * ((x-mu)/FWHM)^2)
        # G(x): exp(-4*ln(2) * ((x-mu)/FWHM)^2)
        sigma = np.maximum(np.asarray(sigma, dtype=float), 1e-12)
        z = (x - float(mu)) / sigma

        # Лоренцева компонента (нормирована: L(mu)=1)
        L = 1.0 / (1.0 + 4.0 * (z ** 2))

        # Гауссова компонента (нормирована: G(mu)=1, ширина задаётся FWHM)
        G = np.exp(-4.0 * np.log(2.0) * (z ** 2))

        # Псевдо-Войгт
        return A * (eta * L + (1 - eta) * G)


    def calculate_tail_weighted_r2(y_true, y_pred, tail_weight_max=10.0):
        """
        Tail-weighted R² (bounded, rank-based).

        Цель: сильнее штрафовать ошибку на "хвостах" (низких интенсивностях),
        но без взрывных весов, чтобы score оставался сопоставим с обычным R².

        Идея:
        - сортируем y_true по возрастанию (самый "хвост" = минимальные y)
        - назначаем веса в диапазоне [1, tail_weight_max]:
            w(low y) = tail_weight_max, w(high y) = 1
        - считаем взвешенный R²
        """
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        n = len(y_true)
        if n == 0:
            return 0.0
        if n == 1:
            return 1.0

        # rank_low: 0 для минимальных y (хвост), n-1 для максимальных y (вершина)
        order = np.argsort(y_true)
        rank_low = np.empty(n, dtype=float)
        rank_low[order] = np.arange(n, dtype=float)

        tw = float(max(1.0, tail_weight_max))
        w = 1.0 + (tw - 1.0) * (1.0 - (rank_low / (n - 1.0)))
        w = w / np.sum(w)

        y_mean_w = np.sum(w * y_true)
        ss_res = np.sum(w * (y_true - y_pred) ** 2)
        ss_tot = np.sum(w * (y_true - y_mean_w) ** 2)
        if ss_tot == 0:
            return 0.0
        return 1.0 - (ss_res / ss_tot)


    def estimate_initial_sigma(x, y, peak_idx, peak_y):
        """
        Оценка начальной ширины sigma на основе данных пика.

        Использует ширину на половине высоты (FWHM approximation).
        """
        half_height = peak_y / 2

        # Ищем пересечение слева от пика
        left_idx = peak_idx
        while left_idx > 0 and y[left_idx] > half_height:
            left_idx -= 1

        # Ищем пересечение справа от пика
        right_idx = peak_idx
        while right_idx < len(y) - 1 and y[right_idx] > half_height:
            right_idx += 1

        # Ширина на половине высоте (FWHM)
        fwhm = float(x[right_idx] - x[left_idx])
        dx = float(x[1] - x[0]) if len(x) > 1 else 1.0
        # Так как pseudo_voigt использует sigma как FWHM, возвращаем FWHM
        return max(fwhm, dx * 2.0)


    # ==============================================================================
    # === СЕКЦИЯ 3: СЕГМЕНТАЦИЯ ===
    # Разбиение пика на сегменты по анализу формы
    # ==============================================================================

    def split_peak_into_segments(x, y, max_segments=3):
        """
        Разбиение пика на сегменты на основе анализа формы.

        Возвращает:
        - список кортежей [(x_start, x_end), ...]
        """
        # Ранее здесь был "костыль" с поиском локальных минимумов, который мог
        # искусственно ограничивать/резать пик.
        #
        # В текущей архитектуре сегментация для piecewise-fit выполняется внутри
        # fit_peak_with_domain_components (через оптимизацию границ), поэтому
        # отдельная эвристика по локальным минимумам не нужна.
        x = np.asarray(x, dtype=float)
        if len(x) == 0:
            return []
        return [(float(x[0]), float(x[-1]))]


    # ==============================================================================
    # === СЕКЦИЯ 2: PIECEWISE ЯДРО ===
    # Реализация чистой piecewise-модели без масок и сумм
    # ==============================================================================
    #
    # АРХИТЕКТУРНОЕ РАЗДЕЛЕНИЕ РОЛЕЙ:
    # ========================================
    #
    # 1) FINAL MODEL (piecewise_multi_voigt)
    #    - Используется для ФИНАЛЬНОЙ аппроксимации пика
    #    - Визуализируется как итоговый fit
    #    - Используется в физическом смысле
    #    - НЕ агрегирует компоненты
    #
    # 2) COMPLEXITY SELECTION SCORE (calculate_aggregated_score)
    #    - Вспомогательная диагностическая метрика
    #    - Используется ТОЛЬКО для выбора числа компонент
    #    - НЕ является моделью пика
    #    - НЕ возвращается наружу
    #    - НЕ визуализируется как fit
    #
    # ПОЧЕМУ ДВА ПОДХОДА?
    # ============================
    #
    # Piecewise-модель с жёсткой сегментацией:
    # - НЕ увеличивает выразительность при добавлении компонент
    # - Каждая точка X описывается РОВНО одной компонентой
    # - Поэтому R²(piecewise) почти не меняется при увеличении n
    #
    # Следовательно:
    # - R²(piecewise) НЕ МОЖЕТ использоваться для выбора сложности
    # - Нужна диагностическая метрика, агрегирующая все компоненты
    #
    # ВАЖНО:
    # =======
    # - Финальная модель ВСЕГДА piecewise
    # - Slider управляет ТОЛЬКО диагностической метрикой
    # - Диагностическая метрика НЕ является аппроксимацией
    #
    # ==============================================================================

    def piecewise_multi_voigt(x, *params, segments=None):
        """
        ЧИСТАЯ piecewise-функция для curve_fit.

        ФИНАЛЬНАЯ МОДЕЛЬ: y(x) = Voigt_i(x), если x ∈ segment_i

        Ключевые принципы:
        - Каждая точка x аппроксимируется РОВНО ОДНОЙ компонентой
        - НЕ суммируются компоненты
        - НЕ используются маски
        - Voigt остается симметричным внутри своего сегмента

        params = [A1, mu1, sigma1, eta1, A2, mu2, sigma2, eta2, ...]
        segments = [(x_start_1, x_end_1), (x_start_2, x_end_2), ...]

        Возвращает:
        - y_pred: предсказанные значения (только piecewise, не сумма!)
        """
        y_pred = np.zeros_like(x)
        num_components = len(params) // 4

        # Для каждой компоненты вычисляем её значения ТОЛЬКО на своём сегменте
        for i in range(num_components):
            idx = i * 4
            A, mu, sigma, eta = params[idx:idx+4]

            # Определяем сегмент этой компоненты
            if segments is not None and i < len(segments):
                x_start, x_end = segments[i]
                # Маска точек, принадлежащих сегменту
                mask = (x >= x_start) & (x <= x_end)
            else:
                # Если сегмент не задан - пропускаем эту компоненту
                continue

            # Вычисляем Voigt ТОЛЬКО на точках сегмента
            if np.any(mask):
                x_segment = x[mask]
                voigt_values = pseudo_voigt(x_segment, A, mu, sigma, eta)

                # Присваиваем значения соответствующим точкам
                y_pred[mask] = voigt_values

        return y_pred


    def create_piecewise_fit_func(segments):
        """
        Создает замыкание для piecewise_multi_voigt с фиксированными segments.

        Это нужно для curve_fit, который не поддерживает kwargs.
        """
        def fit_func(x, *params):
            return piecewise_multi_voigt(x, *params, segments=segments)

        return fit_func


    def calculate_piecewise_prediction(x_peak, components, segments):
        """
        Вычисление piecewise-предсказания для ФИНАЛЬНОЙ модели.

        Использует ТОТ ЖЕ код, что и оптимизация - гарантирует согласованность.

        Возвращает:
        - y_pred: предсказанные значения (piecewise)
        """
        # Формируем params из components
        params = []
        for comp in components:
            params.extend([comp['A'], comp['mu'], comp['sigma'], comp['eta']])

        # Вычисляем piecewise-предсказание
        y_pred = piecewise_multi_voigt(x_peak, *params, segments=segments)

        return y_pred


    # ==============================================================================
    # === СЕКЦИЯ 4: FIT ОДНОГО ПИКА ===
    # Адаптивная аппроксимация с piecewise-подходом
    # ==============================================================================

    def fit_peak_with_domain_components(x_peak, y_peak, r2_threshold=0.95, max_components=3):
        """
        Piecewise-аппроксимация пика псевдо-Войгтами (БЕЗ сумм), критерий — tail-R²:
        - каждая точка принадлежит ровно одному сегменту → оценивается ровно одной компонентой
        - число компонент растёт строго 1 → 2 → 3 только если tail-R² текущей модели ниже порога
        - сегментация НЕ фиксируется заранее: границы сегментов подстраиваем, минимизируя SSE
        """
        x_peak = np.asarray(x_peak, dtype=float)
        y_peak = np.asarray(y_peak, dtype=float)
        r2_threshold = float(r2_threshold)  # явная зависимость от slider
        max_components = int(max_components)

        N = len(x_peak)
        # Линейный baseline на пик (по концам интервала) — нужен, чтобы корректно описывать "хвосты"
        # и не заставлять Voigt "тащить" фон.
        def _baseline_endpoints() -> dict:
            if len(x_peak) == 0:
                return {'x1': 0.0, 'y1': 0.0, 'x2': 0.0, 'y2': 0.0}
            x0 = float(x_peak[0])
            x1 = float(x_peak[-1])
            y0 = float(y_peak[0]) if len(y_peak) > 0 else 0.0
            y1 = float(y_peak[-1]) if len(y_peak) > 0 else y0
            return {'x1': x0, 'y1': y0, 'x2': x1, 'y2': y1}

        def _baseline_vec(xs: np.ndarray) -> np.ndarray:
            xs = np.asarray(xs, dtype=float)
            if len(x_peak) == 0:
                return np.zeros_like(xs, dtype=float)
            ep = _baseline_endpoints()
            x0 = float(ep['x1'])
            x1 = float(ep['x2'])
            y0 = float(ep['y1'])
            y1 = float(ep['y2'])
            if x1 == x0:
                return np.full_like(xs, y0, dtype=float)
            t = (xs - x0) / (x1 - x0)
            return y0 + (y1 - y0) * t

        y_peak_corr = y_peak - _baseline_vec(x_peak)
        if N < 4:
            # Фолбэк на очень коротком массиве
            segments = [(float(x_peak.min()), float(x_peak.max()))] if N else []
            max_idx = int(np.argmax(y_peak)) if N else 0
            mu = float(x_peak[max_idx]) if N else 0.0
            A = float(y_peak_corr[max_idx]) if N else 0.0
            sigma = float(abs(x_peak[1] - x_peak[0])) if N > 1 else 1.0
            comp = {'A': A, 'mu': mu, 'sigma': sigma, 'eta': 0.5, 'support': segments[0] if segments else (0.0, 0.0)}
            y_pred_corr = pseudo_voigt(x_peak, comp['A'], comp['mu'], comp['sigma'], comp['eta']) if N else np.array([], dtype=float)
            y_pred = y_pred_corr + _baseline_vec(x_peak) if N else np.array([], dtype=float)
            score = float(calculate_tail_weighted_r2(y_peak, y_pred)) if N else 0.0
            outlier_score_threshold = 0.90
            # Строго по правилу проекта: выброс, если score_at_max_components < 0.90
            # (даже если массив короткий и 3 компоненты технически не фитятся)
            is_outlier = (max_components >= 3) and (float(score) < float(outlier_score_threshold))
            return {
                'components': [comp] if segments else [],
                'y_pred': y_pred,
                'score': score,
                'score_at_max_components': score,
                'outlier_score_threshold': outlier_score_threshold,
                'is_outlier': is_outlier,
                'num_components': 1 if segments else 0,
                'segments': segments,
                'baseline': _baseline_endpoints(),
            }

        dx = float(np.median(np.diff(x_peak))) if N > 2 else float(x_peak[1] - x_peak[0])
        y_global_max = float(np.max(y_peak_corr)) if N else 1.0

        def _segments_from_splits(split_indices, n, N_):
            """split_indices: индексы окончания левого сегмента (включительно)."""
            if n <= 1:
                return [(0, N_ - 1)]
            split_indices = sorted(int(s) for s in split_indices)
            segs = []
            start = 0
            for s in split_indices:
                segs.append((start, int(s)))
                start = int(s) + 1
            segs.append((start, N_ - 1))
            return segs

        def _x_segments(seg_idx_ranges):
            return [(float(x_peak[i0]), float(x_peak[i1])) for (i0, i1) in seg_idx_ranges]

        def _piecewise_predict(seg_idx_ranges, components):
            y_pred_local = np.zeros_like(x_peak, dtype=float)
            for (i0, i1), c in zip(seg_idx_ranges, components):
                xs = x_peak[i0:i1+1]
                y_pred_local[i0:i1+1] = pseudo_voigt(xs, c['A'], c['mu'], c['sigma'], c['eta'])
            return y_pred_local

        def _fit_component(i0, i1):
            xs = x_peak[i0:i1+1]
            ys = y_peak_corr[i0:i1+1]

            # начальные оценки
            max_idx = int(np.argmax(ys))
            mu0 = float(xs[max_idx])
            A0 = float(ys[max_idx])
            sigma0 = float(estimate_initial_sigma(xs, ys, max_idx, A0))
            eta0 = 0.5

            # bounds по support сегмента
            x_min_s = float(xs[0])
            x_max_s = float(xs[-1])
            width = float(max(x_max_s - x_min_s, dx))
            A_min, A_max = 0.0, float(max(y_global_max * 2.0, A0 * 2.0, 1e-6))
            mu_min, mu_max = x_min_s, x_max_s
            sigma_min, sigma_max = float(max(dx, 1e-6)), float(max(width * 2.0, dx * 3.0))
            eta_min, eta_max = 0.0, 1.0

            p0 = [A0, mu0, sigma0, eta0]
            bounds = ([A_min, mu_min, sigma_min, eta_min], [A_max, mu_max, sigma_max, eta_max])

            try:
                popt, _ = curve_fit(
                    pseudo_voigt,
                    xs,
                    ys,
                    p0=p0,
                    bounds=bounds,
                    maxfev=15000,
                    method='trf'
                )
                A, mu, sigma, eta = [float(v) for v in popt]
            except Exception:
                A, mu, sigma, eta = float(A0), float(mu0), float(sigma0), float(eta0)

            y_fit = pseudo_voigt(xs, A, mu, sigma, eta)
            sse = float(np.sum((ys - y_fit) ** 2))
            comp = {'A': A, 'mu': mu, 'sigma': sigma, 'eta': eta, 'support': (x_min_s, x_max_s)}
            return comp, sse

        def _clamp_splits(splits, n, N_, min_points):
            """
            Приводит split'ы к допустимым, чтобы:
            - было ровно n сегментов
            - каждый сегмент имел >= min_points точек
            """
            if n <= 1:
                return []
            splits = list(sorted(int(s) for s in splits))
            new_splits = []
            start = 0
            for j in range(n - 1):
                desired = splits[j] if j < len(splits) else (start + min_points - 1)
                lo = start + min_points - 1
                remaining = (n - 1) - j
                hi = N_ - (remaining * min_points) - 1
                s = int(np.clip(desired, lo, hi))
                new_splits.append(s)
                start = s + 1
            return list(sorted(new_splits))

        def _fit_piecewise_for_n(n):
            # минимум точек, чтобы fit не разваливался
            min_points = max(6, int(round(N * 0.08)))
            if n * min_points > N:
                min_points = max(4, N // n)

            # стартовые границы: равные по числу точек сегменты
            splits = []
            for k in range(1, n):
                s = int(round(k * N / n)) - 1
                splits.append(s)
            splits = _clamp_splits(splits, n, N, min_points)

            seg_idx = _segments_from_splits(splits, n, N)

            # первоначальный фит всех сегментов
            comps = []
            seg_sse = []
            for (i0, i1) in seg_idx:
                comp, sse = _fit_component(i0, i1)
                comps.append(comp)
                seg_sse.append(sse)

            # оптимизация границ: координатный спуск по каждому split
            max_shift = max(3, int(round(N * 0.08)))
            for _ in range(6):
                improved = False
                seg_idx = _segments_from_splits(splits, n, N)

                for j in range(n - 1):
                    cur = int(splits[j])
                    left_start = seg_idx[j][0]
                    right_end = seg_idx[j + 1][1]
                    lo = left_start + min_points - 1
                    hi = right_end - min_points
                    if lo > hi:
                        continue

                    cand_lo = max(lo, cur - max_shift)
                    cand_hi = min(hi, cur + max_shift)
                    if cand_lo > cand_hi:
                        continue

                    fixed_sse = float(np.sum([seg_sse[k] for k in range(len(seg_sse)) if k not in (j, j+1)]))
                    best_split = cur
                    best_pair_sse = float(seg_sse[j] + seg_sse[j+1])
                    best_pair = None

                    for cand in range(cand_lo, cand_hi + 1):
                        test_splits = list(splits)
                        test_splits[j] = int(cand)
                        test_splits = _clamp_splits(test_splits, n, N, min_points)
                        test_seg = _segments_from_splits(test_splits, n, N)
                        if len(test_seg) != n:
                            continue

                        (l0, l1) = test_seg[j]
                        (r0, r1) = test_seg[j+1]
                        comp_l, sse_l = _fit_component(l0, l1)
                        comp_r, sse_r = _fit_component(r0, r1)
                        pair_sse = float(sse_l + sse_r)
                        if pair_sse < best_pair_sse - 1e-9:
                            best_pair_sse = pair_sse
                            best_split = int(cand)
                            best_pair = (test_splits, test_seg, comp_l, comp_r, float(sse_l), float(sse_r))

                    if best_pair is not None:
                        test_splits, test_seg, comp_l, comp_r, sse_l, sse_r = best_pair
                        splits = test_splits
                        seg_idx = test_seg
                        comps[j] = comp_l
                        comps[j+1] = comp_r
                        seg_sse[j] = sse_l
                        seg_sse[j+1] = sse_r
                        improved = True

                if not improved:
                    break

            # синхронизируем support + предикт
            seg_idx = _segments_from_splits(splits, n, N)
            x_segs = _x_segments(seg_idx)
            for c, seg in zip(comps, x_segs):
                c['support'] = (float(seg[0]), float(seg[1]))

            # === Уточнение: joint fit baseline + piecewise-Voigt при фиксированных сегментах ===
            # Это устраняет систематические смещения хвостов (baseline "учится" вместе с Voigt),
            # но не ломает текущую сегментацию/архитектуру.
            def _fit_func_with_baseline(x, b0, b1, *voigt_params):
                x = np.asarray(x, dtype=float)
                baseline = b0 + b1 * x
                y_pred_local = np.array(baseline, copy=True)
                # индексная piecewise-логика (каждая точка принадлежит ровно одному сегменту)
                if len(x) == N:
                    for i in range(n):
                        p = voigt_params[i * 4:(i + 1) * 4]
                        A, mu, sigma, eta = p
                        i0, i1 = seg_idx[i]
                        xs = x[i0:i1 + 1]
                        y_pred_local[i0:i1 + 1] = baseline[i0:i1 + 1] + pseudo_voigt(xs, A, mu, sigma, eta)
                    return y_pred_local
                # фолбэк (на всякий случай) — по маскам
                for i in range(n):
                    p = voigt_params[i * 4:(i + 1) * 4]
                    A, mu, sigma, eta = p
                    x_start, x_end = x_segs[i]
                    mask = (x >= x_start) & (x <= x_end)
                    y_pred_local[mask] = baseline[mask] + pseudo_voigt(x[mask], A, mu, sigma, eta)
                return y_pred_local

            # p0 для baseline из "концов" + параметры компонент из сегментных fit
            ep0 = _baseline_endpoints()
            x1, y1 = float(ep0['x1']), float(ep0['y1'])
            x2, y2 = float(ep0['x2']), float(ep0['y2'])
            if x2 != x1:
                b1_0 = (y2 - y1) / (x2 - x1)
            else:
                b1_0 = 0.0
            b0_0 = y1 - b1_0 * x1

            p0 = [float(b0_0), float(b1_0)]
            for c in comps:
                p0.extend([float(c['A']), float(c['mu']), float(c['sigma']), float(c['eta'])])

            # bounds: baseline достаточно широкие; для компонент — как раньше, но глобально
            y_min = float(np.min(y_peak)) if N else 0.0
            y_max = float(np.max(y_peak)) if N else 1.0
            y_rng = float(max(1e-6, y_max - y_min))
            x_rng = float(max(1e-12, float(x_peak[-1] - x_peak[0])))

            b0_lo = y_min - 2.0 * y_rng
            b0_hi = y_max + 2.0 * y_rng
            b1_abs = (y_rng / x_rng) * 10.0
            b1_lo, b1_hi = -b1_abs, b1_abs

            lower = [b0_lo, b1_lo]
            upper = [b0_hi, b1_hi]

            # оценка амплитуд относительно стартового baseline
            y_corr0 = y_peak - (b0_0 + b1_0 * x_peak)
            y_corr_max = float(np.max(y_corr0)) if N else 1.0
            y_corr_max = float(max(y_corr_max, 1e-6))

            for i, (c, seg) in enumerate(zip(comps, x_segs)):
                x_min_s, x_max_s = float(seg[0]), float(seg[1])
                width = float(max(x_max_s - x_min_s, dx))
                A0 = float(c['A'])
                A_min, A_max = 0.0, float(max(y_corr_max * 3.0, A0 * 3.0, 1e-6))
                mu_min, mu_max = x_min_s, x_max_s
                sigma_min, sigma_max = float(max(dx, 1e-6)), float(max(width * 2.0, dx * 3.0))
                eta_min, eta_max = 0.0, 1.0
                lower.extend([A_min, mu_min, sigma_min, eta_min])
                upper.extend([A_max, mu_max, sigma_max, eta_max])

            try:
                popt, _ = curve_fit(
                    _fit_func_with_baseline,
                    x_peak,
                    y_peak,
                    p0=p0,
                    bounds=(lower, upper),
                    maxfev=15000,
                    method='trf',
                )
                b0_hat = float(popt[0])
                b1_hat = float(popt[1])
                # обновляем компоненты из joint fit
                comps_hat = []
                for i, seg in enumerate(x_segs):
                    idx = 2 + i * 4
                    A, mu, sigma, eta = [float(v) for v in popt[idx:idx + 4]]
                    comps_hat.append({'A': A, 'mu': mu, 'sigma': sigma, 'eta': eta, 'support': (float(seg[0]), float(seg[1]))})
                comps = comps_hat
                y_pred_total = _fit_func_with_baseline(x_peak, b0_hat, b1_hat, *popt[2:])
                baseline_ep = {
                    'x1': float(x_peak[0]),
                    'y1': float(b0_hat + b1_hat * float(x_peak[0])),
                    'x2': float(x_peak[-1]),
                    'y2': float(b0_hat + b1_hat * float(x_peak[-1])),
                }
            except Exception:
                # фолбэк на старую схему: baseline по концам + сегментные компоненты
                y_pred = _piecewise_predict(seg_idx, comps)
                y_pred_total = y_pred + _baseline_vec(x_peak)
                baseline_ep = _baseline_endpoints()

            score = float(calculate_tail_weighted_r2(y_peak, y_pred_total))
            return {
                'components': comps,
                'y_pred': y_pred_total,
                'score': score,
                'num_components': n,
                'segments': x_segs,
                'baseline': baseline_ep,
            }

        # Последовательное увеличение числа компонент:
        # - "выбранный" результат определяется порогом slider (score >= r2_threshold)
        # - дополнительно ВСЕГДА считаем качество на max_components (для фильтра выбросов)
        chosen_res = None
        last_res = None

        for n in range(1, max_components + 1):
            res = _fit_piecewise_for_n(n)
            last_res = res
            if chosen_res is None and float(res['score']) >= r2_threshold:
                chosen_res = res

        if chosen_res is None:
            chosen_res = last_res

        # Выброс: если даже на 3 компонентах tail-weighted R² < 0.90 → некорректный пик
        outlier_score_threshold = 0.90
        score_at_max = float(last_res['score']) if last_res is not None else 0.0
        is_outlier = (max_components >= 3) and (score_at_max < outlier_score_threshold)

        # Возвращаем выбранный результат + диагностические поля
        chosen_res = dict(chosen_res)
        chosen_res['score_at_max_components'] = score_at_max
        chosen_res['outlier_score_threshold'] = outlier_score_threshold
        chosen_res['is_outlier'] = is_outlier
        # baseline уже определён внутри _fit_piecewise_for_n (joint fit baseline+Voigt),
        # не перезатираем его эвристикой "по концам"
        return chosen_res


    # ==============================================================================
    # === СЕКЦИЯ 5: ВИЗУАЛИЗАЦИЯ И ГЛАВНЫЙ БЛОК ===
    # Обработка пиков, аппроксимация и отображение результатов
    # ==============================================================================

    # Основной блок выполнения
    if df is not None and len(y_coords) > 10 and len(peaks_data) > 0:
        try:
            lp.LetsPlot.setup_html()

            # 1) Перед аппроксимацией сглаживаем ВЕСЬ исходный сигнал фиксированным SG (окно=5, poly=2)
            # (границы пиков уже найдены ранее, здесь сглаживание нужно только для устойчивого fit)
            y_coords_arr = np.asarray(y_coords, dtype=float)
            if len(y_coords_arr) >= 5:
                y_coords_fit = savgol_filter(y_coords_arr, 5, 2)
            else:
                y_coords_fit = y_coords_arr

            # Список для хранения результатов аппроксимации
            fitting_results = []
            plot_elements = []

            # Обрабатываем каждый пик
            for i, peak_info in enumerate(peaks_data):
                left_idx = peak_info['left_bound_idx']
                right_idx = peak_info['right_bound_idx']

                # Данные для fit берём по уточнённым границам
                x_peak = np.array(x_coords[left_idx:right_idx+1])
                y_peak_raw = np.array(y_coords_fit[left_idx:right_idx+1], dtype=float)
                y_peak = y_peak_raw

                # Линейный baseline по концам пика (для корректных "хвостов" в fit/визуализации)
                def _baseline_vec_peak(xs: np.ndarray) -> np.ndarray:
                    xs = np.asarray(xs, dtype=float)
                    if len(x_peak) == 0:
                        return np.zeros_like(xs, dtype=float)
                    x0 = float(x_peak[0])
                    x1 = float(x_peak[-1])
                    y0 = float(y_peak[0]) if len(y_peak) > 0 else 0.0
                    y1 = float(y_peak[-1]) if len(y_peak) > 0 else y0
                    if x1 == x0:
                        return np.full_like(xs, y0, dtype=float)
                    t = (xs - x0) / (x1 - x0)
                    return y0 + (y1 - y0) * t

                # Выполняем аппроксимацию с domain-based подходом
                result = fit_peak_with_domain_components(
                    x_peak, y_peak,
                    r2_threshold=_effective_r2_threshold(),
                    max_components=3
                )

                result['peak_info'] = peak_info
                result['x_peak'] = x_peak
                result['y_peak'] = y_peak
                fitting_results.append(result)

                # Создаем график для этого пика
                plot_data = {
                    'x': x_peak.tolist(),
                    'y': y_peak.tolist(),
                    'type': ['Исходные данные'] * len(x_peak)
                }

                # Добавляем компоненты для визуализации (на всём интервале пика)
                # Piecewise-логика используется ТОЛЬКО для fit и R²
                # Визуализация — диагностическая: показываем симметрию Voigt
                # baseline берём из результата fit (joint baseline+Voigt), чтобы визуализация совпадала с моделью
                bl = result.get('baseline', None)
                if isinstance(bl, dict) and all(k in bl for k in ('x1', 'y1', 'x2', 'y2')) and float(bl['x2']) != float(bl['x1']):
                    x1b, y1b = float(bl['x1']), float(bl['y1'])
                    x2b, y2b = float(bl['x2']), float(bl['y2'])
                    baseline_vals = y1b + (y2b - y1b) * (x_peak - x1b) / (x2b - x1b)
                else:
                    baseline_vals = _baseline_vec_peak(x_peak)
                colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
                for j, comp in enumerate(result['components']):
                    # Вычисляем Voigt НА ВСЁМ интервале пика (без ограничений!)
                    comp_values = pseudo_voigt(x_peak, comp['A'], comp['mu'], comp['sigma'], comp['eta'])
                    model_values = comp_values + baseline_vals

                    # Добавляем все точки (показываем ТОЛЬКО суммарную кривую = baseline + компонента)
                    plot_data['x'] += x_peak.tolist()
                    plot_data['y'] += model_values.tolist()
                    plot_data['type'] += [f'Компонента {j+1} (baseline + Voigt)'] * len(x_peak)

                # Создаем plot без vertical lines
                outlier_suffix = " (ВЫБРОС: tail-R²@3 < 0.90)" if result.get('is_outlier') else ""
                plot = (
                    ggplot(plot_data)
                    + geom_line(aes(x='x', y='y', color='type'), size=1.2)
                    + labs(
                        x=f"2θ (центр: {peak_info['peak_x']:.2f})",
                        y="Интенсивность",
                        title=f"Пик {i+1}: {result['num_components']} компонент(ы), tail-R² = {float(result['score']):.4f}{outlier_suffix}",
                        color="Легенда"
                    )
                    + ggsize(800, 400)
                )

                plot_elements.append({
                    'title': f"Пик {i+1}",
                    'plot': plot,
                    'score': result.get('score', None),
                    'num_components': result['num_components'],
                    'segments': result['segments'],
                    'is_outlier': result.get('is_outlier', False),
                    'score_at_max_components': result.get('score_at_max_components', None)
                })

            # Создаем таблицу результатов
            results_table = []
            outliers_count = 0
            for i, res in enumerate(fitting_results):
                is_outlier = bool(res.get('is_outlier', False))
                if is_outlier:
                    outliers_count += 1
                comp_info = ", ".join([
                    f"μ={c['mu']:.2f}, A={c['A']:.1f}, support=[{c['support'][0]:.1f}, {c['support'][1]:.1f}]"
                    for c in res['components']
                ])
                results_table.append({
                    'Номер': i + 1,
                    'X пика': round(res['peak_info']['peak_x'], 2),
                    'Компонент': res['num_components'],
                    'tail-R²': None if res.get('score') is None else round(float(res['score']), 4),
                    'Выброс': is_outlier,
                    'tail-R²@3': None if res.get('score_at_max_components') is None else round(float(res['score_at_max_components']), 4),
                    'Support': len(res['segments']),
                    'Параметры': comp_info
                })

            df_results = pl.DataFrame(results_table)

            # Формируем результат
            display_plots = mo.vstack([
                mo.md("## 📈 Аппроксимация пиков функциями псевдо-Войгта (Piecewise)"),
                mo.md(f"**Порог tail-R²:** {_effective_r2_threshold():.3f}, **Обработано пиков:** {len(fitting_results)}"),
                mo.md(f"**Фильтр выбросов:** если даже при 3 компонентах tail-R² < 0.90 → пик помечается как выброс. Сейчас выбросов: **{outliers_count}**."),
                mo.md("*Модель: piecewise (каждая точка описывается одной компонентой)*"),
                mo.md("*Визуализация: компоненты показаны полностью для оценки симметрии*"),
                mo.md("### 📊 Графики аппроксимации для каждого пика"),
            ])

            for elem in plot_elements:
                segments_str = ", ".join([f"[{s[0]:.1f}, {s[1]:.1f}]" for s in elem['segments']])
                display_plots = mo.vstack([
                    display_plots,
                    mo.hstack([
                        elem['plot'],
                        mo.md(f"**{elem['title']}**\n\n• Компонент: {elem['num_components']}\n• tail-R² = {float(elem['score']):.4f}\n• Supports: {segments_str}")
                    ])
                ])

            display_plots = mo.vstack([
                display_plots,
                mo.md("### 📋 Таблица результатов"),
                mo.ui.table(df_results)
            ])

        except Exception as e:
            display_plots = mo.vstack([
                mo.md("## ❌ Ошибка аппроксимации"),
                mo.md(f"**Ошибка:** {str(e)}")
            ])
            print(f"Ошибка: {str(e)}")
            import traceback
            traceback.print_exc()
    else:
        display_plots = mo.md("**⏳ Ожидание данных пиков...**")
        fitting_results = []
        y_coords_fit = np.asarray(y_coords, dtype=float) if y_coords is not None else np.array([], dtype=float)

    display_plots
    return fitting_results, y_coords_fit


@app.cell
def _(
    aes,
    df,
    fitting_results,
    geom_line,
    ggplot,
    ggsize,
    labs,
    lp,
    mo,
    np,
    pl,
    x_coords,
    y_coords_fit,
):
    def _():
        # Финальный общий график и таблица метрик компонент в отдельной ячейке
        if df is None or fitting_results is None or len(fitting_results) == 0:
            final_view = mo.md("**⏳ Ожидание результатов аппроксимации...**")
        else:
            lp.LetsPlot.setup_html()

            def pseudo_voigt(x, A, mu, sigma, eta):
                x = np.asarray(x, dtype=float)
                # sigma трактуем как FWHM (единая параметризация как в fit-ячейке)
                sigma = np.maximum(np.asarray(sigma, dtype=float), 1e-12)
                z = (x - float(mu)) / sigma
                L = 1.0 / (1.0 + 4.0 * (z ** 2))
                G = np.exp(-4.0 * np.log(2.0) * (z ** 2))
                return A * (eta * L + (1 - eta) * G)

            x_all = np.asarray(x_coords, dtype=float)
            y_all = np.asarray(y_coords_fit, dtype=float)

            # ---------- График: данные + компоненты + baseline ----------
            global_plot_data = {
                'x': x_all.tolist(),
                'y': y_all.tolist(),
                'type': ['Данные (SG 5,2)'] * len(x_all),
            }

            def _interp_y(xq: float) -> float:
                return float(np.interp(xq, x_all, y_all))

            def _baseline_at(xq: float, x1: float, y1: float, x2: float, y2: float) -> float:
                if x2 == x1:
                    return float(y1)
                return float(y1 + (y2 - y1) * (xq - x1) / (x2 - x1))

            # Компоненты рисуем только на их support.
            # Baseline рисуем ОДИН на пик (между общими границами support всех компонент).
            # Выбросы не показываем и не используем в метриках
            fitting_results_valid = [r for r in fitting_results if not bool(r.get('is_outlier', False))]

            for peak_idx, res in enumerate(fitting_results_valid, 1):
                comps = list(res.get('components', []))
                if len(comps) == 0:
                    continue
                # ВАЖНО: в финальном графике есть ДВА независимых baseline:
                # 1) baseline модели (fit): b0 + b1*x (лежит внутри модели baseline+Voigt)
                # 2) baseline финального графика (хорда): прямая через интенсивности на границах пика
                #
                # Хорда строится по данным (SG 5,2) на общих границах support всех компонент.
                peak_x1 = float(min(c['support'][0] for c in comps))
                peak_x2 = float(max(c['support'][1] for c in comps))
                peak_y1 = _interp_y(peak_x1)
                peak_y2 = _interp_y(peak_x2)

                # Хорда: одна линия на пик
                global_plot_data['x'] += [peak_x1, peak_x2]
                global_plot_data['y'] += [peak_y1, peak_y2]
                global_plot_data['type'] += [f'Хорда Пик {peak_idx}'] * 2

                # baseline модели (из joint fit). Если нет — фолбэк на хорду.
                bl_fit = res.get('baseline', None)
                has_fit_bl = isinstance(bl_fit, dict) and all(k in bl_fit for k in ('x1', 'y1', 'x2', 'y2')) and float(bl_fit['x2']) != float(bl_fit['x1'])

                for comp_idx, comp in enumerate(res['components'], 1):
                    # Модель на финальном графике рисуем на всём пике (а не только на support компоненты)
                    mask = (x_all >= peak_x1) & (x_all <= peak_x2)
                    if not np.any(mask):
                        continue

                    xs = x_all[mask]
                    # МОДЕЛЬ = baseline_fit + Voigt (логика фита baseline+Voigt)
                    y_fit_base = (
                        np.array([_baseline_at(xv, float(bl_fit['x1']), float(bl_fit['y1']), float(bl_fit['x2']), float(bl_fit['y2'])) for xv in xs], dtype=float)
                        if has_fit_bl
                        else np.array([_baseline_at(xv, peak_x1, peak_y1, peak_x2, peak_y2) for xv in xs], dtype=float)
                    )
                    ys = y_fit_base + pseudo_voigt(xs, comp['A'], comp['mu'], comp['sigma'], comp['eta'])
                    global_plot_data['x'] += xs.tolist()
                    global_plot_data['y'] += ys.tolist()
                    global_plot_data['type'] += [f'Пик {peak_idx} / Модель {comp_idx} (fit baseline + Voigt)'] * len(xs)

            global_plot = (
                ggplot(global_plot_data)
                + geom_line(aes(x='x', y='y', color='type'), size=1.0)
                + labs(x="2θ", y="Интенсивность", title="Финальный график: данные + компоненты + baseline", color="Легенда")
                + ggsize(1200, 450)
            )

            # ---------- Таблица: метрики по компонентам ----------
            def _fwhm(x_grid: np.ndarray, y_corr: np.ndarray, x_mu: float, half_h: float) -> float | None:
                if half_h <= 0 or len(x_grid) < 3:
                    return None
                i_mu = int(np.argmin(np.abs(x_grid - x_mu)))

                xl = None
                for k in range(i_mu, 0, -1):
                    if y_corr[k] >= half_h and y_corr[k - 1] < half_h:
                        x_a, y_a = x_grid[k - 1], y_corr[k - 1]
                        x_b, y_b = x_grid[k], y_corr[k]
                        t = (half_h - y_a) / (y_b - y_a) if (y_b - y_a) != 0 else 0.0
                        xl = float(x_a + t * (x_b - x_a))
                        break

                xr = None
                for k in range(i_mu, len(x_grid) - 1):
                    if y_corr[k] >= half_h and y_corr[k + 1] < half_h:
                        x_a, y_a = x_grid[k], y_corr[k]
                        x_b, y_b = x_grid[k + 1], y_corr[k + 1]
                        t = (half_h - y_a) / (y_b - y_a) if (y_b - y_a) != 0 else 0.0
                        xr = float(x_a + t * (x_b - x_a))
                        break

                if xl is None or xr is None or xr < xl:
                    return None
                return float(xr - xl)

            component_rows = []
            for peak_idx, res in enumerate(fitting_results_valid, 1):
                is_outlier = bool(res.get('is_outlier', False))
                comps = list(res.get('components', []))
                if len(comps) == 0:
                    continue

                # Хорда (baseline финального графика): между общими границами support
                peak_x1 = float(min(c['support'][0] for c in comps))
                peak_x2 = float(max(c['support'][1] for c in comps))
                peak_y1 = _interp_y(peak_x1)
                peak_y2 = _interp_y(peak_x2)

                # baseline модели (из joint fit). Если нет — фолбэк на хорду.
                bl_fit = res.get('baseline', None)
                has_fit_bl = isinstance(bl_fit, dict) and all(k in bl_fit for k in ('x1', 'y1', 'x2', 'y2')) and float(bl_fit['x2']) != float(bl_fit['x1'])

                for comp_idx, comp in enumerate(res['components'], 1):
                    x1, x2 = float(comp['support'][0]), float(comp['support'][1])

                    mask = (x_all >= x1) & (x_all <= x2)
                    xs = x_all[mask]
                    if len(xs) < 3:
                        continue

                    # Хорда (baseline финального графика)
                    y_chord = np.array([_baseline_at(xv, peak_x1, peak_y1, peak_x2, peak_y2) for xv in xs], dtype=float)

                    # Модель (baseline_fit + Voigt)
                    y_fit_base = (
                        np.array([_baseline_at(xv, float(bl_fit['x1']), float(bl_fit['y1']), float(bl_fit['x2']), float(bl_fit['y2'])) for xv in xs], dtype=float)
                        if has_fit_bl
                        else y_chord
                    )
                    y_model = y_fit_base + pseudo_voigt(xs, comp['A'], comp['mu'], comp['sigma'], comp['eta'])

                    # Для метрик используем превышение МОДЕЛИ над ХОРДОЙ (как вы описали)
                    y_corr = y_model - y_chord

                    x_mu = float(comp['mu'])
                    y_mu_chord = _baseline_at(x_mu, peak_x1, peak_y1, peak_x2, peak_y2)
                    if has_fit_bl:
                        y_mu_fit_base = _baseline_at(x_mu, float(bl_fit['x1']), float(bl_fit['y1']), float(bl_fit['x2']), float(bl_fit['y2']))
                    else:
                        y_mu_fit_base = y_mu_chord
                    y_mu_model = float(y_mu_fit_base + pseudo_voigt(np.array([x_mu], dtype=float), comp['A'], comp['mu'], comp['sigma'], comp['eta'])[0])
                    height = float(y_mu_model - y_mu_chord)

                    fwhm_val = _fwhm(xs, y_corr, x_mu, height / 2.0)
                    # Площадь МЕЖДУ Voigt и baseline внутри support:
                    # интеграл положительной части (модель - хорда)
                    y_diff = np.maximum(y_corr, 0.0)
                    area = float(np.trapezoid(y_diff, xs))
                    area_over_height = None if height <= 0 else float(area / height)

                    component_rows.append({
                        'Пик': peak_idx,
                        'Компонента': comp_idx,
                        '2θ (mu)': round(x_mu, 4),
                        'Высота': round(height, 4),
                        'FWHM': None if fwhm_val is None else round(float(fwhm_val), 4),
                        'Площадь': round(area, 4),
                        'Площадь/Высота': None if area_over_height is None else round(float(area_over_height), 4),
                        'Support_L': round(x1, 4),
                        'Support_R': round(x2, 4),
                        'Выброс': is_outlier,
                    })

            df_components = pl.DataFrame(component_rows) if len(component_rows) > 0 else pl.DataFrame()

            final_view = mo.vstack([
                mo.md("## 🌐 Финальный общий график (все пики/компоненты + baseline)"),
                global_plot,
                mo.md("## 🧾 Таблица метрик по компонентам"),
                mo.ui.table(df_components),
            ])
        return final_view


    _()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
