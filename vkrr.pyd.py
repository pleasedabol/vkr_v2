import marimo

__generated_with = "0.17.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    from io import StringIO
    import lets_plot as lp
    from lets_plot import ggplot, geom_line, geom_point, labs, ggsize, aes, scale_color_manual, geom_hline, geom_vline
    import numpy as np
    from scipy.signal import savgol_filter
    from scipy.signal import find_peaks
    from scipy.optimize import curve_fit
    from lets_plot.mapping import as_discrete
    return (
        aes,
        curve_fit,
        find_peaks,
        geom_line,
        geom_point,
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
def _(pl):
    def process_coordinates_file(file_content):
        try:
            import re

            xs: list[float] = []
            ys: list[float] = []

            # Разбираем файл строго как пары координат, чтобы ошибки были понятными.
            for line_no, raw in enumerate(file_content.splitlines(), 1):
                s = raw.strip()
                if not s:
                    continue

                parts = re.split(r"\s+", s)
                if len(parts) < 2:
                    return None, None, None, f"Ошибка формата в строке {line_no}: ожидаются две колонки чисел (x y)."

                x_raw, y_raw = parts[0], parts[1]

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
    file_upload = mo.ui.file(
        filetypes=[".txt"],
        multiple=False,
        label="Загрузите файл с координатами (.txt)"
    )

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
    df = None
    x_coords = None
    y_coords = None
    error = None
    file_info = None
    y_done = []
    if file_upload.value:
        file_content_bytes = file_upload.value[0].contents
        file_content = file_content_bytes.decode('utf-8')
        file_info = f"Загружен файл: {file_upload.value[0].name}"

        df, x_coords, y_coords, error = process_coordinates_file(file_content)

    if file_upload.value:
        if error:
            result_display = mo.md(f"**❌ Ошибка:** {error}")
        elif df is not None:
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

    result_display
    return df, x_coords, y_coords


@app.cell
def _(aes, df, geom_line, ggplot, ggsize, labs, lp, mo, x_coords, y_coords):
    if df is not None:
        try:
            lp.LetsPlot.setup_html()

            x_coords_list = x_coords if isinstance(x_coords, list) else x_coords.tolist()
            y_coords_list = y_coords if isinstance(y_coords, list) else y_coords.tolist()

            plot_data2 = {
                'x_coord': x_coords_list,
                'y_coord': y_coords_list
            }

            plot2 = ggplot(plot_data2) + \
                   geom_line(aes(x='x_coord', y='y_coord'), color='blue', size=1) + \
                    labs(x='2θ', y='Интенсивность') + \
                   ggsize(1000, 500)

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
        baseline = snip_baseline(y_coords, iterations=iterations.value)
        y_corrected = np.array(y_coords) - baseline

        x_list = x_coords if isinstance(x_coords, list) else x_coords.tolist()
        y_list = y_coords if isinstance(y_coords, list) else y_coords.tolist()
        plot_dataa = {
            "x": x_list * 3,
            "y": y_list + baseline.tolist() + y_corrected.tolist(),
            "type": (["Raw data"] * len(x_list)) +
                    (["Фон (SNIP)"] * len(x_list)) +
                    (["Вычтенный фон"] * len(x_list))
        }

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

        def _odd(k: int) -> int:
            k = int(k)
            return k if (k % 2 == 1) else (k + 1)

        N_odd_max = N if (N % 2 == 1) else (N - 1)
        if N_odd_max < 5:
            return y

        # Окна ограничены долей длины пика, чтобы не размывать узкие пики.
        alpha_min = 0.12
        alpha_max = 0.25

        w_min_cap = max(5, int(np.floor(alpha_min * N)))
        w_max_cap = max(5, int(np.floor(alpha_max * N)))

        w_min_eff = _odd(int(np.clip(min(int(min_window), w_min_cap), 5, N_odd_max)))

        min_required = _odd(int(polyorder) + 2)
        if w_min_eff < min_required:
            w_min_eff = min(min_required, N_odd_max)
            w_min_eff = _odd(w_min_eff)
            if w_min_eff > N_odd_max:
                w_min_eff = N_odd_max

        w_max_eff = _odd(int(np.clip(min(int(max_window), w_max_cap), w_min_eff, N_odd_max)))
        if w_max_eff < w_min_eff:
            w_max_eff = w_min_eff

        # Ранг точки даёт маленькое окно на пиках и большое окно на фоне.
        order = np.argsort(y)
        rank_small = np.empty(N, dtype=int)
        rank_small[order] = np.arange(N)
        rank_greater = (N - 1) - rank_small

        normalized_ranks = rank_greater / (N - 1) if N > 1 else np.zeros(N)

        window_sizes = w_min_eff + (w_max_eff - w_min_eff) * normalized_ranks

        window_sizes = np.round(window_sizes).astype(int)
        window_sizes = np.where(window_sizes % 2 == 0, window_sizes + 1, window_sizes)
        window_sizes = np.clip(window_sizes, w_min_eff, w_max_eff)

        for i in range(N):
            window = int(window_sizes[i])

            half_window = window // 2
            start_idx = max(0, i - half_window)
            end_idx = min(N, i + half_window + 1)

            window_data = y[start_idx:end_idx]

            actual_window = len(window_data)
            if actual_window < polyorder + 1:
                y_smoothed[i] = y[i]
            else:
                if actual_window % 2 == 0:
                    actual_window -= 1
                    window_data = window_data[:actual_window]

                effective_polyorder = min(polyorder, actual_window - 1)

                try:
                    smoothed_window = savgol_filter(window_data, actual_window, effective_polyorder)

                    center_idx = len(smoothed_window) // 2
                    y_smoothed[i] = smoothed_window[center_idx]
                except:
                    y_smoothed[i] = y[i]

        return y_smoothed

    if df is not None and len(y_corrected) > 10:
        try:
            lp.LetsPlot.setup_html()

            y_smoothed = adaptive_savgol_smooth(
                y_corrected,
                min_window=min_window_slider.value,
                max_window=max_window_slider.value,
                polyorder=polyorder_slider.value
            )

            x_coords_list2 = x_coords if isinstance(x_coords, list) else x_coords.tolist()

            plot_data_smooth = {
                "x": x_coords_list2 * 2,
                "y": y_corrected.tolist() + y_smoothed.tolist(),
                "type": (["После вычитания фона (SNIP)"] * len(x_coords_list2)) +
                        (["Адаптивное сглаживание"] * len(x_coords_list2))
            }

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

            threshold = threshold_ratio

            # Граница пика ищется по первому падению ниже порога.
            left_bound_idx = idx
            for ind in range(idx - 1, -1, -1):
                if y[ind] <= threshold:
                    left_bound_idx = ind
                    break

                if ind == 0:
                    left_bound_idx = 0
                    break

            right_bound_idx = idx
            for ind in range(idx + 1, N):
                if y[ind] <= threshold:
                    right_bound_idx = ind
                    break

                if ind == N - 1:
                    right_bound_idx = N - 1
                    break

            left_bound_x = x[left_bound_idx]
            right_bound_x = x[right_bound_idx]
            width = right_bound_x - left_bound_x

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

            peaks_data = find_peaks_with_bounds(
                y_smoothed,
                x_coords,
                min_height=0,
                min_distance=min_distance_slider,
                prominence=prominence_slider.value,
                threshold_ratio= threshold_ratio_slider.value
            )

            min_width = float(min_peak_width_slider.value)
            peaks_data = [p for p in peaks_data if float(p.get('width', 0.0)) >= min_width]

            if len(peaks_data) > 0:
                plot_data_peaks = {
                    "x": x_coords if isinstance(x_coords, list) else x_coords.tolist(),
                    "y": y_smoothed.tolist()
                }

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
        # sigma здесь трактуется как FWHM для обеих частей псевдо-Войгта.
        sigma = np.maximum(np.asarray(sigma, dtype=float), 1e-12)
        z = (x - float(mu)) / sigma

        L = 1.0 / (1.0 + 4.0 * (z ** 2))
        G = np.exp(-4.0 * np.log(2.0) * (z ** 2))

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

        left_idx = peak_idx
        while left_idx > 0 and y[left_idx] > half_height:
            left_idx -= 1

        right_idx = peak_idx
        while right_idx < len(y) - 1 and y[right_idx] > half_height:
            right_idx += 1

        fwhm = float(x[right_idx] - x[left_idx])
        dx = float(x[1] - x[0]) if len(x) > 1 else 1.0
        return max(fwhm, dx * 2.0)


    def split_peak_into_segments(x, y, max_segments=3):
        """
        Разбиение пика на сегменты на основе анализа формы.

        Возвращает:
        - список кортежей [(x_start, x_end), ...]
        """
        # Сегментация теперь выполняется внутри fit_peak_with_domain_components.
        x = np.asarray(x, dtype=float)
        if len(x) == 0:
            return []
        return [(float(x[0]), float(x[-1]))]


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

        # В финальной piecewise-модели каждая точка принадлежит одной компоненте.
        for i in range(num_components):
            idx = i * 4
            A, mu, sigma, eta = params[idx:idx+4]

            if segments is not None and i < len(segments):
                x_start, x_end = segments[i]
                mask = (x >= x_start) & (x <= x_end)
            else:
                continue

            if np.any(mask):
                x_segment = x[mask]
                voigt_values = pseudo_voigt(x_segment, A, mu, sigma, eta)

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
        Вычисление piecewise-предсказания для  модели.

        Возвращает:
        - y_pred: предсказанные значения (piecewise)
        """
        params = []
        for comp in components:
            params.extend([comp['A'], comp['mu'], comp['sigma'], comp['eta']])

        y_pred = piecewise_multi_voigt(x_peak, *params, segments=segments)

        return y_pred


    def fit_peak_with_domain_components(x_peak, y_peak, r2_threshold=0.95, max_components=3):
        """
        Piecewise-аппроксимация пика псевдо-Войгтами (БЕЗ сумм), критерий — tail-R²:
        - каждая точка принадлежит ровно одному сегменту → оценивается ровно одной компонентой
        - число компонент растёт строго 1 → 2 → 3 только если tail-R² текущей модели ниже порога
        - сегментация не фиксируется заранее: границы сегментов подстраиваем
        """
        x_peak = np.asarray(x_peak, dtype=float)
        y_peak = np.asarray(y_peak, dtype=float)
        r2_threshold = float(r2_threshold)
        max_components = int(max_components)

        N = len(x_peak)
        # Линейный baseline удерживает хвосты, чтобы Voigt не тащил на себе фон.
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

        def _baseline_eval(baseline_ep, x_val):
            if not isinstance(baseline_ep, dict):
                return 0.0
            if not all(k in baseline_ep for k in ('x1', 'y1', 'x2', 'y2')):
                return 0.0
            x1 = float(baseline_ep['x1'])
            y1 = float(baseline_ep['y1'])
            x2 = float(baseline_ep['x2'])
            y2 = float(baseline_ep['y2'])
            if x2 == x1:
                return y1
            t = (float(x_val) - x1) / (x2 - x1)
            return y1 + (y2 - y1) * t

        def _endpoint_constraints_ok(components, baseline_ep, num_components):
            """
            Проверка условия "вне support на концах пика" с устойчивостью к шуму:
            - проверяем не одну точку, а 10 крайних точек соответствующего конца;
            - считаем нарушение только если ВСЕ 10 точек выше исходного пика > 10%.

            Правило применяется только для 2-3 компонент.
            """
            if num_components <= 1:
                return True
            if len(components) <= 1:
                return True

            tol_x = float(max(abs(dx) * 0.25, 1e-9))
            y_scale = float(max(1.0, np.max(np.abs(y_peak))))
            tol_y = float(max(1e-9, 1e-6 * y_scale))
            over_threshold_ratio = 0.10
            edge_points = int(min(10, len(x_peak)))

            if edge_points <= 0:
                return True

            def _all_edge_points_exceed(comp, side: str) -> bool:
                if side == 'left':
                    x_edge = np.asarray(x_peak[:edge_points], dtype=float)
                    y_edge = np.asarray(y_peak[:edge_points], dtype=float)
                else:
                    x_edge = np.asarray(x_peak[-edge_points:], dtype=float)
                    y_edge = np.asarray(y_peak[-edge_points:], dtype=float)

                comp_edge = pseudo_voigt(x_edge, comp['A'], comp['mu'], comp['sigma'], comp['eta'])
                base_edge = np.array([_baseline_eval(baseline_ep, xv) for xv in x_edge], dtype=float)
                model_edge = base_edge + comp_edge

                limits = y_edge * (1.0 + over_threshold_ratio)
                exceed_mask = model_edge > (limits + tol_y)
                return bool(np.all(exceed_mask))

            for comp in components:
                support = comp.get('support', (float(x_peak[0]), float(x_peak[-1])))
                s_left = float(support[0])
                s_right = float(support[1])

                check_left_endpoint = s_left > (float(x_peak[0]) + tol_x)
                check_right_endpoint = s_right < (float(x_peak[-1]) - tol_x)

                # Переаппроксимация нужна только при устойчивом завышении края.
                if check_left_endpoint and _all_edge_points_exceed(comp, 'left'):
                    return False

                if check_right_endpoint and _all_edge_points_exceed(comp, 'right'):
                    return False

            return True

        def _fit_component(i0, i1):
            xs = x_peak[i0:i1+1]
            ys = y_peak_corr[i0:i1+1]

            max_idx = int(np.argmax(ys))
            mu0 = float(xs[max_idx])
            A0 = float(ys[max_idx])
            sigma0 = float(estimate_initial_sigma(xs, ys, max_idx, A0))
            eta0 = 0.5

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

        def _fit_piecewise_for_n(n, allow_refit_retries=True):
            max_refit_attempts = 1 if (n <= 1 or not allow_refit_retries) else 12
            best_attempt = None
            last_attempt = None

            for attempt in range(max_refit_attempts):
                min_points = max(6, int(round(N * 0.08)))
                if n * min_points > N:
                    min_points = max(4, N // n)

                # Повторные попытки слегка двигают стартовые границы сегментов.
                splits = []
                if n > 1:
                    rng = np.random.default_rng(1729 + n * 1000 + attempt)
                for k in range(1, n):
                    s = int(round(k * N / n)) - 1
                    if n > 1 and attempt > 0:
                        jitter = max(1, int(round(N * 0.06)))
                        s += int(rng.integers(-jitter, jitter + 1))
                    splits.append(s)
                splits = _clamp_splits(splits, n, N, min_points)

                seg_idx = _segments_from_splits(splits, n, N)

                comps = []
                seg_sse = []
                for (i0, i1) in seg_idx:
                    comp, sse = _fit_component(i0, i1)
                    comps.append(comp)
                    seg_sse.append(sse)

                # Границы сегментов уточняются координатным спуском.
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

                seg_idx = _segments_from_splits(splits, n, N)
                x_segs = _x_segments(seg_idx)
                for c, seg in zip(comps, x_segs):
                    c['support'] = (float(seg[0]), float(seg[1]))

                # Совместный fit baseline и Voigt уменьшает смещение на хвостах.
                def _fit_func_with_baseline(x, b0, b1, *voigt_params):
                    x = np.asarray(x, dtype=float)
                    baseline = b0 + b1 * x
                    y_pred_local = np.array(baseline, copy=True)
                    if len(x) == N:
                        for i in range(n):
                            p = voigt_params[i * 4:(i + 1) * 4]
                            A, mu, sigma, eta = p
                            i0, i1 = seg_idx[i]
                            xs = x[i0:i1 + 1]
                            y_pred_local[i0:i1 + 1] = baseline[i0:i1 + 1] + pseudo_voigt(xs, A, mu, sigma, eta)
                        return y_pred_local
                    for i in range(n):
                        p = voigt_params[i * 4:(i + 1) * 4]
                        A, mu, sigma, eta = p
                        x_start, x_end = x_segs[i]
                        mask = (x >= x_start) & (x <= x_end)
                        y_pred_local[mask] = baseline[mask] + pseudo_voigt(x[mask], A, mu, sigma, eta)
                    return y_pred_local

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
                    y_pred = _piecewise_predict(seg_idx, comps)
                    y_pred_total = y_pred + _baseline_vec(x_peak)
                    baseline_ep = _baseline_endpoints()

                score = float(calculate_tail_weighted_r2(y_peak, y_pred_total))
                endpoint_constraint_satisfied = _endpoint_constraints_ok(comps, baseline_ep, n)
                attempt_result = {
                    'components': comps,
                    'y_pred': y_pred_total,
                    'score': score,
                    'num_components': n,
                    'segments': x_segs,
                    'baseline': baseline_ep,
                    'endpoint_constraint_satisfied': endpoint_constraint_satisfied,
                }
                last_attempt = attempt_result

                if endpoint_constraint_satisfied:
                    return attempt_result

                if (best_attempt is None) or (float(attempt_result['score']) > float(best_attempt['score'])):
                    best_attempt = attempt_result

            if best_attempt is not None:
                return best_attempt
            return last_attempt

        # Выбранная модель берётся по порогу, но качество на max_components нужно для выбросов.
        chosen_res = None
        last_res = None

        for n in range(1, max_components + 1):
            res = _fit_piecewise_for_n(n, allow_refit_retries=(chosen_res is None))
            last_res = res
            if chosen_res is None and float(res['score']) >= r2_threshold:
                chosen_res = res

        if chosen_res is None:
            chosen_res = last_res

        outlier_score_threshold = 0.90
        score_at_max = float(last_res['score']) if last_res is not None else 0.0
        is_outlier = (max_components >= 3) and (score_at_max < outlier_score_threshold)

        chosen_res = dict(chosen_res)
        chosen_res['score_at_max_components'] = score_at_max
        chosen_res['outlier_score_threshold'] = outlier_score_threshold
        chosen_res['is_outlier'] = is_outlier
        return chosen_res


    if df is not None and len(y_coords) > 10 and len(peaks_data) > 0:
        try:
            lp.LetsPlot.setup_html()

            # Перед fit сглаживаем весь сигнал фиксированным SG(5, 2).
            y_coords_arr = np.asarray(y_coords, dtype=float)
            if len(y_coords_arr) >= 5:
                y_coords_fit = savgol_filter(y_coords_arr, 5, 2)
            else:
                y_coords_fit = y_coords_arr

            fitting_results = []
            plot_elements = []

            for i, peak_info in enumerate(peaks_data):
                left_idx = peak_info['left_bound_idx']
                right_idx = peak_info['right_bound_idx']

                x_peak = np.array(x_coords[left_idx:right_idx+1])
                y_peak_raw = np.array(y_coords_fit[left_idx:right_idx+1], dtype=float)
                y_peak = y_peak_raw

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

                result = fit_peak_with_domain_components(
                    x_peak, y_peak,
                    r2_threshold=_effective_r2_threshold(),
                    max_components=3
                )

                result['peak_info'] = peak_info
                result['x_peak'] = x_peak
                result['y_peak'] = y_peak
                fitting_results.append(result)

                plot_data = {
                    'x': x_peak.tolist(),
                    'y': y_peak.tolist(),
                    'type': ['Исходные данные'] * len(x_peak)
                }

                # Компоненты на графике показываются целиком, чтобы видеть симметрию Voigt.
                bl = result.get('baseline', None)
                if isinstance(bl, dict) and all(k in bl for k in ('x1', 'y1', 'x2', 'y2')) and float(bl['x2']) != float(bl['x1']):
                    x1b, y1b = float(bl['x1']), float(bl['y1'])
                    x2b, y2b = float(bl['x2']), float(bl['y2'])
                    baseline_vals = y1b + (y2b - y1b) * (x_peak - x1b) / (x2b - x1b)
                else:
                    baseline_vals = _baseline_vec_peak(x_peak)
                colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
                for j, comp in enumerate(result['components']):
                    comp_values = pseudo_voigt(x_peak, comp['A'], comp['mu'], comp['sigma'], comp['eta'])
                    model_values = comp_values + baseline_vals

                    plot_data['x'] += x_peak.tolist()
                    plot_data['y'] += model_values.tolist()
                    plot_data['type'] += [f'Компонента {j+1} (baseline + Voigt)'] * len(x_peak)

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
        if df is None or fitting_results is None or len(fitting_results) == 0:
            final_view = mo.md("**⏳ Ожидание результатов аппроксимации...**")
        else:
            lp.LetsPlot.setup_html()

            def pseudo_voigt(x, A, mu, sigma, eta):
                x = np.asarray(x, dtype=float)
                # sigma здесь оставлен в той же FWHM-параметризации, что и в fit.
                sigma = np.maximum(np.asarray(sigma, dtype=float), 1e-12)
                z = (x - float(mu)) / sigma
                L = 1.0 / (1.0 + 4.0 * (z ** 2))
                G = np.exp(-4.0 * np.log(2.0) * (z ** 2))
                return A * (eta * L + (1 - eta) * G)

            x_all = np.asarray(x_coords, dtype=float)
            y_all = np.asarray(y_coords_fit, dtype=float)

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

            fitting_results_valid = [r for r in fitting_results if not bool(r.get('is_outlier', False))]

            for peak_idx, res in enumerate(fitting_results_valid, 1):
                comps = list(res.get('components', []))
                if len(comps) == 0:
                    continue
                # Старый финальный baseline строится хордой по краям пика.
                peak_x1 = float(min(c['support'][0] for c in comps))
                peak_x2 = float(max(c['support'][1] for c in comps))
                peak_y1 = _interp_y(peak_x1)
                peak_y2 = _interp_y(peak_x2)

                global_plot_data['x'] += [peak_x1, peak_x2]
                global_plot_data['y'] += [peak_y1, peak_y2]
                global_plot_data['type'] += [f'Хорда Пик {peak_idx}'] * 2

                bl_fit = res.get('baseline', None)
                has_fit_bl = isinstance(bl_fit, dict) and all(k in bl_fit for k in ('x1', 'y1', 'x2', 'y2')) and float(bl_fit['x2']) != float(bl_fit['x1'])

                for comp_idx, comp in enumerate(res['components'], 1):
                    mask = (x_all >= peak_x1) & (x_all <= peak_x2)
                    if not np.any(mask):
                        continue

                    xs = x_all[mask]
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

                peak_x1 = float(min(c['support'][0] for c in comps))
                peak_x2 = float(max(c['support'][1] for c in comps))
                peak_y1 = _interp_y(peak_x1)
                peak_y2 = _interp_y(peak_x2)

                bl_fit = res.get('baseline', None)
                has_fit_bl = isinstance(bl_fit, dict) and all(k in bl_fit for k in ('x1', 'y1', 'x2', 'y2')) and float(bl_fit['x2']) != float(bl_fit['x1'])

                for comp_idx, comp in enumerate(res['components'], 1):
                    x1, x2 = float(comp['support'][0]), float(comp['support'][1])

                    # FWHM считаем на всём пике, чтобы не потерять пересечения у сложных форм.
                    mask_fwhm = (x_all >= peak_x1) & (x_all <= peak_x2)
                    xs_fwhm = x_all[mask_fwhm]
                    if len(xs_fwhm) < 3:
                        continue

                    y_chord_fwhm = np.array([_baseline_at(xv, peak_x1, peak_y1, peak_x2, peak_y2) for xv in xs_fwhm], dtype=float)

                    y_fit_base_fwhm = (
                        np.array([_baseline_at(xv, float(bl_fit['x1']), float(bl_fit['y1']), float(bl_fit['x2']), float(bl_fit['y2'])) for xv in xs_fwhm], dtype=float)
                        if has_fit_bl
                        else y_chord_fwhm
                    )
                    y_model_fwhm = y_fit_base_fwhm + pseudo_voigt(xs_fwhm, comp['A'], comp['mu'], comp['sigma'], comp['eta'])

                    y_corr_fwhm = y_model_fwhm - y_chord_fwhm

                    x_mu = float(comp['mu'])
                    y_mu_chord = _baseline_at(x_mu, peak_x1, peak_y1, peak_x2, peak_y2)
                    if has_fit_bl:
                        y_mu_fit_base = _baseline_at(x_mu, float(bl_fit['x1']), float(bl_fit['y1']), float(bl_fit['x2']), float(bl_fit['y2']))
                    else:
                        y_mu_fit_base = y_mu_chord
                    y_mu_model = float(y_mu_fit_base + pseudo_voigt(np.array([x_mu], dtype=float), comp['A'], comp['mu'], comp['sigma'], comp['eta'])[0])
                    height = float(y_mu_model - y_mu_chord)

                    fwhm_val = _fwhm(xs_fwhm, y_corr_fwhm, x_mu, height / 2.0)

                    # Площадь берём как положительный зазор модели над хордой.
                    y_diff = np.maximum(y_corr_fwhm, 0.0)
                    area = float(np.trapezoid(y_diff, xs_fwhm))
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
def _(mo):
    final_polynomial_order_input = mo.ui.number(
        start=1,
        stop=25,
        step=1,
        value=9,
        label="Степень полинома шума (1-25)"
    )

    mo.vstack([
        mo.md("## ⚙️ Настройка финального полинома шума"),
        mo.md("Полином строится по точкам вне диапазонов аппроксимированных пиков и отображается как независимая оценка шума."),
        final_polynomial_order_input,
    ])
    return (final_polynomial_order_input,)


@app.cell
def _(
    aes,
    df,
    final_polynomial_order_input,
    fitting_results,
    geom_line,
    geom_point,
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
        if df is None or fitting_results is None or len(fitting_results) == 0:
            polynomial_final_view = mo.md("**⏳ Ожидание результатов аппроксимации...**")
        else:
            try:
                lp.LetsPlot.setup_html()

                def pseudo_voigt(x, A, mu, sigma, eta):
                    x = np.asarray(x, dtype=float)
                    sigma = np.maximum(np.asarray(sigma, dtype=float), 1e-12)
                    z = (x - float(mu)) / sigma
                    L = 1.0 / (1.0 + 4.0 * (z ** 2))
                    G = np.exp(-4.0 * np.log(2.0) * (z ** 2))
                    return A * (eta * L + (1 - eta) * G)

                def _baseline_at(xq: float, x1: float, y1: float, x2: float, y2: float) -> float:
                    if x2 == x1:
                        return float(y1)
                    return float(y1 + (y2 - y1) * (xq - x1) / (x2 - x1))

                def _fit_polynomial_with_required_points(noise_x, noise_y, required_x, required_y, degree):
                    all_x_for_scale = np.concatenate([noise_x, required_x]) if len(required_x) > 0 else noise_x
                    x_center = float((np.min(all_x_for_scale) + np.max(all_x_for_scale)) / 2.0)
                    x_scale = float((np.max(all_x_for_scale) - np.min(all_x_for_scale)) / 2.0)
                    if x_scale == 0:
                        x_scale = 1.0

                    def _design(xs):
                        xs = np.asarray(xs, dtype=float)
                        t = (xs - x_center) / x_scale
                        return np.vander(t, N=int(degree) + 1, increasing=True)

                    A = _design(noise_x)
                    if len(required_x) == 0:
                        coeffs, *_ = np.linalg.lstsq(A, noise_y, rcond=None)
                    else:
                        C = _design(required_x)
                        c0, *_ = np.linalg.lstsq(C, required_y, rcond=None)
                        _, s, vh = np.linalg.svd(C, full_matrices=True)
                        tol = np.finfo(float).eps * max(C.shape) * (float(s[0]) if len(s) else 1.0)
                        rank = int(np.sum(s > tol))
                        null_space = vh[rank:].T

                        if null_space.shape[1] == 0:
                            coeffs = c0
                        else:
                            z, *_ = np.linalg.lstsq(A @ null_space, noise_y - A @ c0, rcond=None)
                            coeffs = c0 + null_space @ z

                    def _predict(xs):
                        return _design(xs) @ coeffs

                    return _predict

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

                def _model_boundary_y(x_val: float, comps: list[dict], bl_fit, side: str) -> float:
                    has_fit_bl = (
                        isinstance(bl_fit, dict)
                        and all(k in bl_fit for k in ('x1', 'y1', 'x2', 'y2'))
                        and float(bl_fit['x2']) != float(bl_fit['x1'])
                    )
                    y_fit_base = (
                        _baseline_at(x_val, float(bl_fit['x1']), float(bl_fit['y1']), float(bl_fit['x2']), float(bl_fit['y2']))
                        if has_fit_bl
                        else 0.0
                    )
                    if side == 'left':
                        comp = min(comps, key=lambda c: float(c['support'][0]))
                    else:
                        comp = max(comps, key=lambda c: float(c['support'][1]))
                    return float(y_fit_base + pseudo_voigt(np.array([x_val], dtype=float), comp['A'], comp['mu'], comp['sigma'], comp['eta'])[0])

                x_all = np.asarray(x_coords, dtype=float)
                y_all = np.asarray(y_coords_fit, dtype=float)
                fitting_results_valid = [r for r in fitting_results if not bool(r.get('is_outlier', False))]

                peak_mask = np.zeros(len(x_all), dtype=bool)
                peak_ranges = []
                required_points = []
                for res in fitting_results_valid:
                    comps = list(res.get('components', []))
                    if len(comps) == 0:
                        continue
                    peak_x1 = float(min(c['support'][0] for c in comps))
                    peak_x2 = float(max(c['support'][1] for c in comps))
                    if peak_x2 < peak_x1:
                        peak_x1, peak_x2 = peak_x2, peak_x1
                    bl_fit = res.get('baseline', None)
                    peak_ranges.append((peak_x1, peak_x2, comps, bl_fit))
                    peak_mask |= (x_all >= peak_x1) & (x_all <= peak_x2)
                    required_points.append((peak_x1, _model_boundary_y(peak_x1, comps, bl_fit, 'left')))
                    required_points.append((peak_x2, _model_boundary_y(peak_x2, comps, bl_fit, 'right')))

                required_points = sorted(required_points, key=lambda p: p[0])
                required_x_list = []
                required_y_list = []
                for x_req, y_req in required_points:
                    if len(required_x_list) > 0 and np.isclose(x_req, required_x_list[-1], rtol=0.0, atol=1e-9):
                        required_y_list[-1] = float((required_y_list[-1] + y_req) / 2.0)
                    else:
                        required_x_list.append(float(x_req))
                        required_y_list.append(float(y_req))

                noise_mask = ~peak_mask
                noise_x = x_all[noise_mask]
                noise_y = y_all[noise_mask]
                required_x = np.asarray(required_x_list, dtype=float)
                required_y = np.asarray(required_y_list, dtype=float)
                requested_degree = int(np.clip(round(float(final_polynomial_order_input.value)), 1, 15))

                if len(noise_x) < 2:
                    polynomial_final_view = mo.vstack([
                        mo.md("## 🌐 Финальный график с независимым полиномом шума"),
                        mo.md("**⚠️ Недостаточно точек вне пиков для построения полинома шума.**"),
                    ])
                else:
                    min_required_degree = max(1, len(peak_ranges) * 2)
                    degree = int(max(requested_degree, min_required_degree))
                    polynomial = _fit_polynomial_with_required_points(
                        noise_x,
                        noise_y,
                        required_x,
                        required_y,
                        degree,
                    )

                    order = np.argsort(x_all)
                    x_sorted = x_all[order]
                    y_poly = polynomial(x_sorted)

                    plot_data = {
                        'x': x_all.tolist(),
                        'y': y_all.tolist(),
                        'type': ['Данные (SG 5,2)'] * len(x_all),
                    }

                    plot_data['x'] += x_sorted.tolist()
                    plot_data['y'] += y_poly.tolist()
                    plot_data['type'] += [f'Полином шума, степень {degree}'] * len(x_sorted)

                    for peak_idx, (peak_x1, peak_x2, comps, bl_fit) in enumerate(peak_ranges, 1):
                        mask = (x_all >= peak_x1) & (x_all <= peak_x2)
                        if not np.any(mask):
                            continue

                        xs = x_all[mask]
                        has_fit_bl = (
                            isinstance(bl_fit, dict)
                            and all(k in bl_fit for k in ('x1', 'y1', 'x2', 'y2'))
                            and float(bl_fit['x2']) != float(bl_fit['x1'])
                        )
                        y_fit_base = (
                            np.array([_baseline_at(xv, float(bl_fit['x1']), float(bl_fit['y1']), float(bl_fit['x2']), float(bl_fit['y2'])) for xv in xs], dtype=float)
                            if has_fit_bl
                            else np.zeros_like(xs, dtype=float)
                        )

                        for comp_idx, comp in enumerate(comps, 1):
                            ys = y_fit_base + pseudo_voigt(xs, comp['A'], comp['mu'], comp['sigma'], comp['eta'])
                            plot_data['x'] += xs.tolist()
                            plot_data['y'] += ys.tolist()
                            plot_data['type'] += [f'Пик {peak_idx} / Компонента {comp_idx} (fit baseline + Voigt)'] * len(xs)

                    polynomial_plot = (
                        ggplot(plot_data)
                        + geom_line(aes(x='x', y='y', color='type'), size=1.0)
                        + labs(
                            x="2θ",
                            y="Интенсивность",
                            title="Финальный график: компоненты fit baseline + Voigt и независимый полином шума",
                            color="Легенда"
                        )
                        + ggsize(1200, 450)
                    )
                    if len(required_x) > 0:
                        polynomial_plot = polynomial_plot + geom_point(
                            aes(x='x', y='y'),
                            data={'x': required_x.tolist(), 'y': required_y.tolist()},
                            color='black',
                            size=4,
                        )

                    component_rows = []
                    for peak_idx, (peak_x1, peak_x2, comps, bl_fit) in enumerate(peak_ranges, 1):
                        mask = (x_all >= peak_x1) & (x_all <= peak_x2)
                        xs_metrics = x_all[mask]
                        if len(xs_metrics) < 3:
                            continue

                        has_fit_bl = (
                            isinstance(bl_fit, dict)
                            and all(k in bl_fit for k in ('x1', 'y1', 'x2', 'y2'))
                            and float(bl_fit['x2']) != float(bl_fit['x1'])
                        )
                        y_fit_base = (
                            np.array([_baseline_at(xv, float(bl_fit['x1']), float(bl_fit['y1']), float(bl_fit['x2']), float(bl_fit['y2'])) for xv in xs_metrics], dtype=float)
                            if has_fit_bl
                            else np.zeros_like(xs_metrics, dtype=float)
                        )
                        y_poly_base = polynomial(xs_metrics)

                        for comp_idx, comp in enumerate(comps, 1):
                            x_mu = float(comp['mu'])
                            y_model = y_fit_base + pseudo_voigt(xs_metrics, comp['A'], comp['mu'], comp['sigma'], comp['eta'])
                            y_corr = y_model - y_poly_base

                            y_mu_fit_base = (
                                _baseline_at(x_mu, float(bl_fit['x1']), float(bl_fit['y1']), float(bl_fit['x2']), float(bl_fit['y2']))
                                if has_fit_bl
                                else 0.0
                            )
                            y_mu_model = float(y_mu_fit_base + pseudo_voigt(np.array([x_mu], dtype=float), comp['A'], comp['mu'], comp['sigma'], comp['eta'])[0])
                            y_mu_poly = float(polynomial(np.array([x_mu], dtype=float))[0])
                            height = float(y_mu_model - y_mu_poly)

                            fwhm_val = _fwhm(xs_metrics, y_corr, x_mu, height / 2.0)
                            tail_cutoff = 0.1 * height if height > 0 else 0.0
                            y_diff = np.where(y_corr >= tail_cutoff, y_corr, 0.0)
                            area = float(np.trapezoid(y_diff, xs_metrics))
                            area_over_height = None if height <= 0 else float(area / height)

                            component_rows.append({
                                'Пик': peak_idx,
                                'Компонента': comp_idx,
                                '2θ (mu)': round(x_mu, 4),
                                'Высота': round(height, 4),
                                'FWHM': None if fwhm_val is None else round(float(fwhm_val), 4),
                                'Площадь': round(area, 4),
                                'Площадь/Высота': None if area_over_height is None else round(float(area_over_height), 4),
                                'Support_L': round(float(comp['support'][0]), 4),
                                'Support_R': round(float(comp['support'][1]), 4),
                            })

                    df_poly_components = pl.DataFrame(component_rows) if len(component_rows) > 0 else pl.DataFrame()

                    view_items = [
                        mo.md("## 🌐 Финальный график с независимым полиномом шума"),
                        mo.md(f"**Степень полинома шума:** {degree}"),
                        mo.md(f"**Обязательных точек прохождения:** {len(required_x)}"),
                    ]
                    if degree != requested_degree:
                        view_items.append(mo.md(f"**⚠️ Использована степень {degree} вместо {requested_degree}: степень не может быть ниже количества пиков × 2.**"))
                    view_items.append(polynomial_plot)
                    view_items.extend([
                        mo.md("## 🧾 Таблица метрик по компонентам относительно полиномиального шума"),
                        mo.ui.table(df_poly_components),
                    ])

                    polynomial_final_view = mo.vstack(view_items)

            except Exception as e:
                polynomial_final_view = mo.vstack([
                    mo.md("## ❌ Ошибка построения финального полиномиального baseline"),
                    mo.md(f"**Ошибка:** {str(e)}"),
                ])

        return polynomial_final_view


    _()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
