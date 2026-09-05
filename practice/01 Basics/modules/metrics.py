import numpy as np


def ED_distance(ts1: np.ndarray, ts2: np.ndarray) -> float:
    """
    Calculate the Euclidean distance

    Parameters
    ----------
    ts1: the first time series
    ts2: the second time series

    Returns
    -------
    ed_dist: euclidean distance between ts1 and ts2
    """
    # Вычисляем сумму квадратов разностей и извлекаем корень
    return np.sqrt(np.sum((ts1 - ts2) ** 2))


def norm_ED_distance(ts1: np.ndarray, ts2: np.ndarray) -> float:
    """
    Calculate the normalized Euclidean distance

    Parameters
    ----------
    ts1: the first time series
    ts2: the second time series

    Returns
    -------
    norm_ed_dist: normalized Euclidean distance between ts1 and ts2s
    """

    n = len(ts1)
    
    # Вычисляем средние и стандартные отклонения
    mu1 = np.mean(ts1)
    mu2 = np.mean(ts2)
    sigma1 = np.std(ts1)
    sigma2 = np.std(ts2)
    
    # Вычисляем скалярное произведение
    dot_product = np.dot(ts1, ts2)
    
    # Вычисляем нормализованное евклидово расстояние по формуле
    # ED_norm(T1, T2) = sqrt(|2n(1 - (<T1,T2> - n*mu1*mu2) / (n*sigma1*sigma2))|)
    numerator = dot_product - n * mu1 * mu2
    denominator = n * sigma1 * sigma2
    
    # Избегаем деления на ноль
    if denominator == 0:
        return 0.0
    
    distance_squared = 2 * n * (1 - numerator / denominator)
    
    # Берем абсолютное значение для избежания отрицательных чисел из-за погрешностей
    return np.sqrt(np.abs(distance_squared))


def DTW_distance(ts1: np.ndarray, ts2: np.ndarray, r: float = 1) -> float:
    """
    Calculate DTW distance

    Parameters
    ----------
    ts1: first time series
    ts2: second time series
    r: warping window size
    
    Returns
    -------
    dtw_dist: DTW distance between ts1 and ts2
    """

    n = len(ts1)
    m = len(ts2)
    
    # Инициализируем матрицу расстояний с бесконечностью
    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0
    
    # Заполняем матрицу
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            # Вычисляем квадрат разности между элементами рядов
            cost = (ts1[i-1] - ts2[j-1]) ** 2
            # Берем минимум из трех возможных путей
            dtw[i, j] = cost + min(dtw[i-1, j],    # удаление
                                   dtw[i, j-1],    # вставка
                                   dtw[i-1, j-1])  # соответствие
    
    # Возвращаем корень из накопленного расстояния в последней ячейке
    # return np.sqrt(dtw[n, m])
    return dtw[n, m]
