import numpy as np
import math
import copy

from modules.utils import sliding_window, z_normalize
from modules.metrics import DTW_distance


def apply_exclusion_zone(array: np.ndarray, idx: int, excl_zone: int) -> np.ndarray:
    """
    Apply an exclusion zone to an array (inplace)
    
    Parameters
    ----------
    array: the array to apply the exclusion zone to
    idx: the index around which the window should be centered
    excl_zone: size of the exclusion zone
    
    Returns
    -------
    array: the array which is applied the exclusion zone
    """

    zone_start = max(0, idx - excl_zone)
    zone_stop = min(array.shape[-1], idx + excl_zone)
    array[zone_start : zone_stop + 1] = np.inf

    return array


def topK_match(dist_profile: np.ndarray, excl_zone: int, topK: int = 3, max_distance: float = np.inf) -> dict:
    """
    Search the topK match subsequences based on distance profile
    
    Parameters
    ----------
    dist_profile: distances between query and subsequences of time series
    excl_zone: size of the exclusion zone
    topK: count of the best match subsequences
    max_distance: maximum distance between query and a subsequence `S` for `S` to be considered a match
    
    Returns
    -------
    topK_match_results: dictionary containing results of algorithm
    """

    topK_match_results = {
        'indices': [],
        'distances': []
    } 

    dist_profile_len = len(dist_profile)
    dist_profile = np.copy(dist_profile).astype(float)

    for k in range(topK):
        min_idx = np.argmin(dist_profile)
        min_dist = dist_profile[min_idx]

        if (np.isnan(min_dist)) or (np.isinf(min_dist)) or (min_dist > max_distance):
            break

        dist_profile = apply_exclusion_zone(dist_profile, min_idx, excl_zone)

        topK_match_results['indices'].append(min_idx)
        topK_match_results['distances'].append(min_dist)

    return topK_match_results


class BestMatchFinder:
    """
    Base Best Match Finder
    
    Parameters
    ----------
    excl_zone_frac: exclusion zone fraction
    topK: number of the best match subsequences
    is_normalize: z-normalize or not subsequences before computing distances
    r: warping window size
    """

    def __init__(self, excl_zone_frac: float = 1, topK: int = 3, is_normalize: bool = True, r: float = 0.05) -> None:
        """ 
        Constructor of class BestMatchFinder
        """

        self.excl_zone_frac: float = excl_zone_frac
        self.topK: int = topK
        self.is_normalize: bool = is_normalize
        self.r: float = r


    def _calculate_excl_zone(self, m: int) -> int:
        """
        Calculate the exclusion zone
        
        Parameters
        ----------
        m: length of subsequence
        
        Returns
        -------
        excl_zone: exclusion zone
        """

        excl_zone = math.ceil(m * self.excl_zone_frac)

        return excl_zone


    def perform(self):

        raise NotImplementedError


class NaiveBestMatchFinder(BestMatchFinder):
    """
    Naive Best Match Finder
    """

    def __init__(self, excl_zone_frac: float = 1, topK: int = 3, is_normalize: bool = True, r: float = 0.05):
        super().__init__(excl_zone_frac, topK, is_normalize, r)
        """ 
        Constructor of class NaiveBestMatchFinder
        """


    def perform(self, ts_data: np.ndarray, query: np.ndarray) -> dict:
        """
        Search subsequences in a time series that most closely match the query using the naive algorithm
        
        Parameters
        ----------
        ts_data: time series
        query: query, shorter than time series

        Returns
        -------
        best_match: dictionary containing results of the naive algorithm
        """

        query = copy.deepcopy(query)
        if (len(ts_data.shape) != 2): # time series set
            ts_data = sliding_window(ts_data, len(query))

        N, m = ts_data.shape
        excl_zone = self._calculate_excl_zone(m)

        dist_profile = np.ones((N,))*np.inf
        bsf = np.inf

        bestmatch = {
            'index' : [],
            'distance' : []
        }
        
        # Нормализуем запрос, если требуется
        if self.is_normalize:
            query_norm = z_normalize(query)
        else:
            query_norm = query
        
        # Вычисляем DTW расстояние для каждой подпоследовательности
        for i in range(N):
            # Извлекаем подпоследовательность
            subsequence = ts_data[i]
            
            # Нормализуем подпоследовательность, если требуется
            if self.is_normalize:
                subsequence_norm = z_normalize(subsequence)
            else:
                subsequence_norm = subsequence
            
            # Вычисляем DTW расстояние с ограничением
            # self.r - вещественное число (радиус)
            dist = DTW_distance(query_norm, subsequence_norm, self.r)
            
            # Сохраняем расстояние в профиль
            dist_profile[i] = dist
            
            # Обновляем best-so-far
            if dist < bsf:
                bsf = dist
        
        # Находим topK похожих подпоследовательностей
        topK_results = topK_match(dist_profile, excl_zone, self.topK)
        
        bestmatch['index'] = topK_results['indices']
        bestmatch['distance'] = topK_results['distances']

        return bestmatch



class UCR_DTW(BestMatchFinder):
    """
    UCR-DTW Match Finder
    
    Additional parameters
    ----------
    not_pruned_num: number of non-pruned subsequences
    lb_Kim_num: number of subsequences that pruned by LB_Kim bounding
    lb_KeoghQC_num: number of subsequences that pruned by LB_KeoghQC bounding
    lb_KeoghCQ_num: number of subsequences that pruned by LB_KeoghCQ bounding
    """

    def __init__(self, excl_zone_frac: float = 1, topK: int = 3, is_normalize: bool = True, r: float = 0.05):
        super().__init__(excl_zone_frac, topK, is_normalize, r)
        """ 
        Constructor of class UCR_DTW
        """        

        self.not_pruned_num = 0
        self.lb_Kim_num = 0
        self.lb_KeoghQC_num = 0
        self.lb_KeoghCQ_num = 0


    def _LB_Kim(self, subs1: np.ndarray, subs2: np.ndarray) -> float:
        """
        Compute LB_Kim lower bound between two subsequences
        
        Parameters
        ----------
        subs1: the first subsequence
        subs2: the second subsequence
        
        Returns
        -------
        lb_Kim: LB_Kim lower bound
        """
        
        # LB_KimFL: сумма квадратов разностей первых и последних точек
        lb_Kim = (subs1[0] - subs2[0]) ** 2 + (subs1[-1] - subs2[-1]) ** 2
        
        return lb_Kim


    def _LB_Keogh(self, subs1: np.ndarray, subs2: np.ndarray, r: float) -> float:
        """
        Compute LB_Keogh lower bound between two subsequences
        
        Parameters
        ----------
        subs1: the first subsequence (query)
        subs2: the second subsequence (candidate)
        r: warping window size
        
        Returns
        -------
        lb_Keogh: LB_Keogh lower bound
        """
        
        n = len(subs1)
        
        # Строим верхнюю и нижнюю оболочки для subs1 (запроса)
        U = np.zeros(n)
        L = np.zeros(n)
        
        # Вычисляем оболочки с учетом радиуса r
        for i in range(n):
            # Определяем границы окна
            start = max(0, i - int(np.ceil(r)))
            end = min(n, i + int(np.ceil(r)) + 1)
            
            # Верхняя оболочка - максимум в окне
            U[i] = np.max(subs1[start:end])
            # Нижняя оболочка - минимум в окне
            L[i] = np.min(subs1[start:end])
        
        # Вычисляем LB_Keogh
        lb_Keogh = 0
        for i in range(n):
            if subs2[i] > U[i]:
                lb_Keogh += (subs2[i] - U[i]) ** 2
            elif subs2[i] < L[i]:
                lb_Keogh += (subs2[i] - L[i]) ** 2
        
        return lb_Keogh


    def get_statistics(self) -> dict:
        """
        Return statistics on the number of pruned and non-pruned subsequences of a time series   
        
        Returns
        -------
            dictionary containing statistics
        """

        statistics = {
            'not_pruned_num': self.not_pruned_num,
            'lb_Kim_num': self.lb_Kim_num,
            'lb_KeoghCQ_num': self.lb_KeoghCQ_num,
            'lb_KeoghQC_num': self.lb_KeoghQC_num
        }

        return statistics


    def perform(self, ts_data: np.ndarray, query: np.ndarray) -> dict:
        """
        Search subsequences in a time series that most closely match the query using UCR-DTW algorithm
        
        Parameters
        ----------
        ts_data: time series
        query: query, shorter than time series

        Returns
        -------
        best_match: dictionary containing results of UCR-DTW algorithm
        """

        query = copy.deepcopy(query)
        if (len(ts_data.shape) != 2):  # time series set
            ts_data = sliding_window(ts_data, len(query))

        N, m = ts_data.shape

        excl_zone = self._calculate_excl_zone(m)

        dist_profile = np.ones((N,)) * np.inf
        bsf = np.inf
        
        bestmatch = {
            'index': [],
            'distance': []
        }

        # Нормализуем запрос
        if self.is_normalize:
            query_norm = z_normalize(query)
        else:
            query_norm = query

        # Предварительное вычисление оболочек для запроса (используется в LB_KeoghEC)
        # Для LB_KeoghEQ оболочки строятся вокруг запроса
        U_query = np.zeros(m)
        L_query = np.zeros(m)
        
        for i in range(m):
            start = max(0, i - int(np.ceil(self.r)))
            end = min(m, i + int(np.ceil(self.r)) + 1)
            U_query[i] = np.max(query_norm[start:end])
            L_query[i] = np.min(query_norm[start:end])

        # Проходим по всем подпоследовательностям
        for i in range(N):
            # Извлекаем подпоследовательность
            subsequence = ts_data[i]
            
            # Нормализуем, если требуется
            if self.is_normalize:
                subsequence_norm = z_normalize(subsequence)
            else:
                subsequence_norm = subsequence
            
            # 1. LB_Kim
            lb_kim = self._LB_Kim(query_norm, subsequence_norm)
            if lb_kim > bsf:
                self.lb_Kim_num += 1
                continue
            
            # 2. LB_KeoghEQ (оболочка вокруг запроса)
            lb_keogh_eq = self._LB_Keogh(query_norm, subsequence_norm, self.r)
            if lb_keogh_eq > bsf:
                self.lb_KeoghQC_num += 1
                continue
            
            # 3. LB_KeoghEC (оболочка вокруг кандидата)
            # Строим оболочку для кандидата
            U_candidate = np.zeros(m)
            L_candidate = np.zeros(m)
            
            for j in range(m):
                start = max(0, j - int(np.ceil(self.r)))
                end = min(m, j + int(np.ceil(self.r)) + 1)
                U_candidate[j] = np.max(subsequence_norm[start:end])
                L_candidate[j] = np.min(subsequence_norm[start:end])
            
            # Вычисляем LB_KeoghEC (оболочка вокруг кандидата, сравниваем с запросом)
            lb_keogh_ec = 0
            for j in range(m):
                if query_norm[j] > U_candidate[j]:
                    lb_keogh_ec += (query_norm[j] - U_candidate[j]) ** 2
                elif query_norm[j] < L_candidate[j]:
                    lb_keogh_ec += (query_norm[j] - L_candidate[j]) ** 2
            
            if lb_keogh_ec > bsf:
                self.lb_KeoghCQ_num += 1
                continue
            
            # 4. Если все нижние границы пройдены, вычисляем DTW
            self.not_pruned_num += 1
            dist = DTW_distance(query_norm, subsequence_norm, self.r)
            
            # Сохраняем расстояние в профиль
            dist_profile[i] = dist
            
            # Обновляем best-so-far
            if dist < bsf:
                bsf = dist

        # Находим topK похожих подпоследовательностей
        topK_results = topK_match(dist_profile, excl_zone, self.topK)
        
        bestmatch['index'] = topK_results['indices']
        bestmatch['distance'] = topK_results['distances']

        return bestmatch