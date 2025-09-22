import numpy as np
import pandas as pd

from scipy.stats import skew, kurtosis



def compute_thd_npdata(matrix: np.array) -> pd.DataFrame:
    """
    Calcula el THD (Total Harmonic Distortion) de una señal.

    Parámetros:
      - signal: array 1D con valores de la señal en el dominio tiempo (amplitudes).

    Retorna:
      - thd_f: el valor de THD_F (fracción, no porcentaje).
      - thd_percent: THD expresado en porcentaje.
    """
    N = matrix.shape[1]
    Y = np.fft.rfft(matrix, axis=1)
    Y_mag = np.abs(Y) / N * 2
    funadmental_idx = np.argmax(Y_mag[:,1:], axis=1) + 1
    V1 = np.max(Y_mag, axis=1)
    thd_f = np.sqrt(np.sum(np.square(Y_mag), axis=1) - np.square(V1)) / V1
    return thd_f, funadmental_idx


def compute_cwru_features(matrix: np.array) -> pd.DataFrame:
    """
    Calcula estadísticas descriptivas para cada fila de una matriz 2D.

    Parámetros
    ----------
    matrix : np.ndarray
        Matriz de forma (n_series, n_samples) donde cada fila es una serie temporal.

    Retorna
    -------
    pd.DataFrame
        DataFrame con columnas:
        ['maximum', 'minimum', 'mean', 'std', 'rms', 
         'skewness', 'kurtosis', 'crest_factor', 'form_factor']
    """
    maximum = np.max(matrix, axis=1)
    minimum = np.min(matrix, axis=1)
    mean = np.mean(matrix, axis=1)
    std = np.std(matrix, axis=1, ddof=1)  # ddof=1 → std muestral
    rms = np.sqrt(np.mean(matrix**2, axis=1))
    skewness = skew(matrix, axis=1, bias=False)
    kurt = kurtosis(matrix, axis=1, bias=False)
    crest_factor = maximum / rms
    form_factor = rms / np.mean(np.abs(matrix), axis=1)

    df = pd.DataFrame({
        "maximum": maximum,
        "minimum": minimum,
        "mean": mean,
        "std": std,
        "rms": rms,
        "skewness": skewness,
        "kurtosis": kurt,
        "crest_factor": crest_factor,
        "form_factor": form_factor
    })
    return df


def compute_features(matrix: np.array) -> pd.DataFrame:

    df = compute_cwru_features(matrix)
    df["thd"], df["f0"] = compute_thd_npdata(matrix)

    return df
