
import streamlit as st
import pandas as pd
import numpy as np
import requests

from datetime import date, datetime, timedelta
from arch import arch_model

# ------------------------------------------------------------
# Parámetros globales
# ------------------------------------------------------------

# Serie diaria: TC Sistema bancario SBS (S/ por US$) - Venta (BCRP)
SERIE_TC_SBS_VENTA = "PD04640PD"

# Fecha mínima que quieres usar como histórico
FECHA_INICIO_SERIE = date(2021, 1, 1)

# URL base API BCRP
BCRP_API_URL = "https://estadisticas.bcrp.gob.pe/estadisticas/series/api"

# Feriados simulados para Perú (edita y amplía esta tabla según tus necesidades)
FERIADOS_PE = {
    date(2021, 1, 1),
    date(2021, 4, 1),
    date(2021, 4, 2),
    date(2021, 5, 1),
    date(2021, 7, 28),
    date(2021, 7, 29),
    date(2021, 12, 8),
    date(2021, 12, 25),
    date(2022, 1, 1),
    date(2022, 4, 14),
    date(2022, 4, 15),
    date(2022, 5, 1),
    date(2022, 7, 28),
    date(2022, 7, 29),
    date(2022, 12, 8),
    date(2022, 12, 25),
    date(2023, 1, 1),
    date(2023, 4, 6),
    date(2023, 4, 7),
    date(2023, 5, 1),
    date(2023, 7, 28),
    date(2023, 7, 29),
    date(2023, 12, 8),
    date(2023, 12, 25),
    date(2024, 1, 1),
    date(2024, 3, 28),
    date(2024, 3, 29),
    date(2024, 5, 1),
    date(2024, 7, 28),
    date(2024, 7, 29),
    date(2024, 12, 8),
    date(2024, 12, 25),
    date(2025, 1, 1),
    date(2025, 4, 17),
    date(2025, 4, 18),
    date(2025, 5, 1),
    date(2025, 7, 28),
    date(2025, 7, 29),
    date(2025, 12, 8),
    date(2025, 12, 25),
}


# ------------------------------------------------------------
# Utilidades de fechas
# ------------------------------------------------------------

def es_habil(d: date) -> bool:
    """Retorna True si la fecha es día hábil (no sábado/domingo ni feriado)."""
    return (d.weekday() < 5) and (d not in FERIADOS_PE)


def contar_dias_habiles(fecha_inicio: date, fecha_fin: date) -> int:
    """
    Cuenta cuántos días hábiles hay entre fecha_inicio y fecha_fin,
    excluyendo la fecha de inicio e incluyendo la fecha de fin si es hábil.
    """
    if fecha_fin <= fecha_inicio:
        return 0

    d = fecha_inicio
    contador = 0
    while d < fecha_fin:
        d += timedelta(days=1)
        if es_habil(d):
            contador += 1
    return contador


def generar_fechas_habiles(fecha_inicio: date, n_dias_habiles: int):
    """
    Genera una lista con las próximas n_dias_habiles fechas hábiles
    posteriores a fecha_inicio.
    """
    fechas = []
    d = fecha_inicio
    while len(fechas) < n_dias_habiles:
        d += timedelta(days=1)
        if es_habil(d):
            fechas.append(pd.Timestamp(d))
    return fechas


# ------------------------------------------------------------
# Descarga de datos del BCRP
# ------------------------------------------------------------

MESES_ES = {
    "ENE": 1, "FEB": 2, "MAR": 3, "ABR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AGO": 8, "SET": 9, "OCT": 10, "NOV": 11, "DIC": 12,
}


def parse_periodo_es(s):
    """
    Convierte cadenas tipo '04.Ene.21' en Timestamp.
    Si no puede parsear, devuelve NaT.
    """
    if not isinstance(s, str):
        return pd.NaT
    s = s.strip()
    parts = s.split(".")
    if len(parts) < 3:
        return pd.NaT
    try:
        dia = int(parts[0])
    except Exception:
        return pd.NaT
    mes_abbr = parts[1][:3].upper()
    anio_str = parts[2]
    try:
        if len(anio_str) == 2:
            anio = 2000 + int(anio_str)
        else:
            anio = int(anio_str)
    except Exception:
        return pd.NaT
    mes = MESES_ES.get(mes_abbr)
    if mes is None:
        return pd.NaT
    try:
        return pd.Timestamp(year=anio, month=mes, day=dia)
    except Exception:
        return pd.NaT


@st.cache_data(ttl=86400)
def obtener_dataframe_bcrp(
    fecha_inicio: date | None = None,
    fecha_fin: date | None = None,
) -> pd.DataFrame:
    """
    Descarga la serie de TC SBS (venta) desde la API del BCRP y la devuelve
    como DataFrame con índice datetime y columna 'tc_sbs_venta'.

    Se cachea por 1 día para evitar reconsultar siempre.
    """
    if fecha_inicio is None:
        fecha_inicio = FECHA_INICIO_SERIE
    if fecha_fin is None:
        fecha_fin = date.today()

    periodo_inicial = fecha_inicio.strftime("%Y-%m-%d")
    periodo_final = fecha_fin.strftime("%Y-%m-%d")

    url = f"{BCRP_API_URL}/{SERIE_TC_SBS_VENTA}/json/{periodo_inicial}/{periodo_final}"

    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    periods = data.get("periods", [])
    if not periods:
        raise ValueError("La API del BCRP no devolvió datos para el rango solicitado.")

    series_cfg = data.get("config", {}).get("series", [])
    if series_cfg:
        serie_name = series_cfg[0].get("name", "valor")
    else:
        serie_name = "valor"

    periodos = [p.get("name") for p in periods]
    valores_raw = [p.get("values", [None]) for p in periods]
    valores = [v[0] if isinstance(v, list) and v else None for v in valores_raw]

    df = pd.DataFrame({"Periodo": periodos, serie_name: valores})
    df[serie_name] = pd.to_numeric(df[serie_name], errors="coerce")
    df["Periodo"] = df["Periodo"].apply(parse_periodo_es)
    df = df.dropna(subset=["Periodo"])

    df = df.set_index("Periodo").sort_index()
    df = df.rename(columns={serie_name: "tc_sbs_venta"})

    # Filtrar desde FECHA_INICIO_SERIE por seguridad
    df = df[df.index.date >= FECHA_INICIO_SERIE]

    return df


# ------------------------------------------------------------
# Construcción de TC SUNAT a partir del SBS
# ------------------------------------------------------------

def construir_tc_sunat(df_sbs: pd.DataFrame):
    """
    A partir de la serie de TC SBS (venta) construye la serie de TC SUNAT:

    1. Se reindexa la serie SBS a TODOS los días calendario (incluyendo fines de semana),
       desde la primera fecha disponible hasta 'hoy'.
    2. Se rellena hacia adelante (ffill) cuando falten datos SBS: regla "tomar el TC
       del día inmediato anterior".
    3. Se define TC_SUNAT(t) = TC_SBS_filled(t-1).
    4. Se marcan fines de semana y feriados, y se construye un DataFrame solo con
       días hábiles reales (sin sábados, domingos ni feriados).

    Devuelve:
      - df_full: con todos los días calendario y tc_sunat
      - df_habiles: solo días hábiles reales (lunes-viernes y no feriados)
    """
    if df_sbs.empty:
        df_vacio = df_sbs.copy()
        df_vacio["tc_sunat"] = pd.Series(dtype=float)
        df_vacio["es_fin_de_semana"] = pd.Series(dtype=bool)
        df_vacio["es_feriado"] = pd.Series(dtype=bool)
        df_vacio["es_habil_real"] = pd.Series(dtype=bool)
        return df_vacio, df_vacio

    df = df_sbs.copy()

    # 1) Índice calendario completo: desde el primer dato SBS hasta hoy
    first_date = df.index.min().date()
    today = date.today()
    idx_full = pd.date_range(start=first_date, end=today, freq="D")

    # 2) Reindexar SBS a calendario y rellenar con último valor conocido
    df_full = df.reindex(idx_full)
    df_full.index.name = "Periodo"
    df_full["tc_sbs_venta"] = df_full["tc_sbs_venta"].ffill()

    # 3) SUNAT(t) = SBS_filled(t-1)
    df_full["tc_sunat"] = df_full["tc_sbs_venta"].shift(1)
    df_full["tc_sunat"] = df_full["tc_sunat"].bfill()

    # 4) Marcar fines de semana y feriados
    df_full["es_fin_de_semana"] = df_full.index.weekday >= 5
    df_full["es_feriado"] = [d in FERIADOS_PE for d in df_full.index.date]

    # Día hábil real = no fin de semana, no feriado
    df_full["es_habil_real"] = ~(df_full["es_fin_de_semana"] | df_full["es_feriado"])

    # 5) DataFrame solo con días hábiles reales
    df_habiles = df_full[df_full["es_habil_real"]].copy()

    return df_full, df_habiles


# ------------------------------------------------------------
# Retornos y resumen de simulaciones
# ------------------------------------------------------------

def calcular_retornos_log(df_sunat_habiles: pd.DataFrame) -> pd.Series:
    """
    Calcula retornos logarítmicos diarios del TC SUNAT sobre días hábiles reales.
    """
    serie = df_sunat_habiles["tc_sunat"].dropna().sort_index()
    retornos = np.log(serie / serie.shift(1))
    retornos = retornos.dropna()
    return retornos


def resumen_paths(paths: np.ndarray, fechas_future) -> pd.DataFrame:
    """
    Recibe paths (n_sims, n_steps+1) y una lista de fechas futuras (n_steps)
    y construye un DataFrame con media y percentiles 5, 50 y 95.
    """
    if paths.shape[1] < 2:
        raise ValueError("paths debe tener al menos una columna para el futuro (S0 + pasos).")

    futuros = paths[:, 1:]  # descartamos la columna inicial S0
    media = futuros.mean(axis=0)
    p05 = np.percentile(futuros, 5, axis=0)
    p50 = np.percentile(futuros, 50, axis=0)
    p95 = np.percentile(futuros, 95, axis=0)

    idx = pd.to_datetime(fechas_future)
    df_resumen = pd.DataFrame(
        {
            "media": media,
            "p05": p05,
            "p50": p50,
            "p95": p95,
        },
        index=idx,
    )
    return df_resumen


def calcular_var_cvar(valores: np.ndarray, alpha: float = 0.95) -> tuple[float, float]:
    """
    Calcula VaR y CVaR (Expected Shortfall) para un nivel alpha sobre un array de valores
    (por ejemplo, TCs simulados en la fecha final).
    """
    if valores.ndim != 1:
        valores = valores.ravel()

    var = np.percentile(valores, alpha * 100)
    cola = valores[valores >= var]
    if len(cola) == 0:
        cvar = var
    else:
        cvar = cola.mean()

    return float(var), float(cvar)


# ------------------------------------------------------------
# AR(1)-GARCH(1,1) y simulación
# ------------------------------------------------------------

def ajustar_garch(retornos_log: pd.Series):
    """
    Ajusta un modelo AR(1)-GARCH(1,1) con distribución normal
    a la serie de retornos logarítmicos.
    """
    r = retornos_log.dropna().values
    if len(r) < 100:
        raise ValueError("Se requieren al menos 100 observaciones de retornos para ajustar GARCH.")

    # El modelo puede trabajar directamente con retornos pequeños
    am = arch_model(r, mean="AR", lags=1, vol="GARCH", p=1, q=1, dist="normal")
    res = am.fit(disp="off")
    return res


def simular_arma_garch(
    res,
    retornos_log: pd.Series,
    S0: float,
    n_steps: int,
    n_sims: int,
) -> np.ndarray:
    """
    Simula trayectorias de retornos futuros usando el modelo AR(1)-GARCH(1,1)
    ajustado (res), y luego reconstruye los niveles del tipo de cambio a partir
    de S0.

    Devuelve un array de shape (n_sims, n_steps+1) con:
      - columna 0: S0
      - columnas 1..n_steps: niveles simulados del TC.
    """
    if n_steps <= 0:
        raise ValueError("n_steps debe ser positivo.")
    if n_sims <= 0:
        raise ValueError("n_sims debe ser positivo.")

    params = res.params

    # Nombres típicos de parámetros en arch_model AR(1)-GARCH(1,1)
    # mean: mu, ar[1]
    # vol: omega, alpha[1], beta[1]
    mu = params.get("mu", 0.0)
    # compatibilidad con diferentes nombres de parámetro para AR(1)
    phi = 0.0
    for key in params.index:
        if "ar" in key.lower():
            phi = params[key]
            break

    omega = params.get("omega", 0.0)
    alpha = None
    beta = None
    for key in params.index:
        lk = key.lower()
        if "alpha" in lk and alpha is None:
            alpha = params[key]
        if "beta" in lk and beta is None:
            beta = params[key]
    if alpha is None:
        alpha = 0.0
    if beta is None:
        beta = 0.0

    r_hist = retornos_log.dropna().values
    resid_hist = res.resid.dropna().values
    sigma_hist = res.conditional_volatility.dropna().values

    r_last = r_hist[-1]
    eps_last = resid_hist[-1]
    sigma_last = sigma_hist[-1]

    # Matrices para retornos, sigmas y epsilons simulados
    r_paths = np.zeros((n_sims, n_steps))
    sigma_paths = np.zeros_like(r_paths)
    eps_paths = np.zeros_like(r_paths)

    # Estados iniciales (mismo último estado para todos los caminos)
    r_prev = np.full(n_sims, r_last, dtype=float)
    eps_prev = np.full(n_sims, eps_last, dtype=float)
    sigma_prev = np.full(n_sims, sigma_last, dtype=float)

    for t in range(n_steps):
        # GARCH(1,1): sigma_t^2 = omega + alpha * eps_{t-1}^2 + beta * sigma_{t-1}^2
        sigma2_t = omega + alpha * (eps_prev ** 2) + beta * (sigma_prev ** 2)
        sigma_t = np.sqrt(np.maximum(sigma2_t, 0.0))

        # Innovaciones ~ N(0,1)
        z_t = np.random.randn(n_sims)
        eps_t = sigma_t * z_t

        # AR(1): r_t = mu + phi * r_{t-1} + eps_t
        r_t = mu + phi * r_prev + eps_t

        # Guardar
        r_paths[:, t] = r_t
        sigma_paths[:, t] = sigma_t
        eps_paths[:, t] = eps_t

        # Actualizar estados
        r_prev = r_t
        eps_prev = eps_t
        sigma_prev = sigma_t

    # Reconstruir niveles del tipo de cambio
    paths = np.zeros((n_sims, n_steps + 1), dtype=float)
    paths[:, 0] = S0
    for t in range(1, n_steps + 1):
        paths[:, t] = paths[:, t - 1] * np.exp(r_paths[:, t - 1])

    return paths
