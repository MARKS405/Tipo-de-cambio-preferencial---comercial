
import streamlit as st
import pandas as pd
import numpy as np
import requests

from datetime import date, timedelta
from arch import arch_model

# ------------------------------------------------------------
# Parámetros globales
# ------------------------------------------------------------

SERIE_TC_SBS_VENTA = "PD04640PD"
FECHA_INICIO_SERIE = date(2021, 1, 1)
BCRP_API_URL = "https://estadisticas.bcrp.gob.pe/estadisticas/series/api"

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


# ------------------------
# Utilidades de fechas
# ------------------------

def es_habil(d: date) -> bool:
    return (d.weekday() < 5) and (d not in FERIADOS_PE)


def contar_dias_habiles(fecha_inicio: date, fecha_fin: date) -> int:
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
    fechas = []
    d = fecha_inicio
    while len(fechas) < n_dias_habiles:
        d += timedelta(days=1)
        if es_habil(d):
            fechas.append(pd.Timestamp(d))
    return fechas


# ------------------------
# Descarga de datos BCRP
# ------------------------

MESES_ES = {
    "ENE": 1, "FEB": 2, "MAR": 3, "ABR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AGO": 8, "SET": 9, "OCT": 10, "NOV": 11, "DIC": 12,
}


def parse_periodo_es(s):
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
    df = df[df.index.date >= FECHA_INICIO_SERIE]
    return df


# ------------------------
# Construcción TC SUNAT
# ------------------------

def construir_tc_sunat(df_sbs: pd.DataFrame):
    if df_sbs.empty:
        df_vacio = df_sbs.copy()
        for col in ["tc_sunat", "es_fin_de_semana", "es_feriado", "es_habil_real"]:
            df_vacio[col] = pd.Series(dtype=float if col == "tc_sunat" else bool)
        return df_vacio, df_vacio

    df = df_sbs.copy()
    first_date = df.index.min().date()
    today = date.today()
    idx_full = pd.date_range(start=first_date, end=today, freq="D")

    df_full = df.reindex(idx_full)
    df_full.index.name = "Periodo"
    df_full["tc_sbs_venta"] = df_full["tc_sbs_venta"].ffill()

    df_full["tc_sunat"] = df_full["tc_sbs_venta"].shift(1)
    df_full["tc_sunat"] = df_full["tc_sunat"].bfill()

    df_full["es_fin_de_semana"] = df_full.index.weekday >= 5
    df_full["es_feriado"] = [d in FERIADOS_PE for d in df_full.index.date]
    df_full["es_habil_real"] = ~(df_full["es_fin_de_semana"] | df_full["es_feriado"])

    df_habiles = df_full[df_full["es_habil_real"]].copy()
    return df_full, df_habiles


# ------------------------
# Retornos y resumen
# ------------------------

def calcular_retornos_log(df_sunat_habiles: pd.DataFrame) -> pd.Series:
    serie = df_sunat_habiles["tc_sunat"].dropna().sort_index()
    retornos = np.log(serie / serie.shift(1))
    retornos = retornos.dropna()
    return retornos


def resumen_paths(paths: np.ndarray, fechas_future) -> pd.DataFrame:
    if paths.shape[1] < 2:
        raise ValueError("paths debe tener al menos una columna para el futuro (S0 + pasos).")

    futuros = paths[:, 1:]
    media = futuros.mean(axis=0)
    p05 = np.percentile(futuros, 5, axis=0)
    p50 = np.percentile(futuros, 50, axis=0)
    p95 = np.percentile(futuros, 95, axis=0)

    idx = pd.to_datetime(fechas_future)
    df_resumen = pd.DataFrame(
        {"media": media, "p05": p05, "p50": p50, "p95": p95},
        index=idx,
    )
    return df_resumen


def calcular_var_cvar(valores: np.ndarray, alpha: float = 0.95) -> tuple[float, float]:
    if valores.ndim != 1:
        valores = valores.ravel()
    var = np.percentile(valores, alpha * 100)
    cola = valores[valores >= var]
    cvar = cola.mean() if len(cola) > 0 else var
    return float(var), float(cvar)


# ------------------------
# AR(1)-GARCH(1,1)
# ------------------------

def ajustar_garch(retornos_log: pd.Series):
    r = retornos_log.dropna().values
    if len(r) < 100:
        raise ValueError("Se requieren al menos 100 observaciones de retornos para ajustar GARCH.")
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
    if n_steps <= 0:
        raise ValueError("n_steps debe ser positivo.")
    if n_sims <= 0:
        raise ValueError("n_sims debe ser positivo.")

    params = res.params

    mu = params.get("mu", 0.0)
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

    resid_hist = np.asarray(res.resid)
    sigma_hist = np.asarray(res.conditional_volatility)

    mask_resid = np.isfinite(resid_hist)
    if not mask_resid.all():
        resid_hist = resid_hist[mask_resid]
    mask_sigma = np.isfinite(sigma_hist)
    if not mask_sigma.all():
        sigma_hist = sigma_hist[mask_sigma]

    r_last = r_hist[-1]
    eps_last = resid_hist[-1]
    sigma_last = sigma_hist[-1]

    r_paths = np.zeros((n_sims, n_steps))
    sigma_paths = np.zeros_like(r_paths)
    eps_paths = np.zeros_like(r_paths)

    r_prev = np.full(n_sims, r_last, dtype=float)
    eps_prev = np.full(n_sims, eps_last, dtype=float)
    sigma_prev = np.full(n_sims, sigma_last, dtype=float)

    for t in range(n_steps):
        sigma2_t = omega + alpha * (eps_prev ** 2) + beta * (sigma_prev ** 2)
        sigma_t = np.sqrt(np.maximum(sigma2_t, 0.0))

        z_t = np.random.randn(n_sims)
        eps_t = sigma_t * z_t

        r_t = mu + phi * r_prev + eps_t

        r_paths[:, t] = r_t
        sigma_paths[:, t] = sigma_t
        eps_paths[:, t] = eps_t

        r_prev = r_t
        eps_prev = eps_t
        sigma_prev = sigma_t

    paths = np.zeros((n_sims, n_steps + 1), dtype=float)
    paths[:, 0] = S0
    for t in range(1, n_steps + 1):
        paths[:, t] = paths[:, t - 1] * np.exp(r_paths[:, t - 1])

    return paths
