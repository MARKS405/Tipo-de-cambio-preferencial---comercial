
import streamlit as st
import pandas as pd
import numpy as np
import requests

from datetime import date, timedelta
from arch import arch_model

# ---------------------------------------------------------------------
# Parámetros globales
# ---------------------------------------------------------------------

SERIE_TC_SBS_VENTA = "PD04640PD"  # Tipo de cambio SBS venta (BCRP)
FECHA_INICIO_SERIE = date(2021, 1, 1)
BCRP_API_URL = "https://estadisticas.bcrp.gob.pe/estadisticas/series/api"

# Feriados simulados (actualiza esta tabla manualmente con la lista real)
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

# ---------------------------------------------------------------------
# Utilidades de fechas
# ---------------------------------------------------------------------

def es_habil(d: date) -> bool:
    """True si es día hábil en Perú (lunes-viernes y no feriado)."""
    return (d.weekday() < 5) and (d not in FERIADOS_PE)


def contar_dias_habiles(fecha_inicio: date, fecha_fin: date) -> int:
    """Cuenta días hábiles estrictamente entre fecha_inicio y fecha_fin."""
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
    """Genera una lista de timestamps para los próximos n_dias_habiles."""
    fechas = []
    d = fecha_inicio
    while len(fechas) < n_dias_habiles:
        d += timedelta(days=1)
        if es_habil(d):
            fechas.append(pd.Timestamp(d))
    return fechas


# ---------------------------------------------------------------------
# Descarga de datos BCRP (API JSON)
# ---------------------------------------------------------------------

MESES_ES = {
    "ENE": 1,
    "FEB": 2,
    "MAR": 3,
    "ABR": 4,
    "MAY": 5,
    "JUN": 6,
    "JUL": 7,
    "AGO": 8,
    "SET": 9,
    "OCT": 10,
    "NOV": 11,
    "DIC": 12,
}


def parse_periodo_es(s):
    """Convierte strings tipo '26Set25' o '04.Ene.21' a Timestamp."""
    if not isinstance(s, str):
        return pd.NaT

    s = s.strip()

    if "." in s:
        # Formato '04.Ene.21'
        parts = s.split(".")
        if len(parts) < 3:
            return pd.NaT
        try:
            dia = int(parts[0])
        except Exception:
            return pd.NaT
        mes_abbr = parts[1][:3].upper()
        anio_str = parts[2]
    else:
        # Formato '26Set25'
        try:
            dia = int(s[:2])
        except Exception:
            return pd.NaT
        mes_abbr = s[2:5].upper()
        anio_str = s[5:]

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
    Descarga la serie diaria de TC SBS (venta) desde el BCRP y
    devuelve un DataFrame indexado por fecha con la columna 'tc_sbs_venta'.
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
    df = df[df.index.date >= FECHA_INICIO_SERIE]
    return df


# ---------------------------------------------------------------------
# Construcción serie TC SUNAT
# ---------------------------------------------------------------------

def construir_tc_sunat(df_sbs: pd.DataFrame):
    """
    A partir del TC SBS (venta) construye la serie de TC SUNAT:
    SUNAT(t) = SBS_venta(t-1).
    Devuelve:
      - df_full: todos los días calendario con tc_sunat
      - df_habiles: sólo días hábiles reales (sin fines de semana ni feriados)
    """
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

    # Rellenar gaps en SBS con forward-fill
    df_full["tc_sbs_venta"] = df_full["tc_sbs_venta"].ffill()

    # SUNAT(t) = SBS_venta(t-1)
    df_full["tc_sunat"] = df_full["tc_sbs_venta"].shift(1)
    df_full["tc_sunat"] = df_full["tc_sunat"].bfill()

    # Flags de fin de semana / feriado
    df_full["es_fin_de_semana"] = df_full.index.weekday >= 5
    df_full["es_feriado"] = [d in FERIADOS_PE for d in df_full.index.date]
    df_full["es_habil_real"] = ~(df_full["es_fin_de_semana"] | df_full["es_feriado"])

    df_habiles = df_full[df_full["es_habil_real"]].copy()
    return df_full, df_habiles


# ---------------------------------------------------------------------
# Retornos y métricas
# ---------------------------------------------------------------------

def calcular_retornos_log(df_sunat_habiles: pd.DataFrame) -> pd.Series:
    """
    Calcula retornos log de la serie SUNAT en días hábiles y recorta outliers.
    """
    serie = df_sunat_habiles["tc_sunat"].dropna().sort_index()
    r = np.log(serie / serie.shift(1))
    r = r.dropna()

    # Recorte de outliers (±2% diario en log-retorno aprox.)
    limite = 0.02
    r = r.clip(lower=-limite, upper=limite)

    return r


def resumen_paths(paths: np.ndarray, fechas_future) -> pd.DataFrame:
    """
    paths: shape (n_sims, n_steps+1) con niveles de TC.
    Sólo miramos las columnas futuras (1:).
    """
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


def var_cvar_retorno(S_T: np.ndarray, S0: float, alpha: float = 0.95):
    """
    Calcula VaR/CVaR sobre el retorno total al vencimiento y devuelve
    también el TC implícito asociado a VaR y CVaR.
    """
    S_T = np.asarray(S_T).ravel()
    ret = S_T / S0 - 1.0

    var_ret = np.percentile(ret, alpha * 100.0)
    cola = ret[ret >= var_ret]
    cvar_ret = cola.mean() if len(cola) > 0 else var_ret

    tc_var = S0 * (1.0 + var_ret)
    tc_cvar = S0 * (1.0 + cvar_ret)

    return float(var_ret), float(cvar_ret), float(tc_var), float(tc_cvar)


# ---------------------------------------------------------------------
# Ajuste GARCH y simulación Monte Carlo
# ---------------------------------------------------------------------

def ajustar_garch(retornos_log: pd.Series):
    """
    Ajusta un GARCH(1,1) con media cero sobre retornos en %.
    """
    r = 100.0 * retornos_log.dropna().values  # a porcentajes

    if len(r) < 200:
        raise ValueError("Se requieren al menos 200 retornos para ajustar GARCH.")

    am = arch_model(r, mean="Zero", vol="GARCH", p=1, q=1, dist="normal")
    res = am.fit(disp="off")
    return res


def simular_arma_garch(res_garch, S0: float, n_steps: int, n_sims: int) -> np.ndarray:
    """
    Simula trayectorias del tipo de cambio usando el modelo GARCH(1,1)
    ajustado en `ajustar_garch`.

    - El modelo está estimado sobre retornos EN PORCENTAJE (100 * r_log).
    - Aquí simulamos retornos en %, luego los convertimos a log-retornos
      para construir niveles de TC.

    Devuelve:
        paths: array (n_sims, n_steps) con los niveles simulados de TC.
    """
    params = res_garch.params

    # Helper para encontrar parámetros aunque cambie un poco el nombre
    def get_param(name_like: str):
        for k in params.index:
            if name_like in k:
                return float(params[k])
        raise KeyError(
            f"No se encontró un parámetro que contenga '{name_like}' en "
            f"res_garch.params: {list(params.index)}"
        )

    # GARCH(1,1): sigma_t^2 = omega + alpha * r_{t-1}^2 + beta * sigma_{t-1}^2
    omega = get_param("omega")
    alpha = get_param("alpha")
    beta = get_param("beta")

    # Varianza incondicional (en la escala de %)
    if alpha + beta < 0.999:
        var_uncond = omega / (1.0 - alpha - beta)
    else:
        # Si el proceso es casi no estacionario, usamos la varianza muestral
        var_uncond = float(np.var(res_garch.resid * 100.0))

    if var_uncond <= 0:
        var_uncond = 1.0

    # Inicializamos matrices
    # sigma2 y r_pct están en la escala de PORCENTAJE
    sigma2 = np.full((n_steps + 1, n_sims), var_uncond, dtype=float)
    r_pct = np.zeros((n_steps, n_sims), dtype=float)

    # Simulamos GARCH(1,1)
    for t in range(n_steps):
        eps = np.random.normal(size=n_sims)        # shocks ~ N(0,1)
        sigma_t = np.sqrt(sigma2[t])              # desviación estándar en %
        r_pct[t] = sigma_t * eps                  # retorno en %
        sigma2[t + 1] = omega + alpha * (r_pct[t] ** 2) + beta * sigma2[t]

    # Pasamos de retornos en % a log-retornos en DECIMAL
    r_log = r_pct / 100.0

    # Construimos niveles de tipo de cambio
    log_S0 = np.log(S0)
    log_paths = log_S0 + np.cumsum(r_log, axis=0)   # shape: (n_steps, n_sims)
    paths = np.exp(log_paths).T                     # -> (n_sims, n_steps)

    return paths
