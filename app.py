
import streamlit as st
from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from tc_sunat_model import (
    contar_dias_habiles,
    generar_fechas_habiles,
    obtener_dataframe_bcrp,
    construir_tc_sunat,
    calcular_retornos_log,
    ajustar_garch,
    simular_arma_garch,
    resumen_paths,
    calcular_var_cvar,
)


# ------------------------------------------------------------
# Utilidad para recortar histórico a una ventana ajustada
# ------------------------------------------------------------

def recortar_historico(df_habiles: pd.DataFrame,
                       fecha_inicio: date,
                       n_steps_habiles: int,
                       factor_hist: int = 4,
                       min_points: int = 60,
                       max_points: int = 504) -> pd.DataFrame:
    """
    Recorta el histórico de df_habiles (con índice datetime y columna tc_sunat)
    para mostrar aproximadamente factor_hist veces el horizonte futuro, con
    límites mínimo y máximo de puntos.
    """
    if df_habiles.empty:
        return df_habiles

    mask_before = df_habiles.index.date <= fecha_inicio
    df_before = df_habiles.loc[mask_before]
    if df_before.empty:
        return df_before

    n_target = int(min(max(factor_hist * max(n_steps_habiles, 1), min_points), max_points))
    df_hist_plot = df_before.iloc[-n_target:]
    return df_hist_plot


# ------------------------------------------------------------
# Gráfico 1: histórico + simulaciones (Plotly)
# ------------------------------------------------------------

def plot_hist_y_sim_plotly(df_hist_plot: pd.DataFrame,
                           fechas_future,
                           paths: np.ndarray):
    """
    Gráfico interactivo con Plotly:
      - Histórico reciente del TC SUNAT.
      - Algunas trayectorias simuladas a partir del último día hábil.
    """
    fig = go.Figure()

    # Histórico
    fig.add_trace(go.Scatter(
        x=df_hist_plot.index,
        y=df_hist_plot["tc_sunat"],
        name="TC SUNAT histórico (reciente)",
        mode="lines",
        line=dict(width=2),
        hovertemplate="Fecha: %{x}<br>TC: %{y:.4f}<extra></extra>",
    ))

    if paths.size > 0:
        n_sims = paths.shape[0]
        n_mostrar = min(60, n_sims)
        idx = np.random.choice(n_sims, n_mostrar, replace=False)

        # punto de unión entre histórico y simulaciones
        last_hist_date = df_hist_plot.index[-1]

        for i in idx:
            fig.add_trace(go.Scatter(
                x=[last_hist_date] + list(fechas_future),
                y=paths[i, :],
                mode="lines",
                line=dict(width=1),
                opacity=0.25,
                showlegend=False,
                hovertemplate="Fecha: %{x}<br>Simulación: %{y:.4f}<extra></extra>",
            ))

    fig.update_layout(
        title="Histórico reciente del TC SUNAT + trayectorias simuladas",
        xaxis_title="Fecha",
        yaxis_title="Tipo de cambio (S/ por US$)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=40, r=20, t=60, b=40),
    )

    st.plotly_chart(fig, use_container_width=True)


# ------------------------------------------------------------
# Gráfico 2: escenarios (media + banda) + VaR horizontal
# ------------------------------------------------------------

def plot_escenarios_y_var(df_resumen: pd.DataFrame,
                          var_final: float,
                          alpha: float):
    """
    Gráfico de:
      - Media proyectada en el tiempo.
      - Banda P5–P95.
      - Línea horizontal de VaR (nivel en la fecha final).
    """
    fig = go.Figure()

    # Banda P5–P95
    fig.add_trace(go.Scatter(
        x=df_resumen.index,
        y=df_resumen["p05"],
        mode="lines",
        line=dict(width=0),
        showlegend=False,
        hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=df_resumen.index,
        y=df_resumen["p95"],
        mode="lines",
        line=dict(width=0),
        fill="tonexty",
        name="Banda P5–P95",
        opacity=0.2,
        hovertemplate="Fecha: %{x}<br>P5–P95: %{y:.4f}<extra></extra>",
    ))

    # Media
    fig.add_trace(go.Scatter(
        x=df_resumen.index,
        y=df_resumen["media"],
        mode="lines",
        name="Media proyectada",
        line=dict(width=2),
        hovertemplate="Fecha: %{x}<br>Media: %{y:.4f}<extra></extra>",
    ))

    # Línea horizontal VaR final
    fig.add_trace(go.Scatter(
        x=[df_resumen.index[0], df_resumen.index[-1]],
        y=[var_final, var_final],
        mode="lines",
        name=f"VaR {int(alpha * 100)}% al vencimiento",
        line=dict(dash="dash"),
        hovertemplate=f"VaR {int(alpha * 100)}%: "+"%{y:.4f}<extra></extra>",
    ))

    fig.update_layout(
        title="Escenarios de proyección del TC SUNAT y nivel de VaR al vencimiento",
        xaxis_title="Fecha",
        yaxis_title="Tipo de cambio (S/ por US$)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=40, r=20, t=60, b=40),
    )

    st.plotly_chart(fig, use_container_width=True)


# ------------------------------------------------------------
# Backtesting (si la fecha final está en el pasado)
# ------------------------------------------------------------

def plot_backtesting(df_sunat_full: pd.DataFrame,
                     fecha_inicio: date,
                     fecha_fin: date,
                     tc_proj: float):
    """
    Muestra el histórico del TC SUNAT entre fecha_inicio y fecha_fin
    y una línea horizontal con el TC proyectado (media).
    """
    mask = (
        (df_sunat_full.index.date >= fecha_inicio) &
        (df_sunat_full.index.date <= fecha_fin)
    )
    df_bt = df_sunat_full.loc[mask]

    if df_bt.empty:
        st.info("No hay datos históricos en el rango seleccionado para backtesting.")
        return

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_bt.index,
        y=df_bt["tc_sunat"],
        mode="lines",
        name="TC SUNAT histórico",
        hovertemplate="Fecha: %{x}<br>TC: %{y:.4f}<extra></extra>",
    ))

    fig.add_trace(go.Scatter(
        x=[df_bt.index.min(), df_bt.index.max()],
        y=[tc_proj, tc_proj],
        mode="lines",
        name="TC proyectado (media)",
        line=dict(dash="dash"),
        hovertemplate="TC proyectado: %{y:.4f}<extra></extra>",
    ))

    fig.update_layout(
        title="Backtesting simple del modelo vs TC SUNAT observado",
        xaxis_title="Fecha",
        yaxis_title="Tipo de cambio (S/ por US$)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=40, r=20, t=60, b=40),
    )

    st.plotly_chart(fig, use_container_width=True)


# ------------------------------------------------------------
# App principal
# ------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="Proyección TC SUNAT USD/PEN (AR-GARCH)",
        layout="wide",
    )

    st.title("Proyección del Tipo de Cambio SUNAT (USD/PEN)")
    st.caption(
        "Modelo de simulación Monte Carlo basado en AR(1)-GARCH(1,1) "
        "con datos del BCRP/SBS y cálculo de VaR/CVaR sobre el tipo de cambio."
    )

    hoy = date.today()

    # Sidebar: configuración de horizonte
    st.sidebar.header("Configuración de horizonte")

    fecha_inicio = st.sidebar.date_input(
        "Fecha de inicio",
        value=hoy,
    )

    opcion_horizonte = st.sidebar.radio(
        "¿Cómo quieres definir el horizonte?",
        ["Plazo en días calendario", "Fecha final"],
        index=0,
    )

    if opcion_horizonte == "Plazo en días calendario":
        plazo_dias_cal = st.sidebar.number_input(
            "Plazo (días calendario)",
            min_value=1,
            max_value=365 * 3,
            value=60,
            step=1,
        )
        plazo_dias_cal = int(plazo_dias_cal)
        fecha_final_cal = fecha_inicio + timedelta(days=plazo_dias_cal)
    else:
        fecha_final_cal = st.sidebar.date_input(
            "Fecha final",
            value=hoy + timedelta(days=60),
        )
        plazo_dias_cal = (fecha_final_cal - fecha_inicio).days

    # Cálculo de días hábiles (plazo limpio)
    plazo_dias_habiles = contar_dias_habiles(fecha_inicio, fecha_final_cal)

    st.sidebar.markdown("---")

    # Parámetros de simulación
    st.sidebar.header("Parámetros de simulación")

    n_sims = st.sidebar.number_input(
        "Número de simulaciones",
        min_value=500,
        max_value=50000,
        value=10000,
        step=500,
    )

    alpha_var = st.sidebar.slider(
        "Nivel de confianza para VaR",
        min_value=0.90,
        max_value=0.99,
        value=0.95,
        step=0.01,
    )

    st.sidebar.markdown("---")
    st.sidebar.caption(
        f"Días hábiles (plazo limpio) entre {fecha_inicio} y {fecha_final_cal}: "
        f"**{plazo_dias_habiles}**"
    )
    st.sidebar.caption(
        "Los feriados usados están definidos en el módulo `tc_sunat_model`. "
        "Actualiza la tabla `FERIADOS_PE` con la lista real."
    )

    if st.button("Simular proyecciones"):
        if plazo_dias_habiles <= 0:
            st.error(
                "El horizonte debe tener al menos 1 día hábil. "
                "Revisa la fecha de inicio y la fecha final/plazo."
            )
            return

        # 1) Cargar datos BCRP y construir TC SUNAT
        try:
            df_tc_sbs = obtener_dataframe_bcrp()
        except Exception as e:
            st.error(f"Error al descargar datos del BCRP: {e}")
            return

        df_sunat_full, df_sunat_habiles = construir_tc_sunat(df_tc_sbs)

        # Histórico hasta la fecha de inicio (solo días hábiles reales)
        mask_hist = df_sunat_habiles.index.date <= fecha_inicio
        df_hist = df_sunat_habiles.loc[mask_hist]

        if len(df_hist) < 120:
            st.error(
                "No hay suficientes datos históricos antes de la fecha de inicio "
                "(se recomiendan al menos ~120 días hábiles para ajustar AR-GARCH)."
            )
            return

        # 2) Estimar retornos y ajustar AR-GARCH
        retornos_log = calcular_retornos_log(df_hist)

        try:
            res_garch = ajustar_garch(retornos_log)
        except Exception as e:
            st.error(f"Error al ajustar el modelo AR-GARCH: {e}")
            return

        # 3) Generar fechas hábiles futuras (plazo limpio)
        fechas_future = generar_fechas_habiles(fecha_inicio, plazo_dias_habiles)

        # S0: TC SUNAT del último día hábil <= fecha_inicio
        S0 = float(df_hist["tc_sunat"].iloc[-1])

        # 4) Simular trayectorias
        n_sims_int = int(n_sims)
        try:
            paths = simular_arma_garch(
                res=res_garch,
                retornos_log=retornos_log,
                S0=S0,
                n_steps=plazo_dias_habiles,
                n_sims=n_sims_int,
            )
        except Exception as e:
            st.error(f"Error al simular trayectorias AR-GARCH: {e}")
            return

        # 5) Resumen de trayectorias
        df_resumen = resumen_paths(paths, fechas_future)
        fecha_final_efectiva = df_resumen.index[-1]

        # VaR y CVaR al vencimiento
        tc_final = paths[:, -1]
        var_final, cvar_final = calcular_var_cvar(tc_final, alpha=alpha_var)

        # Resultados numéricos
        st.subheader("Resumen numérico de la proyección")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "TC SUNAT actual (aprox.)",
                f"{S0:.4f}",
            )
        with col2:
            st.metric(
                "TC esperado al vencimiento (media)",
                f"{df_resumen['media'].iloc[-1]:.4f}",
            )
        with col3:
            st.metric(
                f"VaR {int(alpha_var * 100)}% al vencimiento",
                f"{var_final:.4f}",
            )
        with col4:
            st.metric(
                f"CVaR {int(alpha_var * 100)}% al vencimiento",
                f"{cvar_final:.4f}",
            )

        st.write(
            f"- Fecha final **calendario** solicitada: **{fecha_final_cal}**  \n"
            f"- Fecha final **hábil efectiva** (último paso de la simulación): "
            f"**{fecha_final_efectiva.date()}**"
        )

        # 6) Gráfico 1: histórico + simulaciones
        st.subheader("Histórico del TC SUNAT + simulaciones hasta la fecha final")
        df_hist_plot = recortar_historico(
            df_sunat_habiles,
            fecha_inicio=fecha_inicio,
            n_steps_habiles=plazo_dias_habiles,
        )
        plot_hist_y_sim_plotly(df_hist_plot, fechas_future, paths)

        # 7) Gráfico 2: escenarios + VaR
        st.subheader("Escenarios de proyección y nivel de VaR al vencimiento")
        plot_escenarios_y_var(df_resumen, var_final=var_final, alpha=alpha_var)

        # 8) Backtesting (si la fecha final está en el pasado)
        hoy_actual = date.today()
        if fecha_final_cal < hoy_actual:
            st.subheader("Backtesting simple del modelo")
            tc_proj = df_resumen["media"].iloc[-1]
            plot_backtesting(df_sunat_full, fecha_inicio, fecha_final_cal, tc_proj)

        # 9) Nota metodológica corta
        st.subheader("Nota metodológica (resumen)")
        st.info(
            "El modelo utiliza el tipo de cambio SUNAT limpio (días hábiles reales), "
            "calcula retornos logarítmicos y ajusta un AR(1)-GARCH(1,1) con distribución "
            "normal. A partir de ese modelo se simulan trayectorias de Monte Carlo para "
            "el TC hasta la fecha objetivo, y se calcula el VaR/CVaR sobre la distribución "
            "simulada del tipo de cambio al vencimiento."
        )


if __name__ == "__main__":
    main()
