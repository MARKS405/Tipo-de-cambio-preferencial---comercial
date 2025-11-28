
import streamlit as st
import numpy as np
import pandas as pd
from datetime import date, timedelta
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
    var_cvar_retorno,
)


def plot_hist_y_sim(df_hist: pd.DataFrame, fechas_future, paths: np.ndarray):
    """Gráfico interactivo del histórico reciente + trayectorias simuladas."""

    # Histórico: tomamos 1 año hacia atrás desde la fecha de inicio
    fecha_inicio = fechas_future[0].date()
    ventana_dias = 365
    fecha_min = fecha_inicio - timedelta(days=ventana_dias)
    df_recent = df_hist[df_hist.index.date >= fecha_min]

    fig = go.Figure()

    # Serie histórica
    fig.add_trace(
        go.Scatter(
            x=df_recent.index,
            y=df_recent["tc_sunat"],
            mode="lines",
            name="TC SUNAT histórico (reciente)",
            line=dict(width=2),
        )
    )

    # Trayectorias simuladas (limitamos para que no sea visualmente pesado)
    n_sims = paths.shape[0]
    n_mostrar = min(200, n_sims)
    idx = np.random.choice(n_sims, n_mostrar, replace=False)

    fechas_all = [df_hist.index[-1]] + list(pd.to_datetime(fechas_future))

    for i in idx:
        fig.add_trace(
            go.Scatter(
                x=fechas_all,
                y=paths[i, :],
                mode="lines",
                line=dict(width=1),
                opacity=0.25,
                showlegend=False,
            )
        )

    fig.update_layout(
        title="Histórico reciente del TC SUNAT + trayectorias simuladas",
        xaxis_title="Fecha",
        yaxis_title="Tipo de cambio (S/ por US$)",
        hovermode="x unified",
        template="plotly_dark",
        height=450,
    )

    return fig


def plot_escenarios_y_var(df_resumen: pd.DataFrame, tc_var: float):
    """Gráfico de la media esperada, banda P05-P95 y nivel de VaR en TC."""
    x = df_resumen.index

    fig = go.Figure()

    # Banda 5%-95%
    fig.add_trace(
        go.Scatter(
            x=list(x) + list(x[::-1]),
            y=list(df_resumen["p95"]) + list(df_resumen["p05"][::-1]),
            fill="toself",
            fillcolor="rgba(0, 123, 255, 0.2)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            showlegend=True,
            name="Banda P5–P95",
        )
    )

    # Media
    fig.add_trace(
        go.Scatter(
            x=x,
            y=df_resumen["media"],
            mode="lines",
            name="Media proyectada",
            line=dict(width=2, color="#1f77b4"),
        )
    )

    # VaR en TC como línea horizontal
    fig.add_trace(
        go.Scatter(
            x=[x[0], x[-1]],
            y=[tc_var, tc_var],
            mode="lines",
            name="TC preferencial (VaR)",
            line=dict(width=2, dash="dash", color="#ff7f0e"),
        )
    )

    fig.update_layout(
        title="Escenario central, banda de incertidumbre y TC preferencial (VaR)",
        xaxis_title="Fecha",
        yaxis_title="Tipo de cambio (S/ por US$)",
        hovermode="x unified",
        template="plotly_dark",
        height=400,
    )

    return fig


def plot_backtesting(df_sunat_full: pd.DataFrame, fecha_inicio: date, fecha_final: date, tc_proj: float):
    """Backtesting simple: histórico observado vs nivel proyectado (media)."""
    mask = (df_sunat_full.index.date >= fecha_inicio) & (df_sunat_full.index.date <= fecha_final)
    df_bt = df_sunat_full.loc[mask]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df_bt.index,
            y=df_bt["tc_sunat"],
            mode="lines",
            name="TC SUNAT histórico",
            line=dict(width=2),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[df_bt.index[0], df_bt.index[-1]],
            y=[tc_proj, tc_proj],
            mode="lines",
            name="TC proyectado (media)",
            line=dict(width=2, dash="dash"),
        )
    )

    fig.update_layout(
        title="Backtesting simple del modelo vs TC SUNAT observado",
        xaxis_title="Fecha",
        yaxis_title="Tipo de cambio (S/ por US$)",
        hovermode="x unified",
        template="plotly_dark",
        height=400,
    )

    return fig


def main():
    st.set_page_config(page_title="Proyección TC SUNAT USD/PEN", layout="wide")

    st.title("Proyección del Tipo de Cambio SUNAT (USD/PEN)")
    st.caption(
        "Modelo de simulación Monte Carlo basado en un GARCH(1,1) sobre retornos logarítmicos "
        "del TC SUNAT (datos BCRP/SBS) y cálculo de VaR/CVaR sobre el tipo de cambio al vencimiento."
    )

    hoy = date.today()

    # ------------------------------------------------------------------
    # Sidebar – configuración de horizonte
    # ------------------------------------------------------------------
    st.sidebar.header("Configuración de horizonte")

    fecha_inicio = st.sidebar.date_input("Fecha de inicio", value=hoy)

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

    plazo_dias_habiles = contar_dias_habiles(fecha_inicio, fecha_final_cal)

    # ------------------------------------------------------------------
    # Sidebar – parámetros de simulación
    # ------------------------------------------------------------------
    st.sidebar.header("Parámetros de simulación")

    n_sims = st.sidebar.number_input(
        "Número de simulaciones",
        min_value=500,
        max_value=50000,
        value=10000,
        step=500,
    )
    n_sims = int(n_sims)

    nivel_conf = st.sidebar.slider(
        "Nivel de confianza para VaR",
        min_value=0.80,
        max_value=0.99,
        value=0.95,
        step=0.01,
    )

    st.sidebar.markdown("---")
    st.sidebar.caption(
        f"Días hábiles (plazo limpio) entre {fecha_inicio} y {fecha_final_cal}: **{plazo_dias_habiles}**"
    )
    st.sidebar.caption(
        "Los feriados usados están definidos en el módulo `tc_sunat_model`. "
        "Actualiza la tabla `FERIADOS_PE` con la lista real."
    )

    # ------------------------------------------------------------------
    # Botón principal
    # ------------------------------------------------------------------
    if st.button("Simular proyecciones"):
        if plazo_dias_habiles <= 0:
            st.error(
                "El horizonte debe tener al menos 1 día hábil. "
                "Revisa la fecha de inicio y la fecha final/plazo."
            )
            return

        # 1) Cargar datos y construir TC SUNAT
        try:
            df_tc_sbs = obtener_dataframe_bcrp()
            df_sunat_full, df_sunat_habiles = construir_tc_sunat(df_tc_sbs)
        except Exception as e:
            st.error(f"Error al descargar o construir la serie TC SUNAT: {e}")
            return

        # Histórico sólo hasta la fecha de inicio
        mask_hist = df_sunat_habiles.index.date <= fecha_inicio
        df_hist = df_sunat_habiles.loc[mask_hist]

        if len(df_hist) < 250:
            st.error(
                "No hay suficientes datos históricos antes de la fecha de inicio "
                "(se recomienda al menos ~250 días hábiles)."
            )
            return

        # 2) Retornos log + GARCH(1,1)
        try:
            retornos_log = calcular_retornos_log(df_hist)
            res_garch = ajustar_garch(retornos_log)
        except Exception as e:
            st.error(f"Error al ajustar el modelo GARCH: {e}")
            return

        # 3) Generar fechas futuras y simular trayectorias
        fechas_future = generar_fechas_habiles(fecha_inicio, plazo_dias_habiles)
        n_steps = len(fechas_future)

        S0 = float(df_hist["tc_sunat"].iloc[-1])

        try:
            paths = simular_arma_garch(res_garch, S0=S0, n_steps=n_steps, n_sims=n_sims)
        except Exception as e:
            st.error(f"Error al simular trayectorias GARCH: {e}")
            return

        # 4) Resumen numérico y VaR/CVaR
        df_resumen = resumen_paths(paths, fechas_future)
        S_T = paths[:, -1]

        var_ret, cvar_ret, tc_var, tc_cvar = var_cvar_retorno(S_T, S0, alpha=nivel_conf)

        fecha_final_efectiva = fechas_future[-1].date()

        # ------------------------------------------------------------------
        # Layout principal
        # ------------------------------------------------------------------
        st.subheader("Resumen numérico de la proyección")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("TC SUNAT actual (aprox.)", f"{S0:.4f}")
        with col2:
            st.metric("TC esperado al vencimiento (media)", f"{df_resumen['media'].iloc[-1]:.4f}")
        with col3:
            st.metric(f"TC VaR {int(nivel_conf*100)}% al vencimiento", f"{tc_var:.4f}")
        with col4:
            st.metric(f"TC CVaR {int(nivel_conf*100)}% al vencimiento", f"{tc_cvar:.4f}")

        st.write(
            f"- Fecha final calendario solicitada: **{fecha_final_cal}**  
"
            f"- Fecha final hábil efectiva (último paso de la simulación): **{fecha_final_efectiva}**  
"
            f"- VaR sobre retorno total: **{var_ret*100:.2f}%**  
"
            f"- CVaR sobre retorno total: **{cvar_ret*100:.2f}%**"
        )

        # Gráfico 1: histórico + simulaciones
        st.subheader("Histórico del TC SUNAT + simulaciones hasta la fecha final")
        fig_hist = plot_hist_y_sim(df_hist, fechas_future, paths)
        st.plotly_chart(fig_hist, use_container_width=True)

        # Gráfico 2: media, banda y VaR
        st.subheader("Escenarios de proyección y nivel de VaR al vencimiento")
        fig_var = plot_escenarios_y_var(df_resumen, tc_var)
        st.plotly_chart(fig_var, use_container_width=True)

        # Backtesting si aplica
        hoy_actual = date.today()
        if fecha_final_cal < hoy_actual:
            st.subheader("Backtesting simple del modelo")
            tc_proj_media = df_resumen["media"].iloc[-1]
            fig_bt = plot_backtesting(df_sunat_full, fecha_inicio, fecha_final_cal, tc_proj_media)
            st.plotly_chart(fig_bt, use_container_width=True)

        # Nota metodológica
        st.subheader("Nota metodológica (resumen)")
        st.info(
            "El modelo utiliza el tipo de cambio SUNAT limpio (días hábiles reales), "
            "calcula retornos logarítmicos y ajusta un GARCH(1,1) con distribución normal sobre "
            "retornos diarios en porcentaje. A partir de este modelo se simulan trayectorias de "
            "Monte Carlo para el tipo de cambio hasta la fecha objetivo, y se calcula el VaR/CVaR "
            "sobre el retorno total al vencimiento para derivar un TC preferencial."
        )


if __name__ == "__main__":
    main()
