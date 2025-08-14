import os
import tempfile
from datetime import datetime

import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go

# IA
import google.generativeai as genai
import pandas_ta as ta


# =========================
# CONFIGURACIÓN INICIAL
# =========================
st.set_page_config(page_title="Asesor de Trading Experto IA", layout="wide")

API_KEY = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY", ""))

if not API_KEY:
    st.error("⚠️ Falta la GEMINI_API_KEY en .streamlit/secrets.toml o variable de entorno.")
else:
    genai.configure(api_key=API_KEY)

st.title("🤖📈 Asesor de Trading Experto IA")
st.caption("Análisis técnico desde imágenes, históricos automáticos (Yahoo Finance) y recomendaciones estilo asesor profesional (CFDs).")


# =========================
# UTILIDADES
# =========================
def plot_candles(df: pd.DataFrame, title: str):
    fig = go.Figure(data=[
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="OHLC"
        )
    ])
    # EMAs si existen
    for col in ["EMA20", "EMA50", "EMA200"]:
        if col in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df[col], mode="lines", name=col))
    fig.update_layout(title=title, xaxis_rangeslider_visible=False, height=520)
    st.plotly_chart(fig, use_container_width=True)


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    # EMAs comunes
    df["EMA20"] = ta.ema(df["Close"], length=20)
    df["EMA50"] = ta.ema(df["Close"], length=50)
    df["EMA200"] = ta.ema(df["Close"], length=200)
    # RSI
    df["RSI14"] = ta.rsi(df["Close"], length=14)
    # MACD
    macd = ta.macd(df["Close"], fast=12, slow=26, signal=9)
    if macd is not None and not macd.empty:
        df["MACD"] = macd["MACD_12_26_9"]
        df["MACD_signal"] = macd["MACDs_12_26_9"]
        df["MACD_hist"] = macd["MACDh_12_26_9"]
    return df


def llm_text(prompt: str) -> str:
    model = genai.GenerativeModel("gemini-1.5-flash")
    resp = model.generate_content(prompt)
    return getattr(resp, "text", "").strip()


def llm_image(prompt: str, image_path: str) -> str:
    """Sube imagen a Gemini y genera análisis."""
    # Subir archivo a Gemini (la lib necesita un path/bytes, no el objeto de Streamlit)
    uploaded = genai.upload_file(image_path)
    model = genai.GenerativeModel("gemini-1.5-flash")
    resp = model.generate_content([prompt, uploaded])
    return getattr(resp, "text", "").strip()


def build_trading_prompt_from_df(symbol: str, df: pd.DataFrame, period: str, interval: str) -> str:
    tail_txt = df.tail(60)[["Open","High","Low","Close","EMA20","EMA50","EMA200","RSI14"]].round(4).to_string()
    last_close = float(df["Close"].iloc[-1])
    last_rsi = float(df["RSI14"].iloc[-1]) if pd.notnull(df["RSI14"].iloc[-1]) else "NA"

    prompt = f"""
Actúa como un asesor de trading experto para CFDs.
Instrumento: {symbol}
Periodo solicitado: {period}, Intervalo: {interval}
Último cierre: {last_close}
RSI(14) último: {last_rsi}

Datos recientes (últimas ~60 velas con EMAs y RSI):
{tail_txt}

Objetivo:
1) Diagnóstico técnico (tendencia: alcista/bajista/lateral) y contexto (cruces EMA, RSI, MACD si aplica).
2) Zonas/ niveles relevantes (S/R) y confluencias.
3) Estrategia sugerida (long/short), con:
   - Entrada sugerida (o condiciones de confirmación)
   - Stop Loss y Take Profit (con ratio R:R)
   - Tipo de operativa por temporalidad (scalp/intradía/swing)
4) Escenarios futuros probables y gestión del riesgo (apalancamiento razonable).
5) Si no hay setup A+, recomienda esperar y qué confirmaciones mirar.

No inventes precios inexistentes. Usa lenguaje claro y profesional.
"""
    return prompt


# =========================
# INTERFAZ
# =========================
st.sidebar.header("Opciones")
mode = st.sidebar.radio("Selecciona modo", ["📷 Analizar imagen de gráfico", "📊 Analizar activo por símbolo (histórico)"])

st.sidebar.markdown("---")
st.sidebar.markdown("**Sugerencias de símbolo:** `BTC-USD`, `AAPL`, `TSLA`, `GOLD`, `EURUSD=X`, `^GSPC` (S&P500)")


# =========================
# MODO: IMAGEN
# =========================
if mode == "📷 Analizar imagen de gráfico":
    file = st.file_uploader("Sube una imagen de tu gráfico (TradingView, MetaTrader, broker, etc.)", type=["png", "jpg", "jpeg", "webp"])
    if file is not None:
        st.image(file, caption="Gráfico subido", use_container_width=True)

        # Guardar a archivo temporal (necesario para google-generativeai)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            tmp.write(file.getbuffer())
            image_path = tmp.name

        with st.spinner("Analizando gráfico con IA..."):
            prompt_img = (
                "Eres un asesor de trading experto. Analiza el gráfico subido y responde conciso:\n"
                "1) Tendencia principal (alcista/bajista/lateral) y por qué.\n"
                "2) Patrones técnicos visibles (canales, triángulos, HCH, doble techo/suelo) y zonas S/R.\n"
                "3) Lectura de indicadores visibles (RSI, MACD, EMAs/Bollinger) si aparecen.\n"
                "4) Estrategia en CFDs (long/short): entrada sugerida o condiciones, SL, TP, ratio R:R.\n"
                "5) Gestión del riesgo y temporalidad recomendada.\n"
                "Evita generalidades, entrega niveles aproximados si son legibles. Si faltan datos, pide confirmaciones."
            )
            try:
                analysis = llm_image(prompt_img, image_path)
                st.subheader("📈 Análisis Técnico del Gráfico (IA)")
                st.write(analysis if analysis else "No se obtuvo texto del modelo.")
            except Exception as e:
                st.error(f"Error analizando imagen: {e}")


# =========================
# MODO: SÍMBOLO
# =========================
if mode == "📊 Analizar activo por símbolo (histórico)":
    col1, col2, col3 = st.columns([1.2, 1, 1])
    with col1:
        symbol = st.text_input("Símbolo / Ticker", value="BTC-USD", help="Ej: AAPL, TSLA, BTC-USD, EURUSD=X, GOLD")
    with col2:
        period = st.selectbox("Periodo", ["7d", "1mo", "3mo", "6mo", "1y", "2y"], index=2)
    with col3:
        interval = st.selectbox("Intervalo", ["15m", "30m", "1h", "4h", "1d"], index=2)

    if st.button("🔎 Analizar activo"):
        with st.spinner("Descargando histórico..."):
            try:
                df = yf.download(symbol, period=period, interval=interval, auto_adjust=True, progress=False)
            except Exception as e:
                df = pd.DataFrame()
                st.error(f"Error al descargar datos: {e}")

        if df.empty or len(df) < 20:
            st.warning("No se encontraron datos suficientes para ese símbolo/periodo/intervalo.")
        else:
            df = compute_indicators(df)

            st.subheader(f"📊 Histórico de {symbol}")
            plot_candles(df, f"{symbol} — {period} @ {interval}")

            # Panel técnico rápido
            last = df.iloc[-1]
            st.markdown("### 🔎 Lectura técnica rápida")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Cierre", f"{last['Close']:.4f}")
            c2.metric("RSI(14)", f"{last['RSI14']:.2f}" if pd.notnull(last['RSI14']) else "NA")
            c3.metric("EMA20 vs EMA50", "▲" if last["EMA20"] > last["EMA50"] else "▼")
            c4.metric("EMA50 vs EMA200", "▲" if last["EMA50"] > last["EMA200"] else "▼")

            # Recomendación con IA
            with st.spinner("Generando recomendación del asesor..."):
                try:
                    prompt = build_trading_prompt_from_df(symbol, df, period, interval)
                    advice = llm_text(prompt)
                    st.subheader("🧠 Recomendación del Asesor (IA)")
                    st.write(advice if advice else "No se obtuvo texto del modelo.")
                except Exception as e:
                    st.error(f"Ocurrió un error al generar la recomendación: {e}")

            # (Opcional) Noticias básicas con yfinance (si el activo las ofrece)
            try:
                tk = yf.Ticker(symbol)
                news = tk.news if hasattr(tk, "news") else []
                if news:
                    st.markdown("### 📰 Últimos titulares (referenciales)")
                    for n in news[:5]:
                        st.markdown(f"- {n.get('title','(sin título)')}")
            except Exception:
                pass


# =========================
# FOOTER
# =========================
st.markdown("---")
st.caption(
    "Este asistente entrega opiniones educativas basadas en datos e IA. No constituye asesoría financiera. "
    "Opera con gestión de riesgo adecuada."
)
