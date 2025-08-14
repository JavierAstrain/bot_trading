import os
import tempfile
from datetime import datetime

import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go

# IA
import google.generativeai as genai

# ====== intentar pandas_ta; si no, usar fallback propio ======
HAS_PANDAS_TA = True
try:
    import pandas_ta as ta  # pip install pandas-ta
except Exception:
    HAS_PANDAS_TA = False


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
@st.cache_data(show_spinner=False)
def yf_download(symbol: str, period: str, interval: str) -> pd.DataFrame:
    return yf.download(symbol, period=period, interval=interval, auto_adjust=True, progress=False)

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
    for col in ["EMA20", "EMA50", "EMA200"]:
        if col in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df[col], mode="lines", name=col))
    fig.update_layout(title=title, xaxis_rangeslider_visible=False, height=520)
    st.plotly_chart(fig, use_container_width=True)


# ---- Indicadores de fallback (si no hay pandas_ta) ----
def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()

def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1/length, adjust=False).mean()
    roll_down = down.ewm(alpha=1/length, adjust=False).mean()
    rs = roll_up / roll_down
    return 100 - (100 / (1 + rs))

def macd(series: pd.Series, fast=12, slow=26, signal=9):
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if HAS_PANDAS_TA:
        df["EMA20"] = ta.ema(df["Close"], length=20)
        df["EMA50"] = ta.ema(df["Close"], length=50)
        df["EMA200"] = ta.ema(df["Close"], length=200)
        df["RSI14"] = ta.rsi(df["Close"], length=14)
        macd_df = ta.macd(df["Close"], fast=12, slow=26, signal=9)
        if macd_df is not None and not macd_df.empty:
            df["MACD"] = macd_df["MACD_12_26_9"]
            df["MACD_signal"] = macd_df["MACDs_12_26_9"]
            df["MACD_hist"] = macd_df["MACDh_12_26_9"]
    else:
        df["EMA20"] = ema(df["Close"], 20)
        df["EMA50"] = ema(df["Close"], 50)
        df["EMA200"] = ema(df["Close"], 200)
        df["RSI14"] = rsi(df["Close"], 14)
        macd_line, signal_line, hist = macd(df["Close"])
        df["MACD"] = macd_line
        df["MACD_signal"] = signal_line
        df["MACD_hist"] = hist
    return df

def llm_text(prompt: str) -> str:
    model = genai.GenerativeModel("gemini-1.5-flash")
    resp = model.generate_content(prompt)
    return getattr(resp, "text", "").strip()

def llm_image(prompt: str, image_path: str) -> str:
    uploaded = genai.upload_file(image_path)
    model = genai.GenerativeModel("gemini-1.5-flash")
    resp = model.generate_content([prompt, uploaded])
    return getattr(resp, "text", "").strip()

def build_trading_prompt_from_df(symbol: str, df: pd.DataFrame, period: str, interval: str) -> str:
    subset_cols = [c for c in ["Open","High","Low","Close","EMA20","EMA50","EMA200","RSI14"] if c in df.columns]
    tail_txt = df.tail(60)[subset_cols].round(4).to_string()
    last_close = float(df["Close"].iloc[-1])
    last_rsi = df["RSI14"].iloc[-1] if "RSI14" in df.columns else None
    last_rsi = float(last_rsi) if pd.notnull(last_rsi) else "NA"

    prompt = f"""
Actúa como un asesor de trading experto para CFDs.
Instrumento: {symbol}
Periodo solicitado: {period}, Intervalo: {interval}
Último cierre: {last_close}
RSI(14) último: {last_rsi}

Datos recientes (últimas ~60 velas con EMAs y RSI):
{tail_txt}

Objetivo:
1) Diagnóstico técnico (tendencia y contexto: cruces EMA, RSI, MACD).
2) Niveles relevantes (S/R) y confluencias.
3) Estrategia sugerida (long/short) con: entrada, Stop Loss, Take Profit y ratio R:R.
4) Escenarios probables y gestión del riesgo (apalancamiento razonable).
5) Si no hay setup A+, sugiere esperar y qué confirmaciones mirar.

No inventes precios inexistentes. Usa lenguaje claro y profesional.
"""
    return prompt


# =========================
# INTERFAZ
# =========================
st.sidebar.header("Opciones")
mode = st.sidebar.radio("Selecciona modo", ["📷 Analizar imagen de gráfico", "📊 Analizar activo por símbolo (histórico)"])
st.sidebar.markdown("---")
st.sidebar.markdown("**Ejemplos:** `BTC-USD`, `AAPL`, `TSLA`, `GOLD`, `EURUSD=X`, `^GSPC`")


# =========================
# MODO: IMAGEN
# =========================
if mode == "📷 Analizar imagen de gráfico":
    file = st.file_uploader("Sube una imagen de tu gráfico (TradingView, MetaTrader, broker, etc.)", type=["png", "jpg", "jpeg", "webp"])
    if file is not None:
        st.image(file, caption="Gráfico subido", use_container_width=True)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            tmp.write(file.getbuffer())
            image_path = tmp.name

        with st.spinner("Analizando gráfico con IA..."):
            prompt_img = (
                "Eres un asesor de trading experto. Analiza el gráfico subido y responde conciso:\n"
                "1) Tendencia (alcista/bajista/lateral) y justificación.\n"
                "2) Patrones (canales, triángulos, HCH, doble techo/suelo) y zonas S/R.\n"
                "3) Lectura de indicadores visibles (RSI, MACD, EMAs/Bollinger) si aparecen.\n"
                "4) Estrategia CFD (long/short): condiciones de entrada, SL, TP, ratio R:R.\n"
                "5) Gestión de riesgo y temporalidad más apropiada.\n"
                "Si faltan datos, pide confirmaciones específicas."
            )
            try:
                analysis = llm_image(prompt_img, image_path)
                st.subheader("📈 Análisis Técnico del Gráfico (IA)")
                st.write(analysis or "No se obtuvo texto del modelo.")
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
                df = yf_download(symbol, period, interval)
            except Exception as e:
                df = pd.DataFrame()
                st.error(f"Error al descargar datos: {e}")

        if df.empty or len(df) < 20:
            st.warning("No se encontraron datos suficientes para ese símbolo/periodo/intervalo.")
        else:
            df = compute_indicators(df)

            st.subheader(f"📊 Histórico de {symbol}")
            plot_candles(df, f"{symbol} — {period} @ {interval}")

            last = df.iloc[-1]
            st.markdown("### 🔎 Lectura técnica rápida")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Cierre", f"{last['Close']:.4f}")
            rsi_val = last['RSI14'] if 'RSI14' in df.columns and pd.notnull(last['RSI14']) else None
            c2.metric("RSI(14)", f"{rsi_val:.2f}" if rsi_val is not None else "NA")
            c3.metric("EMA20 vs EMA50", "▲" if last.get("EMA20", 0) > last.get("EMA50", 0) else "▼")
            c4.metric("EMA50 vs EMA200", "▲" if last.get("EMA50", 0) > last.get("EMA200", 0) else "▼")

            with st.spinner("Generando recomendación del asesor..."):
                try:
                    prompt = build_trading_prompt_from_df(symbol, df, period, interval)
                    advice = llm_text(prompt)
                    st.subheader("🧠 Recomendación del Asesor (IA)")
                    st.write(advice or "No se obtuvo texto del modelo.")
                except Exception as e:
                    st.error(f"Ocurrió un error al generar la recomendación: {e}")

            # Noticias referenciales (si yfinance las provee)
            try:
                tk = yf.Ticker(symbol)
                news = getattr(tk, "news", [])
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
