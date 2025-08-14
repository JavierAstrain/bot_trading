import os
import tempfile
import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import google.generativeai as genai

# ====== intentar pandas_ta; si falla, usar fallback propio ======
HAS_PANDAS_TA = True
try:
    import pandas_ta as ta  # pip install pandas-ta (paquete con guion)
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
st.caption("Análisis técnico desde imágenes, históricos automáticos (Yahoo Finance), fundamentales, gestión de riesgo y Q&A estilo asesor profesional (CFDs).")

# --- Estado para Q&A ---
if "last_symbol" not in st.session_state: st.session_state.last_symbol = None
if "last_df" not in st.session_state: st.session_state.last_df = None
if "last_period" not in st.session_state: st.session_state.last_period = None
if "last_interval" not in st.session_state: st.session_state.last_interval = None
if "last_image_analysis" not in st.session_state: st.session_state.last_image_analysis = None
if "chat" not in st.session_state: st.session_state.chat = []


# =========================
# UTILIDADES GENERALES
# =========================
def fmt_num(v, ndigits=4):
    try:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return "NA"
        if hasattr(v, "__len__") and not isinstance(v, (str, bytes)):
            if len(v) > 0:
                v = v[0]
        return f"{float(v):.{ndigits}f}"
    except Exception:
        return str(v)

@st.cache_data(show_spinner=False)
def yf_download(symbol: str, period: str, interval: str) -> pd.DataFrame:
    df = yf.download(symbol, period=period, interval=interval, auto_adjust=True, progress=False)
    # Fuerza numéricos
    for c in [col for col in ["Open", "High", "Low", "Close"] if c in df.columns]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def plot_candles(df: pd.DataFrame, title: str):
    fig = go.Figure(data=[
        go.Candlestick(
            x=df.index, open=df["Open"], high=df["High"],
            low=df["Low"], close=df["Close"], name="OHLC"
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
    if df.empty:
        return df
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

def atr(df: pd.DataFrame, length: int = 14):
    high_low = (df["High"] - df["Low"]).abs()
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close  = (df["Low"]  - df["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.ewm(alpha=1/length, adjust=False).mean()

def position_size(balance: float, risk_pct: float, entry: float, stop: float, contract_value: float = 1.0):
    risk_amount = balance * (risk_pct / 100.0)
    stop_distance = abs(entry - stop)
    if stop_distance <= 0:
        return 0
    units = risk_amount / (stop_distance * contract_value)
    return max(0, units)

@st.cache_data(show_spinner=False, ttl=3600)
def fundamentals_and_news(symbol: str):
    tk = yf.Ticker(symbol)
    info = {}
    try:
        fin = getattr(tk, "fast_info", {})
        info["market_cap"] = getattr(fin, "market_cap", None) or getattr(getattr(tk, "info", {}), "marketCap", None)
        info["pe"] = getattr(fin, "trailing_pe", None) or getattr(getattr(tk, "info", {}), "trailingPE", None)
        info["div_yield"] = getattr(getattr(tk, "info", {}), "dividendYield", None)
        info["currency"] = getattr(fin, "currency", None) or getattr(getattr(tk, "info", {}), "currency", None)
    except Exception:
        pass
    news = []
    try:
        news = tk.news if hasattr(tk, "news") else []
    except Exception:
        pass
    return info, news

# ===== LLM helpers =====
def llm_text(prompt: str) -> str:
    model = genai.GenerativeModel("gemini-1.5-flash")
    resp = model.generate_content(prompt)
    return getattr(resp, "text", "").strip()

def llm_image(prompt: str, image_path: str) -> str:
    uploaded = genai.upload_file(image_path)
    model = genai.GenerativeModel("gemini-1.5-flash")
    resp = model.generate_content([prompt, uploaded])
    return getattr(resp, "text", "").strip()

def df_compact_summary(df: pd.DataFrame, rows: int = 60) -> str:
    if df is None or df.empty:
        return "SIN_DATOS"
    cols = [c for c in ["Open","High","Low","Close","EMA20","EMA50","EMA200","RSI14","ATR14","MACD","MACD_signal","MACD_hist"] if c in df.columns]
    if not cols:
        cols = [c for c in df.columns if c in ["Open","High","Low","Close"]][:4] or df.columns[:6]
    return df.tail(rows)[cols].round(4).to_string()

def build_trading_prompt_from_df(symbol: str, df: pd.DataFrame, period: str, interval: str) -> str:
    cols = [c for c in ["Open","High","Low","Close","EMA20","EMA50","EMA200","RSI14","ATR14","MACD","MACD_signal","MACD_hist"] if c in df.columns]
    tail_txt = df.tail(80)[cols].round(4).to_string()
    last = df.iloc[-1]
    last_close = float(last["Close"])
    last_rsi = float(last["RSI14"]) if "RSI14" in df.columns and pd.notnull(last["RSI14"]) else "NA"
    last_atr = float(last["ATR14"]) if "ATR14" in df.columns and pd.notnull(last["ATR14"]) else "NA"

    prompt = f"""
Actúa como asesor de trading experto para CFDs.
Instrumento: {symbol} | Periodo: {period} | Intervalo: {interval}
Último cierre: {last_close}
RSI(14): {last_rsi} | ATR(14): {last_atr}

Datos recientes:
{tail_txt}

Entrega:
1) Lectura técnica (tendencia, cruces EMA, RSI, MACD, soportes/resistencias).
2) Estrategia principal (long o short) y alternativa si aplica.
3) Niveles concretos: entrada(s), SL (puedes referenciar ATR), TP(s) y ratio riesgo:beneficio.
4) Escenarios probables e invalidaciones.
5) Gestión del riesgo y temporalidad recomendada (scalp/intradía/swing).
Sé específico y accionable. No inventes precios.
"""
    return prompt

def qa_answer(user_q: str) -> str:
    symbol = st.session_state.last_symbol
    df = st.session_state.last_df
    period = st.session_state.last_period
    interval = st.session_state.last_interval
    img_ctx = st.session_state.last_image_analysis

    ctx_parts = []
    if symbol and df is not None and not df.empty:
        ctx_parts.append(f"Contexto del último símbolo: {symbol} (periodo={period}, intervalo={interval}).")
        ctx_parts.append("Muestra de datos recientes con indicadores:")
        ctx_parts.append(df_compact_summary(df))
        last = df.iloc[-1]
        quick = []
        for k in ["Close","EMA20","EMA50","EMA200","RSI14","ATR14"]:
            if k in df.columns and pd.notnull(last.get(k, None)):
                quick.append(f"{k}={fmt_num(last.get(k))}")
        if quick:
            ctx_parts.append("Lectura técnica rápida: " + ", ".join(quick) + ".")
    if img_ctx:
        ctx_parts.append("Resumen del último análisis por imagen:")
        ctx_parts.append(img_ctx)

    ctx = "\n".join(ctx_parts) if ctx_parts else "No hay contexto previo disponible."

    prompt = f"""
Eres un **asesor de trading experto en CFDs**.
Tienes el contexto siguiente (datos históricos y/o análisis visual):

{ctx}

Pregunta del usuario: {user_q}

Responde de forma profesional y accionable:
- Da niveles de entrada, SL, TP y ratio R:R si corresponde.
- Indica sesgo (alcista/bajista/lateral) y temporalidad sugerida.
- Señala confirmaciones/invalidaciones y riesgos.
- Si faltan datos, pide explícitamente lo que necesitas (símbolo, timeframe, etc.).
No inventes precios.
"""
    return llm_text(prompt)


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
                "1) Tendencia principal (alcista/bajista/lateral) y por qué.\n"
                "2) Patrones visibles (canales, triángulos, HCH, doble techo/suelo) y zonas S/R.\n"
                "3) Lectura de indicadores visibles (RSI, MACD, EMAs/Bollinger) si aparecen.\n"
                "4) Estrategia CFD (long/short): condiciones de entrada, SL, TP, ratio R:R.\n"
                "5) Gestión del riesgo y temporalidad recomendada.\n"
                "Si faltan datos, pide confirmaciones específicas."
            )
            try:
                analysis = llm_image(prompt_img, image_path)
                st.subheader("📈 Análisis Técnico del Gráfico (IA)")
                st.write(analysis or "No se obtuvo texto del modelo.")
                st.session_state.last_image_analysis = analysis
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
            df["ATR14"] = atr(df, 14)

            # Guardar contexto para Q&A
            st.session_state.last_symbol = symbol
            st.session_state.last_df = df
            st.session_state.last_period = period
            st.session_state.last_interval = interval

            st.subheader(f"📊 Histórico de {symbol}")
            plot_candles(df, f"{symbol} — {period} @ {interval}")

            # Panel técnico rápido
            st.markdown("### 🔎 Lectura técnica rápida")
            c1, c2, c3, c4 = st.columns(4)
            last = df.iloc[-1]
            c1.metric("Cierre", fmt_num(last.get("Close")))
            c2.metric("RSI(14)", fmt_num(last.get("RSI14"), 2))

            def arrow(a, b):
                try:
                    if a is None or b is None or pd.isna(a) or pd.isna(b):
                        return "—"
                    return "▲" if float(a) > float(b) else "▼"
                except Exception:
                    return "—"

            c3.metric("EMA20 vs EMA50", arrow(last.get("EMA20"), last.get("EMA50")))
            c4.metric("EMA50 vs EMA200", arrow(last.get("EMA50"), last.get("EMA200")))

            # Fundamentales + noticias
            info, news = fundamentals_and_news(symbol)
            st.markdown("### 🧾 Fundamentales (referenciales)")
            colA, colB, colC, colD = st.columns(4)
            colA.metric("Market Cap", f"{info.get('market_cap', 'NA')}")
            colB.metric("P/E (TTM)", f"{info.get('pe', 'NA')}")
            colC.metric("Div. Yield", f"{info.get('div_yield', 'NA')}")
            colD.metric("Moneda", f"{info.get('currency', 'NA')}")

            if news:
                st.markdown("### 📰 Noticias recientes")
                for n in news[:5]:
                    st.markdown(f"- {n.get('title','(sin título)')}")

            # Gestión de riesgo con ATR
            st.markdown("### 🛡️ Gestión de riesgo (ATR)")
            colR1, colR2, colR3, colR4 = st.columns(4)
            atr_val = last.get("ATR14")
            colR1.metric("ATR(14)", fmt_num(atr_val))
            balance = colR2.number_input("Balance (USD/moneda)", value=10000.0, min_value=0.0, step=100.0)
            risk_pct = colR3.number_input("Riesgo % por trade", value=1.0, min_value=0.0, step=0.25)
            rr      = colR4.selectbox("Riesgo:Beneficio", ["1:1", "1:1.5", "1:2", "1:3"], index=2)

            if atr_val is not None and pd.notnull(last.get("Close", None)):
                entry_long  = float(last["Close"])
                sl_long     = entry_long - 1.5 * float(atr_val)
                tp_long     = entry_long + (float(rr.split(":")[1]) / float(rr.split(":")[0])) * (entry_long - sl_long)

                entry_short = float(last["Close"])
                sl_short    = entry_short + 1.5 * float(atr_val)
                tp_short    = entry_short - (float(rr.split(":")[1]) / float(rr.split(":")[0])) * (sl_short - entry_short)

                size_long  = position_size(balance, risk_pct, entry_long, sl_long)
                size_short = position_size(balance, risk_pct, entry_short, sl_short)

                st.markdown("#### Sugerencias rápidas (basadas en ATR)")
                cL, cS = st.columns(2)
                cL.write(f"**Largo (CFD)** → Entrada: `{fmt_num(entry_long)}`, SL: `{fmt_num(sl_long)}`, TP: `{fmt_num(tp_long)}`, Tamaño aprox: `{fmt_num(size_long, 2)}` unidades")
                cS.write(f"**Corto (CFD)** → Entrada: `{fmt_num(entry_short)}`, SL: `{fmt_num(sl_short)}`, TP: `{fmt_num(tp_short)}`, Tamaño aprox: `{fmt_num(size_short, 2)}` unidades")
            else:
                st.info("ATR no disponible para esta combinación de periodo/intervalo.")

            # Recomendación IA
            with st.spinner("Generando recomendación del asesor..."):
                try:
                    prompt = build_trading_prompt_from_df(symbol, df, period, interval)
                    advice = llm_text(prompt)
                    st.subheader("🧠 Recomendación del Asesor (IA)")
                    st.write(advice or "No se obtuvo texto del modelo.")
                except Exception as e:
                    st.error(f"Ocurrió un error al generar la recomendación: {e}")


# =========================
# CHAT: PREGUNTAS ESPECÍFICAS
# =========================
st.markdown("---")
st.header("💬 Preguntas al Asesor")

# Botón limpiar
if st.button("🧹 Limpiar chat", use_container_width=True):
    st.session_state.chat = []
    st.rerun()

# Render historial
for turn in st.session_state.chat:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])

user_q = st.chat_input("Escribe tu pregunta específica (ej: '¿Qué harías con el último análisis de BTC-USD?')")
if user_q:
    st.session_state.chat.append({"role": "user", "content": user_q})
    with st.chat_message("user"):
        st.markdown(user_q)

    with st.chat_message("assistant"):
        with st.spinner("Analizando contexto y respondiendo..."):
            try:
                ans = qa_answer(user_q)
            except Exception as e:
                ans = f"Error generando respuesta: {e}"
            st.markdown(ans)
    st.session_state.chat.append({"role": "assistant", "content": ans})

# =========================
# FOOTER
# =========================
st.markdown("---")
st.caption(
    "Este asistente entrega opiniones educativas basadas en datos e IA. No constituye asesoría financiera. "
    "Opera con gestión de riesgo adecuada."
)

