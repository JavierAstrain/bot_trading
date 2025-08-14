import streamlit as st
import google.generativeai as genai
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- CONFIGURACIÓN DE GEMINI ---
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# --- INTERFAZ STREAMLIT ---
st.set_page_config(page_title="Asesor de Trading Experto", layout="wide")
st.title("🤖📈 Asesor de Trading Experto IA")

st.markdown("""
Este bot actúa como **asesor de trading profesional**, analizando:
- Imágenes de gráficos (análisis técnico con IA)
- Datos históricos (Yahoo Finance)
- Contexto técnico + fundamental
- Recomendaciones y estrategias para CFDs
""")

# --- OPCIONES DEL USUARIO ---
option = st.radio("¿Qué deseas hacer?", ["📷 Subir gráfico (imagen)", "📊 Analizar un activo por símbolo"])

# --- ANALISIS DE IMÁGENES ---
if option == "📷 Subir gráfico (imagen)":
    uploaded_file = st.file_uploader("Sube una imagen de tu gráfico (TradingView, MetaTrader, Broker...)", type=["png", "jpg", "jpeg"])
    
    if uploaded_file:
        st.image(uploaded_file, caption="Gráfico subido", use_column_width=True)
        
        model = genai.GenerativeModel("gemini-1.5-flash")  # Vision
        response = model.generate_content(
            ["Actúa como un asesor de trading experto. Analiza el gráfico y responde con:",
             "1. Tendencia detectada",
             "2. Patrones técnicos (si los hay)",
             "3. Estrategia de entrada/salida",
             "4. Niveles sugeridos de SL y TP",
             "5. Recomendación de gestión de riesgo",
             uploaded_file]
        )
        
        st.subheader("📈 Análisis Técnico del Gráfico")
        st.write(response.text)

# --- ANALISIS DE ACTIVOS POR SÍMBOLO ---
if option == "📊 Analizar un activo por símbolo":
    symbol = st.text_input("Escribe el símbolo (ej: AAPL, BTC-USD, GOLD, EURUSD=X)")
    period = st.selectbox("Periodo a analizar", ["7d", "1mo", "3mo", "6mo", "1y"])
    interval = st.selectbox("Intervalo de velas", ["15m", "1h", "1d"])
    
    if st.button("🔎 Analizar activo") and symbol:
        # Descargar histórico con yfinance
        df = yf.download(symbol, period=period, interval=interval)
        
        if df.empty:
            st.error("No se encontraron datos para este símbolo.")
        else:
            st.subheader(f"📊 Histórico de {symbol}")
            
            # Gráfico interactivo
            fig = go.Figure(data=[go.Candlestick(x=df.index,
                                                 open=df['Open'],
                                                 high=df['High'],
                                                 low=df['Low'],
                                                 close=df['Close'])])
            fig.update_layout(title=f"Precio de {symbol}", xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)
            
            # Preparar datos para análisis con IA
            ultimo_cierre = df["Close"].iloc[-1]
            resumen = df.tail(20).to_string()
            
            model = genai.GenerativeModel("gemini-1.5-flash")
            prompt = f"""
Actúa como un asesor de trading experto. Analiza el activo {symbol}.
Datos recientes:
{resumen}

Precio de cierre más reciente: {ultimo_cierre}

Entrega:
1. Análisis de la tendencia general
2. Recomendación de entrada/salida
3. Posibles escenarios futuros
4. Estrategia de inversión en CFDs
5. Niveles sugeridos de Stop Loss y Take Profit
"""
            response = model.generate_content(prompt)
            
            st.subheader("🧠 Análisis del Activo")
            st.write(response.text)
