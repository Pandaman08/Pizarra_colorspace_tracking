# app.py
import streamlit as st
import cv2
import numpy as np
import os
from PIL import Image

# === Configuración de la página (DEBE SER LA PRIMERA LLAMADA) ===
st.set_page_config(
    page_title="🎨 Pizarra con Color Rojo",
    page_icon="🎨",
    layout="centered"
)

# === Estilos personalizados ===
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #e74c3c;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        margin-bottom: 10px;
    }
    .instructions {
        background-color: #f8f9fa;
        padding: 14px;
        border-radius: 10px;
        border-left: 4px solid #e74c3c;
        margin-bottom: 20px;
        color: #2c3e50;  /* Texto oscuro legible */
        font-size: 15px;
    }
    .status {
        font-weight: bold;
        color: #27ae60;
        font-size: 16px;
    }
    .footer {
        text-align: center;
        color: #7f8c8d;
        font-size: 13px;
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🎨 Pizarra Interactiva con Seguimiento de Color</h1>', unsafe_allow_html=True)

st.markdown("""
<div class="instructions">
    <strong>¿Cómo usarlo?</strong><br>
    1. Haz clic en <b>Iniciar Cámara</b>.<br>
    2. Muestra un objeto <b>rojo brillante</b> (marcador, gorro, etc.) frente a la cámara.<br>
    3. Muévelo para dibujar en el aire.<br>
    4. Usa los botones para limpiar o guardar tu obra.
</div>
""", unsafe_allow_html=True)

if 'canvas' not in st.session_state:
    st.session_state.canvas = None
if 'last_status' not in st.session_state:
    st.session_state.last_status = ""

# === Carpeta de salida ===
os.makedirs("dibujos", exist_ok=True)

# === Controles ===
col1, col2, col3 = st.columns(3)
with col1:
    start = st.button("🟢 Iniciar Cámara", use_container_width=True)
with col2:
    clear = st.button("🗑️ Limpiar", use_container_width=True)
with col3:
    save = st.button("💾 Guardar Dibujo", use_container_width=True)

# === Mensaje de estado ===
status_placeholder = st.empty()

# === Acciones de botones ===
if clear:
    st.session_state.canvas = None
    st.session_state.last_status = "✅ Lienzo limpiado."
if save and st.session_state.canvas is not None:
    filename = "dibujos/dibujo_rojo.png"
    cv2.imwrite(filename, st.session_state.canvas)
    st.session_state.last_status = f"✅ Guardado como `{filename}`"
    img_pil = Image.open(filename)
    st.image(img_pil, caption="Tu dibujo guardado", use_container_width=True)

# Mostrar estado
if st.session_state.last_status:
    status_placeholder.markdown(f'<p class="status">{st.session_state.last_status}</p>', unsafe_allow_html=True)

# === Vista de video ===
video_placeholder = st.empty()

# === Lógica de la cámara ===
if start:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("❌ No se pudo acceder a la cámara.")
    else:
        st.session_state.last_status = "👀 Mueve un objeto rojo para dibujar..."
        status_placeholder.markdown(f'<p class="status">{st.session_state.last_status}</p>', unsafe_allow_html=True)

        # Rangos HSV para rojo
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)
                if st.session_state.canvas is None:
                    st.session_state.canvas = np.zeros_like(frame)

                # Procesamiento
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
                mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
                mask = mask1 + mask2

                kernel = np.ones((5, 5), np.uint8)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                if contours:
                    c = max(contours, key=cv2.contourArea)
                    ((x, y), radius) = cv2.minEnclosingCircle(c)
                    if radius > 10:
                        cv2.circle(st.session_state.canvas, (int(x), int(y)), 5, (0, 0, 255), -1)

                # Combinar
                output = cv2.addWeighted(frame, 0.7, st.session_state.canvas, 0.3, 0)
                output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)

                # Mostrar (¡usando use_container_width!)
                video_placeholder.image(output_rgb, channels="RGB", use_container_width=True)

        except Exception as e:
            pass
        finally:
            cap.release()
else:
    video_placeholder.info("👆 Haz clic en **Iniciar Cámara** para comenzar a dibujar.")

st.markdown('<div class="footer">Basado en OpenCV 3.x con Python • Capítulo 8: Object Tracking</div>', unsafe_allow_html=True)