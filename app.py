# app.py
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import cv2
import numpy as np
import os
from PIL import Image

# === Configuración de la página ===
st.set_page_config(
    page_title="🎨 Pizarra con Color Rojo",
    page_icon="🎨",
    layout="centered"
)

# === Estilos ===
st.markdown("""
<style>
.main-header { text-align: center; color: #e74c3c; margin-bottom: 10px; }
.instructions { background-color: #f8f9fa; padding: 14px; border-radius: 10px; border-left: 4px solid #e74c3c; margin-bottom: 20px; color: #2c3e50; }
.status { font-weight: bold; color: #27ae60; }
.footer { text-align: center; color: #7f8c8d; margin-top: 20px; }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🎨 Pizarra Interactiva con Seguimiento de Color</h1>', unsafe_allow_html=True)

st.markdown("""
<div class="instructions">
    <strong>¿Cómo usarlo?</strong><br>
    1. Permite el acceso a tu cámara.<br>
    2. Muestra un objeto <b>rojo brillante</b> frente a la cámara.<br>
    3. Muévelo para dibujar en el aire.<br>
    4. Usa los botones para limpiar o guardar.
</div>
""", unsafe_allow_html=True)

# === Estado de la sesión ===
if 'canvas' not in st.session_state:
    st.session_state.canvas = None
if 'saved_image' not in st.session_state:
    st.session_state.saved_image = None

os.makedirs("dibujos", exist_ok=True)

# === Controles ===
col1, col2, col3 = st.columns(3)
with col1:
    clear_btn = st.button("🗑️ Limpiar", use_container_width=True)
with col2:
    save_btn = st.button("💾 Guardar Dibujo", use_container_width=True)
with col3:
    st.empty()

# === Acciones ===
if clear_btn:
    st.session_state.canvas = None
    st.session_state.saved_image = None
    st.success("✅ Lienzo limpiado.")

if save_btn and st.session_state.canvas is not None:
    filename = "dibujos/dibujo_rojo.png"
    cv2.imwrite(filename, st.session_state.canvas)
    st.session_state.saved_image = Image.open(filename)
    st.success(f"✅ Guardado como `{filename}`")

# Mostrar imagen guardada
if st.session_state.saved_image is not None:
    st.image(st.session_state.saved_image, caption="Tu dibujo guardado", use_container_width=True)

# === Procesador de video ===
class PizarraProcessor(VideoProcessorBase):
    def __init__(self):
        self.canvas = None

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)  # Espejo

        if self.canvas is None:
            self.canvas = np.zeros_like(img)

        # === Detección de rojo en HSV ===
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = mask1 + mask2

        # === Limpiar ruido ===
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # === Encontrar contornos ===
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            if radius > 10:
                cv2.circle(self.canvas, (int(x), int(y)), 5, (0, 0, 255), -1)

        # === Combinar ===
        output = cv2.addWeighted(img, 0.7, self.canvas, 0.3, 0)

        # Guardar el lienzo en el estado global (para guardar después)
        st.session_state.canvas = self.canvas.copy()

        return output

# === Iniciar WebRTC ===
webrtc_ctx = webrtc_streamer(
    key="pizarra",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=PizarraProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=False,
)

st.markdown('<div class="footer">Basado en OpenCV 3.x con Python • Capítulo 8: Object Tracking</div>', unsafe_allow_html=True)
