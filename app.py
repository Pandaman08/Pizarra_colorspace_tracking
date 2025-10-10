import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode, RTCConfiguration
import cv2
import numpy as np
import os
from PIL import Image
import av

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
if 'clear_canvas' not in st.session_state:
    st.session_state.clear_canvas = False
if 'saved_image' not in st.session_state:
    st.session_state.saved_image = None
if 'canvas_snapshot' not in st.session_state:
    st.session_state.canvas_snapshot = None

os.makedirs("dibujos", exist_ok=True)

# === Controles ===
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("🗑️ Limpiar", use_container_width=True):
        st.session_state.clear_canvas = True
        st.session_state.saved_image = None
        st.success("✅ Lienzo limpiado.")

with col2:
    save_btn = st.button("💾 Guardar Dibujo", use_container_width=True)

with col3:
    brush_size = st.selectbox("✏️ Grosor", [3, 5, 8, 10, 15], index=1)

# === Procesador de video ===
class PizarraProcessor(VideoProcessorBase):
    def __init__(self):
        self.canvas = None
        self.prev_x = None
        self.prev_y = None
        self.brush_size = 5

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)  # Espejo

        # Verificar si se debe limpiar el canvas
        if st.session_state.clear_canvas:
            self.canvas = None
            self.prev_x = None
            self.prev_y = None
            st.session_state.clear_canvas = False

        # Inicializar canvas
        if self.canvas is None:
            self.canvas = np.zeros_like(img)

        # Actualizar tamaño de pincel
        self.brush_size = brush_size

        # === Detección de rojo en HSV ===
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Rango de rojo (el rojo está en dos extremos del espectro HSV)
        lower_red1 = np.array([0, 120, 70])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 120, 70])
        upper_red2 = np.array([180, 255, 255])
        
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)

        # === Limpiar ruido ===
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.dilate(mask, kernel, iterations=1)

        # === Encontrar contornos ===
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Obtener el contorno más grande
            c = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(c)
            
            if area > 300:  # Filtro de área mínima
                # Calcular centro del contorno
                M = cv2.moments(c)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # Dibujar línea desde punto anterior
                    if self.prev_x is not None and self.prev_y is not None:
                        cv2.line(self.canvas, (self.prev_x, self.prev_y), 
                                (cx, cy), (0, 0, 255), self.brush_size)
                    
                    # Actualizar punto anterior
                    self.prev_x = cx
                    self.prev_y = cy
                    
                    # Dibujar círculo indicador en la imagen original
                    cv2.circle(img, (cx, cy), 10, (0, 255, 0), 2)
                    cv2.circle(img, (cx, cy), self.brush_size, (255, 0, 0), -1)
        else:
            # No hay objeto rojo, resetear punto anterior
            self.prev_x = None
            self.prev_y = None

        # === Combinar imagen con canvas ===
        output = cv2.addWeighted(img, 0.6, self.canvas, 0.4, 0)
        
        # Agregar texto de estado
        if contours and cv2.contourArea(max(contours, key=cv2.contourArea)) > 300:
            status = "DIBUJANDO"
            color = (0, 255, 0)
        else:
            status = "ESPERANDO OBJETO ROJO"
            color = (0, 165, 255)
        
        cv2.putText(output, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, color, 2)

        # Guardar snapshot del canvas para poder guardarlo después
        st.session_state.canvas_snapshot = self.canvas.copy()

        return av.VideoFrame.from_ndarray(output, format="bgr24")

# === Acción de guardar (fuera del procesador) ===
if save_btn and st.session_state.canvas_snapshot is not None:
    filename = "dibujos/dibujo_rojo.png"
    cv2.imwrite(filename, st.session_state.canvas_snapshot)
    st.session_state.saved_image = Image.open(filename)
    st.success(f"✅ Guardado como `{filename}`")

# Mostrar imagen guardada
if st.session_state.saved_image is not None:
    with st.expander("🖼️ Ver dibujo guardado", expanded=False):
        st.image(st.session_state.saved_image, caption="Tu última obra maestra", use_container_width=True)

# === Configuración WebRTC ===
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

# === Iniciar WebRTC ===
st.markdown("---")
webrtc_ctx = webrtc_streamer(
    key="pizarra",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=PizarraProcessor,
    media_stream_constraints={"video": True, "audio": False},
    rtc_configuration=RTC_CONFIGURATION,
    async_processing=True,  # ✅ CRÍTICO: Debe ser True
)

# === Tips adicionales ===
with st.expander("💡 Tips para mejores resultados"):
    st.markdown("""
    - **Iluminación**: Asegúrate de tener buena luz
    - **Objeto rojo**: Usa algo de color rojo intenso (marcador, papel, juguete)
    - **Distancia**: Mantén el objeto a 30-60 cm de la cámara
    - **Movimiento**: Muévelo suavemente para dibujar líneas continuas
    - **Limpiar**: Si pierdes el control, presiona "Limpiar" y empieza de nuevo
    """)

st.markdown('<div class="footer">Basado en OpenCV con Python • Capítulo 8: Object Tracking</div>', unsafe_allow_html=True)
