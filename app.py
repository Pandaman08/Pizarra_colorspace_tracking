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
.warning-box { background-color: #fff3cd; padding: 10px; border-radius: 5px; border-left: 4px solid #ffc107; margin: 10px 0; }
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

# === Configuración de calidad ===
st.sidebar.header("⚙️ Configuración")

# Resolución
resolution_options = {
    "Baja (320x240) - Más estable": {"width": 320, "height": 240},
    "Media (640x480) - Balanceado": {"width": 640, "height": 480},
    "Alta (1280x720) - Mejor calidad": {"width": 1280, "height": 720}
}
selected_res = st.sidebar.selectbox(
    "📹 Resolución de cámara",
    list(resolution_options.keys()),
    index=1,
    help="Menor resolución = conexión más estable"
)
resolution = resolution_options[selected_res]

# FPS
fps = st.sidebar.slider("🎬 FPS (cuadros por segundo)", 5, 30, 15, 5,
                        help="Menos FPS = menos lag")

# Grosor de pincel
brush_size = st.sidebar.slider("✏️ Grosor de pincel", 2, 20, 5, 1)

# Sensibilidad
sensitivity = st.sidebar.slider("🎯 Sensibilidad de detección", 100, 1000, 300, 50,
                                help="Área mínima en píxeles para detectar objeto")

st.sidebar.markdown("---")
st.sidebar.info("💡 **Tip**: Si la conexión es inestable, reduce la resolución y FPS")

# === Controles principales ===
col1, col2 = st.columns(2)
with col1:
    if st.button("🗑️ Limpiar Canvas", use_container_width=True):
        st.session_state.clear_canvas = True
        st.session_state.saved_image = None
        st.success("✅ Lienzo limpiado.")

with col2:
    save_btn = st.button("💾 Guardar Dibujo", use_container_width=True)

# === Procesador de video OPTIMIZADO ===
class PizarraProcessor(VideoProcessorBase):
    def __init__(self):
        self.canvas = None
        self.prev_x = None
        self.prev_y = None
        self.frame_count = 0
        self.process_every_n_frames = 1  # Procesar cada frame

    def recv(self, frame):
        self.frame_count += 1
        
        # Obtener frame y voltear
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        
        # Verificar si se debe limpiar el canvas
        if st.session_state.clear_canvas:
            self.canvas = None
            self.prev_x = None
            self.prev_y = None
            st.session_state.clear_canvas = False

        # Inicializar canvas con el tamaño correcto
        if self.canvas is None:
            self.canvas = np.zeros_like(img)

        # === Redimensionar si es necesario para mejor rendimiento ===
        height, width = img.shape[:2]
        process_width = min(width, 640)  # Procesar máximo 640px de ancho
        process_height = int(height * (process_width / width))
        
        if width > process_width:
            img_process = cv2.resize(img, (process_width, process_height))
            scale_x = width / process_width
            scale_y = height / process_height
        else:
            img_process = img.copy()
            scale_x = scale_y = 1

        # === Detección de rojo en HSV (en imagen procesada) ===
        hsv = cv2.cvtColor(img_process, cv2.COLOR_BGR2HSV)
        
        # Rangos de rojo optimizados
        lower_red1 = np.array([0, 120, 70])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 120, 70])
        upper_red2 = np.array([180, 255, 255])
        
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)

        # === Limpiar ruido (operaciones morfológicas) ===
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        # === Encontrar contornos ===
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detected = False
        if contours:
            # Obtener el contorno más grande
            c = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(c)
            
            if area > sensitivity:
                detected = True
                # Calcular centro del contorno
                M = cv2.moments(c)
                if M["m00"] != 0:
                    cx = int((M["m10"] / M["m00"]) * scale_x)
                    cy = int((M["m01"] / M["m00"]) * scale_y)
                    
                    # Dibujar línea desde punto anterior
                    if self.prev_x is not None and self.prev_y is not None:
                        # Calcular distancia para evitar líneas muy largas
                        dist = np.sqrt((cx - self.prev_x)**2 + (cy - self.prev_y)**2)
                        if dist < 100:  # Solo dibujar si el movimiento es razonable
                            cv2.line(self.canvas, (self.prev_x, self.prev_y), 
                                    (cx, cy), (0, 0, 255), brush_size)
                    
                    # Actualizar punto anterior
                    self.prev_x = cx
                    self.prev_y = cy
                    
                    # Dibujar indicadores en la imagen original
                    cv2.circle(img, (cx, cy), 15, (0, 255, 0), 2)
                    cv2.circle(img, (cx, cy), brush_size, (0, 255, 255), -1)
        
        if not detected:
            # No hay objeto rojo, resetear punto anterior
            self.prev_x = None
            self.prev_y = None

        # === Combinar imagen con canvas ===
        output = cv2.addWeighted(img, 0.7, self.canvas, 0.3, 0)
        
        # === Agregar información en pantalla ===
        status_text = "🎨 DIBUJANDO" if detected else "⏳ ESPERANDO..."
        status_color = (0, 255, 0) if detected else (0, 165, 255)
        
        # Fondo semi-transparente para el texto
        cv2.rectangle(output, (5, 5), (300, 45), (0, 0, 0), -1)
        cv2.rectangle(output, (5, 5), (300, 45), status_color, 2)
        
        cv2.putText(output, status_text, (15, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        # Guardar snapshot del canvas
        st.session_state.canvas_snapshot = self.canvas.copy()

        return av.VideoFrame.from_ndarray(output, format="bgr24")

# === Acción de guardar ===
if save_btn and st.session_state.canvas_snapshot is not None:
    timestamp = st.session_state.get('save_count', 0)
    st.session_state.save_count = timestamp + 1
    filename = f"dibujos/dibujo_{timestamp:03d}.png"
    cv2.imwrite(filename, st.session_state.canvas_snapshot)
    st.session_state.saved_image = Image.open(filename)
    st.success(f"✅ Guardado como `{filename}`")

# Mostrar imagen guardada
if st.session_state.saved_image is not None:
    with st.expander("🖼️ Ver dibujo guardado", expanded=False):
        st.image(st.session_state.saved_image, caption="Tu última obra maestra", use_container_width=True)

# === Configuración WebRTC OPTIMIZADA ===
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
        {"urls": ["stun:stun2.l.google.com:19302"]},
    ],
    "iceTransportPolicy": "all",
})

# === Restricciones de medios optimizadas ===
media_stream_constraints = {
    "video": {
        "width": {"ideal": resolution["width"]},
        "height": {"ideal": resolution["height"]},
        "frameRate": {"ideal": fps, "max": fps},
    },
    "audio": False
}

# === Iniciar WebRTC ===
st.markdown("---")
st.markdown("### 📹 Vista de cámara")

webrtc_ctx = webrtc_streamer(
    key="pizarra-optimizada",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=PizarraProcessor,
    media_stream_constraints=media_stream_constraints,
    rtc_configuration=RTC_CONFIGURATION,
    async_processing=True,
)

# Mostrar estado de conexión
if webrtc_ctx.state.playing:
    st.success("✅ Conexión activa")
else:
    st.warning("⚠️ Esperando conexión...")

# === Tips adicionales ===
with st.expander("💡 Solución de problemas"):
    st.markdown("""
    **Si la conexión es inestable:**
    1. 🔽 Reduce la resolución a "Baja" o "Media"
    2. 📉 Baja los FPS a 10 o 15
    3. 🌐 Verifica tu conexión a internet
    4. 🔌 Cierra otras aplicaciones que usen la cámara
    5. 🔄 Recarga la página si se congela
    
    **Si no detecta el color rojo:**
    - Usa un objeto rojo BRILLANTE (no oscuro)
    - Mejora la iluminación de tu espacio
    - Ajusta la sensibilidad en la barra lateral
    - Prueba con diferentes objetos rojos
    
    **Mejores objetos para usar:**
    - ✅ Marcador rojo permanente
    - ✅ Papel rojo brillante
    - ✅ Juguete rojo
    - ❌ Evita: ropa, fondos complejos
    """)

st.markdown('<div class="footer">🚀 Optimizado para WebRTC • OpenCV + Streamlit</div>', unsafe_allow_html=True)
