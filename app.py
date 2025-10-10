import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode, RTCConfiguration
import cv2
import numpy as np
import os
from PIL import Image
import av
import requests

# === Configuración de la página ===
st.set_page_config(
    page_title="🎨 Pizarra con Color Rojo",
    page_icon="🎨",
    layout="centered"
)

# === Función para obtener servidores TURN de Twilio ===
@st.cache_data(ttl=3600)
def get_ice_servers():
    """Obtiene servidores TURN desde Twilio"""
    try:
        account_sid = st.secrets["twilio"]["account_sid"]
        auth_token = st.secrets["twilio"]["auth_token"]
        
        url = f"https://api.twilio.com/2010-04-01/Accounts/{account_sid}/Tokens.json"
        response = requests.post(url, auth=(account_sid, auth_token), timeout=5)
        
        if response.status_code == 201:
            data = response.json()
            ice_servers = data.get('ice_servers', [])
            st.sidebar.success("✅ Twilio TURN activo")
            return ice_servers
        else:
            st.sidebar.warning(f"⚠️ Twilio error {response.status_code}, usando STUN")
            return [{"urls": ["stun:stun.l.google.com:19302"]}]
            
    except Exception as e:
        st.sidebar.error(f"❌ Error: {str(e)[:50]}")
        return [{"urls": ["stun:stun.l.google.com:19302"]}]

# === Estilos ===
st.markdown("""
<style>
.main-header { text-align: center; color: #e74c3c; margin-bottom: 10px; }
.instructions { background-color: #f8f9fa; padding: 14px; border-radius: 10px; border-left: 4px solid #e74c3c; margin-bottom: 20px; color: #2c3e50; }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🎨 Pizarra Interactiva</h1>', unsafe_allow_html=True)

st.markdown("""
<div class="instructions">
    <strong>¿Cómo usarlo?</strong><br>
    1. Permite el acceso a tu cámara.<br>
    2. Muestra un objeto <b>rojo brillante</b>.<br>
    3. Muévelo para dibujar.
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

# === Configuración sidebar OPTIMIZADA ===
st.sidebar.header("⚙️ Configuración")

# Resolución - Valores más bajos por defecto
resolution_options = {
    "Muy Baja (240x180) - Rápida": {"width": 240, "height": 180},
    "Baja (320x240) - Estable": {"width": 320, "height": 240},
    "Media (480x360) - Balanceada": {"width": 480, "height": 360},
    "Alta (640x480) - Calidad": {"width": 640, "height": 480},
}
selected_res = st.sidebar.selectbox(
    "📹 Resolución",
    list(resolution_options.keys()),
    index=1  # Por defecto: Baja
)
resolution = resolution_options[selected_res]

# FPS más bajo por defecto
fps = st.sidebar.slider("🎬 FPS", 5, 20, 10, 5)

# Procesamiento - Nuevo control
skip_frames = st.sidebar.slider("⚡ Saltar frames (mayor = más rápido)", 0, 3, 1, 1,
                                help="Procesa 1 de cada N frames")

brush_size = st.sidebar.slider("✏️ Grosor", 2, 20, 5, 1)
sensitivity = st.sidebar.slider("🎯 Sensibilidad", 100, 1000, 300, 50)

# === Controles principales ===
col1, col2 = st.columns(2)
with col1:
    if st.button("🗑️ Limpiar", use_container_width=True):
        st.session_state.clear_canvas = True
        st.session_state.saved_image = None
        st.success("✅ Limpiado")

with col2:
    save_btn = st.button("💾 Guardar", use_container_width=True)

# === Procesador de video OPTIMIZADO ===
class PizarraProcessor(VideoProcessorBase):
    def __init__(self):
        self.canvas = None
        self.prev_x = None
        self.prev_y = None
        self.frame_count = 0
        self.skip_frames = skip_frames

    def recv(self, frame):
        self.frame_count += 1
        
        # Saltar frames para mejorar rendimiento
        if self.skip_frames > 0 and self.frame_count % (self.skip_frames + 1) != 0:
            # Retornar frame anterior sin procesar
            if hasattr(self, 'last_output'):
                return self.last_output
        
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        
        if st.session_state.clear_canvas:
            self.canvas = None
            self.prev_x = None
            self.prev_y = None
            st.session_state.clear_canvas = False

        if self.canvas is None:
            self.canvas = np.zeros_like(img)

        # === OPTIMIZACIÓN: Reducir tamaño para procesamiento ===
        height, width = img.shape[:2]
        process_scale = 0.5  # Procesar al 50% del tamaño
        small_img = cv2.resize(img, (int(width * process_scale), int(height * process_scale)))

        # Detección de rojo en imagen reducida
        hsv = cv2.cvtColor(small_img, cv2.COLOR_BGR2HSV)
        
        lower_red1 = np.array([0, 120, 70])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 120, 70])
        upper_red2 = np.array([180, 255, 255])
        
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)

        # Operaciones morfológicas reducidas
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.dilate(mask, kernel, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detected = False
        if contours:
            c = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(c)
            
            # Ajustar sensibilidad por escala
            adjusted_sensitivity = sensitivity * (process_scale ** 2)
            
            if area > adjusted_sensitivity:
                detected = True
                M = cv2.moments(c)
                if M["m00"] != 0:
                    # Escalar coordenadas de vuelta al tamaño original
                    cx = int((M["m10"] / M["m00"]) / process_scale)
                    cy = int((M["m01"] / M["m00"]) / process_scale)
                    
                    if self.prev_x is not None and self.prev_y is not None:
                        dist = np.sqrt((cx - self.prev_x)**2 + (cy - self.prev_y)**2)
                        if dist < 150:
                            cv2.line(self.canvas, (self.prev_x, self.prev_y), 
                                    (cx, cy), (0, 0, 255), brush_size)
                    
                    self.prev_x = cx
                    self.prev_y = cy
                    
                    # Indicadores visuales más ligeros
                    cv2.circle(img, (cx, cy), 12, (0, 255, 0), 2)
        
        if not detected:
            self.prev_x = None
            self.prev_y = None

        # Combinar con menos mezcla (más rápido)
        output = cv2.addWeighted(img, 0.75, self.canvas, 0.25, 0)
        
        # Texto simplificado
        status = "DRAW" if detected else "WAIT"
        color = (0, 255, 0) if detected else (100, 100, 255)
        cv2.putText(output, status, (10, 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        st.session_state.canvas_snapshot = self.canvas.copy()
        
        # Guardar para frames saltados
        output_frame = av.VideoFrame.from_ndarray(output, format="bgr24")
        self.last_output = output_frame
        
        return output_frame

# === Guardar imagen ===
if save_btn and st.session_state.canvas_snapshot is not None:
    timestamp = st.session_state.get('save_count', 0)
    st.session_state.save_count = timestamp + 1
    filename = f"dibujos/dibujo_{timestamp:03d}.png"
    cv2.imwrite(filename, st.session_state.canvas_snapshot)
    st.session_state.saved_image = Image.open(filename)
    st.success(f"✅ Guardado: {filename}")

if st.session_state.saved_image is not None:
    with st.expander("🖼️ Ver guardado"):
        st.image(st.session_state.saved_image, use_container_width=True)

# === Configurar ICE servers ===
ice_servers = get_ice_servers()

RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": ice_servers,
    "iceTransportPolicy": "all",
})

# === Restricciones de medios OPTIMIZADAS ===
media_stream_constraints = {
    "video": {
        "width": {"ideal": resolution["width"], "max": resolution["width"]},
        "height": {"ideal": resolution["height"], "max": resolution["height"]},
        "frameRate": {"ideal": fps, "max": fps},
        "facingMode": "user",
    },
    "audio": False
}

# === Iniciar WebRTC ===
st.markdown("---")

webrtc_ctx = webrtc_streamer(
    key="pizarra-optimizada",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=PizarraProcessor,
    media_stream_constraints=media_stream_constraints,
    rtc_configuration=RTC_CONFIGURATION,
    async_processing=True,
)

# === Estado simple ===
col1, col2 = st.columns(2)
with col1:
    if webrtc_ctx.state.playing:
        st.success("🟢 ACTIVO")
    else:
        st.info("⏸️ Presiona START")

with col2:
    st.metric("Config", f"{resolution['width']}x{resolution['height']} @ {fps}fps")

# === Info sidebar ===
st.sidebar.markdown("---")
if webrtc_ctx.state.playing:
    st.sidebar.success("🟢 Conectado")
else:
    st.sidebar.warning("🟡 Detenido")

with st.expander("⚡ Tips de optimización"):
    st.markdown("""
    **Para mejorar velocidad:**
    1. 🔽 Usa "Muy Baja" o "Baja" resolución
    2. 📉 Baja FPS a 5-10
    3. ⚡ Aumenta "Saltar frames" a 2-3
    4. 🌐 Verifica tu internet (mínimo 2 Mbps)
    5. 🔌 Cierra otras apps/pestañas
    
    **Problema común:**
    Si usas API Key (SK...) en lugar de Account SID (AC...), 
    algunas funciones de Twilio pueden ser limitadas.
    """)

st.markdown('<div style="text-align: center; color: #7f8c8d; margin-top: 20px;">🚀 Optimizado para bajo ancho de banda</div>', unsafe_allow_html=True)
