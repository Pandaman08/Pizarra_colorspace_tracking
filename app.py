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
@st.cache_data(ttl=3600)  # Cache por 1 hora
def get_ice_servers():
    """Obtiene servidores TURN desde Twilio"""
    try:
        # Leer credenciales desde secrets
        account_sid = st.secrets["twilio"]["account_sid"]
        auth_token = st.secrets["twilio"]["auth_token"]
        
        # Endpoint de Twilio para generar tokens
        url = f"https://api.twilio.com/2010-04-01/Accounts/{account_sid}/Tokens.json"
        
        # Hacer request con autenticación
        response = requests.post(url, auth=(account_sid, auth_token))
        
        if response.status_code == 201:
            data = response.json()
            ice_servers = data.get('ice_servers', [])
            st.sidebar.success("✅ Servidores TURN de Twilio activos")
            return ice_servers
        else:
            st.sidebar.warning("⚠️ Error con Twilio, usando STUN público")
            return [{"urls": ["stun:stun.l.google.com:19302"]}]
            
    except KeyError:
        st.sidebar.error("❌ Falta configurar secrets.toml con credenciales Twilio")
        st.sidebar.info("Agrega tus credenciales en `.streamlit/secrets.toml`")
        return [{"urls": ["stun:stun.l.google.com:19302"]}]
    except Exception as e:
        st.sidebar.error(f"❌ Error: {str(e)}")
        return [{"urls": ["stun:stun.l.google.com:19302"]}]

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

# === Configuración sidebar ===
st.sidebar.header("⚙️ Configuración")

# Resolución
resolution_options = {
    "Baja (320x240)": {"width": 320, "height": 240},
    "Media (640x480)": {"width": 640, "height": 480},
    "Alta (1280x720)": {"width": 1280, "height": 720}
}
selected_res = st.sidebar.selectbox(
    "📹 Resolución",
    list(resolution_options.keys()),
    index=1
)
resolution = resolution_options[selected_res]

fps = st.sidebar.slider("🎬 FPS", 10, 30, 15, 5)
brush_size = st.sidebar.slider("✏️ Grosor", 2, 20, 5, 1)
sensitivity = st.sidebar.slider("🎯 Sensibilidad", 100, 1000, 300, 50)

# === Controles principales ===
col1, col2 = st.columns(2)
with col1:
    if st.button("🗑️ Limpiar Canvas", use_container_width=True):
        st.session_state.clear_canvas = True
        st.session_state.saved_image = None
        st.success("✅ Lienzo limpiado.")

with col2:
    save_btn = st.button("💾 Guardar Dibujo", use_container_width=True)

# === Procesador de video ===
class PizarraProcessor(VideoProcessorBase):
    def __init__(self):
        self.canvas = None
        self.prev_x = None
        self.prev_y = None

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        
        if st.session_state.clear_canvas:
            self.canvas = None
            self.prev_x = None
            self.prev_y = None
            st.session_state.clear_canvas = False

        if self.canvas is None:
            self.canvas = np.zeros_like(img)

        # Detección de rojo
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        lower_red1 = np.array([0, 120, 70])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 120, 70])
        upper_red2 = np.array([180, 255, 255])
        
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detected = False
        if contours:
            c = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(c)
            
            if area > sensitivity:
                detected = True
                M = cv2.moments(c)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    if self.prev_x is not None and self.prev_y is not None:
                        dist = np.sqrt((cx - self.prev_x)**2 + (cy - self.prev_y)**2)
                        if dist < 100:
                            cv2.line(self.canvas, (self.prev_x, self.prev_y), 
                                    (cx, cy), (0, 0, 255), brush_size)
                    
                    self.prev_x = cx
                    self.prev_y = cy
                    
                    cv2.circle(img, (cx, cy), 15, (0, 255, 0), 2)
                    cv2.circle(img, (cx, cy), brush_size, (0, 255, 255), -1)
        
        if not detected:
            self.prev_x = None
            self.prev_y = None

        output = cv2.addWeighted(img, 0.7, self.canvas, 0.3, 0)
        
        status_text = "🎨 DIBUJANDO" if detected else "⏳ ESPERANDO..."
        status_color = (0, 255, 0) if detected else (0, 165, 255)
        
        cv2.rectangle(output, (5, 5), (300, 45), (0, 0, 0), -1)
        cv2.rectangle(output, (5, 5), (300, 45), status_color, 2)
        cv2.putText(output, status_text, (15, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        st.session_state.canvas_snapshot = self.canvas.copy()

        return av.VideoFrame.from_ndarray(output, format="bgr24")

# === Guardar imagen ===
if save_btn and st.session_state.canvas_snapshot is not None:
    timestamp = st.session_state.get('save_count', 0)
    st.session_state.save_count = timestamp + 1
    filename = f"dibujos/dibujo_{timestamp:03d}.png"
    cv2.imwrite(filename, st.session_state.canvas_snapshot)
    st.session_state.saved_image = Image.open(filename)
    st.success(f"✅ Guardado como `{filename}`")

if st.session_state.saved_image is not None:
    with st.expander("🖼️ Ver dibujo guardado", expanded=False):
        st.image(st.session_state.saved_image, use_container_width=True)

# === Configurar ICE servers con Twilio ===
ice_servers = get_ice_servers()

RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": ice_servers,
    "iceTransportPolicy": "all",
})

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
    key="pizarra-twilio",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=PizarraProcessor,
    media_stream_constraints=media_stream_constraints,
    rtc_configuration=RTC_CONFIGURATION,
    async_processing=True,
)

if webrtc_ctx.state.playing:
    st.success("✅ Conexión activa con Twilio TURN")
else:
    st.info("⏳ Presiona START para iniciar")

# === Info en sidebar ===
st.sidebar.markdown("---")
st.sidebar.markdown("### 📡 Estado de conexión")
if webrtc_ctx.state.playing:
    st.sidebar.success("🟢 CONECTADO")
else:
    st.sidebar.warning("🟡 DESCONECTADO")

with st.expander("💡 Verificar configuración Twilio"):
    st.markdown("""
    **Checklist de configuración:**
    
    ✅ Cuenta Twilio creada  
    ✅ ACCOUNT_SID copiado  
    ✅ AUTH_TOKEN copiado  
    ✅ Archivo `.streamlit/secrets.toml` creado  
    ✅ Credenciales pegadas en secrets.toml  
    ✅ App reiniciada después de agregar secrets  
    
    **Si ves "✅ Servidores TURN de Twilio activos" arriba, todo está bien!**
    """)

st.markdown('<div class="footer">🚀 Powered by Twilio TURN • OpenCV + Streamlit</div>', unsafe_allow_html=True)
