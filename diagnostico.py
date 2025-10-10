import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode, RTCConfiguration
import av
import time

st.set_page_config(page_title="🔍 Diagnóstico WebRTC", layout="wide")

st.title("🔍 Diagnóstico de Conexión WebRTC")

st.markdown("""
Esta herramienta te ayuda a identificar problemas con tu conexión WebRTC.
""")

# === Configuración de prueba ===
col1, col2 = st.columns(2)

with col1:
    st.subheader("⚙️ Configuración de prueba")
    
    test_resolution = st.selectbox(
        "Resolución",
        ["320x240", "640x480", "1280x720"],
        index=1
    )
    width, height = map(int, test_resolution.split('x'))
    
    test_fps = st.slider("FPS", 5, 30, 15, 5)
    
    use_multiple_stun = st.checkbox("Usar múltiples servidores STUN", value=True)

with col2:
    st.subheader("📊 Información del sistema")
    st.info(f"""
    **Navegador recomendado:** Chrome, Edge, Firefox
    
    **Requisitos mínimos:**
    - 🌐 Internet: 2 Mbps
    - 💻 CPU: Dual core
    - 📹 Cámara web funcional
    """)

# === Procesador de diagnóstico ===
class DiagnosticProcessor(VideoProcessorBase):
    def __init__(self):
        self.frame_count = 0
        self.start_time = time.time()
        self.last_fps_update = time.time()
        self.fps = 0

    def recv(self, frame):
        self.frame_count += 1
        
        # Calcular FPS real
        current_time = time.time()
        if current_time - self.last_fps_update >= 1.0:
            elapsed = current_time - self.start_time
            self.fps = self.frame_count / elapsed if elapsed > 0 else 0
            self.last_fps_update = current_time
        
        img = frame.to_ndarray(format="bgr24")
        height, width = img.shape[:2]
        
        # Información en pantalla
        info_text = [
            f"Resolucion: {width}x{height}",
            f"FPS Real: {self.fps:.1f}",
            f"Frames: {self.frame_count}",
            f"Tiempo: {int(time.time() - self.start_time)}s"
        ]
        
        import cv2
        y_offset = 30
        for i, text in enumerate(info_text):
            cv2.putText(img, text, (10, y_offset + i*30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Cuadrícula de referencia
        cv2.line(img, (width//2, 0), (width//2, height), (255, 0, 0), 1)
        cv2.line(img, (0, height//2), (width, height//2), (255, 0, 0), 1)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# === Configuración WebRTC ===
if use_multiple_stun:
    ice_servers = [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
        {"urls": ["stun:stun2.l.google.com:19302"]},
        {"urls": ["stun:stun3.l.google.com:19302"]},
        {"urls": ["stun:stun4.l.google.com:19302"]},
    ]
else:
    ice_servers = [{"urls": ["stun:stun.l.google.com:19302"]}]

RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": ice_servers,
    "iceTransportPolicy": "all"
})

media_constraints = {
    "video": {
        "width": {"ideal": width},
        "height": {"ideal": height},
        "frameRate": {"ideal": test_fps, "max": test_fps}
    },
    "audio": False
}

# === Stream de prueba ===
st.markdown("---")
st.subheader("📹 Prueba de cámara")

webrtc_ctx = webrtc_streamer(
    key=f"diagnostic-{width}-{height}-{test_fps}",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=DiagnosticProcessor,
    media_stream_constraints=media_constraints,
    rtc_configuration=RTC_CONFIGURATION,
    async_processing=True,
)

# === Estado de conexión ===
st.markdown("---")
st.subheader("🔌 Estado de conexión")

if webrtc_ctx.state.playing:
    st.success("✅ **Conexión establecida correctamente**")
    st.balloons()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Estado", "🟢 ACTIVO")
    with col2:
        st.metric("Resolución", f"{width}x{height}")
    with col3:
        st.metric("FPS Objetivo", test_fps)
    
else:
    st.warning("⚠️ Esperando conexión...")
    st.info("👆 Presiona 'START' arriba para iniciar la prueba")

# === Recomendaciones ===
st.markdown("---")
with st.expander("💡 Interpretación de resultados"):
    st.markdown("""
    ### ✅ Señales de buena conexión:
    - FPS real cercano al objetivo (±2 FPS)
    - Video fluido sin congelamientos
    - Latencia baja (< 500ms)
    
    ### ⚠️ Problemas comunes:
    
    **1. FPS muy bajo (< 10)**
    - 🔽 Reduce la resolución
    - 📉 Baja los FPS objetivo
    - 🌐 Verifica tu internet
    
    **2. Video se congela**
    - 🔄 Recarga la página
    - 🔌 Cierra otras apps que usen la cámara
    - 🌐 Usa cable ethernet en vez de WiFi
    
    **3. No se conecta**
    - ✅ Verifica permisos de cámara en el navegador
    - ✅ Prueba otro navegador (Chrome recomendado)
    - ✅ Desactiva VPN/Proxy
    - ✅ Verifica firewall/antivirus
    
    **4. Calidad muy baja**
    - ⬆️ Aumenta resolución gradualmente
    - 💡 Mejora iluminación
    - 📹 Limpia lente de cámara
    """)

with st.expander("🔧 Soluciones avanzadas"):
    st.markdown("""
    ### Si nada funciona, intenta:
    
    1. **Usar servidores TURN (requiere cuenta Twilio)**
       - Crea cuenta gratuita en twilio.com
       - Obtén credenciales
       - Agrega a `secrets.toml`
    
    2. **Ejecutar localmente**
       ```bash
       streamlit run app.py
       ```
       Esto evita problemas de firewall/NAT
    
    3. **Verificar tu red**
       - Prueba en otra red WiFi
       - Usa datos móviles como hotspot
       - Contacta a tu ISP si bloquea WebRTC
    """)
