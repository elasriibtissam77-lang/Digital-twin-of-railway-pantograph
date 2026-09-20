import streamlit as st
import numpy as np
import socket
import threading
import collections
import time
import plotly.graph_objects as go
from datetime import datetime

UDP_IP = "127.0.0.1"
UDP_PORT = 5005
BUFFER_SIZE = 200
REFRESH_MS = 50

@st.cache_resource
def start_udp_listener():
    buf = collections.deque(maxlen=BUFFER_SIZE)
    stats = {"received": 0, "last_ts": 0.0, "connected": False, "errors": 0}

    def _listener():
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        try:
            sock.bind((UDP_IP, UDP_PORT))
            stats["connected"] = True
        except OSError:
            stats["connected"] = False
            return

        sock.settimeout(0.5)

        while True:
            try:
                data, _ = sock.recvfrom(4096)
                recv_ts = time.time()
                fields = data.decode("utf-8", errors="ignore").strip().split(";")

                if len(fields) < 7:
                    continue

                raw = list(map(float, fields[:7]))

                packet = {
                    "f": raw[0],
                    "v": raw[1],
                    "pression": raw[2],
                    "temp": raw[3],
                    "hi": float(np.clip(raw[4], 0.0, 1.0)),
                    "rul": max(0.0, raw[5]),
                    "hauteur": raw[6],
                    "panne": fields[7].strip() if len(fields) >= 8 else "Normal",
                    "ts": recv_ts,
                }

                buf.append(packet)
                stats["received"] += 1
                stats["last_ts"] = recv_ts

            except socket.timeout:
                pass
            except Exception:
                stats["errors"] += 1

    t = threading.Thread(target=_listener, daemon=True)
    t.start()
    return buf, stats

if "hi_history" not in st.session_state:
    st.session_state.hi_history = [1.0] * 120
    st.session_state.rul_history = [200.0] * 120
    st.session_state.last_packet = {
        "f": 14.0,
        "v": 0.5,
        "pression": 7.0,
        "temp": 60.0,
        "hi": 1.0,
        "rul": 200.0,
        "hauteur": 3.2,
        "panne": "Normal",
        "ts": time.time(),
    }

st.set_page_config(
    page_title="ONCF — PantoPrognosis",
    page_icon="🚄",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"], .stApp {
    min-height:100vh;
    background-color:#0e1117 !important;
    color:#e0e0e0;
    font-size:18px !important;
}
[data-testid="stHeader"] { background:rgba(0,0,0,0); height:2rem; }
.block-container {
    max-width:100% !important;
    padding-top:0.35rem !important;
    padding-bottom:0.25rem !important;
    padding-left:1.4rem !important;
    padding-right:1.4rem !important;
    min-height:calc(100vh - 2rem);
}
.hdr {
    background:linear-gradient(90deg,#1a1c23,#0e1117);
    padding:22px 26px;
    border-radius:14px;
    border-left:5px solid #00d4ff;
    margin-bottom:14px;
    display:flex;
    justify-content:space-between;
    align-items:center;
    min-height:115px;
}
.hdr h1 { font-size:2.55rem !important; }
.hdr p { font-size:1.18rem !important; }
.kpi-box {
    background:rgba(255,255,255,0.04);
    border:1px solid rgba(255,255,255,0.09);
    border-radius:14px;
    padding:24px 16px;
    text-align:center;
    min-height:150px;
}
.kpi-label {
    font-size:1.18rem !important;
    text-transform:uppercase;
    letter-spacing:1.5px;
    color:#aeb7c2;
    font-weight:650;
    margin-bottom:8px;
}
.kpi-value {
    font-size:2.85rem !important;
    font-weight:750;
}
.kpi-badge {
    display:inline-block;
    font-size:1rem !important;
    padding:5px 13px;
    border-radius:20px;
    margin-top:7px;
    font-weight:700;
}
.panel {
    background:#1a1c23;
    border-radius:14px;
    padding:14px;
    border:1px solid #2d2f39;
}
.ptitle {
    color:#00d4ff;
    font-size:1.35rem !important;
    font-weight:750;
    margin-bottom:8px;
}
.lag-bar {
    display:flex;
    gap:16px;
    align-items:center;
    font-size:1.05rem !important;
    color:#888;
    margin-top:6px;
}
.lag-item { display:flex; align-items:center; gap:5px; }
.lag-dot { width:8px; height:8px; border-radius:50%; }
[data-testid="stMetricLabel"] p {
    font-size:1.08rem !important;
    font-weight:650 !important;
}
[data-testid="stMetricValue"] {
    font-size:2.35rem !important;
    font-weight:750 !important;
}
div[data-testid="stVerticalBlock"] { gap:0.65rem; }
footer { visibility:hidden; height:0 !important; }
</style>
""", unsafe_allow_html=True)

buf, stats = start_udp_listener()

new_packets = []
while buf:
    new_packets.append(buf.popleft())

if new_packets:
    p = new_packets[-1]
    st.session_state.last_packet = p
    udp_state = "LIVE"
else:
    p = st.session_state.last_packet
    udp_state = "TIMEOUT" if time.time() - p["ts"] > 2.0 else "CACHE"

f = p["f"]
v = p["v"]
pression = p["pression"]
temp_carbone = p["temp"]
hi = p["hi"]
rul = p["rul"]
hauteur = p["hauteur"]
nom_panne = p.get("panne", "Normal").lower()

if "arc" in nom_panne or "electrique" in nom_panne:
    fault_type = "Arc électrique"
elif "fuite" in nom_panne:
    fault_type = "Fuite d'air"
elif "fissure" in nom_panne:
    fault_type = "Fissure"
else:
    fault_type = "Normal"

if fault_type == "Normal":
    status_text = "✅ NORMAL"
    status_short = "OPÉRATIONNEL"
    sc = "#00ff88"
else:
    status_text = "🚨 PANNE DÉTECTÉE"
    status_short = "CRITIQUE"
    sc = "#ff4b4b"

rul_color = "#ff4b4b" if rul < 30 else ("#ffaa00" if rul < 80 else "#00d4ff")
fill_hi = "rgba(0,255,136,0.12)" if fault_type == "Normal" else "rgba(255,75,75,0.12)"
fill_rul = "rgba(255,75,75,0.12)" if rul < 30 else ("rgba(255,170,0,0.12)" if rul < 80 else "rgba(0,212,255,0.12)")

st.session_state.hi_history.append(hi)
st.session_state.rul_history.append(rul)
st.session_state.hi_history = st.session_state.hi_history[-240:]
st.session_state.rul_history = st.session_state.rul_history[-240:]

age_ms = (time.time() - p["ts"]) * 1000.0
pkt_recv = stats.get("received", 0)

badge_color = {"LIVE": "#00ff88", "CACHE": "#888", "TIMEOUT": "#ff4b4b"}.get(udp_state, "#888")
lag_color = "#00ff88" if age_ms < 150 else ("#ffaa00" if age_ms < 500 else "#ff4b4b")

st.markdown(f"""
<div class="hdr">
  <div>
    <h1 style='margin:0;'>🚄 ONCF <span style='color:#00d4ff'>PantoPrognosis</span></h1>
    <p style='margin:5px 0 0;color:#888;'>
      Maintenance Prédictive — LGV Al Boraq &nbsp;|&nbsp;
      {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}
    </p>
    <div class="lag-bar">
      <div class="lag-item"><span class="lag-dot" style="background:{badge_color}"></span><span style="color:{badge_color};font-weight:700">{udp_state}</span></div>
      <div class="lag-item"><span class="lag-dot" style="background:{lag_color}"></span><span>Délai : <b style="color:{lag_color}">{age_ms:.0f} ms</b></span></div>
      <div class="lag-item">Paquets reçus : <b style="color:#00d4ff">{pkt_recv}</b></div>
    </div>
  </div>
  <div style='background:{sc}1a;color:{sc};padding:13px 34px;border-radius:30px;border:1px solid {sc}55;font-weight:700;font-size:1.3rem;text-align:center;'>
    {status_text}
  </div>
</div>
""", unsafe_allow_html=True)

c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    st.markdown(f"""<div class="kpi-box"><div class="kpi-label">RUL estimée*</div><div class="kpi-value" style="color:{rul_color}">{rul:.0f}<small style="font-size:1.2rem"> min</small></div><span class="kpi-badge" style="color:{rul_color};background:{rul_color}18;border:1px solid {rul_color}44;">{"CRITIQUE" if rul < 30 else ("ALERTE" if rul < 80 else "NOMINAL")}</span></div>""", unsafe_allow_html=True)

with c2:
    st.markdown(f"""<div class="kpi-box"><div class="kpi-label">État de santé (HI)</div><div class="kpi-value" style="color:{sc}">{hi*100:.1f}%</div><span class="kpi-badge" style="color:{sc};background:{sc}18;border:1px solid {sc}44;">{status_short}</span></div>""", unsafe_allow_html=True)

with c3:
    st.markdown(f"""<div class="kpi-box"><div class="kpi-label">Diagnostic AMDEC</div><div class="kpi-value" style="font-size:1.8rem !important;margin-top:16px;color:{sc};">{fault_type}</div></div>""", unsafe_allow_html=True)

with c4:
    st.markdown(f"""<div class="kpi-box"><div class="kpi-label">Force de contact</div><div class="kpi-value">{f:.1f}<small style="font-size:1.2rem"> N</small></div></div>""", unsafe_allow_html=True)

with c5:
    p_color = "#ff4b4b" if pression < 5.5 else "#e0e0e0"
    st.markdown(f"""<div class="kpi-box"><div class="kpi-label">Pression coussin</div><div class="kpi-value" style="color:{p_color}">{pression:.2f}<small style="font-size:1.2rem"> bar</small></div></div>""", unsafe_allow_html=True)

g1, g2, g3 = st.columns([3, 2, 1])

with g1:
    st.markdown('<div class="panel"><div class="ptitle">📈 Health Index — Évolution Temps Réel</div>', unsafe_allow_html=True)
    fig_hi = go.Figure()
    fig_hi.add_hrect(y0=0, y1=0.40, fillcolor="rgba(255,75,75,0.07)", line_width=0, annotation_text="Zone critique", annotation_font_size=14, annotation_font_color="#ff4b4b")
    fig_hi.add_hrect(y0=0.40, y1=0.75, fillcolor="rgba(255,170,0,0.05)", line_width=0, annotation_text="Zone alerte", annotation_font_size=14, annotation_font_color="#ffaa00")
    fig_hi.add_trace(go.Scatter(y=st.session_state.hi_history, mode="lines", line=dict(color=sc, width=2.8), fill="tozeroy", fillcolor=fill_hi))
    fig_hi.update_layout(height=385, margin=dict(l=0,r=0,t=8,b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", xaxis=dict(visible=False), yaxis=dict(range=[0,1.05], tickformat=".0%", gridcolor="rgba(255,255,255,0.04)"), showlegend=False, font=dict(size=15, color="#aeb7c2"))
    st.plotly_chart(fig_hi, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with g2:
    st.markdown('<div class="panel"><div class="ptitle">⏱️ RUL — Indicateur du prototype</div>', unsafe_allow_html=True)
    fig_rul = go.Figure()
    fig_rul.add_hrect(y0=0, y1=30, fillcolor="rgba(255,75,75,0.08)", line_width=0)
    fig_rul.add_hrect(y0=30, y1=80, fillcolor="rgba(255,170,0,0.06)", line_width=0)
    fig_rul.add_trace(go.Scatter(y=st.session_state.rul_history, mode="lines", line=dict(color=rul_color, width=2.8), fill="tozeroy", fillcolor=fill_rul))
    fig_rul.update_layout(height=385, margin=dict(l=0,r=0,t=8,b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", xaxis=dict(visible=False), yaxis=dict(range=[0,210], gridcolor="rgba(255,255,255,0.04)"), showlegend=False, font=dict(size=15, color="#aeb7c2"))
    st.plotly_chart(fig_rul, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with g3:
    st.markdown('<div class="panel"><div class="ptitle">👁️ État visuel</div>', unsafe_allow_html=True)
    cam = np.zeros((270, 270, 3), dtype=np.uint8) + 18
    cam[133:137, :] = [0, 180, 220]
    cam[:, 133:137] = [0, 180, 220]

    if fault_type == "Arc électrique":
        cam[90:180, 105:165] = [200, 40, 40]
    elif fault_type == "Fissure":
        cam[35:235, 130:140] = [220, 40, 40]
    elif fault_type == "Fuite d'air":
        cam[105:165, 105:165] = [220, 40, 40]

    st.image(cam, use_container_width=True)
    st.markdown(f"""<div style='margin-top:10px;font-family:monospace;font-size:1.02rem;color:#aeb7c2;line-height:1.45;'>ÉTAT : <span style='color:{sc};font-weight:700'>{fault_type}</span><br>Hauteur : {hauteur:.2f} m</div>""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="panel" style="min-height:160px;">', unsafe_allow_html=True)

t1, t2, t3, t4, t5 = st.columns(5)
t1.metric("Force Contact", f"{f:.1f} N")
t2.metric("Vibrations RMS", f"{v:.3f} g")
t3.metric("Temp. Carbone", f"{temp_carbone:.1f} °C")
t4.metric("Pression Coussin", f"{pression:.2f} bar")
t5.metric("Vitesse Al Boraq", "320 km/h")

st.markdown("</div>", unsafe_allow_html=True)

if udp_state == "TIMEOUT":
    st.error("Perte du flux UDP : les dernières valeurs reçues restent affichées.")

st.caption("* La RUL affichée est un indicateur expérimental du prototype.")

time.sleep(REFRESH_MS / 1000.0)
st.rerun()
