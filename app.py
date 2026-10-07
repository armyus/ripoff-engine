import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
from features import TitleFeatureExtractor

st.set_page_config(page_title="The Rip-Off Engine", layout="wide", initial_sidebar_state="expanded")

# ==========================================
# 1. CUSTOM CSS: Cyberpunk / Retro Terminal
# ==========================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;900&family=Inter:wght@400;600;800&display=swap');

    /* Global Dark/Obsidian Background */
    .stApp {
        background-color: #0d0f17 !important;
        background-image: 
            linear-gradient(rgba(13, 15, 23, 0.95), rgba(13, 15, 23, 0.95)),
            linear-gradient(90deg, rgba(255,255,255,0.03) 1px, transparent 1px),
            linear-gradient(rgba(255,255,255,0.03) 1px, transparent 1px) !important;
        background-size: 100% 100%, 40px 40px, 40px 40px !important;
        color: #e0e5ff;
        font-family: 'Inter', sans-serif;
    }
    
    /* Remove padding & sterile gray areas */
    .css-18e3th9, .css-1d391kg { padding-top: 1rem; }
    header[data-testid="stHeader"] { background: transparent; }
    
    /* Titles & Headers */
    h1, h2, h3 { font-family: 'Orbitron', sans-serif; }
    .neon-title {
        text-align: center;
        font-size: 3.5rem;
        font-weight: 900;
        text-transform: uppercase;
        background: -webkit-linear-gradient(#00c8ff, #00f59b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 20px rgba(0, 200, 255, 0.3);
        margin-bottom: 0.2rem;
    }
    .neon-subtitle {
        text-align: center;
        font-family: 'Courier New', Courier, monospace;
        color: #8c9eff;
        font-size: 1.1rem;
        margin-bottom: 3rem;
        letter-spacing: 1px;
    }

    /* Glassmorphism Cards */
    .glass-card {
        background: rgba(22, 27, 46, 0.75);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        text-align: center;
    }
    
    /* Neon Verdict Cards & Pulsing */
    @keyframes pulse-steal { 0% { box-shadow: 0 0 15px #00f59b; } 50% { box-shadow: 0 0 30px #00f59b, inset 0 0 15px rgba(0,245,155,0.2); } 100% { box-shadow: 0 0 15px #00f59b; } }
    @keyframes pulse-fair { 0% { box-shadow: 0 0 15px #00c8ff; } 50% { box-shadow: 0 0 30px #00c8ff, inset 0 0 15px rgba(0,200,255,0.2); } 100% { box-shadow: 0 0 15px #00c8ff; } }
    @keyframes pulse-ripoff { 0% { box-shadow: 0 0 15px #ff2e55; } 50% { box-shadow: 0 0 35px #ff2e55, inset 0 0 15px rgba(255,46,85,0.2); } 100% { box-shadow: 0 0 15px #ff2e55; } }
    
    .verdict-steal { animation: pulse-steal 2s infinite; border: 2px solid #00f59b; background: rgba(0, 245, 155, 0.1); }
    .verdict-fair { animation: pulse-fair 2s infinite; border: 2px solid #00c8ff; background: rgba(0, 200, 255, 0.1); }
    .verdict-ripoff { animation: pulse-ripoff 2s infinite; border: 2px solid #ff2e55; background: rgba(255, 46, 85, 0.15); }
    
    .verdict-title { font-size: 3.5rem; margin: 0; font-family: 'Orbitron', sans-serif; letter-spacing: 2px; }
    .steal-text { color: #00f59b; text-shadow: 0 0 10px #00f59b; }
    .fair-text { color: #00c8ff; text-shadow: 0 0 10px #00c8ff; }
    .ripoff-text { color: #ff2e55; text-shadow: 0 0 10px #ff2e55; }

    /* Pill Badges */
    .badge {
        display: inline-block; padding: 6px 14px; margin: 4px; border-radius: 20px;
        font-family: 'Orbitron', sans-serif; font-size: 0.85rem; font-weight: 700; letter-spacing: 1px;
    }
    .badge-pos { background: rgba(0, 245, 155, 0.15); color: #00f59b; border: 1px solid #00f59b; box-shadow: 0 0 8px rgba(0,245,155,0.4); }
    .badge-neg { background: rgba(255, 46, 85, 0.15); color: #ff2e55; border: 1px solid #ff2e55; box-shadow: 0 0 8px rgba(255,46,85,0.4); }
    .badge-dim { background: rgba(255, 255, 255, 0.05); color: #6c757d; border: 1px solid #343a40; }

    /* Metrics Values */
    .metric-value { font-size: 2.5rem; font-weight: 800; color: #ffffff; margin-top: 10px; font-family: 'Orbitron', sans-serif; }
    .metric-label { font-size: 0.9rem; font-weight: 600; color: #8c9eff; text-transform: uppercase; letter-spacing: 1.5px; }
    .delta-pos { color: #ff2e55; font-size: 1.2rem; text-shadow: 0 0 5px rgba(255,46,85,0.5); }
    .delta-neg { color: #00f59b; font-size: 1.2rem; text-shadow: 0 0 5px rgba(0,245,155,0.5); }

</style>
""", unsafe_allow_html=True)

# Header
st.markdown("<div class='neon-title'>THE RIP-OFF ENGINE</div>", unsafe_allow_html=True)
st.markdown("<div class='neon-subtitle'>/// FAIR MARKET VALUATION & DEAL CLASSIFICATION SYSTEM ///</div>", unsafe_allow_html=True)

# ==========================================
# 2. MODEL LOADING & STATE MANAGEMENT
# ==========================================
@st.cache_resource
def load_models():
    return joblib.load('models/preprocessor.pkl'), joblib.load('models/best_model.pkl')

try:
    preprocessor, model = load_models()
except Exception as e:
    st.error(f"SYSTEM FAILURE: Missing Neural Link (Models). Error: {e}")
    st.stop()

# Initialize session state for inputs
if 'title' not in st.session_state:
    st.session_state.title = "Pokemon Emerald GBA Complete in Box CIB Authentic Tested Mint"
    st.session_state.platform = "N64"
    st.session_state.condition = "Mint"
    st.session_state.feedback = 99.0
    st.session_state.ratings = 350
    st.session_state.price = 45.0

def load_preset(ptype):
    if ptype == "steal":
        st.session_state.title = "Zelda Ocarina of Time N64 CIB Tested Working"
        st.session_state.platform = "N64"
        st.session_state.condition = "Mint"
        st.session_state.feedback = 98.0
        st.session_state.ratings = 120
        st.session_state.price = 25.0
    else:
        st.session_state.title = "Super Mario Bros NES Loose Flaw Scratch Untested L@@K"
        st.session_state.platform = "NES"
        st.session_state.condition = "Untested"
        st.session_state.feedback = 82.0
        st.session_state.ratings = 12
        st.session_state.price = 199.99

# ==========================================
# 3. SIDEBAR CONTROLS
# ==========================================
with st.sidebar:
    st.markdown("<h2 style='font-family: Orbitron; color: #fff;'>SYS_CONTROL</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.button("⚡ STEAL DEAL", on_click=load_preset, args=("steal",), use_container_width=True)
    with col2:
        st.button("💀 SCALPER", on_click=load_preset, args=("ripoff",), use_container_width=True)
    
    st.markdown("---")
    
    with st.form("inference_form"):
        title_input = st.text_area("TARGET_TITLE", value=st.session_state.title)
        
        platforms = ['NES', 'SNES', 'N64', 'GameCube', 'PS1', 'PS2', 'Sega Genesis']
        platform_idx = platforms.index(st.session_state.platform) if st.session_state.platform in platforms else 2
        platform_input = st.selectbox("HARDWARE_PLATFORM", options=platforms, index=platform_idx)
        
        conditions = ['Mint', 'Good', 'Fair', 'Poor', 'Untested']
        cond_idx = conditions.index(st.session_state.condition) if st.session_state.condition in conditions else 0
        condition_input = st.selectbox("ASSET_CONDITION", options=conditions, index=cond_idx)
        
        feedback_input = st.slider("SELLER_TRUST_INDEX (%)", 50.0, 100.0, float(st.session_state.feedback), 0.1)
        ratings_input = st.number_input("SELLER_REP_COUNT", 0, 5000, int(st.session_state.ratings))
        price_input = st.number_input("REQUESTED_FUNDS ($)", 0.0, 10000.0, float(st.session_state.price), 1.0)
        
        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button(">> EXECUTE SCAN", use_container_width=True, type="primary")

# ==========================================
# 4. INFERENCE & VISUALIZATION
# ==========================================
if submitted:
    # Prepare data for pipeline
    input_df = pd.DataFrame([{
        'title': title_input,
        'platform': platform_input,
        'condition': condition_input,
        'seller_feedback_pct': feedback_input,
        'seller_ratings_count': ratings_input,
        'listed_price': price_input
    }])
    
    # Predict
    X_processed = preprocessor.transform(input_df)
    predicted_fair_value = float(model.predict(X_processed)[0])
    
    # Logic
    diff = price_input - predicted_fair_value
    diff_pct = (diff / predicted_fair_value) * 100 if predicted_fair_value > 0 else 0
    
    # Determine Classifications
    if price_input <= predicted_fair_value * 0.8:
        v_class = "verdict-steal"
        v_text_class = "steal-text"
        v_label = "STEAL! 🔥"
        gauge_color = "#00f59b"
    elif price_input >= predicted_fair_value * 1.2:
        v_class = "verdict-ripoff"
        v_text_class = "ripoff-text"
        v_label = "RIP-OFF! 🛑"
        gauge_color = "#ff2e55"
    else:
        v_class = "verdict-fair"
        v_text_class = "fair-text"
        v_label = "FAIR DEAL ⚖️"
        gauge_color = "#00c8ff"

    # Hero Verdict Card
    st.markdown(f"""
    <div class="glass-card {v_class}" style="margin-bottom: 30px; padding: 40px;">
        <h1 class="verdict-title {v_text_class}">{v_label}</h1>
        <div style="color: #fff; font-family: 'Courier New', monospace; margin-top: 10px; opacity: 0.8;">
            ANALYSIS COMPLETE // VARIANCE: {diff_pct:+.1f}%
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # 3-Column Glass Metrics
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""
        <div class="glass-card">
            <div class="metric-label">ASKING PRICE</div>
            <div class="metric-value">${price_input:.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="glass-card" style="border-color: rgba(0, 200, 255, 0.4); box-shadow: 0 0 15px rgba(0,200,255,0.1);">
            <div class="metric-label" style="color: #00c8ff;">TRUE FAIR VALUE</div>
            <div class="metric-value">${predicted_fair_value:.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        if diff > 0:
            delta_class, delta_text, arrow = "delta-pos", "OVERPAYMENT", "▲"
        else:
            delta_class, delta_neg, arrow = "delta-neg", "SAVINGS", "▼"
            
        st.markdown(f"""
        <div class="glass-card">
            <div class="metric-label">NET {delta_text if diff > 0 else 'SAVINGS'}</div>
            <div class="metric-value {delta_class if diff > 0 else 'delta-neg'}">
                {arrow} ${abs(diff):.2f} <span style="font-size:1.2rem;">({abs(diff_pct):.1f}%)</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Bottom Row: Plotly Gauge & Keyword Pill Badges
    bottom_c1, bottom_c2 = st.columns([1.2, 1])
    
    with bottom_c1:
        st.markdown("<h3 style='color: #fff; font-size: 1.2rem; text-align: center;'>PRICE SEVERITY GAUGE</h3>", unsafe_allow_html=True)
        
        # Max scale is either 200% of fair value or asking price
        max_val = max(predicted_fair_value * 2.0, price_input * 1.2)
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = price_input,
            delta = {'reference': predicted_fair_value, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
            number = {'prefix': "$", 'font': {'color': gauge_color, 'family': 'Orbitron', 'size': 40}},
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "", 'font': {'size': 24}},
            gauge = {
                'axis': {'range': [0, max_val], 'tickwidth': 1, 'tickcolor': "white"},
                'bar': {'color': "rgba(255,255,255,0.8)", 'thickness': 0.15},
                'bgcolor': "rgba(0,0,0,0)",
                'borderwidth': 0,
                'steps': [
                    {'range': [0, predicted_fair_value * 0.8], 'color': "rgba(0, 245, 155, 0.2)"},
                    {'range': [predicted_fair_value * 0.8, predicted_fair_value * 1.2], 'color': "rgba(0, 200, 255, 0.2)"},
                    {'range': [predicted_fair_value * 1.2, max_val], 'color': "rgba(255, 46, 85, 0.2)"}],
                'threshold': {
                    'line': {'color': "#00c8ff", 'width': 4},
                    'thickness': 0.75,
                    'value': predicted_fair_value}
            }
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={'color': "white", 'family': "Inter"},
            height=300,
            margin=dict(l=20, r=20, t=10, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)

    with bottom_c2:
        st.markdown("<h3 style='color: #fff; font-size: 1.2rem; margin-bottom: 20px;'>NEURAL ENTITY EXTRACTION</h3>", unsafe_allow_html=True)
        st.markdown("<div class='glass-card' style='text-align: left; padding: 20px; height: 300px;'>", unsafe_allow_html=True)
        
        # NLP Feature Engine mapping
        raw_keywords = preprocessor.named_transformers_['title'].named_steps['extractor'].keywords
        title_lower = str(title_input).lower()
        
        pos_flags = ['cib', 'box', 'tested', 'working', 'rare', 'mint']
        neg_flags = ['scratch', 'flaw', 'loose', 'poor', 'untested']
        
        badge_html = ""
        
        # We append Condition dynamically to the UI checks as well
        all_eval_words = raw_keywords + [c.lower() for c in ['Mint', 'Poor', 'Untested']]
        all_eval_words = list(set(all_eval_words)) # deduplicate
        
        for kw in all_eval_words:
            is_present = (kw in title_lower) or (kw == condition_input.lower())
            
            if is_present:
                if kw in pos_flags:
                    badge_html += f"<div class='badge badge-pos'>{kw.upper()} ✓</div>"
                elif kw in neg_flags:
                    badge_html += f"<div class='badge badge-neg'>{kw.upper()} ⚠</div>"
                else:
                    # Neutral present
                    badge_html += f"<div class='badge badge-pos' style='color: #fff; border-color:#fff;'>{kw.upper()}</div>"
            else:
                badge_html += f"<div class='badge badge-dim'>{kw.upper()}</div>"
                
        st.markdown(badge_html, unsafe_allow_html=True)
        
        st.markdown("<hr style='border-color: rgba(255,255,255,0.1);'>", unsafe_allow_html=True)
        st.markdown(f"<span style='color: #8c9eff; font-size: 0.85rem;'>» BASE PLATFORM ID: </span> <span style='color: #fff; font-weight: bold;'>{platform_input}</span>", unsafe_allow_html=True)
        st.markdown(f"<span style='color: #8c9eff; font-size: 0.85rem;'>» HARDWARE STATE: </span> <span style='color: #fff; font-weight: bold;'>{condition_input.upper()}</span>", unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)

else:
    st.markdown("""
    <div style='text-align: center; margin-top: 50px;'>
        <h3 style='color: #8c9eff; font-family: Courier New;'>/// AWAITING INPUT DATA ///</h3>
        <p style='color: #6c757d;'>Initialize scan sequence via the side-panel to evaluate target listing.</p>
    </div>
    """, unsafe_allow_html=True)
