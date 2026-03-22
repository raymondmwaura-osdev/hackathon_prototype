"""
client_app.py
--------------
Customer-facing portal for the HEFI system.
Features:
1. Login with Household ID
2. Data Update Wizard
3. Personal HEFI Dashboard
4. AI Support Chatbot
"""

import streamlit as st
import sqlite3
import pandas as pd
import os
import time
from fairness_index import init_db, run_pipeline, upsert_households, recalculate_with_context, DB_PATH
from chatbot_logic import get_chatbot_response

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="Citizen HEFI Portal", page_icon="🏠", layout="wide")

# Theme selector
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

# Theme toggle (visible at top, not hidden in sidebar)
selected_theme = st.radio(
    "Theme",
    ["Light", "Dark"],
    index=0 if st.session_state.theme == "light" else 1,
    horizontal=True,
    key="theme_mode_select",
)
st.session_state.theme = selected_theme.lower()

# Apply theme CSS

def _get_theme_values(mode: str):
    if mode == "dark":
        return {
            "app_bg": "#0d1117",
            "text": "#c9d1d9",
            "panel": "#161b22",
            "border": "#30363d",
            "card": "#161b22",
            "card_shadow": "rgba(0,0,0,0.5)",
            "nav_bg": "#05204b",
            "nav_text": "white",
            "nav_shadow": "rgba(0,0,0,0.4)",
            "muted": "#8b949e",
            "link": "#58a6ff",
            "input_bg": "#161b22",
            "input_text": "#c9d1d9",
            "input_border": "#30363d",
            "button_start": "#238636",
            "button_end": "#2ea043",
            "button_text": "white",
            "hero_sub": "#94a3b8",
            "footer": "#8b949e",
        }
    return {
        "app_bg": "#f4f6ff",
        "text": "#1f2937",
        "panel": "#ffffff",
        "border": "#e2e8f0",
        "card": "#ffffff",
        "card_shadow": "rgba(0,0,0,0.06)",
        "nav_bg": "#002d72",
        "nav_text": "white",
        "nav_shadow": "rgba(0,0,0,0.2)",
        "muted": "#4b5563",
        "link": "#0e60d2",
        "input_bg": "#ffffff",
        "input_text": "#1f2937",
        "input_border": "#cbd5e1",
        "button_start": "#0e60d2",
        "button_end": "#0d359a",
        "button_text": "white",
        "hero_sub": "#4b5563",
        "footer": "#6b7280",
    }

_theme = _get_theme_values(st.session_state.theme)

st.markdown(f"""
    <style>
    /* Layout */
    .stApp {{ background: {_theme['app_bg']}; color: {_theme['text']}; }}
    .top-nav {{ background: {_theme['nav_bg']}; padding: 14px 28px; display: flex; align-items: center; justify-content: space-between; color: {_theme['nav_text']}; box-shadow: 0 4px 14px {_theme['nav_shadow']}; border-radius: 0 0 16px 16px; }}
    .top-nav .brand {{ display: flex; align-items: center; gap: 10px; font-size: 1.1rem; font-weight: 700; }}
    .top-nav .brand span {{ font-size: 1.5rem; }}
    .top-nav .nav-links a {{ color: rgba(255, 255, 255, 0.9); text-decoration: none; margin-left: 18px; font-weight: 600; }}
    .top-nav .nav-links a:hover {{ color: #ffee77; }}

    /* Cards */
    .info-card {{ background: {_theme['card']}; border-radius: 14px; padding: 22px; margin-bottom: 18px; box-shadow: 0 10px 25px {_theme['card_shadow']}; }}
    .info-card h3 {{ margin-top: 0; color: {_theme['nav_bg']}; }}
    .info-card p {{ margin: 10px 0; color: {_theme['muted']}; }}

    .feature-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 12px; margin-top: 14px; }}
    .feature-card {{ background: {_theme['button_end']}; border: 1px solid {_theme['button_start']}; color: white; padding: 14px; border-radius: 14px; box-shadow: 0 12px 28px rgba(0,0,0,0.3); min-height: 96px; }}
    .feature-card strong {{ display: block; font-size: 1rem; margin-bottom: 6px; }}

    .step-badge {{ display: inline-block; background: rgba(0, 45, 114, 0.1); color: {_theme['nav_bg']}; border-radius: 999px; padding: 6px 14px; font-weight: 700; margin-right: 10px; margin-bottom: 10px; }}

    /* Forms */
    .stTextInput>div>div>input {{ background-color: {_theme['input_bg']}; color: {_theme['input_text']}; border: 1px solid {_theme['input_border']}; border-radius: 8px; }}
    .stNumberInput>div>div>input {{ background-color: {_theme['input_bg']}; color: {_theme['input_text']}; border: 1px solid {_theme['input_border']}; border-radius: 8px; }}
    .stSelectbox>div>div>div>div:first-child {{ background: {_theme['input_bg']}; border: 1px solid {_theme['input_border']}; border-radius: 8px; }}

    /* Buttons */
    .stButton>button {{
        width: 100%; border-radius: 8px; height: 3.2em;
        background: linear-gradient(90deg, {_theme['button_start']}, {_theme['button_end']});
        color: {_theme['button_text']}; border: none; font-weight: 700; letter-spacing: 0.5px;
    }}
    .stButton>button:hover {{ opacity: 0.92; }}

    /* Text */
    .hero-title {{ font-size: 2.4rem; margin: 0; }}
    .hero-sub {{ color: {_theme['hero_sub']}; margin-top: 10px; margin-bottom: 18px; }}

    /* Dashboard Cards */
    .user-card {{ background-color: {_theme['card']}; padding: 20px; border-radius: 16px; border: 1px solid {_theme['border']}; margin-bottom: 20px; }}
    .hefi-badge {{ font-size: 2.4em; font-weight: bold; color: {_theme['nav_bg']}; text-align: center; }}
    .chat-bubble {{ background-color: {_theme['card']}; padding: 12px; border-radius: 10px; margin: 5px 0; border: 1px solid {_theme['border']}; }}

    /* Footer */
    .footer {{ color: {_theme['footer']}; font-size: 0.9rem; text-align: center; padding-top: 14px; }}
    @media (max-width: 940px) {{
        .top-nav {{ flex-direction: column; align-items: start; gap: 10px; padding: 12px; }}
        .top-nav .nav-links {{ width: 100%; display: flex; flex-wrap: wrap; gap: 10px; }}
        .top-nav .nav-links a {{ margin-left: 0; padding: 8px 10px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.15); }}
        .hero-title {{ font-size: 1.8rem; }}
    }}
    </style>
    """, unsafe_allow_html=True)

# Setup navigation state defaults
if "view" not in st.session_state:
    st.session_state.view = "landing"
if "login_mode" not in st.session_state:
    st.session_state.login_mode = False

# ─── Session State ───────────────────────────────────────────────────────────
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Ensure the registry exists so users can register and data is persisted
init_db()

# ─── Helper Functions ────────────────────────────────────────────────────────
def get_user_data(hid):
    if not os.path.exists(DB_PATH):
        return None
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql("SELECT * FROM households WHERE household_id = ?", conn, params=(hid,))
    finally:
        conn.close()
    return df.iloc[0].to_dict() if not df.empty else None

# ─── Landing / Login / Registration ──────────────────────────────────────────
if not st.session_state.authenticated:
    if "view" not in st.session_state:
        st.session_state.view = "landing"

    st.markdown(
        """
        <div style='display:flex; align-items:center; gap:16px; flex-wrap:wrap;'>
            <div style='flex:1; min-width:320px;'>
                <h1 style='margin:0;'>🏠 Citizen Energy Portal</h1>
                <p style='margin:4px 0 12px; color: {_theme["muted"]};'>Your gateway to fair energy tariffs, personalized HEFI insights, and practical savings advice.</p>
                <p style='margin:0; color:#8b949e;'>HEFI is designed to ensure that households with the greatest need receive the most support.</p>
            </div>
            <div style='flex:0 0 220px;'>
                <div style='background:linear-gradient(90deg,{_theme["button_start"]},{_theme["button_end"]}); padding:12px 14px; border-radius:14px; color:{_theme["button_text"]}; font-weight:700; text-align:center;'>
                    New: Register and view your tariff tier instantly
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div class='info-card'>", unsafe_allow_html=True)
    st.markdown("<h3>Why HEFI matters</h3>", unsafe_allow_html=True)
    st.markdown(
        "HEFI helps utilities and communities ensure electricity pricing is fair. "
        "By understanding your household's energy needs, HEFI matches tariff support to those who need it most."
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='info-card'>", unsafe_allow_html=True)
    st.markdown("<h3>How it works</h3>", unsafe_allow_html=True)
    st.markdown(
        "<span class='step-badge'>1</span> Collect basic household information.<br>"
        "<span class='step-badge'>2</span> Compute a fairness score (HEFI) and assign a tariff tier.<br>"
        "<span class='step-badge'>3</span> Use your tier to access subsidies and budget support."
    , unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='info-card'>", unsafe_allow_html=True)
    st.markdown("<h3>Ready to get started?</h3>", unsafe_allow_html=True)
    st.markdown("Use the panel to the right to log in (if already registered) or register a new household.")
    st.markdown(
        "<div class='feature-grid'>"
        "<div class='feature-card'>"
        "<strong>✔️ Quick access</strong>Check your HEFI score instantly with clear guidance."
        "</div>"
        "<div class='feature-card'>"
        "<strong>📝 Update details</strong>Keep your profile up to date for fair calculations and better results."
        "</div>"
        "<div class='feature-card'>"
        "<strong>💬 Get support</strong>Chat with the HEFI assistant anytime for tips and help."
        "</div>"
        "</div>"
    , unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    col_info, col_actions = st.columns([2, 1])
    with col_actions:
        if st.session_state.view == "landing":
            if st.session_state.login_mode:
                st.info("🔒 Login mode active: enter your Household ID to access your dashboard.")
            st.markdown("### Already registered? Log in")
            hid_input = st.text_input("Household ID", placeholder="HH_001", key="login_hid")
            if st.button("Access Dashboard", key="login_btn"):
                user = get_user_data(hid_input)
                if user:
                    st.session_state.authenticated = True
                    st.session_state.user_id = hid_input
                    st.session_state.view = "dashboard"
                    st.rerun()
                else:
                    st.error("Household ID not found. Please register or contact your local utility office.")

            st.markdown("---")
            if st.button("Register for HEFI", key="goto_register"):
                st.session_state.view = "register"
                st.session_state.login_mode = False
                st.rerun()

        elif st.session_state.view == "register":
            st.markdown("### Register a new household")
            st.markdown("Please provide accurate information so HEFI can compute a fair tariff recommendation.")
            with st.form("register_form"):
                hid = st.text_input("Choose a Household ID (e.g., HH_001)", placeholder="HH_XXX")
                col_a, col_b = st.columns(2)
                with col_a:
                    consumption = st.number_input("Monthly Consumption (kWh)", min_value=0.0, value=120.0, step=1.0)
                    income = st.number_input("Monthly Income (₹)", min_value=0, value=10000, step=500)
                    size = st.number_input("Household Size", min_value=1, value=4, step=1)
                with col_b:
                    location = st.selectbox("Location", ["Urban", "Rural"])
                    appliances = st.number_input("Appliance Count", min_value=0, value=5, step=1)
                    renewable = st.selectbox("Renewable Energy Access", ["Yes", "No"])
                    dependency = st.slider("Electricity Dependency", 0, 10, 5)

                register = st.form_submit_button("Register and Calculate HEFI")
                if register:
                    hid = hid.strip()
                    if not hid:
                        st.error("Please enter a valid Household ID.")
                    else:
                        existing = get_user_data(hid)
                        if existing:
                            st.error("Household ID already registered. Please log in instead.")
                        else:
                            new_record = {
                                "household_id": hid,
                                "monthly_electricity_consumption_kwh": float(consumption),
                                "household_income": int(income),
                                "household_size": int(size),
                                "urban_or_rural": location,
                                "appliance_count": int(appliances),
                                "renewable_energy_access": renewable,
                                "electricity_dependency_score": float(dependency),
                            }
                            with st.spinner("Registering household and calculating HEFI..."):
                                df_new = recalculate_with_context(pd.DataFrame([new_record]))
                                upsert_households(df_new)
                                time.sleep(1)

                            st.success(
                                f"Registered! Your HEFI is **{df_new.iloc[0]['hefi_score']}** and your tariff tier is **{df_new.iloc[0]['tariff_tier']}**."
                            )
                            st.session_state.authenticated = True
                            st.session_state.user_id = hid
                            st.session_state.view = "dashboard"
                            st.rerun()

            if st.button("Back to Home", key="back_to_landing"):
                st.session_state.view = "landing"
                st.session_state.login_mode = False
                st.rerun()

        else:
            st.session_state.view = "landing"

    st.stop()

# ─── Authenticated Dashboard ─────────────────────────────────────────────────
user_data = get_user_data(st.session_state.user_id)

# Sidebar Navigation
with st.sidebar:
    st.title(f"Welcome, {st.session_state.user_id}")
    st.markdown("---")
    menu = st.radio("Navigation", ["📈 My HEFI Status", "📝 Update My Details", "💬 Support Chat"])
    if st.button("Logout"):
        st.session_state.authenticated = False
        st.rerun()

# ─── TAB: Dashboard ──────────────────────────────────────────────────────────
if menu == "📈 My HEFI Status":
    st.title("📈 Your Household Energy Fairness Status")
    
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("<div class='user-card'><p style='text-align:center;'>Current HEFI Score</p>", unsafe_allow_html=True)
        st.markdown(f"<div class='hefi-badge'>{user_data['hefi_score']}</div></div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='user-card'><p style='text-align:center;'>Tax/Tariff Tier</p>", unsafe_allow_html=True)
        color = "#238636" if user_data['tariff_tier'] == "Subsidized" else "#d29922" if user_data['tariff_tier'] == "Standard" else "#f85149"
        st.markdown(f"<div class='hefi-badge' style='font-size:1.8em; color:{color};'>{user_data['tariff_tier']}</div></div>", unsafe_allow_html=True)
    with c3:
        st.markdown("<div class='user-card'><p style='text-align:center;'>Last Meter Reading</p>", unsafe_allow_html=True)
        st.markdown(f"<div class='hefi-badge' style='font-size:1.8em;'>{user_data['monthly_electricity_consumption_kwh']} <span style='font-size:0.5em;'>kWh</span></div></div>", unsafe_allow_html=True)

    st.markdown("### 📊 Fairness Breakdown")
    # Small bar chart for components
    components = {
        "Income Vuln": user_data['income_vulnerability'],
        "Household Size": user_data['household_size_factor'],
        "Dependency": user_data['energy_dependency'],
        "Anomaly": user_data['consumption_anomaly']
    }
    st.bar_chart(pd.Series(components))

    with st.expander("🔍 What-if simulator: see how changes affect your HEFI"):
        st.markdown(
            "Try changing your household consumption or income to see how your score and tariff tier could shift. "
            "This simulation does not change your saved data — use the **Update My Details** tab to persist changes."
        )
        sim_col_a, sim_col_b = st.columns(2)
        with sim_col_a:
            sim_consumption = st.number_input(
                "Simulated monthly consumption (kWh)",
                value=float(user_data['monthly_electricity_consumption_kwh']),
                min_value=0.0,
                step=1.0,
                key="sim_consumption",
            )
            sim_income = st.number_input(
                "Simulated monthly income (₹)",
                value=int(user_data['household_income']),
                min_value=0,
                step=500,
                key="sim_income",
            )
        with sim_col_b:
            sim_size = st.number_input(
                "Simulated household size",
                value=int(user_data['household_size']),
                min_value=1,
                step=1,
                key="sim_size",
            )
            sim_renewable = st.selectbox(
                "Simulated renewable access",
                ["Yes", "No"],
                index=0 if user_data['renewable_energy_access'] == "Yes" else 1,
                key="sim_renewable",
            )

        if st.button("Run Simulation", key="sim_run"):
            trial = user_data.copy()
            trial.update({
                "monthly_electricity_consumption_kwh": float(sim_consumption),
                "household_income": int(sim_income),
                "household_size": int(sim_size),
                "renewable_energy_access": sim_renewable,
            })
            sim_df = run_pipeline(pd.DataFrame([trial]), retrain=False)
            sim_row = sim_df.iloc[0]
            st.metric(
                "Simulated HEFI",
                f"{sim_row['hefi_score']:.1f}",
                delta=f"{sim_row['hefi_score'] - user_data['hefi_score']:+.1f}",
            )
            st.markdown(f"**Projected tariff tier:** {sim_row['tariff_tier']}")

    st.info("💡 **Tip**: A higher score means you are categorized as more vulnerable, qualifying you for higher subsidies.")

# ─── TAB: Update Details ─────────────────────────────────────────────────────
elif menu == "📝 Update My Details":
    st.title("📝 Self-Report Dashboard")
    st.markdown("Has your household situation changed? Update your details below to ensure a fair tariff calculation.")
    
    with st.form("update_form"):
        col_a, col_b = st.columns(2)
        with col_a:
            new_size = st.number_input("Household Size", value=int(user_data['household_size']), min_value=1)
            new_income = st.number_input("Monthly Income (₹)", value=int(user_data['household_income']), min_value=0)
        with col_b:
            new_appliances = st.number_input("Appliance Count", value=int(user_data['appliance_count']), min_value=1)
            new_renewable = st.selectbox("Renewable Access", ["Yes", "No"], index=0 if user_data['renewable_energy_access']=="Yes" else 1)
        
        submitted = st.form_submit_button("Securely Update My Details")
        if submitted:
            # 1. Show processing state
            with st.spinner("Synchronizing with HEFI Registry..."):
                updated_record = user_data.copy()
                updated_record.update({
                    "household_size": new_size,
                    "household_income": new_income,
                    "appliance_count": new_appliances,
                    "renewable_energy_access": new_renewable
                })
                
                # 2. Contextual recalculation
                df_updated = recalculate_with_context(pd.DataFrame([updated_record]))
                upsert_households(df_updated)
                time.sleep(1.5)
                # Set a flag to show success outside the form
                st.session_state.update_success = True
                st.rerun()

    if st.session_state.get("update_success"):
        st.toast("Registry Synchronized!", icon="✅")
        st.success("**Update Complete.** Your HEFI score has been automatically refreshed based on the new data.")
        if st.button("View My Updated HEFI Status"):
            st.session_state.update_success = False
            st.rerun()

# ─── TAB: Support Chat ───────────────────────────────────────────────────────
elif menu == "💬 Support Chat":
    st.title("💬 HEFI Assistant")
    st.markdown("Ask our AI assistant about your score, subsidies, or energy fairness issues.")
    
    # Display Chat
    for msg in st.session_state.chat_history:
        st.markdown(f"<div class='chat-bubble'><b>{'You' if msg['role']=='user' else 'Bot'}:</b> {msg['text']}</div>", unsafe_allow_html=True)
    
    # Input
    user_q = st.chat_input("How can I lower my electricity bill?")
    if user_q:
        st.session_state.chat_history.append({"role": "user", "text": user_q})
        response = get_chatbot_response(user_q, user_data)
        st.session_state.chat_history.append({"role": "bot", "text": response})
        st.rerun()
