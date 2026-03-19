"""
app.py
------
Streamlit dashboard for the AI-Powered Household Energy Fairness Index (HEFI).

Launch:
    streamlit run app.py
"""

import os
import io
import time
import sqlite3
import warnings

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

warnings.filterwarnings("ignore")

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HEFI — Household Energy Fairness Index",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Theme controls (dark/light) ─────────────────────────────────────────────────
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

theme = st.sidebar.selectbox(
    "Theme",
    ["Dark", "Light"],
    index=0 if st.session_state.theme == "dark" else 1,
    key="theme_mode_select",
)
st.session_state.theme = theme.lower()


def _get_theme_values(mode: str):
    if mode == "dark":
        return {
            "app_bg": "linear-gradient(135deg, #0d1117 0%, #161b22 60%, #0d1117 100%)",
            "text": "#e6edf3",
            "sidebar_bg": "linear-gradient(180deg, #161b22 0%, #0d1117 100%)",
            "sidebar_border": "#21262d",
            "metric_bg": "linear-gradient(135deg, #1c2128, #21262d)",
            "metric_border": "#30363d",
            "metric_shadow": "rgba(0,0,0,0.4)",
            "heading": "#58a6ff",
            "heading_border": "#21262d",
            "badge_sub": "#1a4731",
            "badge_std": "#1f3a5f",
            "badge_prem": "#4a1a1a",
            "text_muted": "#8b949e",
            "link": "#58a6ff",
            "tab_bg": "#21262d",
            "tab_active_bg": "#1c2128",
            "button_bg": "linear-gradient(90deg, #238636, #2ea043)",
            "button_text": "white",
        }
    return {
        "app_bg": "#f4f6ff",
        "text": "#0f172a",
        "sidebar_bg": "#ffffff",
        "sidebar_border": "#e2e8f0",
        "metric_bg": "#ffffff",
        "metric_border": "#e2e8f0",
        "metric_shadow": "rgba(0,0,0,0.06)",
        "heading": "#0e60d2",
        "heading_border": "#e2e8f0",
        "badge_sub": "#1a4731",
        "badge_std": "#1f3a5f",
        "badge_prem": "#4a1a1a",
        "text_muted": "#4b5563",
        "link": "#0e60d2",
        "tab_bg": "#f3f4f6",
        "tab_active_bg": "#e2e8f0",
        "button_bg": "linear-gradient(90deg, #0e60d2, #0d359a)",
        "button_text": "white",
    }

_theme = _get_theme_values(st.session_state.theme)

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
    }}

    /* Main background */
    .stApp {{
        background: {_theme['app_bg']};
        color: {_theme['text']};
    }}

    /* Sidebar */
    section[data-testid="stSidebar"] {{
        background: {_theme['sidebar_bg']};
        border-right: 1px solid {_theme['sidebar_border']};
    }}
    section[data-testid="stSidebar"] * {{
        color: {_theme['text']} !important;
    }}
    section[data-testid="stSidebar"] a {{
        color: {_theme['link']} !important;
    }}

    /* Metric cards */
    .metric-card {{
        background: {_theme['metric_bg']};
        border: 1px solid {_theme['metric_border']};
        border-radius: 12px;
        padding: 20px 24px;
        text-align: center;
        box-shadow: 0 4px 20px {_theme['metric_shadow']};
        transition: transform 0.2s ease;
    }}
    .metric-card:hover {{ transform: translateY(-2px); }}
    .metric-card .value {{
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #58a6ff, #79c0ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }}
    .metric-card .label {{
        font-size: 0.82rem;
        color: {_theme['text_muted']};
        margin-top: 4px;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }}

    /* Section headings */
    .section-heading {{
        font-size: 1.25rem;
        font-weight: 600;
        color: {_theme['heading']};
        border-bottom: 2px solid {_theme['heading_border']};
        padding-bottom: 8px;
        margin-bottom: 16px;
    }}

    /* Tariff badge */
    .badge-sub   {{ background:{_theme['badge_sub']}; color:#56d364; padding:3px 10px; border-radius:20px; font-size:0.8rem; font-weight:600; }}
    .badge-std   {{ background:{_theme['badge_std']}; color:#58a6ff; padding:3px 10px; border-radius:20px; font-size:0.8rem; font-weight:600; }}
    .badge-prem  {{ background:{_theme['badge_prem']}; color:#f97583; padding:3px 10px; border-radius:20px; font-size:0.8rem; font-weight:600; }}

    /* Hide Streamlit branding */
    #MainMenu, footer {{ visibility: hidden; }}

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{ gap: 8px; }}
    .stTabs [data-baseweb="tab"] {{
        background: {_theme['tab_bg']};
        border-radius: 8px 8px 0 0;
        color: {_theme['text_muted']};
        font-weight: 500;
    }}
    .stTabs [aria-selected="true"] {{
        background: {_theme['tab_active_bg']};
        color: {_theme['heading']} !important;
        border-bottom: 2px solid {_theme['heading']};
    }}
    .stTabs [data-baseweb="tab"] span {{
        color: {_theme['text_muted']} !important;
    }}
    .stTabs [aria-selected="true"] span {{
        color: {_theme['heading']} !important;
    }}
    /* Buttons */
    .stButton > button {{
        background: {_theme['button_bg']};
        color: {_theme['button_text']};
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.5rem 1.2rem;
        transition: opacity 0.2s;
    }}
    .stButton > button:hover {{ opacity: 0.85; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ─── Helpers ──────────────────────────────────────────────────────────────────

# Matplotlib global style (adjusts with theme)
if st.session_state.theme == "dark":
    plt.rcParams.update(
        {
            "figure.facecolor": "#161b22",
            "axes.facecolor": "#0d1117",
            "axes.edgecolor": "#30363d",
            "axes.labelcolor": "#8b949e",
            "xtick.color": "#8b949e",
            "ytick.color": "#8b949e",
            "text.color": "#e6edf3",
            "grid.color": "#21262d",
            "grid.linewidth": 0.6,
        }
    )
else:
    plt.rcParams.update(
        {
            "figure.facecolor": "#ffffff",
            "axes.facecolor": "#f8fafc",
            "axes.edgecolor": "#cbd5e1",
            "axes.labelcolor": "#0f172a",
            "xtick.color": "#1f2937",
            "ytick.color": "#1f2937",
            "text.color": "#0f172a",
            "grid.color": "#e2e8f0",
            "grid.linewidth": 0.6,
        }
    )

TIER_COLORS = {
    "Subsidized": "#56d364",
    "Standard": "#58a6ff",
    "Premium": "#f97583",
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "rf_model.pkl")


@st.cache_data(show_spinner=False)
def load_pipeline_modules():
    """Lazy import to keep initial load fast."""
    from data_generator import generate_households
    from fairness_index import run_pipeline, init_db, upsert_households
    from collectors import MeterStream, GovtAPI, FieldAppCollector, simulate_ingestion_log

    return generate_households, run_pipeline, init_db, upsert_households, MeterStream, GovtAPI, FieldAppCollector, simulate_ingestion_log


def metric_card(col, value, label):
    col.markdown(
        f"""
        <div class="metric-card">
            <div class="value">{value}</div>
            <div class="label">{label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "## ⚡ HEFI Dashboard",
    )
    st.markdown("*Household Energy Fairness Index*")
    st.markdown("---")

    data_source = st.radio(
        "**Data source**",
        ["🎲 Simulate dataset", "📂 Upload CSV"],
        index=0,
    )

    n_households = 5_000
    if data_source == "🎲 Simulate dataset":
        n_households = st.slider(
            "Number of households", min_value=500, max_value=10_000,
            value=5_000, step=500,
        )

    retrain = st.checkbox("Re-train model", value=False)
    run_btn = st.button("▶ Run Analysis", width="stretch")

    st.markdown("---")
    st.markdown(
        """
        **Tariff tiers**
        - <span class='badge-sub'>Subsidized</span>  HEFI 70–100
        - <span class='badge-std'>Standard</span>    HEFI 40–69
        - <span class='badge-prem'>Premium</span>     HEFI 0–39
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.caption("© 2026 HEFI Prototype · AI-Powered Energy Equity")


# ─── Main header ──────────────────────────────────────────────────────────────
st.markdown(
    """
    <h1 style="
        background: linear-gradient(90deg,#58a6ff,#79c0ff,#a371f7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.4rem; font-weight: 700; margin-bottom: 4px;">
        ⚡ AI-Powered Household Energy Fairness Index
    </h1>
    <p style="color:#8b949e; font-size:1rem; margin-top:0;">
        Equitable electricity tariff classification using multi-dimensional household vulnerability scoring.
    </p>
    """,
    unsafe_allow_html=True,
)
st.markdown("---")


# ─── Session state ────────────────────────────────────────────────────────────
if "result_df" not in st.session_state:
    st.session_state.result_df = None

uploaded_df = None

if data_source == "📂 Upload CSV":
    uploaded_file = st.file_uploader(
        "Upload households CSV", type=["csv"], label_visibility="collapsed"
    )
    if uploaded_file:
        uploaded_df = pd.read_csv(uploaded_file)
        st.success(f"Loaded **{len(uploaded_df):,}** rows from upload.")


@st.cache_data(show_spinner=False)
def get_cached_results(data_source, n_households, retrain, uploaded_df_bytes=None):
    """
    Cache the heavy ML and HEFI computations.
    If input parameters don't change, Streamlit grabs results from cache.
    """
    generate_households, run_pipeline, init_db, upsert_households, *rest = load_pipeline_modules()

    # Ensure DB is initialized
    init_db()

    if data_source == "🎲 Simulate dataset":
        raw_df = generate_households(n=n_households)
    else:
        if uploaded_df_bytes is None:
            return None
        raw_df = pd.read_csv(io.BytesIO(uploaded_df_bytes))

    result = run_pipeline(raw_df, retrain=retrain)
    
    # Sync results to SQLite registry for Phase 2 compatibility
    upsert_households(result)
    
    return result


@st.cache_data(show_spinner=False)
def load_registry_df(cache_bust: int = 0):
    """Load the latest household registry from the SQLite DB.

    The `cache_bust` parameter is used to invalidate the cached result when
    a refresh is explicitly requested.
    """
    from fairness_index import init_db

    init_db()
    registry_path = os.path.join(BASE_DIR, "data", "registry.db")
    if not os.path.exists(registry_path):
        return pd.DataFrame()

    conn = sqlite3.connect(registry_path)
    try:
        df = pd.read_sql("SELECT * FROM households", conn)
    finally:
        conn.close()
    return df


# If no analysis has run yet, show the current registry in the dashboard
if st.session_state.result_df is None:
    registry_df = load_registry_df(st.session_state.get("_registry_refresh", 0))
    if not registry_df.empty:
        st.session_state.result_df = registry_df


# ─── Run analysis ─────────────────────────────────────────────────────────────
if run_btn:
    uploaded_bytes = None
    if data_source == "📂 Upload CSV" and uploaded_df is not None:
        uploaded_bytes = uploaded_file.getvalue()

    result = get_cached_results(data_source, n_households, retrain, uploaded_bytes)

    if result is not None:
        st.session_state.result_df = result
        st.success("✅ Analysis complete!")
    else:
        st.error("Please upload a CSV file first.")
        st.stop()

# ─── Results display ──────────────────────────────────────────────────────────
if st.session_state.result_df is not None:
    df = st.session_state.result_df

    # ── KPI cards ─────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    metric_card(c1, f"{len(df):,}", "Households Analysed")
    metric_card(c2, f"{df['hefi_score'].mean():.1f}", "Average HEFI Score")
    metric_card(
        c3,
        f"{int((df['tariff_tier'] == 'Subsidized').sum()):,}",
        "Subsidized Households",
    )
    metric_card(
        c4,
        f"{int((df['tariff_tier'] == 'Premium').sum()):,}",
        "Premium Households",
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📊 Visualizations", "🛰️ Digital Ingestion", "🗂 Data Table", "ℹ️ Methodology"]
    )

    # ── Tab 1: Visualizations ─────────────────────────────────────────────────
    with tab1:
        col_a, col_b = st.columns(2)

        # 1. HEFI Histogram
        with col_a:
            st.markdown("<div class='section-heading'>HEFI Score Distribution</div>", unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(6, 3.8))
            scores = df["hefi_score"]

            # Colored zones
            ax.axvspan(0,  40, alpha=0.08, color="#f97583")
            ax.axvspan(40, 70, alpha=0.08, color="#58a6ff")
            ax.axvspan(70,100, alpha=0.08, color="#56d364")

            ax.hist(scores, bins=40, color="#58a6ff", edgecolor="#0d1117", linewidth=0.4)
            ax.axvline(scores.mean(), color="#ffa657", linestyle="--", linewidth=1.5,
                       label=f"Mean = {scores.mean():.1f}")
            ax.set_xlabel("HEFI Score")
            ax.set_ylabel("Households")
            ax.legend(fontsize=8)
            ax.grid(axis="y", alpha=0.4)
            # Zone labels
            for x, lbl, clr in [(20,"Premium","#f97583"),(55,"Standard","#58a6ff"),(85,"Subsidized","#56d364")]:
                ax.text(x, ax.get_ylim()[1]*0.92, lbl, ha="center", fontsize=7.5,
                        color=clr, fontweight="bold")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        # 2. Tariff Pie Chart
        with col_b:
            st.markdown("<div class='section-heading'>Tariff Tier Breakdown</div>", unsafe_allow_html=True)
            tier_counts = df["tariff_tier"].value_counts()
            order = ["Subsidized", "Standard", "Premium"]
            sizes  = [tier_counts.get(t, 0) for t in order]
            colors = [TIER_COLORS[t] for t in order]

            fig, ax = plt.subplots(figsize=(6, 3.8))
            wedges, texts, autotexts = ax.pie(
                sizes, labels=order, colors=colors,
                autopct="%1.1f%%", startangle=140,
                wedgeprops={"edgecolor": "#0d1117", "linewidth": 2},
                pctdistance=0.78,
            )
            for t in texts:
                t.set_fontsize(9); t.set_color("#e6edf3")
            for at in autotexts:
                at.set_fontsize(8); at.set_color("#0d1117"); at.set_fontweight("bold")
            ax.set_title("Proportion of households per tier", fontsize=9, color="#8b949e", pad=10)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        st.markdown("<br>", unsafe_allow_html=True)

        # 3. Scatter: Consumption vs HEFI
        st.markdown("<div class='section-heading'>Consumption vs. HEFI Score</div>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(12, 4.5))
        for tier in ["Premium", "Standard", "Subsidized"]:
            sub = df[df["tariff_tier"] == tier]
            ax.scatter(
                sub["monthly_electricity_consumption_kwh"],
                sub["hefi_score"],
                c=TIER_COLORS[tier], s=10, alpha=0.55,
                label=tier, rasterized=True,
            )
        ax.set_xlabel("Monthly Electricity Consumption (kWh)")
        ax.set_ylabel("HEFI Score")
        ax.axhline(70, color="#56d364", linestyle="--", linewidth=0.9, alpha=0.7)
        ax.axhline(40, color="#f97583", linestyle="--", linewidth=0.9, alpha=0.7)
        ax.legend(fontsize=8, markerscale=2)
        ax.grid(alpha=0.35)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.markdown("<br>", unsafe_allow_html=True)

        # 4. HEFI Components heatmap (sample 500)
        st.markdown("<div class='section-heading'>HEFI Component Correlations</div>", unsafe_allow_html=True)
        comp_cols = [
            "income_vulnerability", "household_size_factor",
            "energy_dependency", "consumption_anomaly", "hefi_score",
        ]
        sample = df[comp_cols].sample(min(500, len(df)), random_state=42)
        fig, ax = plt.subplots(figsize=(8, 3.5))
        corr = sample.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        cmap = sns.diverging_palette(220, 20, as_cmap=True)
        sns.heatmap(
            corr, annot=True, fmt=".2f", cmap=cmap,
            linewidths=0.5, linecolor="#21262d",
            annot_kws={"size": 8}, ax=ax,
            cbar_kws={"shrink": 0.7},
        )
        ax.set_title("Correlation matrix of HEFI components", fontsize=8.5, color="#8b949e")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # ── Tab 2: Digital Ingestion (NEW) ─────────────────────────────────────────
    with tab2:
        st.markdown("<div class='section-heading'>Real-time Data Ingestion Stream</div>", unsafe_allow_html=True)
        
        ca, cb = st.columns([2, 1])
        
        with cb:
            st.info("**Collector Health**\n\n🟢 Meter-Mesh: 98%\n🟢 Govt-API: Active\n🟢 Field-Sync: Online")
            if st.button("Trigger Mock Data Sync", width="stretch"):
                with st.spinner("Processing ingestion pipeline..."):
                    # Ensure DB and table exist
                    generate_households, run_pipeline, init_db, upsert_households, MeterStream, GovtAPI, FieldAppCollector, *rest = load_pipeline_modules()
                    init_db()
                    
                    # 1. Fetch random target from DB
                    conn = sqlite3.connect(os.path.join(os.path.dirname(__file__), "data", "registry.db"))
                    try:
                        target = pd.read_sql("SELECT household_id FROM households ORDER BY RANDOM() LIMIT 1", conn)
                        if not target.empty:
                            hid = target.iloc[0]["household_id"]
                            
                            # 2. Mock external calls
                            generate_households, run_pipeline, init_db, upsert_households, MeterStream, GovtAPI, FieldAppCollector, *rest = load_pipeline_modules()
                            
                            meter = MeterStream([hid])
                            govt = GovtAPI()
                            field = FieldAppCollector()
                            
                            new_data = {
                                "household_id": hid,
                                **meter.poll_consumption(1)[0],
                                **govt.get_socio_economic_data(hid),
                                **field.sync_field_data()
                            }
                            
                            # 3. Recalculate HEFI
                            updated_df = run_pipeline(pd.DataFrame([new_data]), retrain=False)
                            upsert_households(updated_df)
                            
                            st.toast(f"Updated {hid} via Digital Ingestion!", icon="⚡")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.warning("No households in database. Simulate data first.")
                    finally:
                        conn.close()
        
        with ca:
            st.markdown("### 🖥️ Live System Console")
            log_container = st.empty()
            
            # Simple mock log stream
            dummy_logs = []
            *rest, simulate_ingestion_log = load_pipeline_modules()
            for i in range(10):
                dummy_logs.append(f"[{time.strftime('%H:%M:%S')}] {simulate_ingestion_log()}")
            
            log_container.code("\n".join(dummy_logs))
            
            st.markdown("---")
            st.markdown("**Continuous Intelligence Activity**")
            st.progress(0.85, text="ML Re-scoring background worker active")

    # ── Tab 3: Data Table ─────────────────────────────────────────────────────
    with tab3:
        st.markdown("<div class='section-heading'>Full Results Table</div>", unsafe_allow_html=True)

        if st.button("🔄 Refresh from registry"):
            st.session_state._registry_refresh = st.session_state.get("_registry_refresh", 0) + 1
            st.session_state.result_df = None
            st.rerun()

        search_id = st.text_input("Search by Household ID", placeholder="HH_001")
        tier_filter = st.multiselect(
            "Filter by tariff tier",
            ["Subsidized", "Standard", "Premium"],
            default=["Subsidized", "Standard", "Premium"],
        )
        view_df = df[df["tariff_tier"].isin(tier_filter)]

        if search_id.strip():
            pattern = search_id.strip()
            view_df = view_df[view_df["household_id"].str.contains(pattern, case=False, na=False)]

        display_cols = [
            "household_id",
            "monthly_electricity_consumption_kwh",
            "household_income",
            "household_size",
            "urban_or_rural",
            "renewable_energy_access",
            "electricity_dependency_score",
            "hefi_score",
            "tariff_tier",
        ]
        st.dataframe(
            view_df[[c for c in display_cols if c in view_df.columns]],
            width="stretch",
            height=420,
        )

        # Download
        csv_bytes = view_df.to_csv(index=False).encode()
        st.download_button(
            label="⬇ Download filtered results as CSV",
            data=csv_bytes,
            file_name="hefi_results.csv",
            mime="text/csv",
        )

    # ── Tab 4: Methodology ────────────────────────────────────────────────────
    with tab4:
        st.markdown(
            """
            ## Methodology

            ### HEFI Formula
            The **Household Energy Fairness Index** combines four equally-interpretable components,
            scaled from 0–100. Higher scores indicate higher vulnerability.

            | Component | Weight | Description |
            |-----------|--------|-------------|
            | Income Vulnerability | 30 % | Inverted income percentile rank |
            | Household Size Factor | 20 % | Normalized household member count |
            | Energy Dependency | 30 % | Electricity dependency score / 10 |
            | Consumption Anomaly | 20 % | Under-consumption relative to appliances |

            An ML-based adjustment (±5 pts) from the **RandomForest** model refines the score.

            ### Tariff Classification
            | HEFI Range | Tariff |
            |-----------|--------|
            | 70 – 100 | 🟢 Subsidized |
            | 40 – 69  | 🔵 Standard   |
            | 0  – 39  | 🔴 Premium    |

            ### Data Generation
            Synthetic households are drawn from log-normal income distributions (urban vs. rural),
            correlated appliance counts, and Poisson-distributed household sizes — ensuring
            realistic heterogeneity across 5,000+ records.
            """,
            unsafe_allow_html=False,
        )

else:
    # ── Placeholder state ─────────────────────────────────────────────────────
    st.markdown(
        """
        <div style="
            text-align:center; padding: 80px 20px;
            background: linear-gradient(135deg,#1c2128,#21262d);
            border-radius: 16px; border: 1px solid #30363d;">
            <div style="font-size:4rem;">⚡</div>
            <h2 style="color:#58a6ff; margin:16px 0 8px;">Ready to Analyse</h2>
            <p style="color:#8b949e; max-width:500px; margin:auto;">
                Select a data source in the sidebar and click
                <strong style="color:#e6edf3;">▶ Run Analysis</strong>
                to compute HEFI scores and visualise fairness distribution.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
