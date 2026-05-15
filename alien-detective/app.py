"""
app.py
AI-Powered Data Detective: ML Mystery Game
==========================================
Main Streamlit application entry point.

Run with:
    streamlit run app.py
"""

import sys
import os

# Make src/ importable from project root
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import pandas as pd

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Alien Signal Detective",
    page_icon="🛸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Internal imports ──────────────────────────────────────────────────────────
from src import game_state as gs
from src.data_generator import load_or_generate
from src.ml_models import SignalClassifier, SignalClusterer
from src import visualizations as viz
from src import llm_engine as llm

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Dark space theme */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #0a0e1a !important;
    color: #c8d8e8;
}
[data-testid="stSidebar"] {
    background-color: #0d1117 !important;
    border-right: 1px solid #1e2d40;
}
[data-testid="stHeader"] { background: transparent; }

.metric-card {
    background: #111827;
    border: 1px solid #1e3a5f;
    border-radius: 10px;
    padding: 16px;
    text-align: center;
    margin: 4px 0;
}
.metric-value { font-size: 2rem; font-weight: 700; color: #00e5ff; }
.metric-label { font-size: 0.8rem; color: #90a4ae; margin-top: 4px; }

.clue-card {
    background: #0d1f30;
    border-left: 3px solid #f48fb1;
    border-radius: 6px;
    padding: 10px 14px;
    margin: 6px 0;
    font-family: monospace;
    font-size: 0.82rem;
}
.section-title {
    color: #00e5ff;
    font-size: 1.1rem;
    font-weight: 600;
    border-bottom: 1px solid #1e3a5f;
    padding-bottom: 4px;
    margin-bottom: 12px;
}
.mission-badge {
    display: inline-block;
    background: #0d2137;
    border: 1px solid #00e5ff44;
    border-radius: 12px;
    padding: 3px 12px;
    font-size: 0.75rem;
    color: #00e5ff;
    margin-bottom: 8px;
}
.aria-box {
    background: linear-gradient(135deg, #0a1628, #091523);
    border: 1px solid #00e5ff44;
    border-radius: 10px;
    padding: 14px 18px;
    color: #a8d8ea;
    font-style: italic;
    font-size: 0.9rem;
    margin: 10px 0;
}
.theory-box {
    background: #0d1f30;
    border: 1px solid #f48fb155;
    border-radius: 10px;
    padding: 16px;
}
.verdict-correct   { color: #66bb6a; font-weight: 700; font-size: 1.4rem; }
.verdict-partial   { color: #ffa726; font-weight: 700; font-size: 1.4rem; }
.verdict-incorrect { color: #ef5350; font-weight: 700; font-size: 1.4rem; }

/* Override Streamlit defaults */
.stButton > button {
    background: #0d2137;
    color: #00e5ff;
    border: 1px solid #00e5ff55;
    border-radius: 8px;
}
.stButton > button:hover {
    background: #163450;
    border-color: #00e5ff;
}
div[data-testid="stMetricValue"] { color: #00e5ff; }
</style>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
#  INITIALISE STATE
# ═════════════════════════════════════════════════════════════════════════════
gs.init_state()


# ═════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════

def render_sidebar():
    with st.sidebar:
        st.markdown("## 🛸 Alien Detective")
        st.markdown(f"**Analyst:** {gs.get('player_name') or '—'}")

        progress = gs.progress_pct()
        st.progress(progress / 100, text=f"Case Progress: {progress:.0f}%")

        st.markdown("---")
        st.markdown("### Missions")

        for mission in gs.MISSIONS[1:]:   # skip 'intro'
            icon     = gs.MISSION_ICONS.get(mission, "○")
            title    = gs.MISSION_TITLES.get(mission, mission)
            unlocked = gs.mission_is_unlocked(mission)
            done     = mission in (gs.get("missions_completed") or [])
            current  = gs.get("current_mission") == mission

            if done:
                label = f"✅ {icon} {title}"
            elif current:
                label = f"▶ {icon} {title}"
            elif unlocked:
                label = f"○ {icon} {title}"
            else:
                label = f"🔒 {title}"

            if unlocked and not current:
                if st.button(label, key=f"nav_{mission}", use_container_width=True):
                    gs.advance_to(mission)
                    st.rerun()
            else:
                st.markdown(f"<div style='padding:4px 8px;color:{'#00e5ff' if current else '#556'};'>{label}</div>",
                            unsafe_allow_html=True)

        st.markdown("---")

        # Score & time
        col1, col2 = st.columns(2)
        col1.metric("Score", gs.get("score") or 0)
        col2.metric("Time", gs.elapsed_time())

        st.markdown("---")

        # LLM Settings
        with st.expander("⚙️ ARIA Settings"):
            model = st.text_input(
                "Ollama Model",
                value=gs.get("llm_model") or "llama3.2",
                help="Run `ollama pull llama3.2` to install",
            )
            if model != gs.get("llm_model"):
                gs.set("llm_model", model)
                gs.set("ollama_available", None)   # re-check

            if st.button("Check ARIA Status"):
                ok, msg = llm.check_ollama_available(model)
                gs.set("ollama_available", ok)
                if ok:
                    st.success(msg)
                else:
                    st.warning(msg)
            else:
                avail = gs.get("ollama_available")
                if avail is True:
                    st.success("ARIA Online")
                elif avail is False:
                    st.warning("ARIA Offline — run `ollama serve`")

        if st.button("🔄 Restart Game", use_container_width=True):
            gs.reset_game()
            st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
#  PAGE: INTRO
# ═════════════════════════════════════════════════════════════════════════════

def page_intro():
    st.markdown("""
    <div style='text-align:center;padding:30px 0 10px'>
        <div style='font-size:5rem'>🛸</div>
        <h1 style='color:#00e5ff;font-size:2.4rem;margin:0'>ALIEN SIGNAL DETECTIVE</h1>
        <p style='color:#90a4ae;font-size:1rem;margin-top:8px'>
            An AI-Powered ML Mystery Game · Deep Space Monitoring Station ECHO-7
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style='background:#0d1f30;border:1px solid #1e3a5f;border-radius:12px;padding:20px 28px;max-width:700px;margin:0 auto'>
        <h3 style='color:#f48fb1'>📡 TRANSMISSION RECEIVED — STARDATE 2031.74</h3>
        <p style='color:#c8d8e8;line-height:1.7'>
        Station ECHO-7 has intercepted <b style='color:#00e5ff'>500+ anomalous signals</b>
        originating from an uncharted region beyond the outer heliosphere.
        Initial scans show five distinct signal classes — but buried within the noise
        are coordinated transmissions that defy all known communication protocols.
        </p>
        <p style='color:#c8d8e8;line-height:1.7'>
        Your mission: use <b style='color:#00e5ff'>machine learning</b> to classify the signals,
        <b style='color:#00e5ff'>cluster</b> their origins, decode <b style='color:#f48fb1'>hidden clues</b>,
        and submit your final theory to ARIA — our on-board AI — before the transmission window closes.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        name = st.text_input(
            "Enter your analyst designation:",
            placeholder="e.g.  Dr. Reyes",
            value=gs.get("player_name") or "",
        )
        if st.button("🚀  Begin Investigation", use_container_width=True):
            if not name.strip():
                st.error("Please enter your analyst name to begin.")
            else:
                gs.set("player_name", name.strip())
                gs.set("start_time", __import__("datetime").datetime.now())
                gs.award_points(10, "Game started")
                gs.complete_mission("intro")
                st.rerun()

    # Capability overview
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    for col, icon, title, desc in [
        (c1, "🤖", "ML Classification", "RandomForest classifies 5 signal types"),
        (c2, "🗺️", "Cluster Analysis",  "KMeans + DBSCAN maps signal origins"),
        (c3, "🔍", "NLP Clue Decoder",  "ARIA decodes embedded clue fragments"),
        (c4, "📊", "Analytics Dash",    "Plotly charts + live evidence board"),
    ]:
        col.markdown(f"""
        <div class='metric-card'>
            <div style='font-size:2rem'>{icon}</div>
            <div style='color:#00e5ff;font-weight:600;margin:6px 0'>{title}</div>
            <div style='color:#90a4ae;font-size:0.8rem'>{desc}</div>
        </div>
        """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
#  PAGE: SIGNAL LAB
# ═════════════════════════════════════════════════════════════════════════════

def page_signal_lab():
    _mission_header("signal_lab")

    # Load / generate dataset
    if gs.get("dataset") is None:
        with st.spinner("Loading signal database…"):
            df = load_or_generate("data")
            gs.set("dataset", df)
            gs.award_points(20, "Signal database loaded")
    else:
        df = gs.get("dataset")

    # ARIA briefing
    _aria_briefing("signal_lab", {"n_signals": len(df)})

    # ── Stats row ─────────────────────────────────────────────────────────────
    st.markdown("<div class='section-title'>Database Overview</div>",
                unsafe_allow_html=True)

    cols = st.columns(5)
    for col, cls in zip(cols, ["Beacon", "Warning", "DataTransmission",
                                "Anomaly", "Interference"]):
        count = int((df["signal_class"] == cls).sum())
        col.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value'>{count}</div>
            <div class='metric-label'>{cls}</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Charts ────────────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["📍 Origin Map", "📊 Frequencies", "⏱ Timeline"])

    with tab1:
        st.plotly_chart(viz.signal_scatter(df), use_container_width=True)

    with tab2:
        col_a, col_b = st.columns(2)
        col_a.plotly_chart(viz.frequency_histogram(df), use_container_width=True)
        col_b.plotly_chart(viz.class_distribution_pie(df), use_container_width=True)

    with tab3:
        st.plotly_chart(viz.timeline_chart(df), use_container_width=True)

    # ── Data explorer ─────────────────────────────────────────────────────────
    with st.expander("🔎 Browse Signal Database"):
        filter_cls = st.multiselect(
            "Filter by class",
            options=df["signal_class"].unique().tolist(),
            default=df["signal_class"].unique().tolist(),
        )
        filtered = df[df["signal_class"].isin(filter_cls)]
        st.dataframe(
            filtered[["signal_id","timestamp","frequency_mhz","amplitude_db",
                       "duration_sec","modulation","signal_class"]].head(100),
            use_container_width=True,
            height=300,
        )

    _complete_button("signal_lab", "Proceed to Signal Classification →")


# ═════════════════════════════════════════════════════════════════════════════
#  PAGE: CLASSIFY
# ═════════════════════════════════════════════════════════════════════════════

def page_classify():
    _mission_header("classify")

    df = gs.get("dataset")
    if df is None:
        st.warning("Return to Signal Lab first to load the dataset.")
        return

    _aria_briefing("classify", {"accuracy": 90})

    clf = SignalClassifier()

    col_left, col_right = st.columns([1, 2])
    with col_left:
        st.markdown("<div class='section-title'>Train Classifier</div>",
                    unsafe_allow_html=True)
        st.markdown("""
        **Model:** RandomForest (150 trees)
        **Target:** 5 signal classes
        **Features:** frequency, amplitude, duration, pulse rate, noise ratio, modulation, sector coords
        """)

        run_train = st.button("🤖  Train RandomForest", use_container_width=True)
        if clf.is_ready() and gs.get("df_classified") is not None:
            st.success("Model already trained ✓")

    if run_train or (clf.is_ready() and gs.get("df_classified") is None):
        with st.spinner("Training model on 500 signals…"):
            metrics = clf.train(df)
            df_classified = clf.predict(df)

            gs.set("classifier_metrics", metrics)
            gs.set("df_classified", df_classified)
            gs.award_points(30, f"Classifier trained — accuracy {metrics['accuracy']:.1%}")

        # ARIA summary
        summary = llm.summarize_ml_results(metrics, model=gs.get("llm_model"))
        st.markdown(f"<div class='aria-box'>🤖 <b>ARIA:</b> {summary}</div>",
                    unsafe_allow_html=True)

    metrics = gs.get("classifier_metrics")
    df_classified = gs.get("df_classified")

    if metrics and df_classified is not None:
        # ── Metrics ───────────────────────────────────────────────────────────
        st.markdown("<div class='section-title'>Model Performance</div>",
                    unsafe_allow_html=True)
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Accuracy",   f"{metrics['accuracy']:.1%}")
        mc2.metric("F1 Score",   f"{metrics['f1_weighted']:.4f}")
        mc3.metric("CV Mean",    f"{metrics['cv_mean']:.1%}")
        mc4.metric("CV Std",     f"±{metrics['cv_std']:.4f}")

        # ── Charts ────────────────────────────────────────────────────────────
        tab1, tab2, tab3, tab4 = st.tabs([
            "Confusion Matrix", "Feature Importance",
            "Class Distribution", "Confidence",
        ])
        with tab1:
            st.plotly_chart(
                viz.confusion_matrix_heatmap(
                    metrics["confusion_matrix"], metrics["labels"]
                ), use_container_width=True
            )
        with tab2:
            st.plotly_chart(
                viz.feature_importance_bar(metrics["feature_importance"]),
                use_container_width=True
            )
        with tab3:
            col_a, col_b = st.columns(2)
            col_a.plotly_chart(viz.class_distribution_pie(df, "signal_class"),
                               use_container_width=True)
            col_b.plotly_chart(viz.class_distribution_pie(df_classified, "predicted_class"),
                               use_container_width=True)
        with tab4:
            st.plotly_chart(viz.confidence_histogram(df_classified),
                            use_container_width=True)

        # ── Anomaly probability explorer ──────────────────────────────────────
        with st.expander("🔴 High-Anomaly-Probability Signals"):
            high_anom = (df_classified[df_classified["anomaly_probability"] > 0.4]
                         [["signal_id","frequency_mhz","modulation","sector_x",
                           "sector_y","predicted_class","anomaly_probability",
                           "clue_fragment"]]
                         .sort_values("anomaly_probability", ascending=False)
                         .head(20))
            st.dataframe(high_anom, use_container_width=True)

        _complete_button("classify", "Proceed to Origin Mapping →")


# ═════════════════════════════════════════════════════════════════════════════
#  PAGE: CLUSTER
# ═════════════════════════════════════════════════════════════════════════════

def page_cluster():
    _mission_header("cluster")

    df = gs.get("df_classified") or gs.get("dataset")
    if df is None:
        st.warning("Complete the Classification mission first.")
        return

    _aria_briefing("cluster", {"sector": "7-Delta (~200, 250)"})

    col_left, col_right = st.columns([1, 3])
    with col_left:
        st.markdown("<div class='section-title'>Clustering Config</div>",
                    unsafe_allow_html=True)
        n_clusters = st.slider("KMeans clusters", 3, 8, 5)
        dbscan_eps = st.slider("DBSCAN ε (neighbourhood radius)", 10, 60, 25)
        run_cluster = st.button("🗺️  Run Clustering", use_container_width=True)

    if run_cluster or gs.get("df_clustered") is None:
        with st.spinner("Clustering signals…"):
            clust = SignalClusterer()
            df_km  = clust.fit_kmeans(df, n_clusters=n_clusters)
            df_db  = clust.fit_dbscan(df_km, eps=dbscan_eps)
            df_pca = clust.pca_projection(df_db)
            score  = clust.anomaly_concentration_score(df_pca)

            gs.set("df_clustered", df_pca)
            gs.set("mystery_score", score)
            gs.award_points(25, f"Clustering complete — mystery score {score:.2f}")

        if score > 0.65:
            st.warning(
                f"⚠️  ARIA ALERT: Anomaly concentration score is {score:.2f} — "
                "anomaly signals are unusually tightly clustered. Something is out there."
            )

    df_clustered = gs.get("df_clustered")
    mystery_score = gs.get("mystery_score") or 0.0

    if df_clustered is not None:
        # ── Charts ────────────────────────────────────────────────────────────
        tab1, tab2, tab3 = st.tabs(["KMeans Map", "DBSCAN Map", "PCA Projection"])

        with tab1:
            col_a, col_b = st.columns([3, 1])
            col_a.plotly_chart(
                viz.cluster_scatter(df_clustered, "kmeans_cluster"),
                use_container_width=True
            )
            col_b.plotly_chart(viz.mystery_score_gauge(mystery_score),
                               use_container_width=True)
        with tab2:
            st.plotly_chart(
                viz.cluster_scatter(df_clustered, "dbscan_cluster",
                                    highlight_anomalies=True),
                use_container_width=True
            )
        with tab3:
            st.plotly_chart(viz.pca_scatter(df_clustered), use_container_width=True)

        # ── Cluster stats ─────────────────────────────────────────────────────
        with st.expander("📊 Cluster Statistics"):
            stats = (df_clustered.groupby("kmeans_cluster")
                     .agg(count=("signal_id","count"),
                          avg_freq=("frequency_mhz","mean"),
                          avg_noise=("noise_ratio","mean"),
                          cx=("sector_x","mean"),
                          cy=("sector_y","mean"))
                     .round(2))
            st.dataframe(stats, use_container_width=True)

        _complete_button("cluster", "Proceed to Clue Decoder →")


# ═════════════════════════════════════════════════════════════════════════════
#  PAGE: CLUE DECODER
# ═════════════════════════════════════════════════════════════════════════════

def page_clue_decoder():
    _mission_header("clue_decoder")

    df = gs.get("df_classified") or gs.get("dataset")
    if df is None:
        st.warning("Complete earlier missions first.")
        return

    n_clues = len(gs.get_clues())
    _aria_briefing("clue_decoder", {"n_clues": n_clues})

    # Pull all anomaly signals that have clue fragments
    clue_signals = df[
        (df["signal_class"] == "Anomaly") & (df["clue_fragment"].str.len() > 0)
    ].copy()

    st.markdown(f"**{len(clue_signals)} anomaly signals carry embedded clue fragments.**"
                " Select one and let ARIA analyse it.")

    # ── Signal selector ───────────────────────────────────────────────────────
    col_sel, col_meta = st.columns([1, 1])
    with col_sel:
        st.markdown("<div class='section-title'>Select Signal</div>",
                    unsafe_allow_html=True)
        sig_options = clue_signals["signal_id"].tolist()
        chosen_id   = st.selectbox("Signal ID", sig_options)

        if st.button("🔍  Decode Clue with ARIA", use_container_width=True):
            row = clue_signals[clue_signals["signal_id"] == chosen_id].iloc[0]
            meta = row[["frequency_mhz","amplitude_db","duration_sec",
                         "modulation","sector_x","sector_y"]].to_dict()

            with st.spinner("ARIA analysing clue…"):
                analysis = llm.analyze_clue(
                    row["clue_fragment"], meta, model=gs.get("llm_model")
                )

            gs.add_clue(row["clue_fragment"], analysis)
            st.success("Clue added to your evidence board!")
            st.rerun()

        if st.button("💡  Get Hint (−10 pts)", use_container_width=True):
            clues = [c["fragment"] for c in gs.get_clues()]
            hint  = llm.get_hint("clue_decoder", clues, model=gs.get("llm_model"))
            gs.deduct_hint_penalty()
            st.markdown(f"<div class='aria-box'>🤖 <b>ARIA Hint:</b> {hint}</div>",
                        unsafe_allow_html=True)

    with col_meta:
        if chosen_id:
            row = clue_signals[clue_signals["signal_id"] == chosen_id].iloc[0]
            st.markdown("<div class='section-title'>Signal Metadata</div>",
                        unsafe_allow_html=True)
            meta_display = {
                "Frequency":  f"{row['frequency_mhz']} MHz",
                "Amplitude":  f"{row['amplitude_db']} dB",
                "Duration":   f"{row['duration_sec']}s",
                "Modulation": row["modulation"],
                "Sector":     f"({row['sector_x']}, {row['sector_y']})",
                "Clue Fragment": row["clue_fragment"],
            }
            for k, v in meta_display.items():
                is_clue = k == "Clue Fragment"
                st.markdown(
                    f"**{k}:** "
                    f"<span style='color:{'#f48fb1' if is_clue else '#c8d8e8'};font-family:monospace'>{v}</span>",
                    unsafe_allow_html=True,
                )

    # ── Decoded clues ─────────────────────────────────────────────────────────
    clues = gs.get_clues()
    if clues:
        st.markdown("---")
        st.markdown(f"<div class='section-title'>Decoded Clues ({len(clues)})</div>",
                    unsafe_allow_html=True)
        for clue in clues:
            st.markdown(f"""
            <div class='clue-card'>
                <b>Fragment:</b> {clue['fragment']}<br>
                <span style='color:#a8d8ea'><b>ARIA Analysis:</b> {clue.get('analysis','—')}</span><br>
                <span style='color:#556'>{clue['timestamp']}</span>
            </div>
            """, unsafe_allow_html=True)

        if len(clues) >= 2:
            _complete_button("clue_decoder", "Proceed to Case Board →",
                             min_clues=2)
    else:
        st.info("Decode at least 2 clues to proceed to the Case Board.")


# ═════════════════════════════════════════════════════════════════════════════
#  PAGE: CASE BOARD
# ═════════════════════════════════════════════════════════════════════════════

def page_case_board():
    _mission_header("case_board")

    df = gs.get("df_clustered") or gs.get("df_classified") or gs.get("dataset")
    if df is None:
        st.warning("Complete earlier missions first.")
        return

    st.markdown("""
    Review all collected evidence before submitting your final theory.
    """)

    # ── Evidence log ──────────────────────────────────────────────────────────
    col_ev, col_chart = st.columns([1, 2])

    with col_ev:
        st.markdown("<div class='section-title'>Evidence Log</div>",
                    unsafe_allow_html=True)
        log = gs.get_evidence_log()
        for entry in reversed(log[-15:]):
            st.markdown(
                f"<div style='font-size:0.8rem;color:#90a4ae'>{entry['time']}</div>"
                f"<div style='font-size:0.85rem;margin-bottom:6px'>{entry['event']}</div>",
                unsafe_allow_html=True
            )

        st.markdown("---")
        st.markdown("<div class='section-title'>Your Score</div>",
                    unsafe_allow_html=True)
        score = gs.get("score") or 0
        clues = len(gs.get_clues())
        hints = gs.get("hints_used") or 0
        st.metric("Points Earned", score)
        st.metric("Clues Decoded", clues)
        st.metric("Hints Used",    hints)

    with col_chart:
        tab1, tab2 = st.tabs(["Signal Radar", "Anomaly Gauge"])
        with tab1:
            if "signal_class" in df.columns:
                st.plotly_chart(viz.anomaly_radar(df), use_container_width=True)
        with tab2:
            mystery_score = gs.get("mystery_score") or 0.0
            st.plotly_chart(viz.mystery_score_gauge(mystery_score),
                            use_container_width=True)
            st.markdown(f"""
            <div class='aria-box'>
                🤖 <b>ARIA:</b> Anomaly concentration score is
                <b style='color:#f48fb1'>{mystery_score:.2f}</b>.
                {'This indicates a <b>highly coordinated</b> signal source.' if mystery_score > 0.65
                 else 'Signals show moderate spatial coherence.'}
            </div>
            """, unsafe_allow_html=True)

    # ── Clue summary ──────────────────────────────────────────────────────────
    clues = gs.get_clues()
    if clues:
        st.markdown("<div class='section-title'>Decoded Clue Summary</div>",
                    unsafe_allow_html=True)
        cols = st.columns(min(len(clues), 3))
        for col, clue in zip(cols, clues[:3]):
            col.markdown(f"""
            <div class='theory-box'>
                <div style='color:#f48fb1;font-size:0.75rem;font-family:monospace'>{clue['fragment'][:50]}…</div>
                <div style='margin-top:6px;font-size:0.82rem;color:#a8d8ea'>{clue.get('analysis','')[:120]}…</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    _complete_button("case_board", "Ready to submit final theory →")


# ═════════════════════════════════════════════════════════════════════════════
#  PAGE: FINAL DEDUCTION
# ═════════════════════════════════════════════════════════════════════════════

def page_final_deduction():
    _mission_header("final_deduction")

    mystery_score = gs.get("mystery_score") or 0.0
    _aria_briefing("final_deduction", {"mystery_score": mystery_score})

    # Show evidence summary
    ev = gs.build_evidence_summary()
    st.markdown("<div class='section-title'>Evidence Summary</div>",
                unsafe_allow_html=True)
    ec1, ec2, ec3, ec4 = st.columns(4)
    ec1.metric("Anomaly Signals",   ev["anomaly_count"])
    ec2.metric("Mystery Score",     f"{ev['mystery_score']:.2f}")
    ec3.metric("Clues Decoded",     ev["clues_collected"])
    ec4.metric("Classifier Acc.",   f"{ev['classification_accuracy']:.1%}"
               if isinstance(ev["classification_accuracy"], float)
               else str(ev["classification_accuracy"]))

    st.markdown("---")
    st.markdown("<div class='section-title'>Submit Your Theory</div>",
                unsafe_allow_html=True)
    st.markdown("""
    Based on all the evidence you've gathered, what is your theory about the
    anomaly signals? Who or what is sending them, from where, and why?
    """)

    theory = st.text_area(
        "Your Theory",
        value=gs.get("theory_submitted") or "",
        height=140,
        placeholder=(
            "e.g. The anomaly signals are automated beacon pulses from a non-terrestrial probe "
            "stationed at coordinates (200, 250) in Sector 7-Delta. The Fibonacci frequency pattern "
            "and unknown cipher suggest a pre-programmed intelligence-gathering mission…"
        ),
    )

    if st.button("📤  Submit Theory to ARIA", use_container_width=True):
        if len(theory.strip()) < 30:
            st.error("Please write a theory of at least 30 characters.")
        else:
            gs.set("theory_submitted", theory)
            with st.spinner("ARIA evaluating your theory…"):
                result = llm.evaluate_theory(theory, ev, model=gs.get("llm_model"))
            gs.set("theory_result", result)

            bonus = result.get("score", 0)
            gs.award_points(bonus, f"Theory submission — verdict: {result.get('verdict','?')}")
            gs.complete_mission("final_deduction")
            st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
#  PAGE: RESULTS
# ═════════════════════════════════════════════════════════════════════════════

def page_results():
    result = gs.get("theory_result")

    verdict = result.get("verdict", "UNKNOWN") if result else "PENDING"
    score   = result.get("score", 0)         if result else 0

    verdict_class = {
        "CORRECT":           "verdict-correct",
        "PARTIALLY CORRECT": "verdict-partial",
        "INCORRECT":         "verdict-incorrect",
    }.get(verdict, "verdict-partial")

    st.markdown(f"""
    <div style='text-align:center;padding:30px 0'>
        <div style='font-size:4rem'>{'🏆' if verdict=='CORRECT' else '🔍'}</div>
        <h1 style='color:#00e5ff'>Case Closed</h1>
        <div class='{verdict_class}'>{verdict}</div>
        <p style='color:#90a4ae;margin-top:8px'>Theory score: {score}/100</p>
    </div>
    """, unsafe_allow_html=True)

    if result:
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"""
            <div class='theory-box'>
                <div class='section-title'>ARIA's Evaluation</div>
                <p>{result.get('explanation','')}</p>
            </div>
            """, unsafe_allow_html=True)
        with col_b:
            st.markdown(f"""
            <div class='theory-box' style='border-color:#00e5ff44'>
                <div class='section-title'>The Truth Revealed</div>
                <p style='color:#a8d8ea'>{result.get('reveal','')}</p>
            </div>
            """, unsafe_allow_html=True)

    # Final score breakdown
    st.markdown("---")
    st.markdown("<div class='section-title'>Final Scorecard</div>",
                unsafe_allow_html=True)
    total = gs.get("score") or 0
    sc1, sc2, sc3, sc4, sc5 = st.columns(5)
    sc1.metric("Total Score",    total)
    sc2.metric("Theory Points",  score)
    sc3.metric("Clues Found",    len(gs.get_clues()))
    sc4.metric("Hints Used",     gs.get("hints_used") or 0)
    sc5.metric("Time Taken",     gs.elapsed_time())

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔄  Play Again", use_container_width=False):
        gs.reset_game()
        st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
#  HELPER COMPONENTS
# ═════════════════════════════════════════════════════════════════════════════

def _mission_header(mission: str):
    icon  = gs.MISSION_ICONS.get(mission, "•")
    title = gs.MISSION_TITLES.get(mission, mission)
    st.markdown(f"<div class='mission-badge'>{icon} {title}</div>",
                unsafe_allow_html=True)
    st.markdown(f"# {title}")


def _aria_briefing(mission: str, context: dict):
    """Show (and cache) a generated ARIA briefing for a mission."""
    cached = gs.get_briefing(mission)
    if not cached:
        briefing = llm.generate_mission_briefing(
            mission, context, model=gs.get("llm_model")
        )
        gs.cache_briefing(mission, briefing)
        cached = briefing
    st.markdown(f"<div class='aria-box'>🤖 <b>ARIA:</b> {cached}</div>",
                unsafe_allow_html=True)


def _complete_button(mission: str, label: str, min_clues: int = 0):
    """Render the 'complete mission' button with optional clue gate."""
    clues_ok = len(gs.get_clues()) >= min_clues
    if not clues_ok:
        return

    if mission not in (gs.get("missions_completed") or []):
        if st.button(f"✅  {label}", use_container_width=False):
            gs.complete_mission(mission)
            gs.award_points(20, f"Mission {mission} completed")
            st.rerun()
    else:
        st.success(f"Mission complete! {label}")
        if st.button("→ Continue", key=f"cont_{mission}"):
            idx = gs.MISSIONS.index(mission)
            if idx + 1 < len(gs.MISSIONS):
                gs.advance_to(gs.MISSIONS[idx + 1])
            st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
#  ROUTER
# ═════════════════════════════════════════════════════════════════════════════

PAGES = {
    "intro":           page_intro,
    "signal_lab":      page_signal_lab,
    "classify":        page_classify,
    "cluster":         page_cluster,
    "clue_decoder":    page_clue_decoder,
    "case_board":      page_case_board,
    "final_deduction": page_final_deduction,
    "results":         page_results,
}


def main():
    render_sidebar()
    current = gs.get("current_mission") or "intro"
    page_fn = PAGES.get(current, page_intro)
    page_fn()


if __name__ == "__main__":
    main()
