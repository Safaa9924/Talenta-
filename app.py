import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from utils.style import inject_css, sidebar_logo, page_header, kpi_card, PLOTLY_LAYOUT, COLORS
from utils.sample_data import (
    generate_candidates, generate_pipeline_activity, generate_trend, generate_recent_activity
)

st.set_page_config(page_title="CVLora · Dashboard", page_icon="🧠", layout="wide", initial_sidebar_state="expanded")
inject_css()

with st.sidebar:
    sidebar_logo()
    st.caption("HIRING WORKSPACE")
    st.markdown("**Vertex Labs** · Enterprise")
    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

# --------------------------------------------------------------------------
page_header(
    "Overview",
    "Good morning, welcome back 👋",
    "Here's how your hiring pipeline is performing today.",
)

candidates = generate_candidates()
pipeline = generate_pipeline_activity()
trend = generate_trend()
activity = generate_recent_activity()

# ---- KPI row --------------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
with c1:
    kpi_card("Resumes Processed", "12,480", "18.2% vs last month", True)
with c2:
    kpi_card("Avg. Match Score", f"{candidates['match_score'].mean():.1f}%", "3.4 pts", True)
with c3:
    kpi_card("Time-to-Screen", "9.4 sec", "94% faster than manual", True)
with c4:
    kpi_card("Open Positions", "14", "2 closed this week", False)

st.markdown("<div class='cv-divider'></div>", unsafe_allow_html=True)

# ---- Trend + Funnel ---------------------------------------------------
col_left, col_right = st.columns([1.5, 1])

with col_left:
    st.markdown("##### Resumes processed — last 30 days")
    fig = px.area(trend, x="date", y="resumes_processed")
    fig.update_traces(line_color=COLORS["secondary"], fillcolor="rgba(37,99,235,0.12)")
    fig.update_layout(**PLOTLY_LAYOUT, height=280, xaxis_title=None, yaxis_title=None)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

with col_right:
    st.markdown("##### Hiring funnel")
    fig2 = go.Figure(go.Funnel(
        y=pipeline["stage"], x=pipeline["count"],
        marker=dict(color=[COLORS["secondary"], COLORS["secondary"], "#60A5FA", COLORS["accent"], "#34D399", COLORS["primary"]]),
        textinfo="value+percent initial",
    ))
    fig2.update_layout(**PLOTLY_LAYOUT, height=280)
    st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

st.markdown("<div class='cv-divider'></div>", unsafe_allow_html=True)

# ---- Top candidates + activity feed ---------------------------------------
col_a, col_b = st.columns([1.6, 1])

with col_a:
    st.markdown("##### Top recommended candidates")
    top5 = candidates.head(5)
    for _, row in top5.iterrows():
        cc1, cc2, cc3 = st.columns([2.4, 1.3, 1])
        with cc1:
            st.markdown(f"**{row['name']}**  \n<span style='color:#64748B;font-size:13px'>{row['title']} · {row['years_experience']} yrs</span>", unsafe_allow_html=True)
        with cc2:
            from utils.style import match_badge
            st.markdown(match_badge(row["match_score"]), unsafe_allow_html=True)
        with cc3:
            st.button("View profile", key=f"home_view_{row['id']}", use_container_width=True)
        st.markdown("<div style='border-top:1px solid #E2E8F0;margin:8px 0'></div>", unsafe_allow_html=True)

with col_b:
    st.markdown("##### Recent activity")
    for text, source, when in activity:
        st.markdown(
            f"""
            <div style="padding:10px 0;border-bottom:1px solid #E2E8F0;">
                <div style="font-size:13.5px;font-weight:600;">{text}</div>
                <div style="font-size:12px;color:#64748B;">{source} · {when}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("<div class='cv-divider'></div>", unsafe_allow_html=True)
st.caption("CVLora Prototype · Design system: Deep Navy / Royal Blue / Emerald · Inter + Plus Jakarta Sans")
