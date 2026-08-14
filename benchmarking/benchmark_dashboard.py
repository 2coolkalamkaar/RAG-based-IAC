import streamlit as st
import pandas as pd
import json
import os
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Workflow Benchmarks", page_icon="📊", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
* { font-family: 'Inter', sans-serif; }
.stApp { background: #0e1117; color: #e6edf3; }
.metric-box {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
}
.metric-val { font-size: 2.2rem; font-weight: 800; color: #58a6ff; }
.metric-lbl { font-size: 0.9rem; color: #8b949e; text-transform: uppercase; letter-spacing: 1px; }
</style>
""", unsafe_allow_html=True)

st.title("📊 AI Agent Workflow Benchmarks")
st.markdown("A quantitative comparison between Basic, Standard RAG, Advanced RAG, and Secure RAG Terraform Agents.")

data_file = os.path.join(os.path.dirname(__file__), "benchmark_results.json")

if not os.path.exists(data_file):
    st.warning("⚠️ Benchmark results not found! Please wait for `run_benchmark.py` to finish generating `benchmark_results.json`.")
    st.stop()

with open(data_file, 'r') as f:
    data = json.load(f)

df = pd.DataFrame(data)

# Aggregations
summary_df = df.groupby('workflow').agg(
    avg_score=('score', 'mean'),
    avg_time=('time_sec', 'mean'),
    valid_rate=('is_valid', lambda x: x.mean() * 100),
    avg_retries=('retries', 'mean'),
    avg_context=('context_length', 'mean')
).reset_index()

# Ensure standard order
order_map = {"Basic": 1, "Standard RAG": 2, "Advanced RAG": 3, "Secure RAG": 4}
summary_df['order'] = summary_df['workflow'].map(order_map)
summary_df = summary_df.sort_values('order').drop(columns=['order'])

# KPIs
col1, col2, col3, col4 = st.columns(4)
best_score = summary_df.loc[summary_df['avg_score'].idxmax()]
best_valid = summary_df.loc[summary_df['valid_rate'].idxmax()]
best_time = summary_df.loc[summary_df['avg_time'].idxmin()]

col1.markdown(f'<div class="metric-box"><div class="metric-lbl">Highest Quality</div><div class="metric-val">{best_score["workflow"]}</div></div>', unsafe_allow_html=True)
col2.markdown(f'<div class="metric-box"><div class="metric-lbl">Best Validity Rate</div><div class="metric-val">{best_valid["valid_rate"]:.0f}%</div></div>', unsafe_allow_html=True)
col3.markdown(f'<div class="metric-box"><div class="metric-lbl">Fastest Execution</div><div class="metric-val">{best_time["avg_time"]:.1f}s</div></div>', unsafe_allow_html=True)
col4.markdown(f'<div class="metric-box"><div class="metric-lbl">Total Workflows</div><div class="metric-val">{len(summary_df):.0f}</div></div>', unsafe_allow_html=True)

st.divider()

# Charts
c1, c2 = st.columns(2)

with c1:
    st.subheader("🏆 Code Quality Score (out of 5)")
    fig = px.bar(summary_df, x='workflow', y='avg_score', text='avg_score',
                 color='workflow', color_discrete_sequence=['#484f58', '#1f6feb', '#2ea043', '#8957e5'])
    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="#c9d1d9", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

with c2:
    st.subheader("⏱️ Average Execution Time (seconds)")
    fig = px.bar(summary_df, x='workflow', y='avg_time', text='avg_time',
                 color='workflow', color_discrete_sequence=['#484f58', '#1f6feb', '#2ea043', '#8957e5'])
    fig.update_traces(texttemplate='%{text:.1f}s', textposition='outside')
    fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="#c9d1d9", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

c3, c4 = st.columns(2)

with c3:
    st.subheader("🔧 Average Self-Healing Retries")
    fig = px.bar(summary_df, x='workflow', y='avg_retries', text='avg_retries',
                 color='workflow', color_discrete_sequence=['#484f58', '#1f6feb', '#2ea043', '#8957e5'])
    fig.update_traces(texttemplate='%{text:.1f}', textposition='outside')
    fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="#c9d1d9", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

with c4:
    st.subheader("📚 Context Injection Size (Characters)")
    fig = px.bar(summary_df, x='workflow', y='avg_context', text='avg_context',
                 color='workflow', color_discrete_sequence=['#484f58', '#1f6feb', '#2ea043', '#8957e5'])
    fig.update_traces(texttemplate='%{text:.0f}', textposition='outside')
    fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="#c9d1d9", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

st.divider()

st.subheader("📋 Detailed Raw Results")
st.dataframe(df.style.highlight_max(axis=0, subset=['score', 'is_valid']), use_container_width=True)
