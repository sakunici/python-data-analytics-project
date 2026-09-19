import streamlit as st
import pandas as pd
import google.generativeai as genai
import plotly.express as px
from analytics_engine import ECommerceAnalyticsEngine

# ==========================================
# 1. PAGE CONFIGURATION & STYLING
# ==========================================
st.set_page_config(page_title="Executive Dashboard", page_icon="📈", layout="wide")

# BUG FIXED: Advanced Print CSS (Force black text, white background, prevent page breaks)
st.markdown("""
    <style>
    /* UI Typography */
    .main-header { text-align: center; font-size: 3rem; font-weight: 800; margin-bottom: 0px; }
    .sub-header { text-align: center; font-size: 2rem; color: #00E676; margin-bottom: 30px; }
    
    /* KPI Metric Cards */
    div[data-testid="metric-container"] {
        background-color: #1E1E2E; border: 1px solid #30363D;
        padding: 15px; border-radius: 10px; border-bottom: 3px solid #00E676; text-align: center;
    }
    [data-testid="stMetricLabel"] > div {
        font-size: 1.25rem !important; font-weight: 600 !important; color: #E0E0E0 !important;
    }
    
    /* PRINT MEDIA QUERY FOR PDF EXPORT */
    @media print {
        /* Hide all Sidebars, Headers, and Chat Inputs */
        header, [data-testid="stSidebar"], [data-testid="stToolbar"], button, .stChatInputContainer { 
            display: none !important; 
        }
        /* Force White Background */
        .stApp, .main, div.block-container { 
            background-color: white !important; 
        }
        /* Force Black Text */
        p, h1, h2, h3, h4, h5, h6, span, div[data-testid="stMetricValue"], div[data-testid="stMetricLabel"] > div { 
            color: black !important; 
        }
        /* Make KPI Cards lighter for print */
        div[data-testid="metric-container"] {
            background-color: #f8f9fa !important; border: 1px solid #dee2e6 !important;
        }
        /* Prevent Charts from being cut in half */
        .stPlotlyChart { 
            page-break-inside: avoid !important; 
        }
        /* Ensure Data table expanders stay open */
        .streamlit-expanderContent { display: block !important; }
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. CACHED BACKEND INITIALIZATION
# ==========================================
@st.cache_resource
def load_engine():
    data_dir = r"D:\L15-MiniProject\03_Analytics_Project\E-Commerce Sales"
    return ECommerceAnalyticsEngine(data_dir)

try:
    engine = load_engine()
except Exception as e:
    st.error(f"Failed to connect to the Data Engine: {e}")
    st.stop()

# ==========================================
# 3. UI LAYOUT & NAVIGATION
# ==========================================
st.sidebar.title("🤖 AI Configurations")
api_key = st.sidebar.text_input("Enter Gemini API Key", type="password")

st.sidebar.title("📊 Strategic Insights")

modules = {
    "1. Unit Economics & Margins": {"func": engine.run_f1_unit_economics, "x": "product_subcategory", "y": "total_profit", "chart": "bar"},
    "2. Customer LTV : CAC Ratio": {"func": engine.run_f2_ltv_cac, "x": "customer_segment", "y": "ltv_to_cac_ratio", "chart": "bar"},
    "3. RFM Segmentation (Top VIPs)": {"func": engine.run_f3_rfm_segmentation, "x": "customer_id", "y": "monetary_value", "chart": "bar"},
    "4. Cohort Retention": {"func": engine.run_f4_cohort_retention, "x": "month_index", "y": "cohort_month", "chart": "heatmap"},
    "5. Promotion Leakage Audit": {"func": engine.run_f5_promotion_leakage, "x": "campaign_name", "y": "discount_leakage_pct", "chart": "bubble"},
    "6. Logistics SLA & Delivery": {"func": engine.run_f6_logistics_sla, "x": "warehouse", "y": "late_deliveries", "chart": "bar"},
    "7. Returns Root Cause": {"func": engine.run_f7_return_root_cause, "x": "return_reason", "y": "total_returns", "chart": "donut"},
    "8. Multi-Channel Attribution": {"func": engine.run_f8_multichannel_attribution, "x": "sales_channel", "y": "roas", "chart": "bar"},
    "9. Supplier Velocity Scorecard": {"func": engine.run_f9_supplier_velocity, "x": "supplier", "y": "total_units_sold", "chart": "bar"},
    "10. VoC Sentiment Correlation": {"func": engine.run_f10_voc_sentiment, "x": "review_sentiment", "y": "avg_rating", "chart": "line"}
}

selection = st.sidebar.radio("Select Module:", list(modules.keys()), label_visibility="collapsed")

# ==========================================
# 4. DATA EXECUTION & DYNAMIC KPIs
# ==========================================
st.markdown("<div class='main-header'>Enterprise E-Commerce Analytics</div>", unsafe_allow_html=True)
st.markdown(f"<div class='sub-header'>📈 {selection}</div>", unsafe_allow_html=True)

active_module = modules[selection]
polars_df = active_module["func"]()
pandas_df = polars_df.to_pandas()
x_col, y_col = active_module["x"], active_module["y"]

kpi1, kpi2, kpi3 = st.columns(3)

if selection == "4. Cohort Retention":
    # Dedicated KPI calculations for Cohort Analysis
    m1_df = pandas_df[pandas_df["month_index"] == 1]
    avg_m1_retention = float(m1_df["retention_pct"].mean()) if not m1_df.empty else 0.0
    
    post_m0_df = pandas_df[pandas_df["month_index"] > 0]
    max_retention = float(post_m0_df["retention_pct"].max()) if not post_m0_df.empty else 0.0
    
    total_cohorts = pandas_df["cohort_month"].nunique()

    kpi1.metric(label="📅 Active Cohorts Tracked", value=f"{total_cohorts} Months")
    kpi2.metric(label="🔄 Avg Month-1 Retention", value=f"{avg_m1_retention:.1f}%")
    kpi3.metric(label="🎯 Peak Retention (Post-M0)", value=f"{max_retention:.1f}%")

else:
    # Standard Dynamic KPI calculations for ranking/volume modules
    top_category = str(pandas_df.iloc[0][x_col]) if not pandas_df.empty else "N/A"
    raw_y_val = pandas_df.iloc[0][y_col] if not pandas_df.empty else 0.0
    
    try:
        top_value_display = f"{float(raw_y_val):,.2f}"
    except (ValueError, TypeError):
        top_value_display = str(raw_y_val).split(" ")[0]

    if pd.api.types.is_numeric_dtype(pandas_df[y_col]):
        avg_value_display = f"{float(pandas_df[y_col].mean()):,.2f}"
    else:
        avg_value_display = "N/A"

    formatted_y_name = y_col.replace('_', ' ').title()
    kpi1.metric(label="🏆 Top Anchor", value=top_category)
    kpi2.metric(label=f"🚀 Max {formatted_y_name}", value=top_value_display)
    kpi3.metric(label=f"📊 Average {formatted_y_name}", value=avg_value_display)

st.markdown("<br>", unsafe_allow_html=True)

# ==========================================
# 5. ENTERPRISE VISUALIZATIONS (PLOTLY UPGRADES)
# ==========================================
if active_module["chart"] == "bar":
    fig = px.bar(pandas_df, x=x_col, y=y_col, text_auto='.2s', color=y_col, color_continuous_scale="Tealgrn", template="plotly_dark")
    fig.update_traces(textfont_size=13, textposition="outside", cliponaxis=False)

elif active_module["chart"] == "line":
    fig = px.line(pandas_df, x=x_col, y=y_col, markers=True, template="plotly_dark")

elif active_module["chart"] == "donut":
    fig = px.pie(pandas_df, names=x_col, values=y_col, hole=0.4, template="plotly_dark")
    fig.update_traces(textposition='inside', textinfo='percent+label')

elif active_module["chart"] == "bubble":
    fig = px.scatter(pandas_df, x=x_col, y=y_col, size="total_discount_given", color="gross_revenue", 
                     hover_name=x_col, size_max=60, template="plotly_dark", 
                     title="Promotion Impact: Circle Size = Discount Cost | Color = Total Revenue")

elif active_module["chart"] == "heatmap":
    pivot_df = pandas_df.pivot(index=y_col, columns=x_col, values="retention_pct")
    fig = px.imshow(pivot_df, text_auto=".1f", aspect="auto", color_continuous_scale="Tealgrn", template="plotly_dark",
                    title="Customer Retention Cohort (%)")
    fig.update_layout(xaxis_title="Months Since First Purchase", yaxis_title="Cohort Acquisition Month")

st.plotly_chart(fig, use_container_width=True)

with st.expander("🗄️ View Raw Data (Backend Polars Output)"):
    st.dataframe(pandas_df, use_container_width=True, hide_index=True)

# ==========================================
# 6. GENERATIVE AI CFO AGENT (MD&A)
# ==========================================
st.markdown("---")
st.subheader("🤖 Executive MD&A (AI Generated)")

if st.button("✨ Generate Strategic Insights", use_container_width=True):
    if not api_key:
        st.warning("⚠️ Please enter your Gemini API Key in the sidebar first.")
    else:
        with st.spinner("🧠 AI is analyzing the business data..."):
            try:
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel('gemini-3.6-flash')
                data_context = pandas_df.to_markdown(index=False)
                
                prompt = f"""
                You are a Chief Strategy Officer (CSO) analyzing a specific e-commerce data module: '{selection}'.
                Here is the latest data extracted from our DuckDB pipeline:
                {data_context}
                
                Write a concise, executive-grade Management Discussion & Analysis (MD&A) based ONLY on this data.
                Format your response in exactly 3 bullet points:
                1. **Key Observation:** The most critical trend or anomaly in the data.
                2. **Business Impact:** How this affects our revenue, margin, or operational efficiency.
                3. **Strategic Action:** One concrete recommendation for the team. 
                
                Keep the tone professional, objective, and strictly avoid generic filler words. Do not use the $ symbol for LaTeX, use standard text.
                """
                response = model.generate_content(prompt)
                
                # BUG FIXED: Escape Dollar Signs to prevent Streamlit from rendering LaTeX math equations
                safe_text = response.text.replace("$", "\$")
                st.info(safe_text)
                st.success("✅ AI Strategic Analysis Complete!")
            except Exception as e:
                st.error(f"❌ AI Engine Error: {e}")

# ==========================================
# 7. CHAT WITH DATA (THAI / ENGLISH)
# ==========================================
st.markdown("---")
st.subheader("💬 Chat with Data (Talk to your Analytics)")
st.caption("Ask questions about the current data above in Thai or English. (e.g. 'What is the Ad. spending stragegy?')")

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

for msg in st.session_state.chat_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt_input := st.chat_input("Ask AI about this data (Thai or English)..."):
    st.session_state.chat_messages.append({"role": "user", "content": prompt_input})
    with st.chat_message("user"):
        st.markdown(prompt_input)
    
    with st.chat_message("assistant"):
        if not api_key:
            st.warning("⚠️ Please enter your Gemini API Key in the sidebar first.")
        else:
            with st.spinner("🧠 AI is thinking..."):
                try:
                    genai.configure(api_key=api_key)
                    model = genai.GenerativeModel('gemini-3.6-flash')
                    
                    chat_context = f"""
                    You are a Data Analyst Assistant for an E-commerce company.
                    You must answer the user's question based ONLY on the following data context.
                    Reply in the same language the user used to ask (Thai or English).
                    
                    Data Context ({selection}):
                    {pandas_df.to_markdown(index=False)}
                    
                    User Question: {prompt_input}
                    """
                    
                    response = model.generate_content(chat_context)
                    # Escape Dollar Signs for chat too
                    safe_chat_text = response.text.replace("$", "\$")
                    st.markdown(safe_chat_text)
                    st.session_state.chat_messages.append({"role": "assistant", "content": safe_chat_text})
                except Exception as e:
                    st.error(f"❌ AI Chat Error: {e}")