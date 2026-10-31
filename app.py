import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.analytics import (
    clean_properties,
    build_kpis,
    location_summary,
    train_price_model,
    predict_price,
)

# -------------------------------------------------------------------
# EstateIQ — Editorial / property-market research layout
# -------------------------------------------------------------------
st.set_page_config(
    page_title="EstateIQ | Property Market Brief",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@600;700&display=swap');

    :root {
        --paper: #f5f2eb;
        --ink: #202b32;
        --muted: #68747a;
        --line: #dedbd2;
        --rust: #b85e42;
        --sage: #758b78;
        --navy: #263d4a;
    }
    .stApp { background: var(--paper); color: var(--ink); }
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .block-container { max-width: 1440px; padding-top: 1.4rem; padding-bottom: 3rem; }
    h1, h2, h3 { color: var(--ink); }
    h1 { font-family: 'Playfair Display', serif !important; letter-spacing: -1.2px; }
    h2, h3 { font-family: 'Playfair Display', serif !important; }
    [data-testid="stSidebar"] { background: #ebe7de; }
    [data-testid="stMetric"] {
        background: #fffdf8; border: 1px solid var(--line);
        border-radius: 4px; padding: 16px 18px;
    }
    [data-testid="stMetricLabel"] { color: var(--muted) !important; }
    [data-testid="stMetricValue"] { color: var(--ink) !important; }
    .hero {
        background: #263d4a; color: #fffaf1; padding: 34px 38px;
        border-radius: 5px; margin: 10px 0 22px 0;
        background-image: linear-gradient(115deg, rgba(38,61,74,.98), rgba(38,61,74,.88)),
          repeating-linear-gradient(135deg, transparent, transparent 18px, rgba(255,255,255,.04) 19px, rgba(255,255,255,.04) 20px);
    }
    .hero-kicker { color: #d6b59d; font-size: 11px; font-weight: 700; letter-spacing: 2.5px; }
    .hero-title { font-family: 'Playfair Display', serif; font-size: clamp(34px, 4vw, 54px); line-height: 1.08; margin: 9px 0; color: #fffaf1; }
    .hero-sub { color: #d5dfe0; font-size: 14px; max-width: 720px; }
    .section-kicker { color: var(--rust); font-size: 10px; font-weight: 700; letter-spacing: 2px; text-transform: uppercase; }
    .section-title { font-family: 'Playfair Display', serif; font-size: 27px; color: var(--ink); margin: 2px 0 12px 0; }
    .insight-card {
        background: #fffdf8; border: 1px solid var(--line); border-left: 4px solid var(--rust);
        padding: 17px 19px; border-radius: 3px; min-height: 105px;
    }
    .insight-label { color: var(--muted); text-transform: uppercase; letter-spacing: 1.2px; font-size: 10px; font-weight: 700; }
    .insight-value { color: var(--ink); font-family: 'Playfair Display', serif; font-size: 25px; margin-top: 6px; }
    .note { color: var(--muted); font-size: 12px; }
    div[data-testid="stTabs"] button { color: var(--ink); font-weight: 600; }
    div[data-testid="stTabs"] button[aria-selected="true"] { color: var(--rust); border-bottom-color: var(--rust); }
    /* Keep Streamlit's native widgets on the same light visual system. */
    [data-testid="stExpander"] {
        background: #fffdf8 !important;
        border: 1px solid var(--line) !important;
        border-radius: 5px !important;
        overflow: hidden;
    }
    [data-testid="stExpander"] details,
    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] summary:hover {
        background: #fffdf8 !important;
        color: var(--ink) !important;
    }
    [data-testid="stExpander"] summary p,
    [data-testid="stExpander"] summary span,
    [data-testid="stExpander"] summary svg {
        color: var(--ink) !important;
        fill: var(--ink) !important;
    }
    [data-testid="stWidgetLabel"] p,
    [data-testid="stWidgetLabel"] label,
    [data-testid="stFileUploader"] label,
    [data-testid="stMarkdownContainer"] p {
        color: var(--ink);
    }
    [data-testid="stFileUploader"] section {
        background: #fffdf8 !important;
        border: 1px dashed #b9b3a8 !important;
        border-radius: 5px !important;
    }
    [data-testid="stFileUploader"] section div,
    [data-testid="stFileUploader"] section span {
        color: var(--ink) !important;
    }
    [data-baseweb="select"] > div {
        background: #fffdf8 !important;
        border-color: #cfc9bd !important;
        color: var(--ink) !important;
    }
    [data-baseweb="select"] input,
    [data-baseweb="select"] [role="combobox"],
    [data-baseweb="select"] span {
        color: var(--ink) !important;
        -webkit-text-fill-color: var(--ink) !important;
    }
    [data-baseweb="tag"] {
        background: #263d4a !important;
        border: 0 !important;
        color: #fffaf1 !important;
    }
    [data-baseweb="tag"] span,
    [data-baseweb="tag"] svg {
        color: #fffaf1 !important;
        fill: #fffaf1 !important;
    }
    [data-baseweb="popover"] [role="listbox"],
    [data-baseweb="popover"] [role="option"] {
        background: #fffdf8 !important;
        color: var(--ink) !important;
    }
    [data-baseweb="popover"] [role="option"]:hover {
        background: #eee8dc !important;
    }
    input, textarea {
        background: #fffdf8 !important;
        color: var(--ink) !important;
        border-color: #cfc9bd !important;
    }
    .stButton > button, .stDownloadButton > button {
        border-radius: 3px; border: 1px solid var(--navy); background: var(--navy);
        color: #fffaf1 !important; font-weight: 600;
    }
    .stButton > button:hover, .stDownloadButton > button:hover { background: var(--rust); color: white; border-color: var(--rust); }
    div[data-testid="stTabs"] button { color: var(--ink) !important; }
    div[data-testid="stTabs"] button[aria-selected="true"] { color: var(--rust) !important; border-bottom-color: var(--rust) !important; }
    [data-testid="stMetric"] { background: #fffdf8 !important; }
    [data-testid="stMetricLabel"], [data-testid="stMetricValue"] { color: var(--ink) !important; }
    hr { border-color: var(--line); }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header / masthead
mast_left, mast_right = st.columns([3, 1])
with mast_left:
    st.markdown("**ESTATEIQ**  <span style='color:#b85e42'>/</span> PROPERTY MARKET RESEARCH", unsafe_allow_html=True)
with mast_right:
    st.markdown("<div style='text-align:right;color:#68747a;font-size:12px'>DATA DESK · INDIA</div>", unsafe_allow_html=True)

st.markdown(
    """
    <div class="hero">
      <div class="hero-kicker">THE PROPERTY MARKET BRIEF</div>
      <div class="hero-title">A clearer view of<br>where homes are valued.</div>
      <div class="hero-sub">Explore residential property prices, locality differences, rental signals and indicative valuation patterns through one interactive research workspace.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Data input area
with st.expander("DATA & MARKET FILTERS", expanded=True):
    source_col, city_col, bhk_col, type_col = st.columns([1.5, 1.2, 1, 1.2])

    @st.cache_data
    def load_sample():
        return pd.read_csv("data/raw/estateiq_sample_synthetic.csv")

    with source_col:
        uploaded = st.file_uploader("Upload property CSV (optional)", type=["csv"], key="estateiq_upload")
    if uploaded is not None:
        raw = pd.read_csv(uploaded)
        source_label = "Uploaded dataset"
    else:
        raw = load_sample()
        source_label = "Included demonstration dataset"

    try:
        data = clean_properties(raw)
    except Exception as exc:
        st.error(f"Unable to process the selected dataset: {exc}")
        st.info("Required fields: city, locality, property_type, bhk, area_sqft and price_inr.")
        st.stop()

    with city_col:
        all_cities = sorted(data.city.dropna().unique().tolist())
        chosen_cities = st.multiselect("Markets", all_cities, default=all_cities)
    filtered = data[data.city.isin(chosen_cities)] if chosen_cities else data.iloc[0:0]

    with bhk_col:
        all_bhks = sorted(filtered.bhk.dropna().unique().tolist()) if not filtered.empty else []
        chosen_bhks = st.multiselect("Bedrooms", all_bhks, default=all_bhks)
    if chosen_bhks:
        filtered = filtered[filtered.bhk.isin(chosen_bhks)]

    with type_col:
        all_types = sorted(filtered.property_type.dropna().unique().tolist()) if not filtered.empty else []
        chosen_types = st.multiselect("Property category", all_types, default=all_types)
    if chosen_types:
        filtered = filtered[filtered.property_type.isin(chosen_types)]

st.markdown(
    f"<div class='note'>CURRENT VIEW: {source_label.upper()} &nbsp;·&nbsp; {len(filtered):,} MATCHING RECORDS</div>",
    unsafe_allow_html=True,
)

if filtered.empty:
    st.warning("No records match these filters. Change the market or property filters above.")
    st.stop()

kpis = build_kpis(filtered)
metric_cols = st.columns(4)
metric_cols[0].metric("Properties in view", f"{kpis['count']:,}")
metric_cols[1].metric("Median asking price", f"₹{kpis['median_price']/100000:.1f} L")
metric_cols[2].metric("Average rate / sq.ft.", f"₹{kpis['avg_ppsf']:,.0f}")
metric_cols[3].metric("Typical property size", f"{kpis['median_area']:,.0f} sq.ft.")

# Content navigation
overview_tab, location_tab, homes_tab, rental_tab, valuation_tab, records_tab = st.tabs(
    ["01  Market Snapshot", "02  Location Desk", "03  Property Mix", "04  Rental Lens", "05  Valuation Lab", "06  Records"]
)

plot_template = "plotly_white"
plot_layout = dict(
    template=plot_template,
    paper_bgcolor="#fffdf8",
    plot_bgcolor="#fffdf8",
    font=dict(family="DM Sans", color="#26343b"),
    margin=dict(l=20, r=20, t=55, b=20),
)

with overview_tab:
    st.markdown("<div class='section-kicker'>01 / market snapshot</div><div class='section-title'>The market, at a glance</div>", unsafe_allow_html=True)
    col_a, col_b = st.columns([1.35, 1])
    with col_a:
        city_medians = filtered.groupby("city", as_index=False).price_inr.median().sort_values("price_inr", ascending=True)
        fig = px.bar(city_medians, x="price_inr", y="city", orientation="h", text_auto=".2s",
                     title="Median property price by city", color_discrete_sequence=["#b85e42"])
        fig.update_layout(**plot_layout, height=390, xaxis_title="Median price (INR)", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)
    with col_b:
        fig = px.histogram(filtered, x="price_inr", nbins=32, title="How asking prices are distributed",
                           color_discrete_sequence=["#758b78"])
        fig.update_layout(**plot_layout, height=390, xaxis_title="Asking price (INR)", yaxis_title="Listings")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div class='section-kicker'>Market signal</div>", unsafe_allow_html=True)
    top_city = city_medians.sort_values("price_inr", ascending=False).iloc[0]
    lowest_city = city_medians.sort_values("price_inr").iloc[0]
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""<div class="insight-card"><div class="insight-label">Highest median in current view</div><div class="insight-value">{top_city['city']}</div><div class="note">Median listed price: ₹{top_city['price_inr']:,.0f}</div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="insight-card"><div class="insight-label">Lowest median in current view</div><div class="insight-value">{lowest_city['city']}</div><div class="note">Median listed price: ₹{lowest_city['price_inr']:,.0f}</div></div>""", unsafe_allow_html=True)

    if "listing_date" in filtered and filtered.listing_date.notna().any():
        dated = filtered.dropna(subset=["listing_date"]).copy()
        dated["listing_month"] = dated.listing_date.dt.to_period("M").astype(str)
        monthly = dated.groupby("listing_month", as_index=False).price_inr.median()
        if len(monthly) > 1:
            fig = px.line(monthly, x="listing_month", y="price_inr", markers=True,
                          title="Median listing price by listing month", color_discrete_sequence=["#263d4a"])
            fig.update_layout(**plot_layout, height=330, xaxis_title="", yaxis_title="Median price (INR)")
            st.plotly_chart(fig, use_container_width=True)
            st.caption("This is a trend within the available listing sample, not an official property price index.")

with location_tab:
    st.markdown("<div class='section-kicker'>02 / location desk</div><div class='section-title'>Compare neighbourhoods, not just cities</div>", unsafe_allow_html=True)
    loc = location_summary(filtered)
    top_n = st.slider("Number of localities to display", 5, min(30, max(5, len(loc))), min(15, max(5, len(loc))))
    display_loc = loc.head(top_n).sort_values("median_price")
    fig = px.bar(display_loc, x="median_price", y="locality_label", orientation="h", color="city",
                 title="Localities with the highest median asking prices")
    fig.update_layout(**plot_layout, height=max(420, top_n*30), xaxis_title="Median price (INR)", yaxis_title="")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(loc[["city","locality","listings","median_price","avg_price_per_sqft","median_area"]],
                 hide_index=True, use_container_width=True)
    if {"latitude", "longitude"}.issubset(filtered.columns):
        map_data = filtered.dropna(subset=["latitude", "longitude"])
        if not map_data.empty:
            st.subheader("Listing locations")
            st.map(map_data.rename(columns={"latitude":"lat", "longitude":"lon"}))

with homes_tab:
    st.markdown("<div class='section-kicker'>03 / property mix</div><div class='section-title'>What kind of homes make up this sample?</div>", unsafe_allow_html=True)
    left, right = st.columns(2)
    with left:
        fig = px.box(filtered, x="bhk", y="price_inr", color="property_type",
                     title="Asking price by bedroom count")
        fig.update_layout(**plot_layout, height=400, xaxis_title="Bedrooms (BHK)", yaxis_title="Price (INR)")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        size_col = "amenities_count" if filtered["amenities_count"].notna().any() else None
        fig = px.scatter(filtered, x="area_sqft", y="price_inr", color="city", size=size_col,
                         hover_data=["locality","bhk"], title="Property size versus asking price")
        fig.update_layout(**plot_layout, height=400, xaxis_title="Area (sq.ft.)", yaxis_title="Price (INR)")
        st.plotly_chart(fig, use_container_width=True)
    if "furnishing" in filtered and filtered.furnishing.notna().any():
        furnishing = filtered.groupby("furnishing", as_index=False).price_inr.median()
        fig = px.bar(furnishing, x="furnishing", y="price_inr", title="Median asking price by furnishing",
                     color_discrete_sequence=["#758b78"])
        fig.update_layout(**plot_layout, height=320, xaxis_title="", yaxis_title="Median price (INR)")
        st.plotly_chart(fig, use_container_width=True)

with rental_tab:
    st.markdown("<div class='section-kicker'>04 / rental lens</div><div class='section-title'>Look beyond the purchase price</div>", unsafe_allow_html=True)
    rental = filtered.dropna(subset=["monthly_rent_inr","price_inr"]).copy()
    rental = rental[(rental.price_inr > 0) & (rental.monthly_rent_inr >= 0)]
    if rental.empty:
        st.info("The selected dataset does not contain usable monthly rent values.")
    else:
        rental["gross_yield"] = rental.monthly_rent_inr * 12 / rental.price_inr * 100
        a,b,c = st.columns(3)
        a.metric("Median monthly rent", f"₹{rental.monthly_rent_inr.median():,.0f}")
        b.metric("Median gross yield", f"{rental.gross_yield.median():.2f}%")
        c.metric("Records with rent data", f"{len(rental):,}")
        fig = px.box(rental, x="city", y="gross_yield", color="city", title="Gross rental yield by city")
        fig.update_layout(**plot_layout, height=390, showlegend=False, xaxis_title="", yaxis_title="Gross yield (%)")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Gross yield = monthly rent × 12 ÷ property price. It excludes vacancy, taxes, maintenance, financing and transaction costs.")

with valuation_tab:
    st.markdown("<div class='section-kicker'>05 / valuation lab</div><div class='section-title'>An indicative price model</div>", unsafe_allow_html=True)
    st.write("Train a Random Forest model on the currently filtered records, then explore an indicative estimate for a property profile.")
    model_data = filtered.dropna(subset=["price_inr","area_sqft","bhk","city","property_type"])
    if len(model_data) < 30:
        st.warning("At least 30 complete records are recommended to train the estimator.")
    else:
        with st.spinner("Training the model on the current filtered sample..."):
            model, score, mae, _ = train_price_model(model_data)
        score_a, score_b = st.columns(2)
        score_a.metric("Holdout R²", f"{score:.3f}")
        score_b.metric("Holdout mean absolute error", f"₹{mae:,.0f}")
        st.caption("The model uses a random 80/20 train/test split. A low or negative R² indicates weak predictive performance.")
        c1,c2,c3 = st.columns(3)
        city_choice = c1.selectbox("City", sorted(model_data.city.unique()), key="estimate_city")
        available_localities = sorted(model_data[model_data.city == city_choice].locality.unique())
        locality_choice = c2.selectbox("Locality", available_localities, key="estimate_locality")
        bhk_choice = c3.selectbox("Bedrooms", sorted(model_data.bhk.unique()), key="estimate_bhk")
        c4,c5,c6 = st.columns(3)
        area_choice = c4.number_input("Area (sq.ft.)", 200.0, 10000.0, 1200.0, 50.0)
        property_choice = c5.selectbox("Property type", sorted(model_data.property_type.unique()), key="estimate_type")
        age_choice = c6.number_input("Property age (years)", 0, 100, 5)
        estimate = predict_price(model, city_choice, locality_choice, bhk_choice, area_choice, property_choice, age_choice)
        st.markdown(f"""<div class="insight-card"><div class="insight-label">Model estimate · educational only</div><div class="insight-value">₹{estimate:,.0f}</div><div class="note">Approximately ₹{estimate/100000:.2f} lakh for the selected property profile.</div></div>""", unsafe_allow_html=True)
        st.caption("This is not a formal valuation and may not generalize to current market listings.")

with records_tab:
    st.markdown("<div class='section-kicker'>06 / data room</div><div class='section-title'>Inspect the records behind the charts</div>", unsafe_allow_html=True)
    st.dataframe(filtered, use_container_width=True, height=460, hide_index=True)
    st.download_button(
        "Download filtered records · CSV",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="estateiq_filtered_properties.csv",
        mime="text/csv",
    )

st.divider()
footer_left, footer_right = st.columns([3,1])
with footer_left:
    st.markdown("**ESTATEIQ**  ·  Real estate market analytics")
    st.caption("Built with Python, Streamlit, Pandas, Plotly and Scikit-learn.")
with footer_right:
    st.markdown("<div style='text-align:right;color:#68747a;font-size:11px'>RESEARCH WORKSPACE<br>EDUCATIONAL USE</div>", unsafe_allow_html=True)
st.caption("The bundled dataset is synthetic demonstration data, not actual property listings or verified market prices. Replace it with a properly licensed real dataset for real-world conclusions. EstateIQ is not a property valuation service or investment advice.")
