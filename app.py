import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# -------------- PAGE CONFIG & CSS --------------
st.set_page_config(page_title="Green to Brown Utilization Stats", layout="wide", page_icon="✈️")

st.markdown("""
    <style>
    .main { padding: 0rem 1rem; }
    h1 { color: #333; font-size: 2.5rem; font-weight: 600; }
    .metric-container { background-color: #f5f5f5; padding: 1rem; border-radius: 8px; margin: 0.5rem 0; text-align: center; }
    .metric-value { font-size: 2rem; font-weight: bold; margin: 0.5rem 0; }
    .metric-label { font-size: 0.9rem; color: #666; margin-bottom: 0.3rem; }
    .brown-text { color: #8B4513; }
    .green-text { color: #228B22; }
    .stTabs [data-baseweb="tab-list"] { gap: 2rem; }
    .stTabs [data-baseweb="tab"] { height: 50px; padding-left: 20px; padding-right: 20px; font-size: 1.1rem; font-weight: 500; }
    div[data-testid="metric-container"] { background-color: #f5f5f5; border: 1px solid #e0e0e0; padding: 1rem; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
""", unsafe_allow_html=True)

# -------------- FLEXIBLE COLUMN HELPERS --------------
def pick_col(df, candidates, required=True, friendly_name=None):
    """
    Return the first existing column among 'candidates'.
    If not found:
      - if required: raise a Streamlit error and return None
      - else: return None
    """
    cols_norm = {c.strip().lower(): c for c in df.columns}
    for cand in candidates:
        key = cand.strip().lower()
        if key in cols_norm:
            return cols_norm[key]
    if required:
        label = friendly_name or candidates[0]
        st.error(f"Required column not found. Looked for: {candidates} (e.g., '{label}')")
    return None

def map_region(country):
    """
    Map a country name to high-level region buckets: EMEA / AMERICAS / APAC / OTHER.
    This is a simple, pragmatic mapping; extend if you need finer granularity.
    """
    if country is None or pd.isna(country):
        return "OTHER"
    c = str(country).strip().upper()

    emea = {
        # Europe (partial but broad coverage)
        "ALBANIA","ANDORRA","AUSTRIA","BELARUS","BELGIUM","BOSNIA AND HERZEGOVINA","BULGARIA","CROATIA","CYPRUS",
        "CZECH REPUBLIC","CZECHIA","DENMARK","ESTONIA","FINLAND","FRANCE","GERMANY","GREECE","HUNGARY","ICELAND",
        "IRELAND","ITALY","KOSOVO","LATVIA","LIECHTENSTEIN","LITHUANIA","LUXEMBOURG","MALTA","MOLDOVA","MONACO",
        "MONTENEGRO","NETHERLANDS","NORTH MACEDONIA","NORWAY","POLAND","PORTUGAL","ROMANIA","SAN MARINO","SERBIA",
        "SLOVAKIA","SLOVENIA","SPAIN","SWEDEN","SWITZERLAND","UK","UNITED KINGDOM","VATICAN CITY",
        # Middle East / Africa (coarse)
        "UNITED ARAB EMIRATES","UAE","SAUDI ARABIA","QATAR","BAHRAIN","KUWAIT","OMAN","ISRAEL","JORDAN","LEBANON",
        "EGYPT","MOROCCO","ALGERIA","TUNISIA","SOUTH AFRICA","NIGERIA","KENYA","GHANA","ETHIOPIA"
    }
    americas = {
        "USA","UNITED STATES","UNITED STATES OF AMERICA","CANADA","MEXICO","BRAZIL","ARGENTINA","CHILE","COLOMBIA",
        "PERU","URUGUAY","PARAGUAY","BOLIVIA","ECUADOR","VENEZUELA","GUATEMALA","PANAMA","COSTA RICA","DOMINICAN REPUBLIC"
    }
    apac = {
        "CHINA","HONG KONG","TAIWAN","JAPAN","SOUTH KOREA","KOREA, REPUBLIC OF","SINGAPORE","MALAYSIA","THAILAND",
        "VIETNAM","INDONESIA","PHILIPPINES","INDIA","PAKISTAN","SRI LANKA","NEPAL","BANGLADESH","CAMBODIA","LAOS",
        "AUSTRALIA","NEW ZEALAND"
    }

    if c in emea:
        return "EMEA"
    if c in americas:
        return "AMERICAS"
    if c in apac:
        return "APAC"
    return "OTHER"

# -------------- DATA LOADING & PROCESSING --------------
@st.cache_data
def load_and_process_data(uploaded_file):
    """
    Load and harmonize the Excel file, handling both the old and new schemas.
    - Date: prefer 'OriginDeparture Date', else 'POB as text'
    - Weight: prefer 'Volumetric Weight (KG)' (or close variants)
    - Airline: 'Airline' (or close variants)
    - Regions: derive from Origin/Destination Country if missing
    """
    try:
        # Read first sheet automatically, or use sheet 'Export' if present
        xls = pd.ExcelFile(uploaded_file)
        sheet = "Export" if "Export" in xls.sheet_names else xls.sheet_names[0]
        df = pd.read_excel(uploaded_file, sheet_name=sheet, engine="openpyxl")

        # Normalize col names (trim)
        df.columns = df.columns.str.strip()

        # --- Identify columns (flexible) ---
        date_col = pick_col(df, ["OriginDeparture Date", "POB as text", "POB", "Date"], required=True, friendly_name="Date")
        airline_col = pick_col(df, ["Airline", "Carrier", "Airline Name"], required=True, friendly_name="Airline")
        weight_col = pick_col(df, ["Volumetric Weight (KG)", "Volumetric Weight", "Chargeable Weight (KG)", "Weight (KG)", "Weight"], required=True, friendly_name="Weight (KG)")

        # Regions / Countries (optional; we can derive if missing)
        region_lane_col = pick_col(df, ["Region Lane"], required=False)
        origin_region_col = pick_col(df, ["Origin Region"], required=False)
        dest_region_col = pick_col(df, ["Destination Region"], required=False)

        origin_country_col = pick_col(df, ["Origin Country","OriginCountry","Orig Country"], required=False)
        dest_country_col   = pick_col(df, ["Destination Country","DestinationCountry","Dest Country"], required=False)

        origin_iata_col = pick_col(df, ["Origin IATA","OriginIATA","ORI IATA","ORIGIN"], required=False)
        dest_iata_col   = pick_col(df, ["Destination IATA","DestinationIATA","DEST IATA","DESTINATION"], required=False)

        # --- Parse date ---
        if date_col is None:
            return None
        df["Date"] = pd.to_datetime(df[date_col], errors="coerce", infer_datetime_format=True)
        df = df[df["Date"].notna()].copy()

        # Extract month / year / month name
        df["Month"] = df["Date"].dt.month
        df["Year"] = df["Date"].dt.year
        df["Month_Name"] = df["Date"].dt.strftime("%B")

        # --- Weight ---
        df["Weight_KG"] = pd.to_numeric(df[weight_col], errors="coerce").fillna(0.0)

        # --- Airline / UPS flag ---
        df["Airline"] = df[airline_col].astype(str)
        df["Is_UPS"] = df["Airline"].str.upper().str.contains("UPS", na=False)

        # --- Regions ---
        # If we explicitly have Region Lane, keep it. Otherwise create from origin country.
        if region_lane_col:
            df["Region Lane"] = df[region_lane_col].astype(str)
        else:
            # Build Origin/Destination Regions first
            if origin_region_col:
                df["Origin Region"] = df[origin_region_col].astype(str)
            else:
                df["Origin Region"] = df[origin_country_col].map(map_region) if origin_country_col else "OTHER"

            if dest_region_col:
                df["Destination Region"] = df[dest_region_col].astype(str)
            else:
                df["Destination Region"] = df[dest_country_col].map(map_region) if dest_country_col else "OTHER"

            # Use Origin Region as "Region Lane" proxy (like prior screenshots showed repeated names)
            df["Region Lane"] = df["Origin Region"]

        # --- Optional: synthetic cost if you don't have a cost column ---
        # Replace with your real column if available, e.g. df["Cost"] = df["Total Cost (EUR)"]
        if "Cost" not in df.columns:
            np.random.seed(42)
            df["Cost"] = df["Weight_KG"] * np.random.uniform(8, 12, len(df))
        if "Commercial_Cost" not in df.columns:
            df["Commercial_Cost"] = df["Cost"] * 1.15  # Example: 15% more than UPS

        # Keep some helpful fallback columns for Tab 2
        if "Origin Region" not in df.columns:
            df["Origin Region"] = df["Region Lane"]
        if "Destination Region" not in df.columns:
            # If we couldn’t build a destination region, mirror origin
            df["Destination Region"] = df["Region Lane"]

        # Also keep IATA pair if present (nice for drilldowns)
        if origin_iata_col and dest_iata_col:
            df["IATA Pair"] = df[origin_iata_col].astype(str) + "-" + df[dest_iata_col].astype(str)
        else:
            df["IATA Pair"] = df["Origin Region"].astype(str) + "-" + df["Destination Region"].astype(str)

        return df

    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return None

# -------------- METRICS & CHARTS --------------
def calculate_metrics(df):
    metrics = {}
    metrics['brown_volume'] = df[df['Is_UPS']]['Weight_KG'].sum()
    metrics['green_volume'] = df[~df['Is_UPS']]['Weight_KG'].sum()
    metrics['total_volume'] = df['Weight_KG'].sum()

    metrics['utilization'] = (metrics['brown_volume'] / metrics['total_volume'] * 100) if metrics['total_volume'] > 0 else 0.0

    ups_df = df[df['Is_UPS']]
    other_df = df[~df['Is_UPS']]
    metrics['brown_cost'] = ups_df['Cost'].sum()
    metrics['green_cost'] = other_df['Cost'].sum()
    metrics['brown_cost_kg'] = (metrics['brown_cost'] / metrics['brown_volume']) if metrics['brown_volume'] > 0 else 0.0
    metrics['green_cost_kg'] = (metrics['green_cost'] / metrics['green_volume']) if metrics['green_volume'] > 0 else 0.0

    metrics['savings'] = ups_df['Commercial_Cost'].sum() - ups_df['Cost'].sum()
    return metrics

def format_number(num, decimals=0):
    return f"{num:,.{decimals}f}" if isinstance(decimals, int) and decimals > 0 else f"{num:,.0f}"

def create_utilization_chart(monthly_data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=monthly_data['Month'],
        y=monthly_data['Utilization_%'],
        mode='lines+markers',
        name='Utilization %',
        line=dict(color='#FF6B35', width=3),
        marker=dict(size=10),
        text=[f"{val:.1f}%" for val in monthly_data['Utilization_%']],
        textposition='top center',
        hovertemplate='%{x}<br>Utilization: %{y:.1f}%<extra></extra>'
    ))
    fig.update_layout(
        title='BT Utilization % by Month and Year',
        xaxis_title='Month',
        yaxis_title='%',
        yaxis=dict(range=[0, 100], showgrid=True, gridcolor='#E0E0E0'),
        height=400,
        hovermode='x',
        showlegend=False,
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(showgrid=False)
    )
    return fig

# -------------- APP --------------
def main():
    # Title
    col1, col2, col3 = st.columns([2, 3, 1])
    with col2:
        st.markdown("<h1 style='text-align: center;'>Green to Brown <span style='color: #008B8B;'>Overall Utilization Stats</span> <span style='color: #FFA500;'>YoY</span></h1>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Choose Excel file", type=['xlsx', 'xls'], label_visibility="collapsed")

    if uploaded_file is None:
        st.info("👆 Please upload an Excel file to view the dashboard")
        with st.expander("📋 Column detection logic we use"):
            st.markdown("""
            **Date:** `OriginDeparture Date` → fallback: `POB as text`/`POB`/`Date`  
            **Airline:** `Airline` → fallback: `Carrier` / `Airline Name`  
            **Weight:** `Volumetric Weight (KG)` → fallback: `Volumetric Weight` / `Chargeable Weight (KG)` / `Weight (KG)` / `Weight`  
            **Regions:** if `Region Lane` / `Origin Region` / `Destination Region` missing, we derive from countries.  
            """)
        return

    with st.spinner('Processing data...'):
        df = load_and_process_data(uploaded_file)

    if df is None or df.empty:
        st.error("No valid data to display")
        return

    # Current year filter
    current_year = datetime.now().year
    if current_year not in df['Year'].unique():
        current_year = df['Year'].max()
    df_current = df[df['Year'] == current_year].copy()

    tab1, tab2 = st.tabs(["📊 Year Overview", "📈 Monthly Analysis"])

    # ---------------- TAB 1 ----------------
    with tab1:
        st.markdown("### This Year To Date:")

        months_order = ['January','February','March','April','May','June','July','August','September','October','November','December']
        monthly_metrics = []
        for month in range(1, 13):
            month_df = df_current[df_current['Month'] == month]
            if not month_df.empty:
                m = calculate_metrics(month_df)
                monthly_metrics.append({
                    'Month': months_order[month-1],
                    'Brown Volume (kg)': m['brown_volume'],
                    'Green Volume (kg)': m['green_volume'],
                    'Utilization%': m['utilization'],
                    'Savings': m['savings'],
                    'Weight Impact': m['brown_volume'] + m['green_volume'],
                    'Brown Cost/kg': m['brown_cost_kg'],
                    'Green Cost/kg': m['green_cost_kg']
                })

        if monthly_metrics:
            metrics_df = pd.DataFrame(monthly_metrics)
            total = calculate_metrics(df_current)
            total_row = {
                'Month': 'Total',
                'Brown Volume (kg)': total['brown_volume'],
                'Green Volume (kg)': total['green_volume'],
                'Utilization%': total['utilization'],
                'Savings': total['savings'],
                'Weight Impact': total['total_volume'],
                'Brown Cost/kg': total['brown_cost_kg'],
                'Green Cost/kg': total['green_cost_kg']
            }

            display_df = metrics_df.copy()
            display_df['Brown Volume (kg)'] = display_df['Brown Volume (kg)'].apply(lambda x: format_number(x))
            display_df['Green Volume (kg)'] = display_df['Green Volume (kg)'].apply(lambda x: format_number(x))
            display_df['Utilization%'] = display_df['Utilization%'].apply(lambda x: f"{x:.1f}%")
            display_df['Savings'] = display_df['Savings'].apply(lambda x: f"${format_number(x)}")
            display_df['Weight Impact'] = display_df['Weight Impact'].apply(lambda x: f"({format_number(x)})")
            display_df['Brown Cost/kg'] = display_df['Brown Cost/kg'].apply(lambda x: f"${x:.2f}")
            display_df['Green Cost/kg'] = display_df['Green Cost/kg'].apply(lambda x: f"${x:.2f}")

            total_display = {
                'Month': 'Total',
                'Brown Volume (kg)': format_number(total_row['Brown Volume (kg)']),
                'Green Volume (kg)': format_number(total_row['Green Volume (kg)']),
                'Utilization%': f"{total_row['Utilization%']:.1f}%",
                'Savings': f"${format_number(total_row['Savings'])}",
                'Weight Impact': f"({format_number(total_row['Weight Impact'])})",
                'Brown Cost/kg': f"${total_row['Brown Cost/kg']:.2f}",
                'Green Cost/kg': f"${total_row['Green Cost/kg']:.2f}"
            }

            display_df = pd.concat([display_df, pd.DataFrame([total_display])], ignore_index=True)
            st.dataframe(display_df, use_container_width=True, hide_index=True, height=500)

            # Utilization Chart
            st.markdown("### BT Utilization % by Month and Year")
            chart_data = metrics_df.rename(columns={'Utilization%':'Utilization_%'}).copy()
            fig = create_utilization_chart(chart_data[['Month','Utilization_%']])
            st.plotly_chart(fig, use_container_width=True)

            # YTD (kept simple, no LY in the file)
            col1, _ = st.columns(2)
            with col1:
                st.markdown("### YTD Snapshot")
                comp_df = pd.DataFrame({
                    'Metric': ['Utilization %', 'Brown Cost/kg', 'Green Cost/kg'],
                    f'{current_year}': [f"{total['utilization']:.1f}%", f"${total['brown_cost_kg']:.2f}", f"${total['green_cost_kg']:.2f}"]
                })
                st.dataframe(comp_df, use_container_width=True, hide_index=True)

    # ---------------- TAB 2 ----------------
    with tab2:
        st.markdown("### Green to Brown <span style='color: #008B8B;'>Monthly Utilization Stats</span>", unsafe_allow_html=True)

        available_months = sorted(df_current['Month'].unique())
        months_order = ['January','February','March','April','May','June','July','August','September','October','November','December']
        month_names = [months_order[m-1] for m in available_months if 1 <= m <= 12]

        selected_month_name = st.selectbox("Select Month", month_names)
        selected_month = months_order.index(selected_month_name) + 1

        df_month = df_current[df_current['Month'] == selected_month]
        if df_month.empty:
            st.info("No data for the selected month.")
            return

        # Key metrics
        month_metrics = calculate_metrics(df_month)
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("BT Utilization", f"{month_metrics['utilization']:.1f}%")
        c2.metric("% Effective", "—")  # optional KPI
        c3.metric("This Year Volume", f"{month_metrics['total_volume']/1_000_000:.2f}M kg")
        c4.metric("Savings (UPS vs Commercial)", f"${month_metrics['savings']/1_000_000:.2f}M")
        c5.metric("Brown Volume", f"{month_metrics['brown_volume']/1_000:.1f}K kg")
        c6.metric("Green Volume", f"{month_metrics['green_volume']/1_000:.1f}K kg")

        # Two columns for tables
        t1, t2 = st.columns(2)

        with t1:
            st.markdown("#### Utilization by Region (Origin Region proxy)")
            reg_rows = []
            for region, g in df_month.groupby("Region Lane"):
                m = calculate_metrics(g)
                reg_rows.append({
                    'Region': region,
                    'Utilization %': f"{m['utilization']:.1f}%",
                    'Brown Volume (kg)': format_number(m['brown_volume']),
                    'Green Volume (kg)': format_number(m['green_volume']),
                    'Brown Cost/kg': f"${m['brown_cost_kg']:.2f}",
                    'Green Cost/kg': f"${m['green_cost_kg']:.2f}",
                    'Savings ($)': f"${format_number(m['savings'])}"
                })
            st.dataframe(pd.DataFrame(reg_rows), use_container_width=True, hide_index=True, height=420)

        with t2:
            st.markdown("#### Savings Impact by Region")
            sav_rows = []
            for region, g in df_month.groupby("Region Lane"):
                m = calculate_metrics(g)
                sav_rows.append({
                    'Region': region,
                    'BT Volume (kg)': format_number(m['brown_volume']),
                    'Actual Savings ($)': f"${format_number(m['savings'])}"
                })
            st.dataframe(pd.DataFrame(sav_rows), use_container_width=True, hide_index=True, height=420)

        st.markdown("#### Analysis by Region Pair (Origin-Destination)")
        pair_rows = []
        for (o, d), g in df_month.groupby(["Origin Region", "Destination Region"]):
            m = calculate_metrics(g)
            pair_rows.append({
                'Region Pair': f"{o}-{d}",
                'Utilization %': f"{m['utilization']:.1f}%",
                'BT Volume (kg)': format_number(m['brown_volume']),
                'Green Volume (kg)': format_number(m['green_volume']),
                'Brown Cost/kg': f"${m['brown_cost_kg']:.2f}",
                'Green Cost/kg': f"${m['green_cost_kg']:.2f}",
                'Savings ($)': f"${format_number(m['savings'])}"
            })
        # Limit display to top 50 by BT volume for readability
        pair_df = pd.DataFrame(pair_rows)
        if not pair_df.empty:
            pair_df["BT Volume (num)"] = pair_df["BT Volume (kg)"].str.replace(",","").astype(float)
            pair_df = pair_df.sort_values("BT Volume (num)", ascending=False).drop(columns=["BT Volume (num)"]).head(50)
        st.dataframe(pair_df, use_container_width=True, hide_index=True, height=500)

if __name__ == "__main__":
    main()
