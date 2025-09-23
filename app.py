import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# -------------------- PAGE & CSS --------------------
st.set_page_config(page_title="Green to Brown Utilization Stats", layout="wide", page_icon="✈️")
st.markdown("""
<style>
.main { padding:0rem 1rem; }
h1 { color:#333; font-size:2.5rem; font-weight:600; }
.stTabs [data-baseweb="tab-list"]{ gap:2rem; }
.stTabs [data-baseweb="tab"]{ height:50px; padding:0 20px; font-size:1.1rem; font-weight:500; }
div[data-testid="metric-container"]{
  background:#f5f5f5; border:1px solid #e0e0e0; padding:1rem; border-radius:8px; box-shadow:0 2px 4px rgba(0,0,0,0.05);
}
.brown { color:#8B4513; font-weight:700; }
.green { color:#228B22; font-weight:700; }
</style>
""", unsafe_allow_html=True)

# -------------------- IATA → REGION MAPPING --------------------
# Coarse mapping good enough for analytics (expand as you see these in your file)
EUROPE_CODES = {
    "AMS","CDG","ORY","FRA","MUC","DUS","BER","HAM","CGN","BRU","LGG","LHR","LGW","STN","LTN","MAN","BHX","EDI",
    "DUB","MAD","BCN","VLC","AGP","PMI","LIS","OPO","MXP","LIN","FCO","VCE","ATH","ZRH","GVA","VIE","PRG","WAW",
    "KRK","OSL","CPH","ARN","HEL","RIX","TLL","VNO","BUD","BEG","OTP","SOF","SKG","IST"
}
NORTH_AMERICA_CODES = {"JFK","EWR","BOS","PHL","IAD","DCA","CLT","ATL","MIA","MCO","TPA","DFW","IAH","ORD","MDW","MSP","DTW","LAX","SFO","SEA","PHX","SAN","LAS","YYZ","YUL","YVR","YYC","YOW","YEG"}
LATIN_AMERICA_CODES = {"MEX","GDL","MTY","BOG","MDE","LIM","SCL","EZE","GRU","GIG","GUA","PTY","SJO","SAL","SDQ"}
APAC_CODES = {"HKG","SIN","BKK","KUL","CGK","SGN","HAN","TPE","NRT","HND","KIX","CTS","ICN","GMP","PVG","SHA","PEK","PKX","CAN","SZX","DEL","BOM","BLR","MAA","SYD","MEL","AKL"}
ISMEA_CODES = {"DXB","DWC","AUH","DOH","BAH","KWI","MCT","AMM","BEY","JED","RUH","DMM","IKA","THR","KHI","LHE","ISB","CMB","DAC","KTM"}

def iata_to_region(code: str) -> str:
    if not isinstance(code, str) or len(code) < 3:
        return "OTHER"
    c = code.strip().upper()
    if c in EUROPE_CODES: return "EUROPE"
    if c in NORTH_AMERICA_CODES: return "NORTH AMERICA"
    if c in LATIN_AMERICA_CODES: return "LATIN AMERICA"
    if c in APAC_CODES: return "APAC"
    if c in ISMEA_CODES: return "ISMEA"
    return "OTHER"

# -------------------- DATA LOADING --------------------
@st.cache_data
def load_data(file):
    # read smallest set of needed columns, from 'Export' if present
    xls = pd.ExcelFile(file)
    sheet = "Export" if "Export" in xls.sheet_names else xls.sheet_names[0]

    # Use only necessary columns (if present)
    wanted = ["POB as text","Airline","Volumetric Weight (KG)","Origin IATA","Destination IATA"]
    usecols = [c for c in pd.ExcelFile(file).parse(sheet, nrows=0).columns if c in wanted]
    df = pd.read_excel(file, sheet_name=sheet, usecols=usecols or None, engine="openpyxl")

    # tidy names
    df.columns = df.columns.str.strip()

    # keep only rows with date & airline
    if "POB as text" not in df.columns or "Airline" not in df.columns:
        st.error("Required columns not found. Need at least: 'POB as text' and 'Airline'.")
        return pd.DataFrame()

    df["Date"] = pd.to_datetime(df["POB as text"], errors="coerce", infer_datetime_format=True)
    df = df[df["Date"].notna() & df["Airline"].notna() & (df["Airline"].astype(str).str.strip() != "")]
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Month_Name"] = df["Date"].dt.strftime("%B")

    # weight
    if "Volumetric Weight (KG)" in df.columns:
        df["Weight_KG"] = pd.to_numeric(df["Volumetric Weight (KG)"], errors="coerce").fillna(0.0)
    else:
        df["Weight_KG"] = 0.0

    # airline split
    df["Is_UPS"] = df["Airline"].astype(str).str.upper().str.contains("UPS", na=False)

    # regions from IATA
    if "Origin IATA" in df.columns and "Destination IATA" in df.columns:
        df["Origin IATA"] = df["Origin IATA"].astype(str).str.upper()
        df["Destination IATA"] = df["Destination IATA"].astype(str).str.upper()
        df["Origin Region"] = df["Origin IATA"].map(iata_to_region)
        df["Destination Region"] = df["Destination IATA"].map(iata_to_region)
        # region family: only if both ends belong to the same macro region; otherwise CROSS-REGION
        df["Region Family"] = np.where(
            df["Origin Region"] == df["Destination Region"], df["Origin Region"], "CROSS-REGION"
        )
        # lane label
        df["Lane"] = df["Origin IATA"] + "-" + df["Destination IATA"]
    else:
        df["Origin Region"] = "OTHER"
        df["Destination Region"] = "OTHER"
        df["Region Family"] = "OTHER"
        df["Lane"] = "UNKNOWN-UNKNOWN"

    return df

# -------------------- METRICS --------------------
def monthly_metrics(df_year):
    """Return dict-of-series for table with months as columns."""
    months = range(1,13)
    # volumes (row counts)
    brown_vol = {m: int((df_year["Month"]==m) & (df_year["Is_UPS"])).sum() for m in months}
    green_vol = {m: int((df_year["Month"]==m) & (~df_year["Is_UPS"])).sum() for m in months}
    # kgs
    brown_kg = {m: float(df_year.loc[(df_year["Month"]==m) & (df_year["Is_UPS"]), "Weight_KG"].sum()) for m in months}
    green_kg = {m: float(df_year.loc[(df_year["Month"]==m) & (~df_year["Is_UPS"]), "Weight_KG"].sum()) for m in months}
    # utilization %
    util = {}
    for m in months:
        tot = brown_vol[m] + green_vol[m]
        util[m] = (brown_vol[m] / tot * 100.0) if tot > 0 else 0.0
    return brown_vol, green_vol, brown_kg, green_kg, util

def kpi_block(df_month):
    """Return dict of the same metrics for a given month dataframe."""
    brown_v = int((df_month["Is_UPS"]).sum())
    green_v = int((~df_month["Is_UPS"]).sum())
    tot_v = brown_v + green_v
    util = (brown_v / tot_v * 100.0) if tot_v > 0 else 0.0
    brown_kg = float(df_month.loc[df_month["Is_UPS"], "Weight_KG"].sum())
    green_kg = float(df_month.loc[~df_month["Is_UPS"], "Weight_KG"].sum())
    return {"brown_v":brown_v, "green_v":green_v, "tot_v":tot_v,
            "util":util, "brown_kg":brown_kg, "green_kg":green_kg}

def fmt_int(n):   return f"{int(n):,}"
def fmt_kg(x):    return f"{x:,.0f}"
def fmt_pct(p):   return f"{p:.1f}%"

# -------------------- CHART --------------------
def utilization_chart(month_names, values):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=month_names, y=values, mode="lines+markers",
        name="Utilization %", line=dict(color="#FF6B35", width=3),
        marker=dict(size=10),
        text=[f"{v:.1f}%" for v in values], textposition="top center",
        hovertemplate="%{x}<br>Utilization: %{y:.1f}%<extra></extra>"
    ))
    fig.update_layout(
        title="BT Utilization % by Month and Year",
        xaxis_title="Month",
        yaxis_title="%",
        yaxis=dict(range=[0,100], showgrid=True, gridcolor="#E0E0E0"),
        xaxis=dict(showgrid=False),
        height=400, hovermode="x", showlegend=False,
        plot_bgcolor="white", paper_bgcolor="white"
    )
    return fig

# -------------------- APP --------------------
def main():
    # Title
    _, c, _ = st.columns([2,3,1])
    with c:
        st.markdown("<h1 style='text-align:center;'>Green to Brown <span style='color:#008B8B;'>Overall Utilization Stats</span> <span style='color:#FFA500;'>YoY</span></h1>", unsafe_allow_html=True)

    xfile = st.file_uploader("Choose Excel file", type=["xlsx","xls"], label_visibility="collapsed")
    if not xfile:
        st.info("👆 Upload the Excel file to begin")
        return

    with st.spinner("Processing data..."):
        df = load_data(xfile)

    if df.empty:
        st.error("No valid data to display (check required columns and non-empty rows).")
        return

    # Current year view
    current_year = datetime.now().year
    if current_year not in df["Year"].unique():
        current_year = df["Year"].max()
    df_cur = df[df["Year"] == current_year].copy()

    months_order = ['January','February','March','April','May','June','July','August','September','October','November','December']

    tab1, tab2 = st.tabs(["📊 Year Overview", "📈 Monthly Analysis"])

    # ================= TAB 1 =================
    with tab1:
        st.markdown("### This Year To Date")

        bvol, gvol, bkg, gkg, util = monthly_metrics(df_cur)

        # Build a table similar to your screenshots (metrics as rows, months as columns + Total)
        cols = [m for m in months_order]
        table = pd.DataFrame({
            "Metric": [
                "Brown Volume (#)", "Green Volume (#)",
                "Brown KG", "Green KG",
                "Utilization %"
            ]
        })

        def month_series(d, is_pct=False):
            arr = []
            for m in range(1,13):
                val = d.get(m, 0)
                arr.append(f"{val:.1f}%" if is_pct else f"{val:,.0f}")
            return arr

        table = pd.concat([
            table,
            pd.DataFrame({
                m: [
                    f"{bvol[i+1]:,}", f"{gvol[i+1]:,}",
                    f"{bkg[i+1]:,.0f}", f"{gkg[i+1]:,.0f}",
                    f"{util[i+1]:.1f}%"
                ] for i, m in enumerate(months_order)
            })
        ], axis=1)

        # Totals column
        tot_bv = sum(bvol.values())
        tot_gv = sum(gvol.values())
        tot_bkg = sum(bkg.values())
        tot_gkg = sum(gkg.values())
        tot_util = (tot_bv / (tot_bv + tot_gv) * 100.0) if (tot_bv + tot_gv) > 0 else 0.0
        table["Total"] = [f"{tot_bv:,}", f"{tot_gv:,}", f"{tot_bkg:,.0f}", f"{tot_gkg:,.0f}", f"{tot_util:.1f}%"]

        st.dataframe(table, use_container_width=True, hide_index=True, height=430)

        st.markdown("### BT Utilization % by Month and Year")
        util_vals = [util[m] for m in range(1,13)]
        fig = utilization_chart(months_order, util_vals)
        st.plotly_chart(fig, use_container_width=True)

    # ================= TAB 2 =================
    with tab2:
        st.markdown("### Green to Brown <span style='color:#008B8B;'>Monthly Utilization Stats</span>", unsafe_allow_html=True)

        available_months = sorted(df_cur["Month"].unique())
        month_names = [months_order[m-1] for m in available_months if 1 <= m <= 12]
        sel_name = st.selectbox("Select Month", month_names)
        sel_m = months_order.index(sel_name) + 1

        df_m = df_cur[df_cur["Month"] == sel_m]
        if df_m.empty:
            st.info("No data for the selected month.")
            return

        # KPIs
        k = kpi_block(df_m)
        c1,c2,c3,c4,c5,c6 = st.columns(6)
        c1.metric("BT Utilization", fmt_pct(k["util"]))
        c2.metric("% Effective", "—")  # (not computed since no targets required)
        c3.metric("This Year Volume", f"{k['tot_v']:,}")
        c4.metric("—", "—")
        c5.metric("Brown KG", fmt_kg(k["brown_kg"]))
        c6.metric("Green KG", fmt_kg(k["green_kg"]))

        # ----- Utilization by Region (Region Family) -----
        left, right = st.columns(2)

        with left:
            st.markdown("#### Utilization by Region")
            rows = []
            for region, g in df_m.groupby("Region Family"):
                kk = kpi_block(g)
                rows.append({
                    "REGION": region,
                    "Brown Volume (#)": fmt_int(kk["brown_v"]),
                    "Green Volume (#)": fmt_int(kk["green_v"]),
                    "Brown KG": fmt_kg(kk["brown_kg"]),
                    "Green KG": fmt_kg(kk["green_kg"]),
                    "Utilization %": fmt_pct(kk["util"])
                })
            reg_df = pd.DataFrame(rows)
            # Sort by utilization desc
            reg_df["_s"] = reg_df["Utilization %"].str.rstrip("%").astype(float)
            reg_df = reg_df.sort_values("_s", ascending=False).drop(columns="_s")
            st.dataframe(reg_df, use_container_width=True, hide_index=True, height=420)

        with right:
            st.markdown("#### By Lane (Origin IATA → Destination IATA)")
            lane_rows = []
            # show top 30 lanes by total shipments that month
            lane_counts = df_m.groupby("Lane").size().sort_values(ascending=False).head(30).index
            for lane in lane_counts:
                g = df_m[df_m["Lane"] == lane]
                kk = kpi_block(g)
                lane_rows.append({
                    "Lane": lane,
                    "Brown Volume (#)": fmt_int(kk["brown_v"]),
                    "Green Volume (#)": fmt_int(kk["green_v"]),
                    "Brown KG": fmt_kg(kk["brown_kg"]),
                    "Green KG": fmt_kg(kk["green_kg"]),
                    "Utilization %": fmt_pct(kk["util"])
                })
            st.dataframe(pd.DataFrame(lane_rows), use_container_width=True, hide_index=True, height=420)

        # ----- Optional month-over-month pivots -----
        with st.expander("📈 Month-over-month pivots"):
            # Region Family MoM
            rf = (df_cur
                  .groupby(["Month","Region Family"])
                  .agg(Shipments=("Airline","size"),
                       Brown=("Is_UPS","sum"),
                       KG=("Weight_KG","sum"))
                  .reset_index())
            rf["Util %"] = (rf["Brown"] / rf["Shipments"] * 100).fillna(0.0)
            reg_pvt = rf.pivot(index="Region Family", columns="Month", values="Util %").fillna(0.0)
            reg_pvt.columns = [months_order[m-1] for m in reg_pvt.columns]
            st.markdown("**Utilization % by Region Family (MoM)**")
            st.dataframe(reg_pvt.round(1), use_container_width=True)

            # Lane MoM (top 15 lanes overall)
            top_lanes = (df_cur.groupby("Lane").size().sort_values(ascending=False).head(15).index)
            lf = (df_cur[df_cur["Lane"].isin(top_lanes)]
                  .groupby(["Month","Lane"])
                  .agg(Shipments=("Airline","size"),
                       Brown=("Is_UPS","sum"))
                  .reset_index())
            lf["Util %"] = (lf["Brown"] / lf["Shipments"] * 100).fillna(0.0)
            lane_pvt = lf.pivot(index="Lane", columns="Month", values="Util %").fillna(0.0)
            lane_pvt.columns = [months_order[m-1] for m in lane_pvt.columns]
            st.markdown("**Utilization % by Lane (MoM, Top 15)**")
            st.dataframe(lane_pvt.round(1), use_container_width=True)

if __name__ == "__main__":
    main()
