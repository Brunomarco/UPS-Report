import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
from io import BytesIO
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
EUROPE_CODES = {
    "AMS","CDG","ORY","FRA","MUC","DUS","BER","HAM","CGN","BRU","LGG","LHR","LGW","STN","LTN","MAN","BHX","EDI",
    "DUB","MAD","BCN","VLC","AGP","PMI","LIS","OPO","MXP","LIN","FCO","VCE","ATH","ZRH","GVA","VIE","PRG","WAW",
    "KRK","OSL","CPH","ARN","HEL","RIX","TLL","VNO","BUD","BEG","OTP","SOF","SKG","IST"
}
NORTH_AMERICA_CODES = {"JFK","EWR","BOS","PHL","IAD","DCA","CLT","ATL","MIA","MCO","TPA","DFW","IAH","ORD","MDW","MSP","DTW","LAX","SFO","SEA","PHX","SAN","LAS","YYZ","YUL","YVR","YYC","YOW","YEG"}
LATIN_AMERICA_CODES = {"MEX","GDL","MTY","BOG","MDE","LIM","SCL","EZE","GRU","GIG","GUA","PTY","SJO","SAL","SDQ"}
APAC_CODES = {"HKG","SIN","BKK","KUL","CGK","SGN","HAN","TPE","NRT","HND","KIX","CTS","ICN","GMP","PVG","SHA","PEK","PKX","CAN","SZX","SYD","MEL","AKL","DEL","BOM","BLR","MAA"}
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

MONTH_NAMES = ['January','February','March','April','May','June','July','August','September','October','November','December']

# -------------------- DATA LOADING (fast & memory-aware) --------------------
@st.cache_data(show_spinner=False)
def load_data_from_bytes(xlsx_bytes: bytes) -> pd.DataFrame:
    """
    Read only the needed columns from the Excel, convert dtypes to save RAM,
    and compute helper columns. Cached by file bytes.
    """
    bio = BytesIO(xlsx_bytes)
    xls = pd.ExcelFile(bio)
    sheet = "Export" if "Export" in xls.sheet_names else xls.sheet_names[0]

    # Only the columns we truly need:
    wanted = ["POB as text","Airline","Volumetric Weight (KG)","Origin IATA","Destination IATA"]
    # Determine which of them exist (header-only parse)
    header_df = pd.read_excel(BytesIO(xlsx_bytes), sheet_name=sheet, nrows=0, engine="openpyxl")
    usecols = [c for c in header_df.columns if c in wanted]

    df = pd.read_excel(BytesIO(xlsx_bytes), sheet_name=sheet, usecols=usecols or None, engine="openpyxl")

    # Clean & filter rows (only if Airline & POB as text present)
    if "POB as text" not in df.columns or "Airline" not in df.columns:
        return pd.DataFrame()

    df.columns = df.columns.str.strip()
    df["Airline"] = df["Airline"].astype("string").str.strip()
    df = df[ df["Airline"].notna() & (df["Airline"] != "") ]

    # Parse dates
    df["Date"] = pd.to_datetime(df["POB as text"], errors="coerce", infer_datetime_format=True)
    df = df[df["Date"].notna()]

    # Extract date parts
    df["Year"]  = df["Date"].dt.year.astype("int16")
    df["Month"] = df["Date"].dt.month.astype("int8")

    # Weight (float32 to halve RAM vs float64)
    if "Volumetric Weight (KG)" in df.columns:
        df["Weight_KG"] = pd.to_numeric(df["Volumetric Weight (KG)"], errors="coerce").fillna(0.0).astype("float32")
    else:
        df["Weight_KG"] = np.zeros(len(df), dtype="float32")

    # Brown vs Green
    df["Is_UPS"] = df["Airline"].str.upper().str.contains("UPS", na=False).astype("bool")

    # IATA → Region + Lane
    if "Origin IATA" in df.columns and "Destination IATA" in df.columns:
        df["Origin IATA"] = df["Origin IATA"].astype("string").str.upper()
        df["Destination IATA"] = df["Destination IATA"].astype("string").str.upper()
        # Category reduces memory a lot on 900k+ rows
        df["Origin IATA"] = df["Origin IATA"].astype("category")
        df["Destination IATA"] = df["Destination IATA"].astype("category")

        # map regions
        df["Origin Region"] = df["Origin IATA"].astype(str).map(iata_to_region).astype("category")
        df["Destination Region"] = df["Destination IATA"].astype(str).map(iata_to_region).astype("category")
        df["Region Family"] = np.where(
            df["Origin Region"] == df["Destination Region"], df["Origin Region"].astype(str), "CROSS-REGION"
        ).astype("category")

        df["Lane"] = (df["Origin IATA"].astype(str) + "-" + df["Destination IATA"].astype(str)).astype("category")
    else:
        df["Origin Region"] = pd.Categorical(["OTHER"] * len(df))
        df["Destination Region"] = pd.Categorical(["OTHER"] * len(df))
        df["Region Family"] = pd.Categorical(["OTHER"] * len(df))
        df["Lane"] = pd.Categorical(["UNKNOWN-UNKNOWN"] * len(df))

    # Make Airline categorical to save memory
    df["Airline"] = df["Airline"].astype("category")

    return df

# -------------------- AGG HELPERS (vectorized) --------------------
def build_monthly_table(df_year: pd.DataFrame) -> pd.DataFrame:
    """Return formatted table like the screenshot (rows=metrics, columns=months+Total)."""
    # Group once by Month and Is_UPS
    g = (df_year
         .groupby(["Month","Is_UPS"], observed=True)
         .agg(Shipments=("Airline","size"),
              KG=("Weight_KG","sum"))
         .reset_index())

    # Pivot to Brown/Green
    ship = g.pivot(index="Month", columns="Is_UPS", values="Shipments").fillna(0).rename(columns={True:"Brown", False:"Green"})
    kg   = g.pivot(index="Month", columns="Is_UPS", values="KG").fillna(0).rename(columns={True:"Brown", False:"Green"})

    # Ensure all months exist 1..12
    ship = ship.reindex(range(1,13), fill_value=0)
    kg   = kg.reindex(range(1,13), fill_value=0.0)

    util = (ship["Brown"] / (ship["Brown"] + ship["Green"]).replace(0, np.nan) * 100).fillna(0.0)

    # Build display table
    df_out = pd.DataFrame({"Metric": ["Brown Volume (#)","Green Volume (#)","Brown KG","Green KG","Utilization %"]})

    for m in range(1,13):
        df_out[MONTH_NAMES[m-1]] = [
            f"{int(ship.loc[m,'Brown']):,}",
            f"{int(ship.loc[m,'Green']):,}",
            f"{kg.loc[m,'Brown']:,.0f}",
            f"{kg.loc[m,'Green']:,.0f}",
            f"{util.loc[m]:.1f}%"
        ]

    # Totals column
    tot_brown_v = int(ship["Brown"].sum())
    tot_green_v = int(ship["Green"].sum())
    tot_brown_kg = float(kg["Brown"].sum())
    tot_green_kg = float(kg["Green"].sum())
    tot_util = (tot_brown_v / (tot_brown_v + tot_green_v) * 100) if (tot_brown_v + tot_green_v) > 0 else 0.0

    df_out["Total"] = [
        f"{tot_brown_v:,}",
        f"{tot_green_v:,}",
        f"{tot_brown_kg:,.0f}",
        f"{tot_green_kg:,.0f}",
        f"{tot_util:.1f}%"
    ]
    return df_out, util.reindex(range(1,13), fill_value=0.0)

def kpi_for_slice(df_slice: pd.DataFrame):
    g = (df_slice.groupby("Is_UPS", observed=True)
         .agg(Shipments=("Airline","size"),
              KG=("Weight_KG","sum"))
         .reindex([True, False], fill_value=0))
    brown_v = int(g.loc[True, "Shipments"]) if True in g.index else 0
    green_v = int(g.loc[False, "Shipments"]) if False in g.index else 0
    tot_v = brown_v + green_v
    util = (brown_v / tot_v * 100.0) if tot_v > 0 else 0.0
    brown_kg = float(g.loc[True, "KG"]) if True in g.index else 0.0
    green_kg = float(g.loc[False, "KG"]) if False in g.index else 0.0
    return brown_v, green_v, tot_v, util, brown_kg, green_kg

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
    # Header
    _, c, _ = st.columns([2,3,1])
    with c:
        st.markdown("<h1 style='text-align:center;'>Green to Brown <span style='color:#008B8B;'>Overall Utilization Stats</span> <span style='color:#FFA500;'>YoY</span></h1>", unsafe_allow_html=True)

    xfile = st.file_uploader("Choose Excel file", type=["xlsx","xls"], label_visibility="collapsed")
    if not xfile:
        st.info("👆 Upload the Excel file to begin")
        return

    # Cache key = file bytes (so 930k rows are parsed only once per upload)
    file_bytes = xfile.getvalue()
    with st.spinner("Processing data..."):
        df = load_data_from_bytes(file_bytes)

    if df.empty:
        st.error("No valid data to display. Ensure the file has 'POB as text' and 'Airline'.")
        return

    # Current year
    current_year = datetime.now().year
    if current_year not in df["Year"].unique():
        current_year = int(df["Year"].max())
    df_cur = df[df["Year"] == current_year]

    tab1, tab2 = st.tabs(["📊 Year Overview", "📈 Monthly Analysis"])

    # ============ TAB 1 ============
    with tab1:
        st.markdown("### This Year To Date")
        monthly_table, util_series = build_monthly_table(df_cur)
        st.dataframe(monthly_table, use_container_width=True, hide_index=True, height=430)

        st.markdown("### BT Utilization % by Month and Year")
        fig = utilization_chart(MONTH_NAMES, util_series.values.tolist())
        st.plotly_chart(fig, use_container_width=True)

    # ============ TAB 2 ============
    with tab2:
        st.markdown("### Green to Brown <span style='color:#008B8B;'>Monthly Utilization Stats</span>", unsafe_allow_html=True)

        available_months = sorted(df_cur["Month"].unique())
        month_names = [MONTH_NAMES[m-1] for m in available_months if 1 <= m <= 12]
        sel_name = st.selectbox("Select Month", month_names)
        sel_month = MONTH_NAMES.index(sel_name) + 1

        df_m = df_cur[df_cur["Month"] == sel_month]
        if df_m.empty:
            st.info("No data for the selected month.")
            return

        # KPIs
        b_v, g_v, t_v, util, b_kg, g_kg = kpi_for_slice(df_m)
        c1,c2,c3,c4,c5,c6 = st.columns(6)
        c1.metric("BT Utilization", f"{util:.1f}%")
        c2.metric("% Effective", "—")
        c3.metric("This Year Volume", f"{t_v:,}")
        c4.metric("—", "—")
        c5.metric("Brown KG", f"{b_kg:,.0f}")
        c6.metric("Green KG", f"{g_kg:,.0f}")

        # ----- By Region Family -----
        left, right = st.columns(2)
        with left:
            st.markdown("#### Utilization by Region")
            rg = (df_m
                  .groupby(["Region Family","Is_UPS"], observed=True)
                  .agg(Shipments=("Airline","size"), KG=("Weight_KG","sum"))
                  .reset_index())
            # pivot to columns Brown/Green
            rg_p = (rg.pivot(index="Region Family", columns="Is_UPS", values="Shipments")
                      .rename(columns={True:"Brown", False:"Green"})
                      .fillna(0))
            rg_kg = (rg.pivot(index="Region Family", columns="Is_UPS", values="KG")
                      .rename(columns={True:"BrownKG", False:"GreenKG"})
                      .fillna(0.0))
            tbl = rg_p.join(rg_kg, how="outer").fillna(0)
            tbl["Utilization %"] = (tbl["Brown"] / (tbl["Brown"] + tbl["Green"]).replace(0,np.nan) * 100).fillna(0.0)
            disp = (tbl.reset_index()
                    .assign(**{
                        "Brown Volume (#)": lambda d: d["Brown"].astype(int).map(lambda x: f"{x:,}"),
                        "Green Volume (#)": lambda d: d["Green"].astype(int).map(lambda x: f"{x:,}"),
                        "Brown KG": lambda d: d["BrownKG"].map(lambda x: f"{x:,.0f}"),
                        "Green KG": lambda d: d["GreenKG"].map(lambda x: f"{x:,.0f}"),
                        "Utilization %": lambda d: d["Utilization %"].map(lambda x: f"{x:.1f}%")
                    }))[
                        ["Region Family","Brown Volume (#)","Green Volume (#)","Brown KG","Green KG","Utilization %"]
                    ]
            # Sort by Util %
            disp["_u"] = disp["Utilization %"].str.rstrip("%").astype(float)
            disp = disp.sort_values("_u", ascending=False).drop(columns="_u")
            st.dataframe(disp, use_container_width=True, hide_index=True, height=420)

        with right:
            st.markdown("#### By Lane (Origin IATA → Destination IATA)")
            # top 30 lanes by shipments this month
            top_lanes = (df_m.groupby("Lane").size().sort_values(ascending=False).head(30).index)
            ln = (df_m[df_m["Lane"].isin(top_lanes)]
                  .groupby(["Lane","Is_UPS"], observed=True)
                  .agg(Shipments=("Airline","size"), KG=("Weight_KG","sum"))
                  .reset_index())
            ln_p = (ln.pivot(index="Lane", columns="Is_UPS", values="Shipments")
                      .rename(columns={True:"Brown", False:"Green"})
                      .fillna(0))
            ln_kg = (ln.pivot(index="Lane", columns="Is_UPS", values="KG")
                      .rename(columns={True:"BrownKG", False:"GreenKG"})
                      .fillna(0.0))
            lt = ln_p.join(ln_kg, how="outer").fillna(0)
            lt["Utilization %"] = (lt["Brown"] / (lt["Brown"] + lt["Green"]).replace(0,np.nan) * 100).fillna(0.0)
            disp_lane = (lt.reset_index()
                         .assign(**{
                             "Brown Volume (#)": lambda d: d["Brown"].astype(int).map(lambda x: f"{x:,}"),
                             "Green Volume (#)": lambda d: d["Green"].astype(int).map(lambda x: f"{x:,}"),
                             "Brown KG": lambda d: d["BrownKG"].map(lambda x: f"{x:,.0f}"),
                             "Green KG": lambda d: d["GreenKG"].map(lambda x: f"{x:,.0f}"),
                             "Utilization %": lambda d: d["Utilization %"].map(lambda x: f"{x:.1f}%")
                         }))[
                             ["Lane","Brown Volume (#)","Green Volume (#)","Brown KG","Green KG","Utilization %"]
                         ]
            st.dataframe(disp_lane, use_container_width=True, hide_index=True, height=420)

        # ----- Month-over-month pivots (optional, fast) -----
        with st.expander("📈 Month-over-month pivots"):
            # Region Family MoM
            rf = (df_cur
                  .groupby(["Month","Region Family","Is_UPS"], observed=True)
                  .agg(Shipments=("Airline","size"))
                  .reset_index())
            rf_p = (rf.pivot_table(index=["Region Family","Month"], columns="Is_UPS", values="Shipments", fill_value=0)
                      .rename(columns={True:"Brown", False:"Green"})
                      .reset_index())
            rf_p["Util %"] = (rf_p["Brown"] / (rf_p["Brown"] + rf_p["Green"]).replace(0,np.nan) * 100).fillna(0.0)
            reg_pvt = rf_p.pivot(index="Region Family", columns="Month", values="Util %").fillna(0.0)
            reg_pvt.columns = [MONTH_NAMES[m-1] for m in reg_pvt.columns]
            st.markdown("**Utilization % by Region Family (MoM)**")
            st.dataframe(reg_pvt.round(1), use_container_width=True)

            # Lane MoM (top 15 lanes overall)
            top_lanes_all = (df_cur.groupby("Lane").size().sort_values(ascending=False).head(15).index)
            lf = (df_cur[df_cur["Lane"].isin(top_lanes_all)]
                  .groupby(["Month","Lane","Is_UPS"], observed=True)
                  .agg(Shipments=("Airline","size"))
                  .reset_index())
            lf_p = (lf.pivot_table(index=["Lane","Month"], columns="Is_UPS", values="Shipments", fill_value=0)
                      .rename(columns={True:"Brown", False:"Green"})
                      .reset_index())
            lf_p["Util %"] = (lf_p["Brown"] / (lf_p["Brown"] + lf_p["Green"]).replace(0,np.nan) * 100).fillna(0.0)
            lane_pvt = lf_p.pivot(index="Lane", columns="Month", values="Util %").fillna(0.0)
            lane_pvt.columns = [MONTH_NAMES[m-1] for m in lane_pvt.columns]
            st.markdown("**Utilization % by Lane (MoM, Top 15)**")
            st.dataframe(lane_pvt.round(1), use_container_width=True)

if __name__ == "__main__":
    main()
