import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# ==================== PAGE CONFIGURATION ====================
st.set_page_config(
    page_title="Green to Brown Utilization Stats 2024", 
    layout="wide", 
    page_icon="✈️"
)

# Enhanced CSS for professional styling
st.markdown("""
<style>
    .main { padding: 0rem 1rem; }
    h1 { 
        text-align: center; 
        font-size: 2.5rem;
        background: linear-gradient(90deg, #2E7D32 0%, #FF8C00 50%, #1976D2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
    }
    h3 {
        color: #1976D2;
        border-bottom: 2px solid #E0E0E0;
        padding-bottom: 10px;
        margin-top: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] { 
        gap: 2rem;
        background-color: #F5F5F5;
        padding: 0.5rem;
        border-radius: 10px;
    }
    .stTabs [data-baseweb="tab"] { 
        height: 50px; 
        padding: 0 30px; 
        font-size: 1.1rem; 
        font-weight: 600;
        background-color: white;
        border-radius: 8px;
        border: 2px solid transparent;
    }
    .stTabs [data-baseweb="tab"]:hover {
        border-color: #FF8C00;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #FF8C00 0%, #FFA500 100%);
        color: white;
    }
    div[data-testid="metric-container"] { 
        background: linear-gradient(135deg, #FFFFFF 0%, #F8F9FA 100%);
        border: 1px solid #E0E0E0;
        padding: 1.2rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.07);
        transition: transform 0.2s;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.1);
    }
    [data-testid="stMetricValue"] { 
        font-size: 2rem; 
        font-weight: 700;
        color: #1976D2;
    }
    [data-testid="stMetricDelta"] {
        font-size: 1rem;
    }
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
    }
    .stDataFrame > div {
        border-radius: 10px;
    }
    thead tr th {
        background: linear-gradient(90deg, #1976D2 0%, #2196F3 100%) !important;
        color: white !important;
        font-weight: 600 !important;
        text-align: center !important;
        padding: 12px !important;
    }
    tbody tr {
        border-bottom: 1px solid #F0F0F0;
    }
    tbody tr:hover {
        background-color: #FFF3E0 !important;
    }
    tbody tr td {
        padding: 10px !important;
        text-align: center !important;
    }
    .stSelectbox label {
        font-weight: 600;
        color: #424242;
    }
    .success-box {
        background: linear-gradient(135deg, #4CAF50 0%, #66BB6A 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        font-weight: 600;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ==================== IATA TO REGION MAPPING ====================
def get_region(iata_code):
    """Map IATA codes to regions"""
    if pd.isna(iata_code) or not isinstance(iata_code, str):
        return "OTHER"
    
    code = str(iata_code).strip().upper()
    
    # Handle special codes
    if code in ['0', 'PLY2', ''] or len(code) != 3:
        return "OTHER"
    
    # EUROPE
    europe_codes = {
        'AMS', 'ARN', 'ATH', 'BCN', 'BER', 'BGO', 'BIO', 'BLL', 'BRI', 'BRU',
        'BUD', 'CAG', 'CDG', 'CGN', 'CPH', 'CTA', 'DUB', 'DUS', 'EDI', 'ESB',
        'FCO', 'FKB', 'FRA', 'GDN', 'GVA', 'HAJ', 'HAM', 'HEL', 'HER', 'ISL',
        'IST', 'KRK', 'LCA', 'LGG', 'LGW', 'LHR', 'LIN', 'LIS', 'LJU', 'LPA',
        'LUX', 'MAD', 'MLA', 'MLH', 'MMX', 'MUC', 'MXP', 'OPO', 'ORY', 'OSL',
        'OTP', 'PDL', 'PMI', 'PMO', 'POZ', 'PRG', 'REG', 'RIX', 'RMO', 'RZE',
        'SJJ', 'SKG', 'SKP', 'SOF', 'STR', 'SUF', 'SVG', 'TGD', 'TIA', 'TLL',
        'TFS', 'TOS', 'TRD', 'TUN', 'VIE', 'VLC', 'VNO', 'WAW', 'WRE', 'ZAG', 'ZRH'
    }
    
    # NORTH AMERICA (USA & Canada)
    north_america_codes = {
        'ABQ', 'ALB', 'AMA', 'ATL', 'AUS', 'AZO', 'BDL', 'BGR', 'BHM', 'BNA',
        'BOI', 'BOS', 'BTR', 'BTV', 'BUF', 'BWI', 'CHA', 'CHS', 'CHO', 'CID',
        'CLE', 'CLT', 'CMH', 'CRP', 'CVG', 'DAY', 'DCA', 'DEN', 'DFW', 'DSM',
        'DTW', 'ELP', 'EUG', 'EVV', 'EWR', 'FAR', 'FAT', 'FLL', 'FWA', 'GEG',
        'GNV', 'GPT', 'GRB', 'GRR', 'GSO', 'GSP', 'GTF', 'HNL', 'HOR', 'HOU',
        'HSV', 'IAD', 'IAH', 'ICT', 'IDA', 'IND', 'JAX', 'JAN', 'JFK', 'LAS',
        'LAX', 'LBB', 'LEX', 'LGA', 'LIT', 'MCI', 'MCO', 'MDT', 'MDW', 'MEM',
        'MIA', 'MKE', 'MSN', 'MSP', 'MSY', 'MYR', 'OKC', 'OMA', 'ORD', 'ORF',
        'PBI', 'PDX', 'PHL', 'PHX', 'PIT', 'PNS', 'PSL', 'PWM', 'RAP', 'RDM',
        'RDU', 'RIC', 'RNO', 'ROA', 'ROC', 'RST', 'RSW', 'SAN', 'SAT', 'SAV',
        'SBY', 'SDF', 'SEA', 'SFO', 'SGF', 'SJC', 'SLC', 'SMF', 'SNA', 'STL',
        'SYR', 'TPA', 'TUL', 'TUS', 'TYS', 'YEG', 'YHZ', 'YLW', 'YOW', 'YQB',
        'YUL', 'YVR', 'YWG', 'YXE', 'YYC', 'YYZ', 'YYT'
    }
    
    # LATIN AMERICA (Mexico, Central & South America, Caribbean)
    latin_america_codes = {
        'ASU', 'BEL', 'BGI', 'BJX', 'BLZ', 'BOG', 'BRO', 'CBO', 'CEN', 'CJS',
        'CLO', 'CUL', 'CUN', 'CUU', 'DGO', 'EZE', 'FOR', 'GDL', 'GIG', 'GRU',
        'GUA', 'GYE', 'HMO', 'LAP', 'LIM', 'LPB', 'MEX', 'MFE', 'MGA', 'MID',
        'MTY', 'MVD', 'MXL', 'MZT', 'OAX', 'PAP', 'PTY', 'PVR', 'SAL', 'SAP',
        'SCL', 'SDQ', 'SJD', 'SJO', 'SJU', 'SLP', 'SSA', 'TAM', 'TAP', 'TGZ',
        'TIJ', 'TRC', 'UIO', 'VCP', 'VER', 'VSA', 'BON', 'CUR', 'FDF', 'PTP'
    }
    
    # APAC (Asia Pacific)
    apac_codes = {
        'ADL', 'AKL', 'ASP', 'BBI', 'BDQ', 'BKI', 'BKK', 'BLR', 'BNE', 'BOM',
        'CAN', 'CBR', 'CCJ', 'CCU', 'CEB', 'CEI', 'CFS', 'CGK', 'CHC', 'CJB',
        'CKY', 'CMB', 'CNS', 'CNX', 'COK', 'CTS', 'DAC', 'DEL', 'DRW', 'DUD',
        'DVO', 'FUK', 'GAU', 'GIS', 'GLT', 'GOI', 'HAN', 'HBA', 'HDY', 'HIJ',
        'HKG', 'HND', 'HYD', 'ICN', 'IDR', 'ILO', 'ISB', 'ITM', 'IVC', 'IXC',
        'IXE', 'IXM', 'IXU', 'IZO', 'JAI', 'JKH', 'KCH', 'KHI', 'KIJ', 'KIX',
        'KKC', 'KMI', 'KMJ', 'KMQ', 'KOA', 'KOJ', 'KTM', 'KUL', 'LAD', 'LHE',
        'LKO', 'LLW', 'LST', 'MAA', 'MHG', 'MKY', 'MNL', 'MYJ', 'MYY', 'NAG',
        'NAN', 'NGO', 'NGS', 'NOU', 'NPE', 'NPL', 'NRT', 'NSN', 'OAG', 'OIT',
        'OKA', 'OKJ', 'PAT', 'PEK', 'PER', 'PMR', 'PNH', 'PNQ', 'POM', 'PPS',
        'PQQ', 'PVG', 'ROK', 'ROT', 'RPR', 'SBW', 'SDJ', 'SGN', 'SIN', 'STV',
        'SUB', 'SYD', 'SZX', 'TAC', 'TAK', 'TPE', 'TRG', 'TRV', 'TSV', 'UBJ',
        'UBN', 'UBP', 'VNS', 'VTZ', 'WLG', 'XGB', 'XOP', 'XPJ', 'XRF', 'XYL',
        'YGJ', 'ZAM', 'ZFJ', 'ZFQ'
    }
    
    # ISMEA (India, Middle East, Africa)
    ismea_codes = {
        'ABJ', 'ABV', 'ACC', 'ADD', 'ALA', 'ALG', 'AMD', 'AMM', 'AUH', 'BAH',
        'BEY', 'BJL', 'BKO', 'BSL', 'CAI', 'CMN', 'CPT', 'DAR', 'DME', 'DOH',
        'DUR', 'DWC', 'DXB', 'EBB', 'ELS', 'FIH', 'FNA', 'GIB', 'GIZ', 'GRJ',
        'HRE', 'JED', 'JNB', 'KGL', 'KIV', 'KWI', 'LBV', 'LED', 'LFW', 'LOS',
        'LUN', 'MCT', 'MRU', 'NBO', 'OUA', 'PLZ', 'RUH', 'TBS', 'TLV', 'TNR',
        'TSE', 'UBN'
    }
    
    if code in europe_codes:
        return "EUROPE"
    elif code in north_america_codes:
        return "NORTH AMERICA"
    elif code in latin_america_codes:
        return "LATIN AMERICA"
    elif code in apac_codes:
        return "APAC"
    elif code in ismea_codes:
        return "ISMEA"
    else:
        return "OTHER"

# ==================== DATA PROCESSING ====================
@st.cache_data
def load_and_process_data(uploaded_file):
    """Load Excel file and process data with correct column mapping"""
    try:
        # Read the entire Excel file first to get proper columns
        df_full = pd.read_excel(uploaded_file, engine='openpyxl')
        
        # Get columns by index (0-based)
        # E = index 4 (POB as text)
        # J = index 9 (Origin IATA)
        # K = index 10 (Destination IATA)
        # L = index 11 (Volumetric Weight)
        # M = index 12 (Airline)
        
        df = pd.DataFrame()
        df['POB as text'] = df_full.iloc[:, 4] if df_full.shape[1] > 4 else None
        df['Origin IATA'] = df_full.iloc[:, 9] if df_full.shape[1] > 9 else None
        df['Destination IATA'] = df_full.iloc[:, 10] if df_full.shape[1] > 10 else None
        df['Volumetric Weight (KG)'] = df_full.iloc[:, 11] if df_full.shape[1] > 11 else None
        df['Airline'] = df_full.iloc[:, 12] if df_full.shape[1] > 12 else None
        
        # Remove rows with missing POB as text or Airline
        initial_count = len(df)
        df = df.dropna(subset=['POB as text', 'Airline'])
        removed_count = initial_count - len(df)
        
        if removed_count > 0:
            st.info(f"ℹ️ Removed {removed_count:,} rows with missing date or airline data")
        
        # Clean airline column
        df['Airline'] = df['Airline'].astype(str).str.strip()
        df = df[df['Airline'] != '']
        df = df[df['Airline'] != 'nan']
        
        # Parse dates
        df['Date'] = pd.to_datetime(df['POB as text'], errors='coerce')
        df = df.dropna(subset=['Date'])
        
        # Extract date components
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Month_Name'] = df['Date'].dt.strftime('%B')
        
        # FILTER FOR 2024 ONLY
        df = df[df['Year'] == 2024]
        
        if len(df) == 0:
            st.error("No data found for 2024. Please ensure your file contains 2024 data.")
            return pd.DataFrame()
        
        # Identify Brown (UPS) vs Green (Others)
        df['Is_UPS'] = df['Airline'].str.upper().str.contains('UPS', na=False)
        df['Category'] = df['Is_UPS'].map({True: 'Brown', False: 'Green'})
        
        # Process weight
        df['Weight_KG'] = pd.to_numeric(df['Volumetric Weight (KG)'], errors='coerce').fillna(0)
        
        # Process IATA codes
        df['Origin IATA'] = df['Origin IATA'].astype(str).str.strip().str.upper()
        df['Destination IATA'] = df['Destination IATA'].astype(str).str.strip().str.upper()
        
        # Map to regions
        df['Origin Region'] = df['Origin IATA'].apply(get_region)
        df['Destination Region'] = df['Destination IATA'].apply(get_region)
        
        # Create region pairs (always in format: Origin-Destination)
        df['Region Pair'] = df['Origin Region'] + '-' + df['Destination Region']
        
        # Create lane pairs
        df['Lane Pair'] = df['Origin IATA'] + '-' + df['Destination IATA']
        
        return df
        
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        st.error("Please check that your Excel file has the correct columns:")
        st.error("Column E: POB as text, Column J: Origin IATA, Column K: Destination IATA, Column L: Volumetric Weight, Column M: Airline")
        return pd.DataFrame()

# ==================== CALCULATION FUNCTIONS ====================
def calculate_monthly_stats(df):
    """Calculate statistics by month for 2024"""
    months = ['January', 'February', 'March', 'April', 'May', 'June',
              'July', 'August', 'September', 'October', 'November', 'December']
    
    stats = []
    
    for month_num in range(1, 13):
        df_month = df[df['Month'] == month_num]
        
        # Volume is count of rows
        brown_volume = len(df_month[df_month['Is_UPS'] == True])
        green_volume = len(df_month[df_month['Is_UPS'] == False])
        total_volume = brown_volume + green_volume
        
        # Weight sums
        brown_weight = df_month[df_month['Is_UPS'] == True]['Weight_KG'].sum()
        green_weight = df_month[df_month['Is_UPS'] == False]['Weight_KG'].sum()
        
        # Utilization % = (Brown Volume / Total Volume) * 100
        utilization = (brown_volume / total_volume * 100) if total_volume > 0 else 0
        
        stats.append({
            'Month': months[month_num-1],
            'Month_Num': month_num,
            'Brown_Volume': brown_volume,
            'Green_Volume': green_volume,
            'Total_Volume': total_volume,
            'Brown_Weight': brown_weight,
            'Green_Weight': green_weight,
            'Utilization_%': utilization
        })
    
    return pd.DataFrame(stats)

def format_table_display(stats_df):
    """Format statistics table for professional display"""
    display_data = []
    
    # Row 1: Brown Volume
    row1 = ['🟫 Brown Volume (#)'] + [f"{int(row['Brown_Volume']):,}" for _, row in stats_df.iterrows()]
    display_data.append(row1)
    
    # Row 2: Green Volume
    row2 = ['🟩 Green Volume (#)'] + [f"{int(row['Green_Volume']):,}" for _, row in stats_df.iterrows()]
    display_data.append(row2)
    
    # Row 3: Utilization %
    row3 = ['📊 Utilization %'] + [f"{row['Utilization_%']:.1f}%" for _, row in stats_df.iterrows()]
    display_data.append(row3)
    
    # Row 4: Total Weight Impact
    row4 = ['⚖️ Total Weight (KG)'] + [f"{row['Brown_Weight'] + row['Green_Weight']:,.0f}" for _, row in stats_df.iterrows()]
    display_data.append(row4)
    
    # Row 5: Brown Weight
    row5 = ['🟫 Brown Weight (KG)'] + [f"{row['Brown_Weight']:,.0f}" for _, row in stats_df.iterrows()]
    display_data.append(row5)
    
    # Row 6: Green Weight
    row6 = ['🟩 Green Weight (KG)'] + [f"{row['Green_Weight']:,.0f}" for _, row in stats_df.iterrows()]
    display_data.append(row6)
    
    # Create DataFrame
    columns = ['Metrics'] + list(stats_df['Month'])
    df_display = pd.DataFrame(display_data, columns=columns)
    
    # Add totals column
    total_brown_vol = stats_df['Brown_Volume'].sum()
    total_green_vol = stats_df['Green_Volume'].sum()
    total_brown_weight = stats_df['Brown_Weight'].sum()
    total_green_weight = stats_df['Green_Weight'].sum()
    total_util = (total_brown_vol / (total_brown_vol + total_green_vol) * 100) if (total_brown_vol + total_green_vol) > 0 else 0
    
    df_display['TOTAL 2024'] = [
        f"{total_brown_vol:,}",
        f"{total_green_vol:,}",
        f"{total_util:.1f}%",
        f"{total_brown_weight + total_green_weight:,.0f}",
        f"{total_brown_weight:,.0f}",
        f"{total_green_weight:,.0f}"
    ]
    
    return df_display

def create_utilization_chart(stats_df):
    """Create professional utilization chart"""
    fig = go.Figure()
    
    # Main utilization line
    fig.add_trace(go.Scatter(
        x=stats_df['Month'],
        y=stats_df['Utilization_%'],
        mode='lines+markers+text',
        name='UPS Utilization %',
        line=dict(color='#FF6B35', width=4),
        marker=dict(size=12, color='#FF6B35', line=dict(color='white', width=2)),
        text=[f"{val:.1f}%" for val in stats_df['Utilization_%']],
        textposition='top center',
        textfont=dict(size=11, weight='bold'),
        hovertemplate='<b>%{x}</b><br>Utilization: %{y:.1f}%<br>Brown: %{customdata[0]:,}<br>Green: %{customdata[1]:,}<extra></extra>',
        customdata=np.column_stack((stats_df['Brown_Volume'], stats_df['Green_Volume']))
    ))
    
    # Add average line
    avg_util = stats_df['Utilization_%'].mean()
    fig.add_hline(
        y=avg_util, 
        line_dash="dash", 
        line_color="gray",
        annotation_text=f"2024 Average: {avg_util:.1f}%",
        annotation_position="right"
    )
    
    fig.update_layout(
        title={
            'text': 'UPS (Brown) Utilization % by Month - 2024',
            'font': {'size': 20, 'color': '#1976D2'}
        },
        xaxis_title='Month',
        yaxis_title='Utilization %',
        yaxis=dict(
            range=[0, max(100, stats_df['Utilization_%'].max() + 10)],
            gridcolor='#E0E0E0',
            showgrid=True,
            ticksuffix='%'
        ),
        xaxis=dict(
            showgrid=False,
            tickangle=-45
        ),
        height=450,
        hovermode='x unified',
        plot_bgcolor='#FAFAFA',
        paper_bgcolor='white',
        showlegend=False,
        margin=dict(t=80, b=60)
    )
    
    return fig

def calculate_region_stats(df_month):
    """Calculate statistics by region pair with proper formatting"""
    results = []
    
    # Get unique region pairs and sort them
    region_pairs = sorted(df_month['Region Pair'].unique())
    
    for region_pair in region_pairs:
        df_region = df_month[df_month['Region Pair'] == region_pair]
        
        brown_vol = len(df_region[df_region['Is_UPS'] == True])
        green_vol = len(df_region[df_region['Is_UPS'] == False])
        total_vol = brown_vol + green_vol
        
        # Skip if no volume
        if total_vol == 0:
            continue
            
        brown_weight = df_region[df_region['Is_UPS'] == True]['Weight_KG'].sum()
        green_weight = df_region[df_region['Is_UPS'] == False]['Weight_KG'].sum()
        
        utilization = (brown_vol / total_vol * 100)
        
        results.append({
            'Region Pair': region_pair,
            'Utilization %': f"{utilization:.1f}%",
            'Brown Volume': f"{brown_vol:,}",
            'Green Volume': f"{green_vol:,}",
            'Total Volume': f"{total_vol:,}",
            'Brown KG': f"{brown_weight:,.0f}",
            'Green KG': f"{green_weight:,.0f}",
            '_sort': utilization
        })
    
    df_results = pd.DataFrame(results)
    if not df_results.empty:
        df_results = df_results.sort_values('_sort', ascending=False)
        return df_results.drop('_sort', axis=1)
    return df_results

def calculate_lane_stats(df_month, top_n=30):
    """Calculate statistics by lane pair"""
    # Get top lanes by total volume
    lane_volumes = df_month.groupby('Lane Pair').size().sort_values(ascending=False)
    top_lanes = lane_volumes.head(top_n).index
    
    results = []
    for lane in top_lanes:
        df_lane = df_month[df_month['Lane Pair'] == lane]
        
        brown_vol = len(df_lane[df_lane['Is_UPS'] == True])
        green_vol = len(df_lane[df_lane['Is_UPS'] == False])
        total_vol = brown_vol + green_vol
        
        brown_weight = df_lane[df_lane['Is_UPS'] == True]['Weight_KG'].sum()
        green_weight = df_lane[df_lane['Is_UPS'] == False]['Weight_KG'].sum()
        
        utilization = (brown_vol / total_vol * 100)
        
        results.append({
            'Lane (Origin-Dest)': lane,
            'Utilization %': f"{utilization:.1f}%",
            'Brown Vol': f"{brown_vol:,}",
            'Green Vol': f"{green_vol:,}",
            'Total': f"{total_vol:,}",
            'Brown KG': f"{brown_weight:,.0f}",
            'Green KG': f"{green_weight:,.0f}",
            '_sort': utilization
        })
    
    df_results = pd.DataFrame(results)
    if not df_results.empty:
        df_results = df_results.sort_values('_sort', ascending=False)
        return df_results.drop('_sort', axis=1)
    return df_results

# ==================== MAIN APPLICATION ====================
def main():
    # Professional header
    st.markdown("""
        <h1>Green to Brown Overall Utilization Stats 2024</h1>
    """, unsafe_allow_html=True)
    
    # File upload section
    with st.container():
        uploaded_file = st.file_uploader(
            "📁 Upload Excel File",
            type=['xlsx', 'xls'],
            help="Required: Column E (POB as text), J (Origin IATA), K (Destination IATA), L (Volumetric Weight), M (Airline)"
        )
    
    if not uploaded_file:
        st.info("👆 Please upload your Excel file to begin analysis")
        
        # Instructions box
        st.markdown("""
        <div style='background: linear-gradient(135deg, #E3F2FD 0%, #BBDEFB 100%); padding: 20px; border-radius: 10px; margin: 20px 0;'>
        <h3 style='color: #1565C0;'>📋 Required Excel Columns:</h3>
        <ul style='color: #424242; font-size: 16px;'>
        <li><b>Column E:</b> POB as text (Date)</li>
        <li><b>Column J:</b> Origin IATA Code</li>
        <li><b>Column K:</b> Destination IATA Code</li>
        <li><b>Column L:</b> Volumetric Weight (KG)</li>
        <li><b>Column M:</b> Airline</li>
        </ul>
        <p style='color: #616161; margin-top: 10px;'><i>Note: Only 2024 data will be processed. Rows with missing dates or airline information will be excluded.</i></p>
        </div>
        """, unsafe_allow_html=True)
        st.stop()
    
    # Load and process data
    with st.spinner("🔄 Processing your data for 2024..."):
        df = load_and_process_data(uploaded_file)
    
    if df.empty:
        st.stop()
    
    # Success message with stats
    st.markdown(f"""
    <div class='success-box'>
    ✅ Successfully loaded {len(df):,} shipments from 2024
    </div>
    """, unsafe_allow_html=True)
    
    # Quick summary metrics
    total_brown = len(df[df['Is_UPS'] == True])
    total_green = len(df[df['Is_UPS'] == False])
    overall_util = (total_brown / len(df) * 100)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📦 Total Shipments", f"{len(df):,}")
    with col2:
        st.metric("🟫 UPS (Brown)", f"{total_brown:,}")
    with col3:
        st.metric("🟩 Others (Green)", f"{total_green:,}")
    with col4:
        st.metric("📊 Overall Utilization", f"{overall_util:.1f}%")
    
    # Create tabs
    tab1, tab2 = st.tabs(["📊 Year Overview 2024", "📈 Monthly Analysis"])
    
    # ========== TAB 1: YEAR OVERVIEW ==========
    with tab1:
        st.markdown("### 📅 Monthly Performance Overview - 2024")
        
        # Calculate monthly stats
        monthly_stats = calculate_monthly_stats(df)
        
        # Display formatted table
        display_table = format_table_display(monthly_stats)
        st.dataframe(
            display_table,
            use_container_width=True,
            hide_index=True,
            height=250
        )
        
        # Display chart
        st.markdown("### 📈 Utilization Trend Analysis")
        fig = create_utilization_chart(monthly_stats)
        st.plotly_chart(fig, use_container_width=True)
        
        # Quarterly summary
        st.markdown("### 📊 Quarterly Summary 2024")
        
        # Calculate quarterly stats
        quarterly_data = []
        quarters = ['Q1 (Jan-Mar)', 'Q2 (Apr-Jun)', 'Q3 (Jul-Sep)', 'Q4 (Oct-Dec)']
        quarter_months = [[1,2,3], [4,5,6], [7,8,9], [10,11,12]]
        
        for i, months in enumerate(quarter_months):
            q_data = monthly_stats[monthly_stats['Month_Num'].isin(months)]
            if len(q_data) > 0:
                q_brown = q_data['Brown_Volume'].sum()
                q_green = q_data['Green_Volume'].sum()
                q_total = q_brown + q_green
                q_util = (q_brown / q_total * 100) if q_total > 0 else 0
                quarterly_data.append({
                    'Quarter': quarters[i],
                    'Brown Volume': f"{q_brown:,}",
                    'Green Volume': f"{q_green:,}",
                    'Total Volume': f"{q_total:,}",
                    'Utilization %': f"{q_util:.1f}%"
                })
        
        if quarterly_data:
            q_df = pd.DataFrame(quarterly_data)
            st.dataframe(q_df, use_container_width=True, hide_index=True)
    
    # ========== TAB 2: MONTHLY ANALYSIS ==========
    with tab2:
        st.markdown("### 🔍 Detailed Monthly Analysis - 2024")
        
        # Month selector
        months = ['January', 'February', 'March', 'April', 'May', 'June',
                 'July', 'August', 'September', 'October', 'November', 'December']
        available_months = sorted(df['Month'].unique())
        available_month_names = [months[m-1] for m in available_months if m <= 12]
        
        selected_month_name = st.selectbox(
            "📅 Select Month for Analysis",
            available_month_names,
            index=len(available_month_names)-1 if available_month_names else 0
        )
        
        selected_month = months.index(selected_month_name) + 1
        df_month = df[df['Month'] == selected_month]
        
        if df_month.empty:
            st.warning("No data available for the selected month")
            st.stop()
        
        # Calculate KPIs for selected month
        brown_vol = len(df_month[df_month['Is_UPS'] == True])
        green_vol = len(df_month[df_month['Is_UPS'] == False])
        total_vol = brown_vol + green_vol
        utilization = (brown_vol / total_vol * 100) if total_vol > 0 else 0
        
        brown_kg = df_month[df_month['Is_UPS'] == True]['Weight_KG'].sum()
        green_kg = df_month[df_month['Is_UPS'] == False]['Weight_KG'].sum()
        
        # Display monthly KPIs
        st.markdown(f"### 📊 {selected_month_name} 2024 Key Metrics")
        
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        with col1:
            st.metric("🎯 BT Utilization", f"{utilization:.1f}%")
        
        with col2:
            target = 30.0  # Example target
            effectiveness = (utilization / target * 100) if target > 0 else 0
            st.metric("% Effective", f"{effectiveness:.0f}%")
        
        with col3:
            st.metric("📦 Total Volume", f"{total_vol:,}")
        
        with col4:
            # Compare to previous month if available
            prev_month = selected_month - 1 if selected_month > 1 else 12
            df_prev = df[df['Month'] == prev_month] if prev_month in available_months else pd.DataFrame()
            if not df_prev.empty:
                prev_total = len(df_prev)
                mom_change = ((total_vol - prev_total) / prev_total * 100) if prev_total > 0 else 0
                st.metric("MoM Change", f"{mom_change:+.1f}%")
            else:
                st.metric("MoM Change", "—")
        
        with col5:
            st.metric("🟫 Brown KG", f"{brown_kg:,.0f}")
        
        with col6:
            st.metric("🟩 Green KG", f"{green_kg:,.0f}")
        
        # Region and Lane Analysis
        st.markdown("### 🌍 Regional & Lane Analysis")
        
        left_col, right_col = st.columns(2)
        
        with left_col:
            st.markdown("#### 📍 Utilization by Region Pair")
            region_stats = calculate_region_stats(df_month)
            if not region_stats.empty:
                st.dataframe(
                    region_stats,
                    use_container_width=True,
                    hide_index=True,
                    height=400
                )
            else:
                st.info("No regional data available for this month")
        
        with right_col:
            st.markdown("#### ✈️ Top 30 Lanes (Origin-Destination)")
            lane_stats = calculate_lane_stats(df_month, top_n=30)
            if not lane_stats.empty:
                st.dataframe(
                    lane_stats,
                    use_container_width=True,
                    hide_index=True,
                    height=400
                )
            else:
                st.info("No lane data available for this month")
        
        # Month-over-month analysis
        with st.expander("📊 Advanced Analytics - Month over Month Trends"):
            st.markdown("#### 🌐 Region Pair Utilization % (Monthly Trend)")
            
            # Create region MoM pivot
            region_mom_data = []
            for month_num in available_months:
                if month_num <= 12:
                    df_m = df[df['Month'] == month_num]
                    for region_pair in df_m['Region Pair'].unique():
                        df_r = df_m[df_m['Region Pair'] == region_pair]
                        brown = len(df_r[df_r['Is_UPS'] == True])
                        total = len(df_r)
                        util = (brown / total * 100) if total > 0 else 0
                        region_mom_data.append({
                            'Month': months[month_num-1],
                            'Region Pair': region_pair,
                            'Utilization': util
                        })
            
            if region_mom_data:
                mom_df = pd.DataFrame(region_mom_data)
                pivot = mom_df.pivot(index='Region Pair', columns='Month', values='Utilization')
                pivot = pivot.fillna(0).round(1)
                
                # Display without gradient (matplotlib not available in deployment)
                st.dataframe(
                    pivot,
                    use_container_width=True
                )
            
            st.markdown("#### ✈️ Top 15 Lanes Utilization % (Monthly Trend)")
            
            # Get top 15 lanes overall
            top_lanes = df['Lane Pair'].value_counts().head(15).index
            
            lane_mom_data = []
            for month_num in available_months:
                if month_num <= 12:
                    df_m = df[df['Month'] == month_num]
                    for lane in top_lanes:
                        df_l = df_m[df_m['Lane Pair'] == lane]
                        if not df_l.empty:
                            brown = len(df_l[df_l['Is_UPS'] == True])
                            total = len(df_l)
                            util = (brown / total * 100) if total > 0 else 0
                            lane_mom_data.append({
                                'Month': months[month_num-1],
                                'Lane': lane,
                                'Utilization': util
                            })
            
            if lane_mom_data:
                lane_df = pd.DataFrame(lane_mom_data)
                lane_pivot = lane_df.pivot(index='Lane', columns='Month', values='Utilization')
                lane_pivot = lane_pivot.fillna(0).round(1)
                
                # Display without gradient (matplotlib not available in deployment)
                st.dataframe(
                    lane_pivot,
                    use_container_width=True
                )
        
        # Summary insights
        st.markdown("### 💡 Key Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🏆 Top 5 Performing Regions")
            region_performance = []
            for region in df_month['Origin Region'].unique():
                df_reg = df_month[df_month['Origin Region'] == region]
                brown = len(df_reg[df_reg['Is_UPS'] == True])
                total = len(df_reg)
                if total >= 10:  # Minimum threshold
                    util = brown / total * 100
                    region_performance.append({
                        'Region': region,
                        'Utilization': util,
                        'Volume': total
                    })
            
            if region_performance:
                perf_df = pd.DataFrame(region_performance)
                perf_df = perf_df.sort_values('Utilization', ascending=False).head(5)
                for _, row in perf_df.iterrows():
                    st.write(f"**{row['Region']}**: {row['Utilization']:.1f}% ({row['Volume']:,} shipments)")
        
        with col2:
            st.markdown("#### 📊 Volume Distribution")
            # Show distribution by region
            region_dist = df_month.groupby('Origin Region').size().sort_values(ascending=False).head(5)
            for region, count in region_dist.items():
                pct = (count / len(df_month) * 100)
                st.write(f"**{region}**: {count:,} ({pct:.1f}%)")

if __name__ == "__main__":
    main()
