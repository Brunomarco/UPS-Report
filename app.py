import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
from io import BytesIO
import warnings
warnings.filterwarnings("ignore")

# ==================== PAGE & CSS ====================
st.set_page_config(page_title="Green to Brown Utilization Stats", layout="wide", page_icon="✈️")

# Custom CSS for matching the design
st.markdown("""
<style>
.main { padding: 0rem 1rem; }
h1 { color: #333; font-size: 2.5rem; font-weight: 600; }
.stTabs [data-baseweb="tab-list"] { gap: 2rem; }
.stTabs [data-baseweb="tab"] { 
    height: 50px; 
    padding: 0 20px; 
    font-size: 1.1rem; 
    font-weight: 500; 
}
div[data-testid="metric-container"] {
    background: #f5f5f5; 
    border: 1px solid #e0e0e0; 
    padding: 1rem; 
    border-radius: 8px; 
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
}
.metric-value { font-size: 1.8rem; font-weight: 600; }
.brown-text { color: #8B4513; font-weight: 700; }
.green-text { color: #228B22; font-weight: 700; }
.header-title { 
    text-align: center; 
    margin-bottom: 2rem;
}
</style>
""", unsafe_allow_html=True)

# ==================== REGION MAPPING ====================
def get_region_from_iata(code: str) -> str:
    """Map IATA codes to regions"""
    if not isinstance(code, str) or len(code) != 3:
        return "OTHER"
    
    code = code.strip().upper()
    
    # Europe codes
    EUROPE = {
        "AMS", "CDG", "ORY", "FRA", "MUC", "DUS", "BER", "HAM", "CGN", "BRU", "LGG",
        "LHR", "LGW", "STN", "LTN", "MAN", "BHX", "EDI", "GLA", "DUB", "ORK", "SNN",
        "MAD", "BCN", "VLC", "AGP", "PMI", "IBZ", "LIS", "OPO", "FAO",
        "MXP", "LIN", "FCO", "VCE", "NAP", "BLQ", "FLR", "PSA",
        "ATH", "SKG", "HER", "ZRH", "GVA", "BSL", "VIE", "SZG",
        "PRG", "BTS", "WAW", "KRK", "GDN", "OSL", "BGO", "TRD",
        "CPH", "BLL", "ARN", "GOT", "MMX", "HEL", "TKU", "OUL",
        "RIX", "TLL", "VNO", "BUD", "BEG", "ZAG", "LJU", "SKP",
        "OTP", "TSR", "SOF", "VAR", "IST", "SAW", "ESB", "AYT"
    }
    
    # Americas codes
    AMERICAS = {
        # USA
        "JFK", "EWR", "LGA", "BOS", "PHL", "IAD", "DCA", "BWI", "CLT", "RDU",
        "ATL", "MIA", "MCO", "TPA", "FLL", "RSW", "PBI", "JAX", "MSY", "IAH",
        "DFW", "AUS", "SAT", "ORD", "MDW", "DTW", "MSP", "STL", "MCI", "MKE",
        "CLE", "CVG", "CMH", "IND", "PIT", "BUF", "BNA", "MEM", "SDF",
        "DEN", "PHX", "LAS", "SLC", "ABQ", "LAX", "SFO", "SJC", "OAK", "SAN",
        "SEA", "PDX", "ANC", "HNL", "OGG",
        # Canada
        "YYZ", "YUL", "YVR", "YYC", "YOW", "YEG", "YWG", "YHZ",
        # Mexico & Central America
        "MEX", "GDL", "MTY", "CUN", "SJD", "PVR", "GUA", "SAL", "TGU", "MGA",
        "PTY", "SJO", "LIR",
        # Caribbean
        "SJU", "STT", "STX", "HAV", "SDQ", "PUJ", "POP", "KIN", "MBJ", "NAS",
        "GCM", "BGI", "POS",
        # South America
        "BOG", "MDE", "CLO", "UIO", "GYE", "LIM", "CUZ", "LPB", "VVI", "CCS",
        "CRB", "GEO", "PBM", "SCL", "EZE", "AEP", "COR", "MDZ", "GRU", "GIG",
        "BSB", "CNF", "SSA", "REC", "FOR", "MAO", "MVD", "ASU", "CAY"
    }
    
    # APAC codes
    APAC = {
        # East Asia
        "HKG", "PVG", "SHA", "PEK", "PKX", "CAN", "SZX", "XIY", "CTU", "CKG",
        "WUH", "NKG", "HGH", "XMN", "FOC", "TAO", "DLC", "TSN", "SHE", "HAK",
        "NRT", "HND", "KIX", "ITM", "NGO", "FUK", "CTS", "OKA", "ICN", "GMP",
        "PUS", "CJU", "TPE", "KHH", "TSA",
        # Southeast Asia
        "SIN", "KUL", "PEN", "LGK", "BKI", "KCH", "BKK", "DMK", "HKT", "CNX",
        "SGN", "HAN", "DAD", "CXR", "PNH", "REP", "VTE", "RGN", "MDL",
        "MNL", "CEB", "DVO", "CGK", "SUB", "DPS", "BDO", "PKU", "MES",
        # South Asia
        "DEL", "BOM", "MAA", "BLR", "HYD", "CCU", "COK", "AMD", "GOI", "ATQ",
        "CMB", "DAC", "KTM", "ISB", "KHI", "LHE", "MLE",
        # Oceania
        "SYD", "MEL", "BNE", "PER", "ADL", "OOL", "CNS", "DRW", "AKL", "WLG",
        "CHC", "ZQN", "NAN", "APW", "POM", "GUM", "SPN"
    }
    
    # ISMEA codes (India, Middle East, Africa)
    ISMEA = {
        # Middle East
        "DXB", "DWC", "AUH", "SHJ", "DOH", "KWI", "BAH", "MCT", "RUH", "JED",
        "DMM", "AMM", "BEY", "TLV", "CAI", "HRG", "SSH", "LXR", "ASW",
        # Africa
        "JNB", "CPT", "DUR", "PLZ", "GBE", "WDH", "MPM", "LAD", "LUN", "HRE",
        "NBO", "MBA", "KGL", "EBB", "DAR", "ZNZ", "ADD", "ABJ", "ACC", "LOS",
        "ABV", "PHC", "DKR", "CMN", "RAK", "TUN", "ALG", "ORN"
    }
    
    # LATIN AMERICA codes (separate from Americas for more granular analysis)
    LATAM = {
        "MEX", "GDL", "MTY", "CUN", "GUA", "SAL", "TGU", "PTY", "SJO",
        "BOG", "MDE", "CLO", "UIO", "LIM", "CUZ", "LPB", "SCL", "EZE", "GRU",
        "GIG", "BSB", "CCS", "MVD", "ASU"
    }
    
    if code in EUROPE:
        return "EUROPE"
    elif code in AMERICAS:
        return "AMERICAS"
    elif code in APAC:
        return "APAC"
    elif code in ISMEA:
        return "ISMEA"
    elif code in LATAM:
        return "LATIN AMERICA"
    else:
        return "OTHER"

# ==================== DATA LOADING ====================
@st.cache_data(show_spinner=False)
def load_data_optimized(file_bytes: bytes) -> pd.DataFrame:
    """Load and process Excel data efficiently for large files"""
    
    # Read Excel with minimal columns - only what we need
    # Columns: A (POB as text), E (Airline), F (Volumetric Weight), L (Origin IATA), M (Destination IATA)
    usecols = [0, 4, 5, 11, 12]  # Column indices for A, E, F, L, M
    
    try:
        df = pd.read_excel(
            BytesIO(file_bytes),
            usecols=usecols,
            engine='openpyxl'
        )
        
        # Rename columns based on expected positions
        df.columns = ['POB as text', 'Airline', 'Volumetric Weight (KG)', 'Origin IATA', 'Destination IATA']
        
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        return pd.DataFrame()
    
    # Clean and filter data
    df = df.dropna(subset=['POB as text', 'Airline'])
    df['Airline'] = df['Airline'].astype(str).str.strip()
    df = df[df['Airline'] != '']
    
    # Parse dates
    df['Date'] = pd.to_datetime(df['POB as text'], errors='coerce')
    df = df.dropna(subset=['Date'])
    
    # Extract date components
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Month_Name'] = df['Date'].dt.strftime('%B')
    
    # Process weight - handle missing values
    df['Weight_KG'] = pd.to_numeric(df['Volumetric Weight (KG)'], errors='coerce').fillna(0)
    
    # Identify UPS Airlines (Brown) vs Others (Green)
    df['Is_Brown'] = df['Airline'].str.upper().str.contains('UPS', na=False)
    df['Category'] = df['Is_Brown'].map({True: 'Brown', False: 'Green'})
    
    # Process IATA codes and regions
    df['Origin IATA'] = df['Origin IATA'].astype(str).str.strip().str.upper()
    df['Destination IATA'] = df['Destination IATA'].astype(str).str.strip().str.upper()
    
    # Map to regions
    df['Origin Region'] = df['Origin IATA'].apply(get_region_from_iata)
    df['Destination Region'] = df['Destination IATA'].apply(get_region_from_iata)
    
    # Create region family (same region or cross-region)
    df['Region Family'] = df.apply(
        lambda x: x['Origin Region'] if x['Origin Region'] == x['Destination Region'] else 'CROSS-REGION',
        axis=1
    )
    
    # Create lanes
    df['Lane'] = df['Origin IATA'] + '-' + df['Destination IATA']
    
    return df

# ==================== CALCULATION FUNCTIONS ====================
def calculate_monthly_stats(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """Calculate monthly statistics for the given year"""
    
    df_year = df[df['Year'] == year].copy()
    
    # Group by month and category
    monthly = df_year.groupby(['Month', 'Month_Name', 'Category']).agg({
        'Airline': 'count',  # Volume (number of shipments)
        'Weight_KG': 'sum'    # Total weight
    }).rename(columns={'Airline': 'Volume'}).reset_index()
    
    # Pivot for easier calculation
    volume_pivot = monthly.pivot_table(
        index=['Month', 'Month_Name'],
        columns='Category',
        values='Volume',
        fill_value=0
    ).reset_index()
    
    weight_pivot = monthly.pivot_table(
        index=['Month', 'Month_Name'],
        columns='Category',
        values='Weight_KG',
        fill_value=0
    ).reset_index()
    
    # Ensure we have both Brown and Green columns
    for col in ['Brown', 'Green']:
        if col not in volume_pivot.columns:
            volume_pivot[col] = 0
        if col not in weight_pivot.columns:
            weight_pivot[col] = 0
    
    # Calculate utilization
    volume_pivot['Total_Volume'] = volume_pivot['Brown'] + volume_pivot['Green']
    volume_pivot['Utilization_%'] = (volume_pivot['Brown'] / volume_pivot['Total_Volume'] * 100).fillna(0)
    
    # Merge weight data
    result = volume_pivot.merge(
        weight_pivot[['Month', 'Brown', 'Green']],
        on='Month',
        suffixes=('_Volume', '_Weight')
    )
    
    # Calculate totals
    totals = pd.DataFrame({
        'Month': [13],
        'Month_Name': ['Total'],
        'Brown_Volume': [result['Brown_Volume'].sum()],
        'Green_Volume': [result['Green_Volume'].sum()],
        'Total_Volume': [result['Total_Volume'].sum()],
        'Brown_Weight': [result['Brown_Weight'].sum()],
        'Green_Weight': [result['Green_Weight'].sum()],
        'Utilization_%': [0]  # Will calculate separately
    })
    
    if totals['Total_Volume'].iloc[0] > 0:
        totals['Utilization_%'] = (totals['Brown_Volume'] / totals['Total_Volume'] * 100).iloc[0]
    
    result = pd.concat([result, totals], ignore_index=True)
    
    return result

def calculate_region_stats(df: pd.DataFrame, selected_month: int = None) -> pd.DataFrame:
    """Calculate statistics by region"""
    
    if selected_month:
        df_filtered = df[df['Month'] == selected_month].copy()
    else:
        df_filtered = df.copy()
    
    # Group by region family and category
    region_stats = df_filtered.groupby(['Region Family', 'Category']).agg({
        'Airline': 'count',
        'Weight_KG': 'sum'
    }).rename(columns={'Airline': 'Volume'}).reset_index()
    
    # Pivot for easier calculation
    volume_pivot = region_stats.pivot_table(
        index='Region Family',
        columns='Category',
        values='Volume',
        fill_value=0
    )
    
    weight_pivot = region_stats.pivot_table(
        index='Region Family',
        columns='Category',
        values='Weight_KG',
        fill_value=0
    )
    
    # Ensure columns exist
    for col in ['Brown', 'Green']:
        if col not in volume_pivot.columns:
            volume_pivot[col] = 0
        if col not in weight_pivot.columns:
            weight_pivot[col] = 0
    
    # Calculate utilization
    result = pd.DataFrame({
        'Region Family': volume_pivot.index,
        'Brown Volume (#)': volume_pivot['Brown'].values,
        'Green Volume (#)': volume_pivot['Green'].values,
        'Brown KG': weight_pivot['Brown'].values,
        'Green KG': weight_pivot['Green'].values
    })
    
    result['Total Volume'] = result['Brown Volume (#)'] + result['Green Volume (#)']
    result['Utilization %'] = (result['Brown Volume (#)'] / result['Total Volume'] * 100).fillna(0)
    
    # Sort by utilization
    result = result.sort_values('Utilization %', ascending=False)
    
    return result

def calculate_lane_stats(df: pd.DataFrame, selected_month: int = None, top_n: int = 30) -> pd.DataFrame:
    """Calculate statistics by lane (Origin-Destination pairs)"""
    
    if selected_month:
        df_filtered = df[df['Month'] == selected_month].copy()
    else:
        df_filtered = df.copy()
    
    # Get top lanes by total volume
    top_lanes = df_filtered.groupby('Lane').size().nlargest(top_n).index
    df_filtered = df_filtered[df_filtered['Lane'].isin(top_lanes)]
    
    # Group by lane and category
    lane_stats = df_filtered.groupby(['Lane', 'Category']).agg({
        'Airline': 'count',
        'Weight_KG': 'sum'
    }).rename(columns={'Airline': 'Volume'}).reset_index()
    
    # Pivot for easier calculation
    volume_pivot = lane_stats.pivot_table(
        index='Lane',
        columns='Category',
        values='Volume',
        fill_value=0
    )
    
    weight_pivot = lane_stats.pivot_table(
        index='Lane',
        columns='Category',
        values='Weight_KG',
        fill_value=0
    )
    
    # Ensure columns exist
    for col in ['Brown', 'Green']:
        if col not in volume_pivot.columns:
            volume_pivot[col] = 0
        if col not in weight_pivot.columns:
            weight_pivot[col] = 0
    
    # Calculate utilization
    result = pd.DataFrame({
        'Lane': volume_pivot.index,
        'Brown Volume (#)': volume_pivot['Brown'].values,
        'Green Volume (#)': volume_pivot['Green'].values,
        'Brown KG': weight_pivot['Brown'].values,
        'Green KG': weight_pivot['Green'].values
    })
    
    result['Total Volume'] = result['Brown Volume (#)'] + result['Green Volume (#)']
    result['Utilization %'] = (result['Brown Volume (#)'] / result['Total Volume'] * 100).fillna(0)
    
    # Sort by utilization
    result = result.sort_values('Utilization %', ascending=False)
    
    return result

def format_region_table(region_stats: pd.DataFrame) -> pd.DataFrame:
    """Format region statistics for display"""
    display_df = region_stats.copy()
    
    # Format numbers
    display_df['Brown Volume (#)'] = display_df['Brown Volume (#)'].apply(lambda x: f"{int(x):,}")
    display_df['Green Volume (#)'] = display_df['Green Volume (#)'].apply(lambda x: f"{int(x):,}")
    display_df['Brown KG'] = display_df['Brown KG'].apply(lambda x: f"{x:,.0f}")
    display_df['Green KG'] = display_df['Green KG'].apply(lambda x: f"{x:,.0f}")
    display_df['Utilization %'] = display_df['Utilization %'].apply(lambda x: f"{x:.1f}%")
    
    # Remove total volume column for display
    display_df = display_df[['Region Family', 'Brown Volume (#)', 'Green Volume (#)', 
                             'Brown KG', 'Green KG', 'Utilization %']]
    
    return display_df

def format_lane_table(lane_stats: pd.DataFrame) -> pd.DataFrame:
    """Format lane statistics for display"""
    display_df = lane_stats.copy()
    
    # Format numbers
    display_df['Brown Volume (#)'] = display_df['Brown Volume (#)'].apply(lambda x: f"{int(x):,}")
    display_df['Green Volume (#)'] = display_df['Green Volume (#)'].apply(lambda x: f"{int(x):,}")
    display_df['Brown KG'] = display_df['Brown KG'].apply(lambda x: f"{x:,.0f}")
    display_df['Green KG'] = display_df['Green KG'].apply(lambda x: f"{x:,.0f}")
    display_df['Utilization %'] = display_df['Utilization %'].apply(lambda x: f"{x:.1f}%")
    
    # Remove total volume column for display
    display_df = display_df[['Lane', 'Brown Volume (#)', 'Green Volume (#)', 
                             'Brown KG', 'Green KG', 'Utilization %']]
    
    return display_df

# ==================== VISUALIZATION FUNCTIONS ====================
def create_utilization_chart(monthly_stats: pd.DataFrame, year: int) -> go.Figure:
    """Create the utilization percentage chart"""
    
    # Filter out the total row
    chart_data = monthly_stats[monthly_stats['Month'] <= 12].copy()
    
    fig = go.Figure()
    
    # Add utilization line
    fig.add_trace(go.Scatter(
        x=chart_data['Month_Name'],
        y=chart_data['Utilization_%'],
        mode='lines+markers+text',
        name='BT Utilization %',
        line=dict(color='#FF8C00', width=3),
        marker=dict(size=10, color='#FF8C00'),
        text=[f"{val:.1f}%" for val in chart_data['Utilization_%']],
        textposition='top center',
        textfont=dict(size=10),
        hovertemplate='%{x}<br>Utilization: %{y:.1f}%<extra></extra>'
    ))
    
    fig.update_layout(
        title=f'BT Utilization % by Month and Year ({year})',
        xaxis_title='Month',
        yaxis_title='%',
        yaxis=dict(range=[0, 100], gridcolor='#E0E0E0'),
        xaxis=dict(gridcolor='#E0E0E0'),
        height=400,
        hovermode='x unified',
        plot_bgcolor='white',
        showlegend=False
    )
    
    return fig

def format_display_table(monthly_stats: pd.DataFrame) -> pd.DataFrame:
    """Format the monthly statistics table for display"""
    
    # Create display dataframe
    display_data = []
    
    # Brown Volume row
    brown_vol = ['Brown Volume (#)']
    for _, row in monthly_stats.iterrows():
        if row['Month'] <= 12:
            brown_vol.append(f"{int(row['Brown_Volume']):,}")
    display_data.append(brown_vol)
    
    # Green Volume row
    green_vol = ['Green Volume (#)']
    for _, row in monthly_stats.iterrows():
        if row['Month'] <= 12:
            green_vol.append(f"{int(row['Green_Volume']):,}")
    display_data.append(green_vol)
    
    # Utilization row
    util_row = ['Utilization%']
    for _, row in monthly_stats.iterrows():
        if row['Month'] <= 12:
            util_row.append(f"{row['Utilization_%']:.1f}%")
    display_data.append(util_row)
    
    # Weight Impact row
    weight_row = ['Weight Impact']
    for _, row in monthly_stats.iterrows():
        if row['Month'] <= 12:
            total_weight = row['Brown_Weight'] + row['Green_Weight']
            weight_row.append(f"{total_weight:,.0f} kg")
    display_data.append(weight_row)
    
    # Brown Cost/kg row
    brown_kg = ['Brown Cost/Kg']
    for _, row in monthly_stats.iterrows():
        if row['Month'] <= 12:
            brown_kg.append(f"${row['Brown_Weight']:,.2f}")
    display_data.append(brown_kg)
    
    # Green Cost/kg row
    green_kg = ['Green Cost/Kg']
    for _, row in monthly_stats.iterrows():
        if row['Month'] <= 12:
            green_kg.append(f"${row['Green_Weight']:,.2f}")
    display_data.append(green_kg)
    
    # YoY Savings row
    savings = ['YoY Savings']
    for _, row in monthly_stats.iterrows():
        if row['Month'] <= 12:
            # Placeholder for YoY savings calculation
            savings.append("—")
    display_data.append(savings)
    
    # Create dataframe with month names as columns
    months = ['January', 'February', 'March', 'April', 'May', 'June', 
              'July', 'August', 'September', 'October', 'November', 'December']
    
    df_display = pd.DataFrame(display_data, columns=['Metric'] + months[:len(brown_vol)-1])
    
    # Add totals column
    total_stats = monthly_stats[monthly_stats['Month'] == 13].iloc[0] if len(monthly_stats[monthly_stats['Month'] == 13]) > 0 else None
    
    if total_stats is not None:
        df_display['Total'] = [
            f"{int(total_stats['Brown_Volume']):,}",
            f"{int(total_stats['Green_Volume']):,}",
            f"{total_stats['Utilization_%']:.1f}%",
            f"{total_stats['Brown_Weight'] + total_stats['Green_Weight']:,.0f} kg",
            f"${total_stats['Brown_Weight']:,.2f}",
            f"${total_stats['Green_Weight']:,.2f}",
            "—"
        ]
    
    return df_display
