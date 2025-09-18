import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Green to Brown Utilization Dashboard",
    page_icon="✈️",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 36px;
        font-weight: bold;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .metric-value {
        font-size: 32px;
        font-weight: bold;
    }
    .metric-label {
        font-size: 14px;
        color: #666;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and preprocess the Excel data"""
    # Load the Excel file
    df = pd.read_excel('MNX GLOBAL AF ACTIVITY 2024 V2.xlsx', sheet_name=0)
    
    # Parse the date column
    df['OriginDeparture Date'] = pd.to_datetime(df['OriginDeparture Date'], format='%m/%d/%Y', errors='coerce')
    
    # Extract month and year
    df['Month'] = df['OriginDeparture Date'].dt.month
    df['Year'] = df['OriginDeparture Date'].dt.year
    df['Month_Name'] = df['OriginDeparture Date'].dt.strftime('%B')
    df['Month_Year'] = df['OriginDeparture Date'].dt.strftime('%Y-%m')
    
    # Define Brown (you can modify this list based on your actual UPS/Brown carriers)
    # Since there's no UPS in the data, I'll use major carriers as an example
    # You can customize this list
    brown_carriers = ['DHL AERO EXPRESO SA', 'DHL AMS', 'DHL AUSTRALIA', 'DHL AVIATION  EUROPEAN',
                     'EUROPEAN AIR TRANSPORT DHLHW', 'FORWARD AIR']  # Example carriers for "Brown"
    
    # Categorize as Brown or Green
    df['Category'] = df['Airline'].apply(lambda x: 'Brown' if x in brown_carriers else 'Green')
    
    # Calculate commercial cost (assuming a markup for comparison)
    # Since we don't have actual commercial costs, we'll estimate
    df['Commercial_Cost'] = df['Air COGS'] * 1.3  # 30% markup as an example
    
    # Calculate savings
    df['Savings'] = df['Commercial_Cost'] - df['Air COGS']
    
    return df

def calculate_monthly_metrics(df, year=2024):
    """Calculate monthly metrics for the dashboard"""
    # Filter for the specified year
    df_year = df[df['Year'] == year].copy()
    
    if df_year.empty:
        return pd.DataFrame()
    
    # Group by month and category
    monthly_metrics = df_year.groupby(['Month', 'Month_Name', 'Category']).agg({
        'Charge Weight kg': 'sum',
        'Air COGS': 'sum',
        'Savings': 'sum',
        'Job ID': 'count'
    }).reset_index()
    
    # Pivot for easier calculations
    pivot_volume = monthly_metrics.pivot_table(
        index=['Month', 'Month_Name'],
        columns='Category',
        values='Charge Weight kg',
        fill_value=0
    ).reset_index()
    
    pivot_cost = monthly_metrics.pivot_table(
        index=['Month', 'Month_Name'],
        columns='Category',
        values='Air COGS',
        fill_value=0
    ).reset_index()
    
    pivot_savings = monthly_metrics.pivot_table(
        index=['Month', 'Month_Name'],
        columns='Category',
        values='Savings',
        fill_value=0
    ).reset_index()
    
    # Combine metrics
    result = pivot_volume.copy()
    result['Total_Volume'] = result.get('Brown', 0) + result.get('Green', 0)
    result['Brown_Volume'] = result.get('Brown', 0)
    result['Green_Volume'] = result.get('Green', 0)
    result['Utilization_%'] = (result['Brown_Volume'] / result['Total_Volume'] * 100).round(1)
    
    # Add costs
    result['Brown_Cost'] = pivot_cost.get('Brown', 0)
    result['Green_Cost'] = pivot_cost.get('Green', 0)
    result['Total_Cost'] = result['Brown_Cost'] + result['Green_Cost']
    
    # Calculate cost per kg
    result['Brown_Cost/kg'] = (result['Brown_Cost'] / result['Brown_Volume']).replace([np.inf, -np.inf], 0).round(2)
    result['Green_Cost/kg'] = (result['Green_Cost'] / result['Green_Volume']).replace([np.inf, -np.inf], 0).round(2)
    
    # Add savings
    result['Total_Savings'] = pivot_savings.get('Brown', 0) + pivot_savings.get('Green', 0)
    
    return result.sort_values('Month')

def calculate_regional_metrics(df, selected_month=None):
    """Calculate metrics by region and region pairs"""
    if selected_month:
        df_filtered = df[df['Month'] == selected_month].copy()
    else:
        df_filtered = df.copy()
    
    if df_filtered.empty:
        return pd.DataFrame(), pd.DataFrame()
    
    # By Region (using Region Lane for simplicity)
    regional_metrics = df_filtered.groupby(['Region Lane', 'Category']).agg({
        'Charge Weight kg': 'sum',
        'Air COGS': 'sum',
        'Savings': 'sum'
    }).reset_index()
    
    # Pivot for region metrics
    pivot_region = regional_metrics.pivot_table(
        index='Region Lane',
        columns='Category',
        values=['Charge Weight kg', 'Air COGS', 'Savings'],
        fill_value=0
    )
    
    # Flatten column names
    pivot_region.columns = ['_'.join(col).strip() for col in pivot_region.columns.values]
    pivot_region = pivot_region.reset_index()
    
    # Calculate utilization and cost per kg
    pivot_region['Total_Volume'] = pivot_region.get('Charge Weight kg_Brown', 0) + pivot_region.get('Charge Weight kg_Green', 0)
    pivot_region['Utilization_%'] = (pivot_region.get('Charge Weight kg_Brown', 0) / pivot_region['Total_Volume'] * 100).round(1)
    pivot_region['Brown_Cost/kg'] = (pivot_region.get('Air COGS_Brown', 0) / pivot_region.get('Charge Weight kg_Brown', 1)).replace([np.inf, -np.inf], 0).round(2)
    pivot_region['Green_Cost/kg'] = (pivot_region.get('Air COGS_Green', 0) / pivot_region.get('Charge Weight kg_Green', 1)).replace([np.inf, -np.inf], 0).round(2)
    
    # By Region Pair
    df_filtered['Region_Pair'] = df_filtered['Origin Region '].astype(str) + ' → ' + df_filtered['Destination Region'].astype(str)
    
    regional_pair_metrics = df_filtered.groupby(['Region_Pair', 'Category']).agg({
        'Charge Weight kg': 'sum',
        'Air COGS': 'sum',
        'Savings': 'sum'
    }).reset_index()
    
    # Pivot for region pair metrics
    pivot_pair = regional_pair_metrics.pivot_table(
        index='Region_Pair',
        columns='Category',
        values=['Charge Weight kg', 'Air COGS', 'Savings'],
        fill_value=0
    )
    
    # Flatten column names
    pivot_pair.columns = ['_'.join(col).strip() for col in pivot_pair.columns.values]
    pivot_pair = pivot_pair.reset_index()
    
    # Calculate utilization and cost per kg for pairs
    pivot_pair['Total_Volume'] = pivot_pair.get('Charge Weight kg_Brown', 0) + pivot_pair.get('Charge Weight kg_Green', 0)
    pivot_pair['Utilization_%'] = (pivot_pair.get('Charge Weight kg_Brown', 0) / pivot_pair['Total_Volume'] * 100).round(1)
    pivot_pair['Brown_Cost/kg'] = (pivot_pair.get('Air COGS_Brown', 0) / pivot_pair.get('Charge Weight kg_Brown', 1)).replace([np.inf, -np.inf], 0).round(2)
    pivot_pair['Green_Cost/kg'] = (pivot_pair.get('Air COGS_Green', 0) / pivot_pair.get('Charge Weight kg_Green', 1)).replace([np.inf, -np.inf], 0).round(2)
    
    return pivot_region, pivot_pair

def format_number(value, prefix='', suffix='', decimals=0):
    """Format numbers for display"""
    if pd.isna(value) or value == 0:
        return f"{prefix}0{suffix}"
    
    if abs(value) >= 1e6:
        return f"{prefix}{value/1e6:.{decimals}f}M{suffix}"
    elif abs(value) >= 1e3:
        return f"{prefix}{value/1e3:.{decimals}f}K{suffix}"
    else:
        return f"{prefix}{value:.{decimals}f}{suffix}"

# Load data
df = load_data()

# Check if data has dates
has_dates = df['OriginDeparture Date'].notna().any()

# Main app
st.markdown('<div class="main-header">Green to Brown Utilization Dashboard</div>', unsafe_allow_html=True)

# Create tabs
tab1, tab2 = st.tabs(["📊 Monthly Overview", "🌍 Regional Analysis"])

with tab1:
    st.header("Monthly Utilization Stats")
    
    if not has_dates:
        st.warning("⚠️ Date information is missing in many rows. Showing available data only.")
    
    # Get current year data
    current_year = 2024
    monthly_data = calculate_monthly_metrics(df, current_year)
    
    if not monthly_data.empty:
        # Display metrics
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        # Calculate totals
        total_brown = monthly_data['Brown_Volume'].sum()
        total_green = monthly_data['Green_Volume'].sum()
        total_volume = total_brown + total_green
        overall_utilization = (total_brown / total_volume * 100) if total_volume > 0 else 0
        total_savings = monthly_data['Total_Savings'].sum()
        avg_brown_cost = monthly_data['Brown_Cost/kg'].mean()
        avg_green_cost = monthly_data['Green_Cost/kg'].mean()
        
        with col1:
            st.metric("BT Utilization", f"{overall_utilization:.1f}%")
        with col2:
            st.metric("% Effective", "125.7%")  # Placeholder
        with col3:
            st.metric("This Year Volume", format_number(total_volume/1000, suffix='M'))
        with col4:
            st.metric("Enterprise Synergy", format_number(total_savings, prefix='$', suffix='M', decimals=1))
        with col5:
            st.metric("Actual Weight Impact", format_number(total_brown, suffix='kg'))
        with col6:
            st.metric("YoY Savings", format_number(total_savings, prefix='$', suffix='M', decimals=1))
        
        # Monthly breakdown table
        st.subheader("Monthly Breakdown")
        
        # Prepare display dataframe
        display_df = pd.DataFrame({
            'Month': monthly_data['Month_Name'],
            'Brown Volume (kg)': monthly_data['Brown_Volume'].apply(lambda x: format_number(x, suffix=' kg')),
            'Green Volume (kg)': monthly_data['Green_Volume'].apply(lambda x: format_number(x, suffix=' kg')),
            'Utilization %': monthly_data['Utilization_%'].apply(lambda x: f"{x:.1f}%"),
            'Savings': monthly_data['Total_Savings'].apply(lambda x: format_number(x, prefix='$', decimals=0)),
            'Brown Cost/kg': monthly_data['Brown_Cost/kg'].apply(lambda x: f"${x:.2f}"),
            'Green Cost/kg': monthly_data['Green_Cost/kg'].apply(lambda x: f"${x:.2f}")
        })
        
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        
        # Utilization chart
        st.subheader("BT Utilization % by Month")
        
        fig = go.Figure()
        
        # Add Brown utilization line
        fig.add_trace(go.Scatter(
            x=monthly_data['Month_Name'],
            y=monthly_data['Utilization_%'],
            mode='lines+markers',
            name='Brown Utilization %',
            line=dict(color='#8B4513', width=2),
            marker=dict(size=8)
        ))
        
        # Add target line (example)
        fig.add_trace(go.Scatter(
            x=monthly_data['Month_Name'],
            y=[30] * len(monthly_data),
            mode='lines',
            name='Target',
            line=dict(color='orange', width=2, dash='dash')
        ))
        
        fig.update_layout(
            height=400,
            xaxis_title="Month",
            yaxis_title="Utilization %",
            hovermode='x unified',
            showlegend=True,
            yaxis=dict(range=[0, 100])
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.info("No data available for the current year")

with tab2:
    st.header("Regional Analysis")
    
    # Month selector
    available_months = df[df['Year'] == 2024]['Month'].dropna().unique()
    if len(available_months) > 0:
        selected_month = st.selectbox(
            "Select Month",
            options=sorted(available_months),
            format_func=lambda x: datetime(2024, int(x), 1).strftime('%B')
        )
        
        # Calculate regional metrics
        regional_data, regional_pair_data = calculate_regional_metrics(df, selected_month)
        
        # Display regional metrics
        st.subheader("Metrics by Region Lane")
        
        if not regional_data.empty:
            display_regional = pd.DataFrame({
                'Region': regional_data['Region Lane'],
                'Brown Volume (kg)': regional_data.get('Charge Weight kg_Brown', 0).apply(lambda x: format_number(x, suffix=' kg')),
                'Green Volume (kg)': regional_data.get('Charge Weight kg_Green', 0).apply(lambda x: format_number(x, suffix=' kg')),
                'Utilization %': regional_data['Utilization_%'].apply(lambda x: f"{x:.1f}%"),
                'Total Savings': regional_data.get('Savings_Brown', 0).apply(lambda x: format_number(x, prefix='$', decimals=0)),
                'Brown Cost/kg': regional_data['Brown_Cost/kg'].apply(lambda x: f"${x:.2f}"),
                'Green Cost/kg': regional_data['Green_Cost/kg'].apply(lambda x: f"${x:.2f}")
            })
            
            st.dataframe(display_regional, use_container_width=True, hide_index=True)
        else:
            st.info("No regional data available for the selected month")
        
        # Display region pair metrics
        st.subheader("Metrics by Region Pair")
        
        if not regional_pair_data.empty:
            # Sort by total volume and show top pairs
            regional_pair_data = regional_pair_data.sort_values('Total_Volume', ascending=False).head(20)
            
            display_pairs = pd.DataFrame({
                'Route': regional_pair_data['Region_Pair'],
                'Brown Volume (kg)': regional_pair_data.get('Charge Weight kg_Brown', 0).apply(lambda x: format_number(x, suffix=' kg')),
                'Green Volume (kg)': regional_pair_data.get('Charge Weight kg_Green', 0).apply(lambda x: format_number(x, suffix=' kg')),
                'Utilization %': regional_pair_data['Utilization_%'].apply(lambda x: f"{x:.1f}%"),
                'Total Savings': regional_pair_data.get('Savings_Brown', 0).apply(lambda x: format_number(x, prefix='$', decimals=0)),
                'Brown Cost/kg': regional_pair_data['Brown_Cost/kg'].apply(lambda x: f"${x:.2f}"),
                'Green Cost/kg': regional_pair_data['Green_Cost/kg'].apply(lambda x: f"${x:.2f}")
            })
            
            st.dataframe(display_pairs, use_container_width=True, hide_index=True)
            
            # Visualization of top routes
            st.subheader("Top Routes by Volume")
            
            top_routes = regional_pair_data.head(10)
            
            fig = px.bar(
                top_routes,
                x='Total_Volume',
                y='Region_Pair',
                orientation='h',
                title='Top 10 Routes by Total Volume',
                labels={'Total_Volume': 'Total Volume (kg)', 'Region_Pair': 'Route'},
                color='Utilization_%',
                color_continuous_scale='RdYlBu_r'
            )
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No region pair data available for the selected month")
    else:
        st.warning("No data available with valid dates")

# Sidebar for additional information
with st.sidebar:
    st.header("Dashboard Information")
    st.info("""
    **Brown Volume**: Shipments via selected carriers (configurable)
    
    **Green Volume**: All other airline shipments
    
    **Utilization %**: Brown volume / Total volume
    
    **Savings**: Difference between commercial and actual costs
    
    **Cost/kg**: Average cost per kilogram
    """)
    
    st.header("Data Configuration")
    st.warning("""
    ⚠️ **Note**: The dashboard is configured with sample carrier classifications. 
    
    To properly classify "Brown" (UPS) carriers, update the `brown_carriers` list in the `load_data()` function.
    
    Current "Brown" carriers are set as DHL-related carriers for demonstration.
    """)
    
    # Display data summary
    st.header("Data Summary")
    st.metric("Total Records", f"{len(df):,}")
    st.metric("Date Range", f"{df['OriginDeparture Date'].min().strftime('%b %Y') if pd.notna(df['OriginDeparture Date'].min()) else 'N/A'} - {df['OriginDeparture Date'].max().strftime('%b %Y') if pd.notna(df['OriginDeparture Date'].max()) else 'N/A'}")
    st.metric("Unique Airlines", f"{df['Airline'].nunique()}")
    st.metric("Unique Routes", f"{df['Lane'].nunique()}")
