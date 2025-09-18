import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import numpy as np
import os
import traceback

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

def load_data_from_file(file):
    """Load and preprocess the Excel data"""
    try:
        # Read Excel file
        df = pd.read_excel(file, sheet_name=0)
        
        # Check if required columns exist
        required_columns = ['Airline', 'Charge Weight kg', 'Air COGS']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            st.error(f"Missing required columns: {missing_columns}")
            st.info("Available columns: " + ", ".join(df.columns.tolist()))
            return pd.DataFrame()
        
        # Parse dates if column exists
        if 'OriginDeparture Date' in df.columns:
            df['OriginDeparture Date'] = pd.to_datetime(
                df['OriginDeparture Date'], 
                format='%m/%d/%Y', 
                errors='coerce'
            )
            df['Month'] = df['OriginDeparture Date'].dt.month
            df['Year'] = df['OriginDeparture Date'].dt.year
            df['Month_Name'] = df['OriginDeparture Date'].dt.strftime('%B')
            df['Month_Year'] = df['OriginDeparture Date'].dt.strftime('%Y-%m')
        else:
            # Create dummy date columns if not present
            df['Month'] = 1
            df['Year'] = 2024
            df['Month_Name'] = 'January'
            df['Month_Year'] = '2024-01'
        
        # Add Origin and Destination regions if not present
        if 'Origin Region ' not in df.columns:
            df['Origin Region '] = 'Unknown'
        if 'Destination Region' not in df.columns:
            df['Destination Region'] = 'Unknown'
        if 'Region Lane' not in df.columns:
            df['Region Lane'] = df['Origin Region '].astype(str) + df['Destination Region'].astype(str)
        
        # Define Brown carriers - you can customize this list
        brown_carriers = [
            'DHL AERO EXPRESO SA', 'DHL AMS', 'DHL AUSTRALIA', 
            'DHL AVIATION  EUROPEAN', 'EUROPEAN AIR TRANSPORT DHLHW', 
            'FORWARD AIR', 'UPS'  # Added UPS in case it appears
        ]
        
        # Categorize
        df['Category'] = np.where(
            df['Airline'].isin(brown_carriers), 
            'Brown', 
            'Green'
        )
        
        # Calculate commercial cost and savings
        df['Commercial_Cost'] = df['Air COGS'] * 1.3
        df['Savings'] = df['Commercial_Cost'] - df['Air COGS']
        
        # Create region pair column
        df['Region_Pair'] = df['Origin Region '].astype(str) + ' → ' + df['Destination Region'].astype(str)
        
        # Clean data
        df = df.dropna(subset=['Charge Weight kg', 'Air COGS'])
        
        return df
        
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        st.error("Full error details:")
        st.code(traceback.format_exc())
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_data():
    """Try to load data from default file"""
    default_file = 'MNX GLOBAL AF ACTIVITY 2024 V2.xlsx'
    
    if os.path.exists(default_file):
        with open(default_file, 'rb') as f:
            return load_data_from_file(f)
    else:
        return pd.DataFrame()

def calculate_monthly_metrics(df, year=2024):
    """Calculate monthly metrics for the dashboard"""
    try:
        # Filter for the specified year
        df_year = df[df['Year'] == year].copy()
        
        if df_year.empty:
            return pd.DataFrame()
        
        # Aggregate data
        agg_dict = {
            'Charge Weight kg': 'sum',
            'Air COGS': 'sum',
            'Savings': 'sum'
        }
        
        # Group by month and category
        monthly_metrics = df_year.groupby(['Month', 'Month_Name', 'Category']).agg(agg_dict).reset_index()
        
        # Create summary by month
        result = []
        for month in sorted(df_year['Month'].unique()):
            month_data = monthly_metrics[monthly_metrics['Month'] == month]
            month_name = month_data['Month_Name'].iloc[0] if not month_data.empty else ''
            
            brown_data = month_data[month_data['Category'] == 'Brown']
            green_data = month_data[month_data['Category'] == 'Green']
            
            brown_volume = brown_data['Charge Weight kg'].sum() if not brown_data.empty else 0
            green_volume = green_data['Charge Weight kg'].sum() if not green_data.empty else 0
            brown_cost = brown_data['Air COGS'].sum() if not brown_data.empty else 0
            green_cost = green_data['Air COGS'].sum() if not green_data.empty else 0
            total_savings = month_data['Savings'].sum()
            
            total_volume = brown_volume + green_volume
            utilization = (brown_volume / total_volume * 100) if total_volume > 0 else 0
            brown_cost_kg = (brown_cost / brown_volume) if brown_volume > 0 else 0
            green_cost_kg = (green_cost / green_volume) if green_volume > 0 else 0
            
            result.append({
                'Month': month,
                'Month_Name': month_name,
                'Brown_Volume': brown_volume,
                'Green_Volume': green_volume,
                'Total_Volume': total_volume,
                'Utilization_%': utilization,
                'Brown_Cost': brown_cost,
                'Green_Cost': green_cost,
                'Total_Cost': brown_cost + green_cost,
                'Brown_Cost/kg': brown_cost_kg,
                'Green_Cost/kg': green_cost_kg,
                'Total_Savings': total_savings
            })
        
        return pd.DataFrame(result)
    except Exception as e:
        st.error(f"Error calculating monthly metrics: {str(e)}")
        return pd.DataFrame()

def calculate_regional_metrics(df, selected_month=None):
    """Calculate metrics by region and region pairs"""
    try:
        if selected_month:
            df_filtered = df[df['Month'] == selected_month].copy()
        else:
            df_filtered = df.copy()
        
        if df_filtered.empty:
            return pd.DataFrame(), pd.DataFrame()
        
        # Aggregate metrics
        agg_dict = {
            'Charge Weight kg': 'sum',
            'Air COGS': 'sum',
            'Savings': 'sum'
        }
        
        # By Region Lane
        regional_metrics = df_filtered.groupby(['Region Lane', 'Category']).agg(agg_dict).reset_index()
        
        # Process region metrics
        region_results = []
        for region in df_filtered['Region Lane'].unique():
            region_data = regional_metrics[regional_metrics['Region Lane'] == region]
            
            brown_data = region_data[region_data['Category'] == 'Brown']
            green_data = region_data[region_data['Category'] == 'Green']
            
            brown_volume = brown_data['Charge Weight kg'].sum() if not brown_data.empty else 0
            green_volume = green_data['Charge Weight kg'].sum() if not green_data.empty else 0
            brown_cost = brown_data['Air COGS'].sum() if not brown_data.empty else 0
            green_cost = green_data['Air COGS'].sum() if not green_data.empty else 0
            savings = region_data['Savings'].sum()
            
            total_volume = brown_volume + green_volume
            if total_volume > 0:
                utilization = (brown_volume / total_volume * 100)
                brown_cost_kg = (brown_cost / brown_volume) if brown_volume > 0 else 0
                green_cost_kg = (green_cost / green_volume) if green_volume > 0 else 0
                
                region_results.append({
                    'Region Lane': region,
                    'Brown_Volume': brown_volume,
                    'Green_Volume': green_volume,
                    'Total_Volume': total_volume,
                    'Utilization_%': utilization,
                    'Brown_Cost': brown_cost,
                    'Green_Cost': green_cost,
                    'Brown_Cost/kg': brown_cost_kg,
                    'Green_Cost/kg': green_cost_kg,
                    'Total_Savings': savings
                })
        
        pivot_region = pd.DataFrame(region_results)
        
        # By Region Pair
        pair_metrics = df_filtered.groupby(['Region_Pair', 'Category']).agg(agg_dict).reset_index()
        
        # Process region pair metrics
        pair_results = []
        for pair in df_filtered['Region_Pair'].unique():
            pair_data = pair_metrics[pair_metrics['Region_Pair'] == pair]
            
            brown_data = pair_data[pair_data['Category'] == 'Brown']
            green_data = pair_data[pair_data['Category'] == 'Green']
            
            brown_volume = brown_data['Charge Weight kg'].sum() if not brown_data.empty else 0
            green_volume = green_data['Charge Weight kg'].sum() if not green_data.empty else 0
            brown_cost = brown_data['Air COGS'].sum() if not brown_data.empty else 0
            green_cost = green_data['Air COGS'].sum() if not green_data.empty else 0
            savings = pair_data['Savings'].sum()
            
            total_volume = brown_volume + green_volume
            if total_volume > 0:
                utilization = (brown_volume / total_volume * 100)
                brown_cost_kg = (brown_cost / brown_volume) if brown_volume > 0 else 0
                green_cost_kg = (green_cost / green_volume) if green_volume > 0 else 0
                
                pair_results.append({
                    'Region_Pair': pair,
                    'Brown_Volume': brown_volume,
                    'Green_Volume': green_volume,
                    'Total_Volume': total_volume,
                    'Utilization_%': utilization,
                    'Brown_Cost': brown_cost,
                    'Green_Cost': green_cost,
                    'Brown_Cost/kg': brown_cost_kg,
                    'Green_Cost/kg': green_cost_kg,
                    'Total_Savings': savings
                })
        
        pivot_pair = pd.DataFrame(pair_results)
        
        return pivot_region, pivot_pair
    except Exception as e:
        st.error(f"Error calculating regional metrics: {str(e)}")
        return pd.DataFrame(), pd.DataFrame()

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

# Main Application
def main():
    # Main header
    st.markdown('<div class="main-header">Green to Brown Utilization Dashboard</div>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'df' not in st.session_state:
        st.session_state.df = pd.DataFrame()
    
    # File upload section
    with st.sidebar:
        st.header("📁 Data Upload")
        
        uploaded_file = st.file_uploader(
            "Upload Excel File",
            type=['xlsx', 'xls'],
            help="Upload your MNX GLOBAL AF ACTIVITY Excel file"
        )
        
        if uploaded_file is not None:
            with st.spinner('Processing file...'):
                df = load_data_from_file(uploaded_file)
                if not df.empty:
                    st.session_state.df = df
                    st.success(f"✅ Loaded {len(df):,} rows")
        
        # Try to load default file if no upload
        elif st.session_state.df.empty:
            df = load_data()
            if not df.empty:
                st.session_state.df = df
                st.info(f"Loaded default file: {len(df):,} rows")
    
    # Get dataframe from session state
    df = st.session_state.df
    
    if df.empty:
        st.warning("⚠️ Please upload an Excel file to begin")
        st.info("""
        **Expected file format:**
        - Excel file (.xlsx or .xls)
        - Required columns: 'Airline', 'Charge Weight kg', 'Air COGS'
        - Optional columns: 'OriginDeparture Date', 'Origin Region ', 'Destination Region', 'Region Lane'
        
        The file should contain shipping/logistics data with airline information and costs.
        """)
        return
    
    # Check if data has dates
    has_dates = 'OriginDeparture Date' in df.columns and df['OriginDeparture Date'].notna().any()
    
    # Create tabs
    tab1, tab2 = st.tabs(["📊 Monthly Overview", "🌍 Regional Analysis"])
    
    with tab1:
        st.header("Monthly Utilization Stats")
        
        if not has_dates:
            st.warning("⚠️ Date information is missing. Using default values for demonstration.")
        
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
            
            with col1:
                st.metric("BT Utilization", f"{overall_utilization:.1f}%")
            with col2:
                st.metric("% Effective", "125.7%")
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
            
            if len(monthly_data) > 0:
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
                
                # Add target line
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
        
        # Get available months
        if 'Month' in df.columns:
            available_months = df['Month'].dropna().unique()
        else:
            available_months = [1]  # Default to January
        
        if len(available_months) > 0:
            selected_month = st.selectbox(
                "Select Month",
                options=sorted(available_months),
                format_func=lambda x: datetime(2024, int(x), 1).strftime('%B') if pd.notna(x) else 'Unknown'
            )
            
            # Calculate regional metrics
            with st.spinner('Calculating regional metrics...'):
                regional_data, regional_pair_data = calculate_regional_metrics(df, selected_month)
            
            # Display regional metrics
            st.subheader("Metrics by Region Lane")
            
            if not regional_data.empty:
                display_regional = pd.DataFrame({
                    'Region': regional_data['Region Lane'],
                    'Brown Volume (kg)': regional_data['Brown_Volume'].apply(lambda x: format_number(x, suffix=' kg')),
                    'Green Volume (kg)': regional_data['Green_Volume'].apply(lambda x: format_number(x, suffix=' kg')),
                    'Utilization %': regional_data['Utilization_%'].apply(lambda x: f"{x:.1f}%"),
                    'Total Savings': regional_data['Total_Savings'].apply(lambda x: format_number(x, prefix='$', decimals=0)),
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
                regional_pair_data = regional_pair_data.nlargest(20, 'Total_Volume')
                
                display_pairs = pd.DataFrame({
                    'Route': regional_pair_data['Region_Pair'],
                    'Brown Volume (kg)': regional_pair_data['Brown_Volume'].apply(lambda x: format_number(x, suffix=' kg')),
                    'Green Volume (kg)': regional_pair_data['Green_Volume'].apply(lambda x: format_number(x, suffix=' kg')),
                    'Utilization %': regional_pair_data['Utilization_%'].apply(lambda x: f"{x:.1f}%"),
                    'Total Savings': regional_pair_data['Total_Savings'].apply(lambda x: format_number(x, prefix='$', decimals=0)),
                    'Brown Cost/kg': regional_pair_data['Brown_Cost/kg'].apply(lambda x: f"${x:.2f}"),
                    'Green Cost/kg': regional_pair_data['Green_Cost/kg'].apply(lambda x: f"${x:.2f}")
                })
                
                st.dataframe(display_pairs, use_container_width=True, hide_index=True)
                
                # Visualization of top routes
                if len(regional_pair_data) > 0:
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
            st.warning("No month data available")
    
    # Sidebar additional information
    with st.sidebar:
        if not df.empty:
            st.header("Dashboard Information")
            st.info("""
            **Brown Volume**: Shipments via selected carriers
            
            **Green Volume**: All other airline shipments
            
            **Utilization %**: Brown volume / Total volume
            
            **Savings**: Difference between commercial and actual costs
            """)
            
            st.header("Data Configuration")
            st.warning("""
            ⚠️ **Note**: Configure Brown carriers in the code.
            
            Current "Brown" carriers include DHL-related carriers.
            """)
            
            # Display data summary
            st.header("Data Summary")
            st.metric("Total Records", f"{len(df):,}")
            
            if has_dates and 'OriginDeparture Date' in df.columns:
                date_min = df['OriginDeparture Date'].min()
                date_max = df['OriginDeparture Date'].max()
                if pd.notna(date_min) and pd.notna(date_max):
                    st.metric("Date Range", f"{date_min.strftime('%b %Y')} - {date_max.strftime('%b %Y')}")
            
            st.metric("Unique Airlines", f"{df['Airline'].nunique()}")
            
            # Show current Brown carriers
            brown_carriers = [
                'DHL AERO EXPRESO SA', 'DHL AMS', 'DHL AUSTRALIA', 
                'DHL AVIATION  EUROPEAN', 'EUROPEAN AIR TRANSPORT DHLHW', 
                'FORWARD AIR', 'UPS'
            ]
            
            # Show which carriers are actually in the data
            carriers_in_data = df['Airline'].unique()
            brown_in_data = [c for c in brown_carriers if c in carriers_in_data]
            
            if brown_in_data:
                st.write("**Active Brown Carriers:**")
                for carrier in brown_in_data[:5]:  # Show first 5
                    st.write(f"• {carrier}")
                if len(brown_in_data) > 5:
                    st.write(f"• ... and {len(brown_in_data)-5} more")

if __name__ == "__main__":
    main()
