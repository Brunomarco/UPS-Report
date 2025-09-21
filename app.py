import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import numpy as np
import calendar
import io

# Page configuration
st.set_page_config(
    page_title="Green to Brown Monthly Utilization Stats",
    page_icon="📦",
    layout="wide"
)

# Custom CSS to match the screenshot style
st.markdown("""
<style>
    .main-header {
        font-size: 36px;
        font-weight: bold;
        color: #333;
        margin-bottom: 20px;
    }
    
    .metric-container {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 10px;
    }
    
    [data-testid="metric-container"] {
        background-color: rgba(255, 255, 255, 0.95);
        border: 1px solid #e0e0e0;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        font-weight: 600;
    }
    
    .uploadedFile {
        display: none;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data(uploaded_file):
    """Load and process the Excel data"""
    try:
        # Load the Excel file
        df = pd.read_excel(uploaded_file)
        
        # Parse the date column (M/D/YYYY format)
        df['OriginDeparture Date'] = pd.to_datetime(df['OriginDeparture Date'], format='%m/%d/%Y', errors='coerce')
        
        # Extract month and year
        df['Month'] = df['OriginDeparture Date'].dt.month
        df['Year'] = df['OriginDeparture Date'].dt.year
        df['Month_Name'] = df['OriginDeparture Date'].dt.strftime('%B')
        df['Month_Year'] = df['OriginDeparture Date'].dt.strftime('%Y-%m')
        
        # Define UPS/Brown carriers
        # IMPORTANT: Update this list with actual UPS carrier names from your data
        ups_carriers = ['UPS', 'United Parcel Service', 'UPS Airlines', 'UPS SCS']
        
        # Check if any UPS carriers exist in the data
        actual_ups = [c for c in ups_carriers if c in df['Airline'].unique()]
        
        if not actual_ups:
            # If no UPS found, use specific carriers as Brown for demo
            # YOU SHOULD UPDATE THIS LIST with carriers that should be considered as "Brown"
            brown_carriers = ['DHL AERO EXPRESO SA', 'DHL AMS', 'DHL AUSTRALIA', 
                            'DHL AVIATION  EUROPEAN', 'EUROPEAN AIR TRANSPORT DHLHW']
            st.sidebar.warning("⚠️ No 'UPS' found in airline names. Using DHL carriers as Brown for demo. Update the code with actual UPS carrier names.")
        else:
            brown_carriers = actual_ups
            st.sidebar.success(f"✅ Found UPS carriers: {', '.join(actual_ups)}")
        
        # Categorize as Brown (UPS) or Green (all others)
        df['Is_Brown'] = df['Airline'].isin(brown_carriers)
        df['Category'] = df['Is_Brown'].map({True: 'Brown', False: 'Green'})
        
        # Create region pair column
        df['Region_Pair'] = df['Origin Region '].astype(str) + ' → ' + df['Destination Region'].astype(str)
        
        return df, brown_carriers
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.error("Please make sure your Excel file has the following columns:")
        st.error("- Airline, Charge Weight kg, Air COGS, OriginDeparture Date")
        st.error("- Origin Region , Destination Region, Region Lane")
        return pd.DataFrame(), []

def calculate_monthly_metrics(df):
    """Calculate monthly metrics for the current year"""
    # Filter for 2024 (current year)
    df_2024 = df[df['Year'] == 2024].copy()
    
    if df_2024.empty:
        st.warning("No data found for 2024. Showing all available data.")
        df_2024 = df.copy()
    
    # Group by month and category
    monthly_metrics = df_2024.groupby(['Month', 'Month_Name', 'Category']).agg({
        'Charge Weight kg': 'sum',
        'Air COGS': 'sum',
        'Job ID': 'count'
    }).reset_index()
    
    # Pivot to get Brown and Green side by side
    pivot_weight = monthly_metrics.pivot_table(
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
    
    # Merge the data
    result = pivot_weight.copy()
    
    # Ensure columns exist
    if 'Brown' not in result.columns:
        result['Brown'] = 0
    if 'Green' not in result.columns:
        result['Green'] = 0
    
    # Calculate metrics
    result['Brown_Volume_kg'] = result['Brown']
    result['Green_Volume_kg'] = result['Green']
    result['Total_Volume_kg'] = result['Brown_Volume_kg'] + result['Green_Volume_kg']
    
    # Calculate utilization %
    result['Utilization_%'] = np.where(
        result['Total_Volume_kg'] > 0,
        (result['Brown_Volume_kg'] / result['Total_Volume_kg']) * 100,
        0
    )
    
    # Add costs
    result['Brown_Cost'] = pivot_cost.get('Brown', 0) if 'Brown' in pivot_cost.columns else 0
    result['Green_Cost'] = pivot_cost.get('Green', 0) if 'Green' in pivot_cost.columns else 0
    
    # Calculate cost per kg
    result['Brown_Cost_per_kg'] = np.where(
        result['Brown_Volume_kg'] > 0,
        result['Brown_Cost'] / result['Brown_Volume_kg'],
        0
    )
    
    result['Green_Cost_per_kg'] = np.where(
        result['Green_Volume_kg'] > 0,
        result['Green_Cost'] / result['Green_Volume_kg'],
        0
    )
    
    # Calculate savings (assuming commercial rate is 20% higher than UPS rate)
    result['Commercial_Cost'] = result['Brown_Cost'] * 1.2
    result['Savings'] = result['Commercial_Cost'] - result['Brown_Cost']
    
    return result.sort_values('Month')

def calculate_regional_metrics(df, selected_month):
    """Calculate metrics by region for a specific month"""
    # Filter for selected month and year 2024
    df_month = df[(df['Month'] == selected_month) & (df['Year'] == 2024)].copy()
    
    if df_month.empty:
        # Try without year filter
        df_month = df[df['Month'] == selected_month].copy()
    
    if df_month.empty:
        return pd.DataFrame(), pd.DataFrame()
    
    # By Region Lane
    regional_metrics = df_month.groupby(['Region Lane', 'Category']).agg({
        'Charge Weight kg': 'sum',
        'Air COGS': 'sum'
    }).reset_index()
    
    # Pivot for region metrics
    pivot_region_weight = regional_metrics.pivot_table(
        index='Region Lane',
        columns='Category',
        values='Charge Weight kg',
        fill_value=0
    ).reset_index()
    
    pivot_region_cost = regional_metrics.pivot_table(
        index='Region Lane',
        columns='Category',
        values='Air COGS',
        fill_value=0
    ).reset_index()
    
    # Merge and calculate metrics
    region_result = pivot_region_weight.copy()
    
    if 'Brown' not in region_result.columns:
        region_result['Brown'] = 0
    if 'Green' not in region_result.columns:
        region_result['Green'] = 0
    
    region_result['Brown_Volume_kg'] = region_result['Brown']
    region_result['Green_Volume_kg'] = region_result['Green']
    region_result['Total_Volume_kg'] = region_result['Brown_Volume_kg'] + region_result['Green_Volume_kg']
    region_result['Utilization_%'] = np.where(
        region_result['Total_Volume_kg'] > 0,
        (region_result['Brown_Volume_kg'] / region_result['Total_Volume_kg']) * 100,
        0
    )
    
    # Add costs
    region_result['Brown_Cost'] = pivot_region_cost.get('Brown', 0) if 'Brown' in pivot_region_cost.columns else 0
    region_result['Green_Cost'] = pivot_region_cost.get('Green', 0) if 'Green' in pivot_region_cost.columns else 0
    
    # Cost per kg
    region_result['Brown_Cost_per_kg'] = np.where(
        region_result['Brown_Volume_kg'] > 0,
        region_result['Brown_Cost'] / region_result['Brown_Volume_kg'],
        0
    )
    
    region_result['Green_Cost_per_kg'] = np.where(
        region_result['Green_Volume_kg'] > 0,
        region_result['Green_Cost'] / region_result['Green_Volume_kg'],
        0
    )
    
    # Savings
    region_result['Commercial_Cost'] = region_result['Brown_Cost'] * 1.2
    region_result['Savings'] = region_result['Commercial_Cost'] - region_result['Brown_Cost']
    
    # By Region Pair (Origin → Destination)
    pair_metrics = df_month.groupby(['Region_Pair', 'Category']).agg({
        'Charge Weight kg': 'sum',
        'Air COGS': 'sum'
    }).reset_index()
    
    # Pivot for region pair metrics
    pivot_pair_weight = pair_metrics.pivot_table(
        index='Region_Pair',
        columns='Category',
        values='Charge Weight kg',
        fill_value=0
    ).reset_index()
    
    pivot_pair_cost = pair_metrics.pivot_table(
        index='Region_Pair',
        columns='Category',
        values='Air COGS',
        fill_value=0
    ).reset_index()
    
    # Merge and calculate metrics for pairs
    pair_result = pivot_pair_weight.copy()
    
    if 'Brown' not in pair_result.columns:
        pair_result['Brown'] = 0
    if 'Green' not in pair_result.columns:
        pair_result['Green'] = 0
    
    pair_result['Brown_Volume_kg'] = pair_result['Brown']
    pair_result['Green_Volume_kg'] = pair_result['Green']
    pair_result['Total_Volume_kg'] = pair_result['Brown_Volume_kg'] + pair_result['Green_Volume_kg']
    pair_result['Utilization_%'] = np.where(
        pair_result['Total_Volume_kg'] > 0,
        (pair_result['Brown_Volume_kg'] / pair_result['Total_Volume_kg']) * 100,
        0
    )
    
    # Add costs
    pair_result['Brown_Cost'] = pivot_pair_cost.get('Brown', 0) if 'Brown' in pivot_pair_cost.columns else 0
    pair_result['Green_Cost'] = pivot_pair_cost.get('Green', 0) if 'Green' in pivot_pair_cost.columns else 0
    
    # Cost per kg
    pair_result['Brown_Cost_per_kg'] = np.where(
        pair_result['Brown_Volume_kg'] > 0,
        pair_result['Brown_Cost'] / pair_result['Brown_Volume_kg'],
        0
    )
    
    pair_result['Green_Cost_per_kg'] = np.where(
        pair_result['Green_Volume_kg'] > 0,
        pair_result['Green_Cost'] / pair_result['Green_Volume_kg'],
        0
    )
    
    # Savings
    pair_result['Commercial_Cost'] = pair_result['Brown_Cost'] * 1.2
    pair_result['Savings'] = pair_result['Commercial_Cost'] - pair_result['Brown_Cost']
    
    # Sort by total volume
    pair_result = pair_result.sort_values('Total_Volume_kg', ascending=False)
    
    return region_result, pair_result

def format_number(value, decimals=0):
    """Format large numbers with K, M, B suffixes"""
    if pd.isna(value) or value == 0:
        return "0"
    
    if abs(value) >= 1e9:
        return f"{value/1e9:.{decimals}f}B"
    elif abs(value) >= 1e6:
        return f"{value/1e6:.{decimals}f}M"
    elif abs(value) >= 1e3:
        return f"{value/1e3:.{decimals}f}K"
    else:
        return f"{value:.{decimals}f}"

def format_currency(value):
    """Format currency values"""
    if pd.isna(value) or value == 0:
        return "$0"
    return f"${format_number(value, 2)}"

# Main app
def main():
    # Header
    st.markdown('<h1 style="color: #2e7d32;">Green to Brown Monthly Utilization Stats</h1>', 
                unsafe_allow_html=True)
    
    # File uploader
    uploaded_file = st.file_uploader(
        "📁 Upload MNX GLOBAL AF ACTIVITY Excel File",
        type=['xlsx', 'xls'],
        help="Upload your MNX GLOBAL AF ACTIVITY 2024 V2.xlsx file"
    )
    
    if uploaded_file is None:
        st.info("👆 Please upload your Excel file to begin analysis")
        st.markdown("""
        ### Expected File Format:
        The Excel file should contain the following columns:
        - **Airline**: Name of the carrier
        - **Charge Weight kg**: Weight in kilograms (Column S)
        - **Air COGS**: Air cost of goods sold (Column T)
        - **OriginDeparture Date**: Date in M/D/YYYY format
        - **Origin Region**: Origin region
        - **Destination Region**: Destination region
        - **Region Lane**: Combined region information
        """)
        return
    
    # Load data
    with st.spinner('Loading and processing data... This may take a moment for large files.'):
        df, brown_carriers = load_data(uploaded_file)
    
    if df.empty:
        return
    
    # Sidebar info
    with st.sidebar:
        st.header("📊 Dashboard Info")
        st.success(f"✅ File loaded successfully!")
        st.info(f"""
        **Data Summary:**
        - Total Records: {len(df):,}
        - Date Range: {df['OriginDeparture Date'].min().strftime('%b %Y') if pd.notna(df['OriginDeparture Date'].min()) else 'N/A'} to {df['OriginDeparture Date'].max().strftime('%b %Y') if pd.notna(df['OriginDeparture Date'].max()) else 'N/A'}
        
        **Categories:**
        - Brown (UPS): {len(df[df['Category'] == 'Brown']):,} records
        - Green (Others): {len(df[df['Category'] == 'Green']):,} records
        
        **Brown Carriers Identified:**
        {chr(10).join(['• ' + c for c in brown_carriers[:5]])}{'...' if len(brown_carriers) > 5 else ''}
        """)
        
        st.divider()
        
        # Add configuration section
        st.header("⚙️ Configuration")
        
        # Allow user to specify UPS carriers
        st.markdown("**Specify UPS/Brown Carriers:**")
        user_ups_carriers = st.text_area(
            "Enter carrier names (one per line):",
            value='\n'.join(brown_carriers[:3]) if brown_carriers else "UPS\nUnited Parcel Service",
            height=100,
            help="Enter the exact airline names that should be considered as Brown (UPS)"
        )
        
        if st.button("Update Carriers"):
            # Update the categorization based on user input
            new_brown_carriers = [c.strip() for c in user_ups_carriers.split('\n') if c.strip()]
            df['Is_Brown'] = df['Airline'].isin(new_brown_carriers)
            df['Category'] = df['Is_Brown'].map({True: 'Brown', False: 'Green'})
            st.success(f"Updated! Brown carriers: {', '.join(new_brown_carriers[:3])}...")
            st.rerun()
    
    # Create tabs
    tab1, tab2 = st.tabs(["📅 Monthly Overview", "🌍 Regional Analysis"])
    
    with tab1:
        st.header("Monthly Utilization - Year 2024")
        
        # Calculate monthly metrics
        monthly_data = calculate_monthly_metrics(df)
        
        if not monthly_data.empty:
            # Display top-level metrics
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            
            total_brown = monthly_data['Brown_Volume_kg'].sum()
            total_green = monthly_data['Green_Volume_kg'].sum()
            total_volume = total_brown + total_green
            overall_utilization = (total_brown / total_volume * 100) if total_volume > 0 else 0
            total_savings = monthly_data['Savings'].sum()
            avg_brown_cost = monthly_data['Brown_Cost_per_kg'].mean()
            avg_green_cost = monthly_data['Green_Cost_per_kg'].mean()
            
            with col1:
                st.metric("Overall Utilization %", f"{overall_utilization:.1f}%")
            
            with col2:
                st.metric("Total Volume", format_number(total_volume))
            
            with col3:
                st.metric("Brown Volume", format_number(total_brown))
            
            with col4:
                st.metric("Green Volume", format_number(total_green))
            
            with col5:
                st.metric("Total Savings", format_currency(total_savings))
            
            with col6:
                st.metric("Avg Brown Cost/kg", f"${avg_brown_cost:.2f}")
            
            # Monthly metrics table
            st.subheader("📊 Monthly Metrics Table")
            
            # Prepare display table
            display_df = monthly_data[['Month_Name', 'Brown_Volume_kg', 'Green_Volume_kg', 
                                       'Total_Volume_kg', 'Utilization_%', 'Savings',
                                       'Brown_Cost_per_kg', 'Green_Cost_per_kg']].copy()
            
            # Format columns for display
            display_df['Brown_Volume_kg'] = display_df['Brown_Volume_kg'].apply(lambda x: format_number(x))
            display_df['Green_Volume_kg'] = display_df['Green_Volume_kg'].apply(lambda x: format_number(x))
            display_df['Total_Volume_kg'] = display_df['Total_Volume_kg'].apply(lambda x: format_number(x))
            display_df['Utilization_%'] = display_df['Utilization_%'].apply(lambda x: f"{x:.1f}%")
            display_df['Savings'] = display_df['Savings'].apply(format_currency)
            display_df['Brown_Cost_per_kg'] = display_df['Brown_Cost_per_kg'].apply(lambda x: f"${x:.2f}")
            display_df['Green_Cost_per_kg'] = display_df['Green_Cost_per_kg'].apply(lambda x: f"${x:.2f}")
            
            # Rename columns for better display
            display_df.columns = ['Month', 'Brown Volume (kg)', 'Green Volume (kg)', 
                                  'Total Volume (kg)', 'Utilization %', 'Savings ($)',
                                  'Brown Cost/kg', 'Green Cost/kg']
            
            st.dataframe(display_df, use_container_width=True, hide_index=True)
            
            # Download button for the table
            csv = display_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Monthly Metrics as CSV",
                data=csv,
                file_name=f"monthly_metrics_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
            
            # Utilization Chart
            st.subheader("📈 Monthly Utilization % Trend")
            
            fig = go.Figure()
            
            # Add utilization line
            fig.add_trace(go.Scatter(
                x=monthly_data['Month_Name'],
                y=monthly_data['Utilization_%'],
                mode='lines+markers',
                name='Utilization %',
                line=dict(color='#2e7d32', width=3),
                marker=dict(size=10, color='#2e7d32'),
                text=[f"{x:.1f}%" for x in monthly_data['Utilization_%']],
                textposition='top center'
            ))
            
            # Add target line (example: 30%)
            fig.add_trace(go.Scatter(
                x=monthly_data['Month_Name'],
                y=[30] * len(monthly_data),
                mode='lines',
                name='Target (30%)',
                line=dict(color='orange', width=2, dash='dash')
            ))
            
            fig.update_layout(
                xaxis_title="Month",
                yaxis_title="Utilization %",
                yaxis=dict(range=[0, max(monthly_data['Utilization_%'].max() + 10, 40)]),
                hovermode='x unified',
                height=450,
                showlegend=True,
                plot_bgcolor='white',
                paper_bgcolor='white'
            )
            
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#E0E0E0')
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#E0E0E0')
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Volume comparison chart
            st.subheader("📊 Brown vs Green Volume by Month")
            
            fig2 = go.Figure()
            
            fig2.add_trace(go.Bar(
                x=monthly_data['Month_Name'],
                y=monthly_data['Brown_Volume_kg'],
                name='Brown (UPS)',
                marker_color='#8B4513',
                text=[format_number(x) for x in monthly_data['Brown_Volume_kg']],
                textposition='outside'
            ))
            
            fig2.add_trace(go.Bar(
                x=monthly_data['Month_Name'],
                y=monthly_data['Green_Volume_kg'],
                name='Green (Others)',
                marker_color='#2e7d32',
                text=[format_number(x) for x in monthly_data['Green_Volume_kg']],
                textposition='outside'
            ))
            
            fig2.update_layout(
                xaxis_title="Month",
                yaxis_title="Volume (kg)",
                barmode='group',
                height=400,
                showlegend=True,
                plot_bgcolor='white',
                paper_bgcolor='white'
            )
            
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.warning("No monthly data available to display")
    
    with tab2:
        st.header("Regional Analysis")
        
        # Month selector
        available_months = sorted(df[df['Year'] == 2024]['Month'].dropna().unique())
        
        if not available_months:
            available_months = sorted(df['Month'].dropna().unique())
        
        if available_months:
            selected_month = st.selectbox(
                "Select Month for Analysis",
                available_months,
                format_func=lambda x: calendar.month_name[int(x)] if pd.notna(x) else 'Unknown',
                index=len(available_months)-1 if available_months else 0
            )
            
            # Calculate regional metrics
            regional_data, pair_data = calculate_regional_metrics(df, selected_month)
            
            # Display metrics for selected month
            st.subheader(f"📅 Analysis for {calendar.month_name[selected_month]} 2024")
            
            # Summary metrics for the month
            month_df = df[(df['Month'] == selected_month) & (df['Year'] == 2024)]
            if month_df.empty:
                month_df = df[df['Month'] == selected_month]
            
            col1, col2, col3, col4 = st.columns(4)
            
            brown_volume = month_df[month_df['Category'] == 'Brown']['Charge Weight kg'].sum()
            green_volume = month_df[month_df['Category'] == 'Green']['Charge Weight kg'].sum()
            total_volume = brown_volume + green_volume
            utilization = (brown_volume / total_volume * 100) if total_volume > 0 else 0
            
            with col1:
                st.metric("Month Utilization %", f"{utilization:.1f}%")
            with col2:
                st.metric("Brown Volume", format_number(brown_volume))
            with col3:
                st.metric("Green Volume", format_number(green_volume))
            with col4:
                st.metric("Total Volume", format_number(total_volume))
            
            # Regional breakdown
            st.subheader("🌍 By Region Lane")
            
            if not regional_data.empty:
                # Prepare display
                regional_display = regional_data[['Region Lane', 'Brown_Volume_kg', 'Green_Volume_kg',
                                                  'Total_Volume_kg', 'Utilization_%', 'Savings',
                                                  'Brown_Cost_per_kg', 'Green_Cost_per_kg']].copy()
                
                # Format for display
                regional_display['Brown_Volume_kg'] = regional_display['Brown_Volume_kg'].apply(lambda x: format_number(x))
                regional_display['Green_Volume_kg'] = regional_display['Green_Volume_kg'].apply(lambda x: format_number(x))
                regional_display['Total_Volume_kg'] = regional_display['Total_Volume_kg'].apply(lambda x: format_number(x))
                regional_display['Utilization_%'] = regional_display['Utilization_%'].apply(lambda x: f"{x:.1f}%")
                regional_display['Savings'] = regional_display['Savings'].apply(format_currency)
                regional_display['Brown_Cost_per_kg'] = regional_display['Brown_Cost_per_kg'].apply(lambda x: f"${x:.2f}")
                regional_display['Green_Cost_per_kg'] = regional_display['Green_Cost_per_kg'].apply(lambda x: f"${x:.2f}")
                
                regional_display.columns = ['Region', 'Brown Volume (kg)', 'Green Volume (kg)',
                                           'Total Volume (kg)', 'Utilization %', 'Savings ($)',
                                           'Brown Cost/kg', 'Green Cost/kg']
                
                st.dataframe(regional_display, use_container_width=True, hide_index=True)
                
                # Regional utilization chart
                fig3 = px.bar(regional_data.head(10), 
                             x='Region Lane', 
                             y='Utilization_%',
                             title='Top 10 Regions by Utilization %',
                             labels={'Utilization_%': 'Utilization %', 'Region Lane': 'Region'},
                             color='Utilization_%',
                             color_continuous_scale='RdYlGn')
                
                fig3.update_layout(height=400)
                st.plotly_chart(fig3, use_container_width=True)
            else:
                st.info("No regional data available for the selected month")
            
            # Region Pair breakdown
            st.subheader("🔄 By Region Pair (Origin → Destination)")
            
            if not pair_data.empty:
                # Show top 15 pairs
                pair_display = pair_data.head(15)[['Region_Pair', 'Brown_Volume_kg', 'Green_Volume_kg',
                                                   'Total_Volume_kg', 'Utilization_%', 'Savings',
                                                   'Brown_Cost_per_kg', 'Green_Cost_per_kg']].copy()
                
                # Format for display
                pair_display['Brown_Volume_kg'] = pair_display['Brown_Volume_kg'].apply(lambda x: format_number(x))
                pair_display['Green_Volume_kg'] = pair_display['Green_Volume_kg'].apply(lambda x: format_number(x))
                pair_display['Total_Volume_kg'] = pair_display['Total_Volume_kg'].apply(lambda x: format_number(x))
                pair_display['Utilization_%'] = pair_display['Utilization_%'].apply(lambda x: f"{x:.1f}%")
                pair_display['Savings'] = pair_display['Savings'].apply(format_currency)
                pair_display['Brown_Cost_per_kg'] = pair_display['Brown_Cost_per_kg'].apply(lambda x: f"${x:.2f}")
                pair_display['Green_Cost_per_kg'] = pair_display['Green_Cost_per_kg'].apply(lambda x: f"${x:.2f}")
                
                pair_display.columns = ['Route', 'Brown Volume (kg)', 'Green Volume (kg)',
                                       'Total Volume (kg)', 'Utilization %', 'Savings ($)',
                                       'Brown Cost/kg', 'Green Cost/kg']
                
                st.dataframe(pair_display, use_container_width=True, hide_index=True)
                
                # Top routes visualization
                fig4 = px.bar(pair_data.head(10), 
                             x='Total_Volume_kg', 
                             y='Region_Pair',
                             orientation='h',
                             title='Top 10 Routes by Total Volume',
                             labels={'Total_Volume_kg': 'Total Volume (kg)', 'Region_Pair': 'Route'},
                             color='Utilization_%',
                             color_continuous_scale='RdYlGn')
                
                fig4.update_layout(height=400)
                st.plotly_chart(fig4, use_container_width=True)
            else:
                st.info("No region pair data available for the selected month")
        else:
            st.warning("No data available with valid months")

if __name__ == "__main__":
    main()
