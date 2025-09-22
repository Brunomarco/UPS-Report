import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import numpy as np
import time

# Page configuration
st.set_page_config(page_title="UPS Logistics Dashboard", layout="wide")

# Custom CSS for styling
st.markdown("""
    <style>
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .brown-metric { border-left: 5px solid #6F4E37; }
    .green-metric { border-left: 5px solid #228B22; }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: white;
        border-radius: 10px;
        padding-left: 20px;
        padding-right: 20px;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_and_process_data(file_bytes, filename):
    """Load and process the Excel file - optimized version"""
    
    # Use file bytes for consistent hashing
    df = pd.read_excel(file_bytes)
    
    # Set random seed based on filename for consistent random data
    np.random.seed(hash(filename) % (2**32))
    
    # Clean column names once
    df.columns = df.columns.str.strip()
    
    # Optimized date parsing - check specific columns first
    date_columns_priority = ['Tender Date', 'POB as text', 'OriginDeparture Date']
    date_found = False
    
    for col in date_columns_priority:
        if col in df.columns:
            try:
                df['Date'] = pd.to_datetime(df[col], errors='coerce', format='mixed')
                if df['Date'].notna().sum() > len(df) * 0.3:  # At least 30% valid dates
                    date_found = True
                    break
            except:
                pass
    
    # If no date found, create sample dates
    if not date_found:
        # Generate dates for the current year
        current_year = datetime.now().year
        start_date = pd.Timestamp(f'{current_year}-01-01')
        end_date = pd.Timestamp(f'{current_year}-12-31')
        df['Date'] = pd.date_range(start=start_date, end=end_date, periods=len(df))
    
    # Vectorized date operations
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year
    df['Month_Year'] = df['Date'].dt.to_period('M')
    
    # Optimized column detection and data generation
    # Weight column
    if 'Volumetric Weight (KG)' in df.columns:
        df['Weight_KG'] = pd.to_numeric(df['Volumetric Weight (KG)'], errors='coerce').fillna(0)
    elif 'Weight (KG)' in df.columns:
        df['Weight_KG'] = pd.to_numeric(df['Weight (KG)'], errors='coerce').fillna(0)
    elif 'S' in df.columns:
        df['Weight_KG'] = pd.to_numeric(df['S'], errors='coerce').fillna(0)
    else:
        # Generate deterministic sample data
        df['Weight_KG'] = np.random.uniform(10, 100, len(df))
    
    # Cost column
    if 'Cost' in df.columns:
        df['Cost'] = pd.to_numeric(df['Cost'], errors='coerce').fillna(0)
    elif 'T' in df.columns:
        df['Cost'] = pd.to_numeric(df['T'], errors='coerce').fillna(0)
    else:
        # Generate cost based on weight with some variation
        df['Cost'] = df['Weight_KG'] * np.random.uniform(5, 15, len(df))
    
    # Identify UPS shipments
    if 'Airline' in df.columns:
        df['Is_UPS'] = df['Airline'].str.upper().str.contains('UPS', na=False)
    else:
        # Create deterministic sample data
        df['Is_UPS'] = np.random.choice([True, False], size=len(df), p=[0.3, 0.7])
    
    # Add region columns if not present - vectorized operations
    if 'Region Lane' not in df.columns:
        regions = ['EMEA-EMEA', 'AMERICAS-AMERICAS', 'APAC-APAC', 'EMEA-AMERICAS', 'AMERICAS-APAC']
        df['Region Lane'] = np.random.choice(regions, size=len(df))
    
    if 'Origin Region' not in df.columns:
        df['Origin Region'] = df['Region Lane'].str.split('-', expand=True)[0]
    
    if 'Destination Region' not in df.columns:
        # Use negative index for last element
        df['Destination Region'] = df['Region Lane'].str.split('-').str[-1]
    
    # Calculate commercial cost (vectorized)
    df['Commercial_Cost'] = df['Cost'] * np.where(df['Is_UPS'], 1.3, 1.0)
    
    return df

@st.cache_data(ttl=600)  # Cache for 10 minutes
def calculate_all_metrics(df):
    """Pre-calculate all metrics for better performance"""
    
    # Current year data
    current_year = datetime.now().year
    df_current_year = df[df['Year'] == current_year].copy()
    
    # Overall metrics
    overall_metrics = calculate_metrics_vectorized(df_current_year)
    
    # Monthly metrics - vectorized calculation
    monthly_metrics = {}
    if not df_current_year.empty:
        grouped = df_current_year.groupby('Month')
        for month in grouped.groups.keys():
            month_df = grouped.get_group(month)
            monthly_metrics[month] = calculate_metrics_vectorized(month_df)
    
    # Regional metrics - pre-calculate for all months
    regional_metrics = {}
    for month in df_current_year['Month'].unique():
        month_df = df_current_year[df_current_year['Month'] == month]
        regional_metrics[month] = {}
        
        # By region lane
        for region in month_df['Region Lane'].unique():
            region_df = month_df[month_df['Region Lane'] == region]
            regional_metrics[month][region] = calculate_metrics_vectorized(region_df)
    
    return overall_metrics, monthly_metrics, regional_metrics, df_current_year

def calculate_metrics_vectorized(df):
    """Optimized metric calculation using vectorized operations"""
    if df.empty:
        return {
            'brown_volume': 0, 'green_volume': 0, 'total_volume': 0,
            'brown_cost': 0, 'green_cost': 0, 'savings': 0,
            'utilization': 0, 'brown_cost_per_kg': 0, 'green_cost_per_kg': 0
        }
    
    # Use boolean indexing for better performance
    is_ups_mask = df['Is_UPS'] == True
    
    brown_volume = df.loc[is_ups_mask, 'Weight_KG'].sum()
    green_volume = df.loc[~is_ups_mask, 'Weight_KG'].sum()
    total_volume = df['Weight_KG'].sum()
    
    brown_cost = df.loc[is_ups_mask, 'Cost'].sum()
    green_cost = df.loc[~is_ups_mask, 'Cost'].sum()
    
    # Savings calculation
    commercial_cost_ups = df.loc[is_ups_mask, 'Commercial_Cost'].sum()
    savings = commercial_cost_ups - brown_cost
    
    return {
        'brown_volume': brown_volume,
        'green_volume': green_volume,
        'total_volume': total_volume,
        'brown_cost': brown_cost,
        'green_cost': green_cost,
        'savings': savings,
        'utilization': (brown_volume / total_volume * 100) if total_volume > 0 else 0,
        'brown_cost_per_kg': brown_cost / brown_volume if brown_volume > 0 else 0,
        'green_cost_per_kg': green_cost / green_volume if green_volume > 0 else 0
    }

def display_metrics_row(metrics, col_titles):
    """Display a row of metrics - optimized"""
    cols = st.columns(len(col_titles))
    
    for i, (col, title) in enumerate(zip(cols, col_titles)):
        with col:
            # Determine styling class
            if 'Brown' in title:
                style_class = "metric-card brown-metric"
            elif 'Green' in title:
                style_class = "metric-card green-metric"
            else:
                style_class = "metric-card"
            
            st.markdown(f'<div class="{style_class}">', unsafe_allow_html=True)
            
            # Display appropriate metric
            if 'Volume' in title:
                value = metrics['brown_volume'] if 'Brown' in title else metrics['green_volume']
                st.metric(title, f"{value:,.0f} kg")
            elif 'Utilization' in title:
                st.metric(title, f"{metrics['utilization']:.1f}%")
            elif 'Savings' in title:
                st.metric(title, f"${metrics['savings']:,.0f}")
            elif 'Cost/kg' in title:
                value = metrics['brown_cost_per_kg'] if 'Brown' in title else metrics['green_cost_per_kg']
                st.metric(title, f"${value:.2f}")
            
            st.markdown('</div>', unsafe_allow_html=True)

def main():
    st.title("🚚 UPS Logistics Dashboard (Optimized)")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an Excel file", type=['xlsx', 'xls'])
    
    if uploaded_file is not None:
        # Show loading spinner
        with st.spinner('Processing data... Please wait.'):
            # Read file to bytes for consistent caching
            file_bytes = uploaded_file.read()
            uploaded_file.seek(0)  # Reset file pointer
            
            # Load and process data
            df = load_and_process_data(file_bytes, uploaded_file.name)
            
            # Pre-calculate all metrics
            overall_metrics, monthly_metrics, regional_metrics, df_current_year = calculate_all_metrics(df)
        
        # Get current year
        current_year = datetime.now().year
        
        # Create tabs
        tab1, tab2 = st.tabs(["📊 Year Overview", "📈 Monthly Analysis"])
        
        with tab1:
            st.header(f"Year {current_year} Overview")
            
            # Display overall metrics
            st.subheader("Overall Metrics")
            display_metrics_row(
                overall_metrics,
                ["🟫 Brown Volume (UPS)", "🟢 Green Volume (Others)", 
                 "📊 Utilization %", "💰 Total Savings", 
                 "📦 Brown Cost/kg", "📦 Green Cost/kg"]
            )
            
            # Monthly breakdown table - using pre-calculated data
            st.subheader("Monthly Breakdown")
            
            if monthly_metrics:
                monthly_data = []
                for month in sorted(monthly_metrics.keys()):
                    metrics = monthly_metrics[month]
                    monthly_data.append({
                        'Month': datetime(current_year, month, 1).strftime('%B'),
                        'Brown Volume (kg)': f"{metrics['brown_volume']:,.0f}",
                        'Green Volume (kg)': f"{metrics['green_volume']:,.0f}",
                        'Utilization %': f"{metrics['utilization']:.1f}%",
                        'Savings ($)': f"${metrics['savings']:,.0f}",
                        'Brown Cost/kg ($)': f"${metrics['brown_cost_per_kg']:.2f}",
                        'Green Cost/kg ($)': f"${metrics['green_cost_per_kg']:.2f}"
                    })
                
                monthly_df = pd.DataFrame(monthly_data)
                st.dataframe(monthly_df, use_container_width=True, hide_index=True)
                
                # Utilization Chart
                st.subheader("Utilization Trend")
                
                utilization_data = []
                for month in sorted(monthly_metrics.keys()):
                    utilization_data.append({
                        'Month': datetime(current_year, month, 1).strftime('%B'),
                        'Utilization %': monthly_metrics[month]['utilization']
                    })
                
                util_df = pd.DataFrame(utilization_data)
                
                fig = px.line(util_df, x='Month', y='Utilization %', 
                             markers=True, title='UPS Utilization % by Month',
                             line_shape='spline')
                fig.update_traces(line_color='#6F4E37', line_width=3, marker_size=10)
                fig.update_layout(
                    hovermode='x unified',
                    height=400,
                    xaxis_title="Month",
                    yaxis_title="Utilization %",
                    yaxis_range=[0, 100]
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.header("Monthly Analysis")
            
            # Month selector
            if monthly_metrics:
                available_months = sorted(monthly_metrics.keys())
                month_names = [datetime(2024, m, 1).strftime('%B') for m in available_months]
                
                selected_month_name = st.selectbox("Select Month", month_names)
                selected_month = available_months[month_names.index(selected_month_name)]
                
                # Get pre-calculated metrics for selected month
                month_metrics = monthly_metrics[selected_month]
                
                # Overall metrics for selected month
                st.subheader(f"{selected_month_name} Overview")
                display_metrics_row(
                    month_metrics,
                    ["🟫 Brown Volume (UPS)", "🟢 Green Volume (Others)", 
                     "📊 Utilization %", "💰 Total Savings", 
                     "📦 Brown Cost/kg", "📦 Green Cost/kg"]
                )
                
                # Get month data
                df_month = df_current_year[df_current_year['Month'] == selected_month].copy()
                
                if not df_month.empty:
                    # Analysis by Region Lane
                    st.subheader("Analysis by Region Lane")
                    
                    region_data = []
                    if selected_month in regional_metrics:
                        for region, metrics in regional_metrics[selected_month].items():
                            region_data.append({
                                'Region Lane': region,
                                'Brown Volume (kg)': f"{metrics['brown_volume']:,.0f}",
                                'Green Volume (kg)': f"{metrics['green_volume']:,.0f}",
                                'Utilization %': f"{metrics['utilization']:.1f}%",
                                'Savings ($)': f"${metrics['savings']:,.0f}",
                                'Brown Cost/kg ($)': f"${metrics['brown_cost_per_kg']:.2f}",
                                'Green Cost/kg ($)': f"${metrics['green_cost_per_kg']:.2f}"
                            })
                    
                    if region_data:
                        region_df_display = pd.DataFrame(region_data)
                        st.dataframe(region_df_display, use_container_width=True, hide_index=True)
                    
                    # Volume distribution charts
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Pie chart for volume distribution
                        fig_pie = go.Figure(data=[go.Pie(
                            labels=['UPS (Brown)', 'Others (Green)'],
                            values=[month_metrics['brown_volume'], month_metrics['green_volume']],
                            hole=.3,
                            marker_colors=['#6F4E37', '#228B22']
                        )])
                        fig_pie.update_layout(
                            title=f"{selected_month_name} Volume Distribution",
                            height=400
                        )
                        st.plotly_chart(fig_pie, use_container_width=True)
                    
                    with col2:
                        # Bar chart for regional volumes - optimized groupby
                        region_volumes = df_month.groupby(['Region Lane', 'Is_UPS'])['Weight_KG'].sum().reset_index()
                        region_volumes['Type'] = region_volumes['Is_UPS'].map({True: 'UPS', False: 'Others'})
                        
                        fig_bar = px.bar(region_volumes, x='Region Lane', y='Weight_KG', color='Type',
                                        title=f"{selected_month_name} Regional Volumes",
                                        color_discrete_map={'UPS': '#6F4E37', 'Others': '#228B22'},
                                        labels={'Weight_KG': 'Volume (kg)'})
                        fig_bar.update_layout(height=400)
                        st.plotly_chart(fig_bar, use_container_width=True)
            else:
                st.warning("No data available for analysis")
    
    else:
        st.info("Please upload an Excel file to get started")
        st.markdown("""
        ### Expected Excel Format:
        - **Date columns**: Tender Date, POB as text, or OriginDeparture Date
        - **Weight**: Column S or 'Volumetric Weight (KG)'
        - **Cost**: Column T or 'Cost'
        - **Airline**: Should include 'UPS' for UPS shipments
        - **Region columns**: Region Lane, Origin Region, Destination Region
        
        ### Performance Tips:
        - Large files (>10MB) may take a moment to process
        - The dashboard caches processed data for faster subsequent interactions
        - Clear cache using the menu (⋮) → Clear cache if you upload a new file
        """)

if __name__ == "__main__":
    main()
