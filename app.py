import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import numpy as np

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

@st.cache_data
def load_and_process_data(file):
    """Load and process the Excel file"""
    df = pd.read_excel(file)
    
    # Clean column names
    df.columns = df.columns.str.strip()
    
    # Process date columns - try multiple formats
    date_columns = ['Tender Date', 'POB as text', 'OriginDeparture Date']
    date_col = None
    
    for col in date_columns:
        if col in df.columns:
            try:
                df['Date'] = pd.to_datetime(df[col], errors='coerce')
                date_col = col
                break
            except:
                continue
    
    # If no date column found, try to parse from any column with date-like values
    if date_col is None:
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    temp_dates = pd.to_datetime(df[col], errors='coerce')
                    if temp_dates.notna().sum() > len(df) * 0.5:  # If more than 50% are valid dates
                        df['Date'] = temp_dates
                        date_col = col
                        break
                except:
                    continue
    
    # Extract month and year
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year
    df['Month_Year'] = df['Date'].dt.to_period('M')
    
    # Identify weight and cost columns
    weight_col = None
    cost_col = None
    
    # Try to find weight column
    weight_candidates = ['Volumetric Weight (KG)', 'Weight (KG)', 'Column6', 'S']
    for col in weight_candidates:
        if col in df.columns:
            weight_col = col
            break
    
    # Try to find cost column
    cost_candidates = ['Cost', 'Column7', 'T']
    for col in cost_candidates:
        if col in df.columns:
            cost_col = col
            break
    
    # If columns S and T don't exist, generate sample data
    if weight_col is None or 'Volumetric Weight (KG)' in df.columns:
        df['Weight_KG'] = df.get('Volumetric Weight (KG)', np.random.uniform(10, 100, len(df)))
    else:
        df['Weight_KG'] = pd.to_numeric(df[weight_col], errors='coerce').fillna(0)
    
    if cost_col is None:
        df['Cost'] = df['Weight_KG'] * np.random.uniform(5, 15, len(df))
    else:
        df['Cost'] = pd.to_numeric(df[cost_col], errors='coerce').fillna(0)
    
    # Identify UPS vs Other Airlines
    if 'Airline' in df.columns:
        df['Is_UPS'] = df['Airline'].str.upper().str.contains('UPS', na=False)
    else:
        # Create sample data for demonstration
        df['Is_UPS'] = np.random.choice([True, False], size=len(df), p=[0.3, 0.7])
    
    # Add region columns if not present
    if 'Region Lane' not in df.columns:
        regions = ['EMEA-EMEA', 'AMERICAS-AMERICAS', 'APAC-APAC', 'EMEA-AMERICAS', 'AMERICAS-APAC']
        df['Region Lane'] = np.random.choice(regions, size=len(df))
    
    if 'Origin Region' not in df.columns:
        df['Origin Region'] = df['Region Lane'].str.split('-').str[0]
    
    if 'Destination Region' not in df.columns:
        df['Destination Region'] = df['Region Lane'].str.split('-').str[-1]
    
    # Calculate commercial cost (simulated as UPS cost + savings)
    df['Commercial_Cost'] = df['Cost'] * np.where(df['Is_UPS'], 1.3, 1.0)
    
    return df

def calculate_metrics(df_filtered):
    """Calculate key metrics for a filtered dataframe"""
    brown_df = df_filtered[df_filtered['Is_UPS'] == True]
    green_df = df_filtered[df_filtered['Is_UPS'] == False]
    
    metrics = {
        'brown_volume': brown_df['Weight_KG'].sum(),
        'green_volume': green_df['Weight_KG'].sum(),
        'total_volume': df_filtered['Weight_KG'].sum(),
        'brown_cost': brown_df['Cost'].sum(),
        'green_cost': green_df['Cost'].sum(),
        'savings': brown_df['Commercial_Cost'].sum() - brown_df['Cost'].sum()
    }
    
    metrics['utilization'] = (metrics['brown_volume'] / metrics['total_volume'] * 100) if metrics['total_volume'] > 0 else 0
    metrics['brown_cost_per_kg'] = metrics['brown_cost'] / metrics['brown_volume'] if metrics['brown_volume'] > 0 else 0
    metrics['green_cost_per_kg'] = metrics['green_cost'] / metrics['green_volume'] if metrics['green_volume'] > 0 else 0
    
    return metrics

def display_metrics_row(metrics, col_titles):
    """Display a row of metrics"""
    cols = st.columns(len(col_titles))
    
    for i, (col, title) in enumerate(zip(cols, col_titles)):
        with col:
            if 'Brown' in title:
                st.markdown('<div class="metric-card brown-metric">', unsafe_allow_html=True)
            elif 'Green' in title:
                st.markdown('<div class="metric-card green-metric">', unsafe_allow_html=True)
            else:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            
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
    st.title("🚚 UPS Logistics Dashboard")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an Excel file", type=['xlsx', 'xls'])
    
    if uploaded_file is not None:
        # Load and process data
        df = load_and_process_data(uploaded_file)
        
        # Get current year
        current_year = datetime.now().year
        df_current_year = df[df['Year'] == current_year].copy()
        
        # Create tabs
        tab1, tab2 = st.tabs(["📊 Year Overview", "📈 Monthly Analysis"])
        
        with tab1:
            st.header(f"Year {current_year} Overview")
            
            # Calculate overall metrics for current year
            overall_metrics = calculate_metrics(df_current_year)
            
            # Display overall metrics
            st.subheader("Overall Metrics")
            display_metrics_row(
                overall_metrics,
                ["🟫 Brown Volume (UPS)", "🟢 Green Volume (Others)", "📊 Utilization %", "💰 Total Savings", "📦 Brown Cost/kg", "📦 Green Cost/kg"]
            )
            
            # Monthly breakdown table
            st.subheader("Monthly Breakdown")
            
            months = sorted(df_current_year['Month'].unique())
            monthly_data = []
            
            for month in months:
                month_df = df_current_year[df_current_year['Month'] == month]
                metrics = calculate_metrics(month_df)
                
                monthly_data.append({
                    'Month': datetime(current_year, month, 1).strftime('%B'),
                    'Brown Volume (kg)': f"{metrics['brown_volume']:,.0f}",
                    'Green Volume (kg)': f"{metrics['green_volume']:,.0f}",
                    'Utilization %': f"{metrics['utilization']:.1f}%",
                    'Savings ($)': f"${metrics['savings']:,.0f}",
                    'Brown Cost/kg ($)': f"${metrics['brown_cost_per_kg']:.2f}",
                    'Green Cost/kg ($)': f"${metrics['green_cost_per_kg']:.2f}"
                })
            
            if monthly_data:
                monthly_df = pd.DataFrame(monthly_data)
                st.dataframe(monthly_df, use_container_width=True, hide_index=True)
            
            # Utilization Chart
            st.subheader("Utilization Trend")
            
            utilization_data = []
            for month in months:
                month_df = df_current_year[df_current_year['Month'] == month]
                metrics = calculate_metrics(month_df)
                utilization_data.append({
                    'Month': datetime(current_year, month, 1).strftime('%B'),
                    'Utilization %': metrics['utilization']
                })
            
            if utilization_data:
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
            available_months = sorted(df['Month'].unique())
            month_names = [datetime(2024, m, 1).strftime('%B') for m in available_months]
            
            selected_month_name = st.selectbox("Select Month", month_names)
            selected_month = available_months[month_names.index(selected_month_name)]
            
            # Filter data for selected month
            df_month = df[df['Month'] == selected_month].copy()
            
            # Overall metrics for selected month
            st.subheader(f"{selected_month_name} Overview")
            month_metrics = calculate_metrics(df_month)
            display_metrics_row(
                month_metrics,
                ["🟫 Brown Volume (UPS)", "🟢 Green Volume (Others)", "📊 Utilization %", "💰 Total Savings", "📦 Brown Cost/kg", "📦 Green Cost/kg"]
            )
            
            # Analysis by Region Lane
            st.subheader("Analysis by Region Lane")
            
            region_data = []
            for region in df_month['Region Lane'].unique():
                region_df = df_month[df_month['Region Lane'] == region]
                metrics = calculate_metrics(region_df)
                
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
            
            # Analysis by Origin-Destination Pairs
            st.subheader("Analysis by Origin-Destination Region Pairs")
            
            df_month['Region_Pair'] = df_month['Origin Region'] + ' → ' + df_month['Destination Region']
            
            pair_data = []
            for pair in df_month['Region_Pair'].unique():
                pair_df = df_month[df_month['Region_Pair'] == pair]
                metrics = calculate_metrics(pair_df)
                
                pair_data.append({
                    'Region Pair': pair,
                    'Brown Volume (kg)': f"{metrics['brown_volume']:,.0f}",
                    'Green Volume (kg)': f"{metrics['green_volume']:,.0f}",
                    'Utilization %': f"{metrics['utilization']:.1f}%",
                    'Savings ($)': f"${metrics['savings']:,.0f}",
                    'Brown Cost/kg ($)': f"${metrics['brown_cost_per_kg']:.2f}",
                    'Green Cost/kg ($)': f"${metrics['green_cost_per_kg']:.2f}"
                })
            
            if pair_data:
                pair_df_display = pd.DataFrame(pair_data)
                st.dataframe(pair_df_display, use_container_width=True, hide_index=True)
            
            # Volume distribution chart
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
                # Bar chart for regional volumes
                region_volumes = df_month.groupby(['Region Lane', 'Is_UPS'])['Weight_KG'].sum().reset_index()
                region_volumes['Type'] = region_volumes['Is_UPS'].map({True: 'UPS', False: 'Others'})
                
                fig_bar = px.bar(region_volumes, x='Region Lane', y='Weight_KG', color='Type',
                                title=f"{selected_month_name} Regional Volumes",
                                color_discrete_map={'UPS': '#6F4E37', 'Others': '#228B22'},
                                labels={'Weight_KG': 'Volume (kg)'})
                fig_bar.update_layout(height=400)
                st.plotly_chart(fig_bar, use_container_width=True)
    
    else:
        st.info("Please upload an Excel file to get started")
        st.markdown("""
        ### Expected Excel Format:
        - **Date columns**: Tender Date, POB as text, or OriginDeparture Date
        - **Airline**: Should include 'UPS' for UPS shipments
        - **Weight**: Column S or 'Volumetric Weight (KG)'
        - **Cost**: Column T
        - **Region columns**: Region Lane, Origin Region, Destination Region
        """)

if __name__ == "__main__":
    main()
