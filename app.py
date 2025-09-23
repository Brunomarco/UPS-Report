import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(page_title="Green to Brown Utilization Stats", layout="wide", page_icon="✈️")

# Custom CSS to match the design
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    h1 {
        color: #333;
        font-size: 2.5rem;
        font-weight: 600;
    }
    .metric-container {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #666;
        margin-bottom: 0.3rem;
    }
    .brown-text {
        color: #8B4513;
    }
    .green-text {
        color: #228B22;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
        font-size: 1.1rem;
        font-weight: 500;
    }
    div[data-testid="metric-container"] {
        background-color: #f5f5f5;
        border: 1px solid #e0e0e0;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_and_process_data(uploaded_file):
    """Load and process the Excel file"""
    try:
        # Read Excel file
        df = pd.read_excel(uploaded_file)
        
        # Clean column names
        df.columns = df.columns.str.strip()
        
        # Parse dates from OriginDeparture Date (MM/DD/YYYY format)
        if 'OriginDeparture Date' in df.columns:
            df['Date'] = pd.to_datetime(df['OriginDeparture Date'], format='%m/%d/%Y', errors='coerce')
        else:
            st.error("OriginDeparture Date column not found")
            return None
        
        # Remove rows with invalid dates
        df = df[df['Date'].notna()]
        
        # Extract month and year
        df['Month'] = df['Date'].dt.month
        df['Year'] = df['Date'].dt.year
        df['Month_Name'] = df['Date'].dt.strftime('%B')
        
        # Process Weight
        if 'Volumetric Weight (KG)' in df.columns:
            df['Weight_KG'] = pd.to_numeric(df['Volumetric Weight (KG)'], errors='coerce').fillna(0)
        else:
            st.error("Volumetric Weight (KG) column not found")
            return None
        
        # Process Airline - identify UPS
        if 'Airline' in df.columns:
            df['Is_UPS'] = df['Airline'].astype(str).str.upper().str.contains('UPS', na=False)
        else:
            st.error("Airline column not found")
            return None
        
        # Generate synthetic cost data (since not in example)
        # You can replace this with actual cost column when available
        np.random.seed(42)
        df['Cost'] = df['Weight_KG'] * np.random.uniform(8, 12, len(df))
        df['Commercial_Cost'] = df['Cost'] * 1.15  # Assume 15% markup for commercial
        
        # Process regions
        if 'Region Lane' not in df.columns:
            df['Region Lane'] = 'Unknown'
        
        if 'Origin Region' not in df.columns:
            df['Origin Region'] = df['Region Lane'].str.split('-').str[0]
        
        if 'Destination Region' not in df.columns:
            df['Destination Region'] = df['Region Lane'].str.split('-').str[-1]
        
        return df
        
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return None

def calculate_metrics(df):
    """Calculate key metrics"""
    metrics = {}
    
    # Volume calculations
    metrics['brown_volume'] = df[df['Is_UPS']]['Weight_KG'].sum()
    metrics['green_volume'] = df[~df['Is_UPS']]['Weight_KG'].sum()
    metrics['total_volume'] = df['Weight_KG'].sum()
    
    # Utilization percentage
    if metrics['total_volume'] > 0:
        metrics['utilization'] = (metrics['brown_volume'] / metrics['total_volume']) * 100
    else:
        metrics['utilization'] = 0
    
    # Cost calculations
    ups_df = df[df['Is_UPS']]
    other_df = df[~df['Is_UPS']]
    
    metrics['brown_cost'] = ups_df['Cost'].sum()
    metrics['green_cost'] = other_df['Cost'].sum()
    
    # Cost per kg
    metrics['brown_cost_kg'] = metrics['brown_cost'] / metrics['brown_volume'] if metrics['brown_volume'] > 0 else 0
    metrics['green_cost_kg'] = metrics['green_cost'] / metrics['green_volume'] if metrics['green_volume'] > 0 else 0
    
    # Savings (difference between commercial and actual for UPS)
    metrics['savings'] = ups_df['Commercial_Cost'].sum() - ups_df['Cost'].sum()
    
    return metrics

def format_number(num, decimals=0):
    """Format numbers with commas"""
    if decimals == 0:
        return f"{num:,.0f}"
    else:
        return f"{num:,.{decimals}f}"

def create_utilization_chart(monthly_data):
    """Create utilization trend chart"""
    fig = go.Figure()
    
    # Add utilization line
    fig.add_trace(go.Scatter(
        x=monthly_data['Month_Name'],
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
        yaxis=dict(
            range=[0, 100],
            showgrid=True,
            gridcolor='#E0E0E0'
        ),
        height=400,
        hovermode='x',
        showlegend=False,
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(showgrid=False)
    )

    
    return fig

def main():
    # Title with styling
    col1, col2, col3 = st.columns([2, 3, 1])
    with col2:
        st.markdown("<h1 style='text-align: center;'>Green to Brown <span style='color: #008B8B;'>Overall Utilization Stats</span> <span style='color: #FFA500;'>YoY</span></h1>", unsafe_allow_html=True)
    
    # File uploader
    uploaded_file = st.file_uploader("Choose Excel file", type=['xlsx', 'xls'], label_visibility="collapsed")
    
    if uploaded_file is not None:
        # Load and process data
        with st.spinner('Processing data...'):
            df = load_and_process_data(uploaded_file)
        
        if df is None or len(df) == 0:
            st.error("No valid data to display")
            return
        
        # Get current year
        current_year = datetime.now().year
        if current_year not in df['Year'].unique():
            current_year = df['Year'].max()
        
        # Filter for current year
        df_current = df[df['Year'] == current_year].copy()
        
        # Create tabs
        tab1, tab2 = st.tabs(["📊 Year Overview", "📈 Monthly Analysis"])
        
        with tab1:
            st.markdown(f"### This Year To Date:")
            
            # Calculate monthly metrics for current year
            monthly_metrics = []
            months_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                           'July', 'August', 'September', 'October', 'November', 'December']
            
            for month in range(1, 13):
                month_df = df_current[df_current['Month'] == month]
                if len(month_df) > 0:
                    metrics = calculate_metrics(month_df)
                    monthly_metrics.append({
                        'Month': months_order[month-1],
                        'Brown Volume (kg)': metrics['brown_volume'],
                        'Green Volume (kg)': metrics['green_volume'],
                        'Utilization%': metrics['utilization'],
                        'Savings': metrics['savings'],
                        'Weight Impact': metrics['brown_volume'] + metrics['green_volume'],
                        'Brown Cost/kg': metrics['brown_cost_kg'],
                        'Green Cost/kg': metrics['green_cost_kg']
                    })
            
            if monthly_metrics:
                # Create DataFrame for display
                metrics_df = pd.DataFrame(monthly_metrics)
                
                # Calculate totals
                total_metrics = calculate_metrics(df_current)
                
                # Add total row
                total_row = {
                    'Month': 'Total',
                    'Brown Volume (kg)': total_metrics['brown_volume'],
                    'Green Volume (kg)': total_metrics['green_volume'],
                    'Utilization%': total_metrics['utilization'],
                    'Savings': total_metrics['savings'],
                    'Weight Impact': total_metrics['total_volume'],
                    'Brown Cost/kg': total_metrics['brown_cost_kg'],
                    'Green Cost/kg': total_metrics['green_cost_kg']
                }
                
                # Display metrics table
                display_df = metrics_df.copy()
                display_df['Brown Volume (kg)'] = display_df['Brown Volume (kg)'].apply(lambda x: format_number(x))
                display_df['Green Volume (kg)'] = display_df['Green Volume (kg)'].apply(lambda x: format_number(x))
                display_df['Utilization%'] = display_df['Utilization%'].apply(lambda x: f"{x:.1f}%")
                display_df['Savings'] = display_df['Savings'].apply(lambda x: f"${format_number(x)}")
                display_df['Weight Impact'] = display_df['Weight Impact'].apply(lambda x: f"(${format_number(x)})")
                display_df['Brown Cost/kg'] = display_df['Brown Cost/kg'].apply(lambda x: f"${x:.2f}")
                display_df['Green Cost/kg'] = display_df['Green Cost/kg'].apply(lambda x: f"${x:.2f}")
                
                # Add total row for display
                total_display = {
                    'Month': 'Total',
                    'Brown Volume (kg)': format_number(total_row['Brown Volume (kg)']),
                    'Green Volume (kg)': format_number(total_row['Green Volume (kg)']),
                    'Utilization%': f"{total_row['Utilization%']:.1f}%",
                    'Savings': f"${format_number(total_row['Savings'])}",
                    'Weight Impact': f"(${format_number(total_row['Weight Impact'])})",
                    'Brown Cost/kg': f"${total_row['Brown Cost/kg']:.2f}",
                    'Green Cost/kg': f"${total_row['Green Cost/kg']:.2f}"
                }
                
                display_df = pd.concat([display_df, pd.DataFrame([total_display])], ignore_index=True)
                
                # Style the dataframe
                st.dataframe(
                    display_df,
                    use_container_width=True,
                    hide_index=True,
                    height=500
                )
                
                # Utilization Chart
                st.markdown("### BT Utilization % by Month and Year")
                
                chart_data = pd.DataFrame(monthly_metrics)
                chart_data['Utilization_%'] = chart_data['Utilization%']
                
                fig = create_utilization_chart(chart_data)
                st.plotly_chart(fig, use_container_width=True)
                
                # YTD Comparisons section
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("### YTD Comparisons")
                    comparison_data = {
                        'Metric': ['Utilization %', 'Brown Cost/kg', 'Green Cost/kg'],
                        '2023': ['25.3%', '$1.28', '$2.63'],  # Placeholder data
                        '2024': [f"{total_metrics['utilization']:.1f}%", 
                                f"${total_metrics['brown_cost_kg']:.2f}",
                                f"${total_metrics['green_cost_kg']:.2f}"],
                        '2025': ['-', '-', '-']  # Future year
                    }
                    comp_df = pd.DataFrame(comparison_data)
                    st.dataframe(comp_df, use_container_width=True, hide_index=True)
        
        with tab2:
            st.markdown("### Green to Brown <span style='color: #008B8B;'>Monthly Utilization Stats</span>", unsafe_allow_html=True)
            
            # Month selector
            available_months = sorted(df_current['Month'].unique())
            month_names = [months_order[m-1] for m in available_months if m <= 12]
            
            selected_month_name = st.selectbox("Select Month", month_names)
            selected_month = months_order.index(selected_month_name) + 1
            
            # Filter for selected month
            df_month = df_current[df_current['Month'] == selected_month]
            
            if len(df_month) > 0:
                # Calculate overall metrics for the month
                month_metrics = calculate_metrics(df_month)
                
                # Display key metrics
                col1, col2, col3, col4, col5, col6 = st.columns(6)
                
                with col1:
                    st.metric("BT Utilization", f"{month_metrics['utilization']:.1f}%")
                
                with col2:
                    st.metric("% Effective", "125.7%")  # Placeholder
                
                with col3:
                    st.metric("This Year Volume", f"{month_metrics['total_volume']/1000000:.1f}M")
                
                with col4:
                    st.metric("Enterprise Synergy", f"${month_metrics['savings']/1000000:.1f}M")
                
                with col5:
                    st.metric("Actual Weight Impact", f"(${month_metrics['brown_volume']/1000:.1f}K)")
                
                with col6:
                    st.metric("YoY Savings", f"(${month_metrics['savings']/1000:.1f}K)")
                
                # Create two columns for tables
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Utilization by Region")
                    
                    # Group by Region Lane
                    region_metrics = []
                    for region in df_month['Region Lane'].unique():
                        region_df = df_month[df_month['Region Lane'] == region]
                        metrics = calculate_metrics(region_df)
                        
                        region_metrics.append({
                            'ORIG_REGION': region,
                            'Target %': '30%',  # Placeholder
                            'Actual Utilization %': f"{metrics['utilization']:.1f}%",
                            '% Effective': f"{min(metrics['utilization']/30*100, 200):.1f}%" if metrics['utilization'] > 0 else "0%"
                        })
                    
                    region_df_display = pd.DataFrame(region_metrics)
                    st.dataframe(region_df_display, use_container_width=True, hide_index=True, height=400)
                
                with col2:
                    st.markdown("#### Savings Impact")
                    
                    # Group by Region Lane for savings
                    savings_metrics = []
                    for region in df_month['Region Lane'].unique():
                        region_df = df_month[df_month['Region Lane'] == region]
                        metrics = calculate_metrics(region_df)
                        
                        savings_metrics.append({
                            'ORIG_REGION': region,
                            'LY BT Volume': format_number(metrics['brown_volume'] * 0.9),  # Placeholder
                            'BT Volume': format_number(metrics['brown_volume']),
                            'Actual Savings': f"${format_number(metrics['savings'])}",
                            'YoY Savings': f"${format_number(metrics['savings'] * 0.1)}"  # Placeholder
                        })
                    
                    savings_df_display = pd.DataFrame(savings_metrics)
                    st.dataframe(savings_df_display, use_container_width=True, hide_index=True, height=400)
                
                # Region Pair Analysis
                st.markdown("#### Analysis by Origin-Destination Pairs")
                
                # Create region pairs
                df_month['Region_Pair'] = df_month['Origin Region'] + '-' + df_month['Destination Region']
                
                pair_metrics = []
                for pair in df_month['Region_Pair'].unique()[:20]:  # Limit to top 20
                    pair_df = df_month[df_month['Region_Pair'] == pair]
                    metrics = calculate_metrics(pair_df)
                    
                    pair_metrics.append({
                        'Region Pair': pair,
                        'Target %': '30%',
                        'Actual Utilization %': f"{metrics['utilization']:.1f}%",
                        '% Effective': f"{min(metrics['utilization']/30*100, 200):.1f}%" if metrics['utilization'] > 0 else "0%",
                        'BT Volume': format_number(metrics['brown_volume']),
                        'Actual Savings': f"${format_number(metrics['savings'])}"
                    })
                
                if pair_metrics:
                    pair_df_display = pd.DataFrame(pair_metrics)
                    st.dataframe(pair_df_display, use_container_width=True, hide_index=True, height=400)
    
    else:
        st.info("👆 Please upload an Excel file to view the dashboard")
        
        with st.expander("📋 Expected Excel Format"):
            st.markdown("""
            ### Required Columns:
            - **OriginDeparture Date**: Date in MM/DD/YYYY format
            - **Airline**: Airline name (UPS or others)
            - **Volumetric Weight (KG)**: Weight of shipment
            - **Region Lane**: Region information (e.g., EMEAEMEA)
            - **Origin Region**: Origin region
            - **Destination Region**: Destination region
            
            ### Notes:
            - Brown volumes = UPS shipments
            - Green volumes = All other airlines
            - Each row represents one order
            """)

if __name__ == "__main__":
    main()
