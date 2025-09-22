import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Logistics Dashboard",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .stMetric {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    div[data-testid="metric-container"] {
        background-color: white;
        border: 1px solid #e0e0e0;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    .main-header {
        font-size: 32px;
        font-weight: bold;
        color: #1f4788;
        margin-bottom: 30px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data(uploaded_file):
    """Load and prepare the Excel data"""
    try:
        # Read all sheets
        excel_file = pd.ExcelFile(uploaded_file)
        
        # Combine all sheets
        all_data = []
        for sheet_name in excel_file.sheet_names:
            df = pd.read_excel(uploaded_file, sheet_name=sheet_name)
            df['Sheet'] = sheet_name
            all_data.append(df)
        
        df = pd.concat(all_data, ignore_index=True)
        
        # Identify key columns based on common patterns
        # Map columns based on what we found in the Excel
        column_mapping = {
            'weight_kg': 'WEIGHT(KG)',
            'airline': 'Airline',
            'depart_date': 'Depart Date',
            'origin_country': 'PU CTRY',
            'dest_country': 'DEL CTRY',
            'origin_city': 'SHIPPER CITY',
            'dest_city': 'DELIVERY CITY',
            'total_charges': 'TOTAL CHARGES',
            'amount': 'AMOUNT',
            'pieces': 'PIECES',
            'status': 'STATUS',
            'service': 'SVCDESC'
        }
        
        # Check which columns exist and map them
        for key, col_name in column_mapping.items():
            if col_name in df.columns:
                df[key] = df[col_name]
            else:
                # Try to find similar column names
                similar_cols = [c for c in df.columns if col_name.lower() in c.lower()]
                if similar_cols:
                    df[key] = df[similar_cols[0]]
                else:
                    df[key] = np.nan
        
        # Parse dates
        if 'depart_date' in df.columns:
            df['depart_date'] = pd.to_datetime(df['depart_date'], errors='coerce')
            df['month'] = df['depart_date'].dt.month
            df['year'] = df['depart_date'].dt.year
            df['month_year'] = df['depart_date'].dt.to_period('M')
        
        # Identify Brown (UPS) vs Green (Other Airlines)
        if 'airline' in df.columns:
            df['airline'] = df['airline'].fillna('Unknown').astype(str)
            df['category'] = df['airline'].apply(lambda x: 'Brown (UPS)' if 'UPS' in x.upper() else 'Green (Other)')
        else:
            df['category'] = 'Unknown'
        
        # Create region pairs
        if 'origin_country' in df.columns and 'dest_country' in df.columns:
            df['region_pair'] = df['origin_country'].astype(str) + ' → ' + df['dest_country'].astype(str)
        
        # Determine regions based on countries
        def get_region(country):
            if pd.isna(country):
                return 'Unknown'
            country = str(country).upper()
            americas = ['US', 'CA', 'MX', 'BR', 'AR', 'CL', 'CO', 'PE', 'VE']
            emea = ['GB', 'UK', 'FR', 'DE', 'IT', 'ES', 'NL', 'BE', 'CH', 'AT', 'SE', 'NO', 'DK', 'FI', 'PL', 'CZ', 'HU', 'RO', 'BG', 'GR', 'PT', 'IE', 'ZA', 'EG', 'SA', 'AE', 'IL']
            apac = ['CN', 'JP', 'KR', 'IN', 'AU', 'NZ', 'SG', 'MY', 'TH', 'ID', 'PH', 'VN', 'HK', 'TW']
            
            if country in americas:
                return 'AMERICAS'
            elif country in emea:
                return 'EMEA'
            elif country in apac:
                return 'APAC'
            else:
                return 'Other'
        
        df['origin_region'] = df['origin_country'].apply(get_region)
        df['dest_region'] = df['dest_country'].apply(get_region)
        df['region_lane'] = df['origin_region'] + df['dest_region']
        
        # Calculate costs
        numeric_columns = ['weight_kg', 'total_charges', 'amount', 'pieces']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Calculate cost per kg
        if 'total_charges' in df.columns and 'weight_kg' in df.columns:
            df['cost_per_kg'] = df.apply(lambda x: x['total_charges'] / x['weight_kg'] if x['weight_kg'] > 0 else 0, axis=1)
        elif 'amount' in df.columns and 'weight_kg' in df.columns:
            df['cost_per_kg'] = df.apply(lambda x: x['amount'] / x['weight_kg'] if x['weight_kg'] > 0 else 0, axis=1)
        else:
            df['cost_per_kg'] = 0
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

def calculate_metrics(df, groupby_col=None):
    """Calculate key metrics for the dashboard"""
    if groupby_col:
        grouped = df.groupby(groupby_col)
    else:
        grouped = df.groupby(lambda x: True)
    
    metrics = []
    
    for name, group in grouped:
        brown_data = group[group['category'] == 'Brown (UPS)']
        green_data = group[group['category'] == 'Green (Other)']
        
        total_volume = group['weight_kg'].sum()
        brown_volume = brown_data['weight_kg'].sum()
        green_volume = green_data['weight_kg'].sum()
        
        utilization_pct = (brown_volume / total_volume * 100) if total_volume > 0 else 0
        
        # Calculate costs
        brown_cost = brown_data['total_charges'].sum() if 'total_charges' in brown_data.columns else brown_data['amount'].sum() if 'amount' in brown_data.columns else 0
        green_cost = green_data['total_charges'].sum() if 'total_charges' in green_data.columns else green_data['amount'].sum() if 'amount' in green_data.columns else 0
        
        brown_cost_per_kg = (brown_cost / brown_volume) if brown_volume > 0 else 0
        green_cost_per_kg = (green_cost / green_volume) if green_volume > 0 else 0
        
        # Calculate savings (assuming green cost is commercial rate)
        savings = green_cost_per_kg * brown_volume - brown_cost if green_cost_per_kg > 0 else 0
        
        metric_dict = {
            'Brown Volume (kg)': round(brown_volume, 2),
            'Green Volume (kg)': round(green_volume, 2),
            'Total Volume (kg)': round(total_volume, 2),
            'Utilization %': round(utilization_pct, 2),
            'Savings ($)': round(savings, 2),
            'Brown Cost/kg': round(brown_cost_per_kg, 2),
            'Green Cost/kg': round(green_cost_per_kg, 2)
        }
        
        if groupby_col:
            metric_dict[groupby_col] = name
        
        metrics.append(metric_dict)
    
    return pd.DataFrame(metrics)

def create_utilization_chart(df):
    """Create utilization percentage chart"""
    monthly_metrics = calculate_metrics(df, 'month_year')
    monthly_metrics = monthly_metrics.sort_values('month_year')
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=monthly_metrics['month_year'].astype(str),
        y=monthly_metrics['Utilization %'],
        mode='lines+markers',
        name='Utilization %',
        line=dict(color='#8B4513', width=3),
        marker=dict(size=10)
    ))
    
    fig.update_layout(
        title='UPS Utilization % by Month',
        xaxis_title='Month',
        yaxis_title='Utilization %',
        height=400,
        yaxis=dict(range=[0, 100]),
        hovermode='x unified'
    )
    
    return fig

def create_volume_chart(df):
    """Create volume comparison chart"""
    monthly_metrics = calculate_metrics(df, 'month_year')
    monthly_metrics = monthly_metrics.sort_values('month_year')
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=monthly_metrics['month_year'].astype(str),
        y=monthly_metrics['Brown Volume (kg)'],
        name='Brown (UPS)',
        marker_color='#8B4513'
    ))
    
    fig.add_trace(go.Bar(
        x=monthly_metrics['month_year'].astype(str),
        y=monthly_metrics['Green Volume (kg)'],
        name='Green (Other)',
        marker_color='#90EE90'
    ))
    
    fig.update_layout(
        title='Monthly Volume Comparison',
        xaxis_title='Month',
        yaxis_title='Volume (kg)',
        barmode='group',
        height=400,
        hovermode='x unified'
    )
    
    return fig

def main():
    st.markdown('<div class="main-header">📦 Logistics Performance Dashboard</div>', unsafe_allow_html=True)
    
    # File upload
    uploaded_file = st.file_uploader("Choose Excel file", type=['xlsx', 'xls'])
    
    if uploaded_file is not None:
        # Load data
        df = load_data(uploaded_file)
        
        if df.empty:
            st.error("No data could be loaded from the file.")
            return
        
        # Create tabs
        tab1, tab2 = st.tabs(["📊 Year Overview", "🔍 Monthly Analysis"])
        
        with tab1:
            st.header("Year-to-Date Performance")
            
            # Filter for current year
            current_year = datetime.now().year
            if 'year' in df.columns:
                year_data = df[df['year'] == current_year]
            else:
                year_data = df
            
            # Calculate overall metrics
            overall_metrics = calculate_metrics(year_data)
            
            if not overall_metrics.empty:
                # Display key metrics in columns
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    st.metric("Brown Volume (kg)", f"{overall_metrics.iloc[0]['Brown Volume (kg)']:,.0f}")
                
                with col2:
                    st.metric("Green Volume (kg)", f"{overall_metrics.iloc[0]['Green Volume (kg)']:,.0f}")
                
                with col3:
                    st.metric("Utilization %", f"{overall_metrics.iloc[0]['Utilization %']:.1f}%")
                
                with col4:
                    st.metric("Savings ($)", f"${overall_metrics.iloc[0]['Savings ($)']:,.0f}")
                
                with col5:
                    st.metric("Brown Cost/kg", f"${overall_metrics.iloc[0]['Brown Cost/kg']:.2f}")
            
            # Monthly metrics table
            st.subheader("Monthly Breakdown")
            monthly_metrics = calculate_metrics(year_data, 'month_year')
            
            if not monthly_metrics.empty:
                monthly_metrics = monthly_metrics.sort_values('month_year')
                
                # Format the dataframe for display
                display_df = monthly_metrics.copy()
                display_df['month_year'] = display_df['month_year'].astype(str)
                
                # Style the dataframe
                st.dataframe(
                    display_df.style.format({
                        'Brown Volume (kg)': '{:,.0f}',
                        'Green Volume (kg)': '{:,.0f}',
                        'Total Volume (kg)': '{:,.0f}',
                        'Utilization %': '{:.1f}%',
                        'Savings ($)': '${:,.0f}',
                        'Brown Cost/kg': '${:.2f}',
                        'Green Cost/kg': '${:.2f}'
                    }),
                    use_container_width=True
                )
                
                # Charts
                col1, col2 = st.columns(2)
                
                with col1:
                    st.plotly_chart(create_utilization_chart(year_data), use_container_width=True)
                
                with col2:
                    st.plotly_chart(create_volume_chart(year_data), use_container_width=True)
        
        with tab2:
            st.header("Monthly Detailed Analysis")
            
            # Month selector
            if 'month_year' in df.columns:
                available_months = sorted(df['month_year'].dropna().unique())
                selected_month = st.selectbox(
                    "Select Month",
                    options=available_months,
                    format_func=lambda x: str(x)
                )
                
                # Filter data for selected month
                month_data = df[df['month_year'] == selected_month]
                
                # Display month overview
                st.subheader(f"Overview for {selected_month}")
                month_metrics = calculate_metrics(month_data)
                
                if not month_metrics.empty:
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Volume", f"{month_metrics.iloc[0]['Total Volume (kg)']:,.0f} kg")
                    
                    with col2:
                        st.metric("UPS Utilization", f"{month_metrics.iloc[0]['Utilization %']:.1f}%")
                    
                    with col3:
                        st.metric("Total Savings", f"${month_metrics.iloc[0]['Savings ($)']:,.0f}")
                    
                    with col4:
                        avg_cost = (month_metrics.iloc[0]['Brown Cost/kg'] + month_metrics.iloc[0]['Green Cost/kg']) / 2
                        st.metric("Avg Cost/kg", f"${avg_cost:.2f}")
                
                # Analysis by Region
                st.subheader("Analysis by Region")
                
                if 'region_lane' in month_data.columns:
                    region_metrics = calculate_metrics(month_data, 'region_lane')
                    
                    if not region_metrics.empty:
                        st.dataframe(
                            region_metrics.style.format({
                                'Brown Volume (kg)': '{:,.0f}',
                                'Green Volume (kg)': '{:,.0f}',
                                'Total Volume (kg)': '{:,.0f}',
                                'Utilization %': '{:.1f}%',
                                'Savings ($)': '${:,.0f}',
                                'Brown Cost/kg': '${:.2f}',
                                'Green Cost/kg': '${:.2f}'
                            }),
                            use_container_width=True
                        )
                        
                        # Create region visualization
                        fig_region = px.bar(
                            region_metrics,
                            x='region_lane',
                            y=['Brown Volume (kg)', 'Green Volume (kg)'],
                            title='Volume by Region Lane',
                            labels={'value': 'Volume (kg)', 'variable': 'Category'},
                            color_discrete_map={'Brown Volume (kg)': '#8B4513', 'Green Volume (kg)': '#90EE90'}
                        )
                        st.plotly_chart(fig_region, use_container_width=True)
                
                # Analysis by Region Pairs
                st.subheader("Analysis by Origin-Destination Pairs")
                
                if 'origin_region' in month_data.columns and 'dest_region' in month_data.columns:
                    month_data['region_pair_full'] = month_data['origin_region'] + ' → ' + month_data['dest_region']
                    pair_metrics = calculate_metrics(month_data, 'region_pair_full')
                    
                    if not pair_metrics.empty:
                        # Sort by total volume
                        pair_metrics = pair_metrics.sort_values('Total Volume (kg)', ascending=False)
                        
                        st.dataframe(
                            pair_metrics.head(10).style.format({
                                'Brown Volume (kg)': '{:,.0f}',
                                'Green Volume (kg)': '{:,.0f}',
                                'Total Volume (kg)': '{:,.0f}',
                                'Utilization %': '{:.1f}%',
                                'Savings ($)': '${:,.0f}',
                                'Brown Cost/kg': '${:.2f}',
                                'Green Cost/kg': '${:.2f}'
                            }),
                            use_container_width=True
                        )
                        
                        # Create Sankey diagram for top flows
                        top_pairs = pair_metrics.head(10)
                        
                        if len(top_pairs) > 0:
                            sources = []
                            targets = []
                            values = []
                            
                            for _, row in top_pairs.iterrows():
                                if 'region_pair_full' in row:
                                    parts = row['region_pair_full'].split(' → ')
                                    if len(parts) == 2:
                                        sources.append(parts[0])
                                        targets.append(parts[1])
                                        values.append(row['Total Volume (kg)'])
                            
                            if sources:
                                # Create unique labels
                                all_labels = list(set(sources + targets))
                                label_dict = {label: i for i, label in enumerate(all_labels)}
                                
                                fig_sankey = go.Figure(data=[go.Sankey(
                                    node=dict(
                                        pad=15,
                                        thickness=20,
                                        line=dict(color="black", width=0.5),
                                        label=all_labels,
                                        color=["#1f4788" if "AMERICAS" in label else "#90EE90" if "EMEA" in label else "#FFA500" for label in all_labels]
                                    ),
                                    link=dict(
                                        source=[label_dict[src] for src in sources],
                                        target=[label_dict[tgt] for tgt in targets],
                                        value=values,
                                        color="rgba(100, 100, 100, 0.3)"
                                    )
                                )])
                                
                                fig_sankey.update_layout(
                                    title="Top 10 Origin-Destination Flows",
                                    height=500
                                )
                                
                                st.plotly_chart(fig_sankey, use_container_width=True)
            else:
                st.info("No data available for monthly analysis")
    
    else:
        st.info("Please upload an Excel file to begin")
        
        # Show sample structure
        st.subheader("Expected Data Structure")
        st.markdown("""
        The Excel file should contain the following information:
        - **Airline**: Carrier information (UPS or other airlines)
        - **Weight/Volume**: Shipment weight in kg
        - **Dates**: Departure dates for temporal analysis
        - **Locations**: Origin and destination information
        - **Costs**: Charges or amounts for cost calculations
        """)

if __name__ == "__main__":
    main()
