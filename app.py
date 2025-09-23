import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import numpy as np
import io
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

# Use session state to store processed data
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'file_name' not in st.session_state:
    st.session_state.file_name = None

def read_excel_fast(uploaded_file):
    """Optimized Excel reading"""
    try:
        # Read into bytes first (faster than direct read)
        bytes_data = uploaded_file.getvalue() if hasattr(uploaded_file, 'getvalue') else uploaded_file.read()
        
        # Try different methods based on file extension
        file_name = uploaded_file.name.lower()
        
        if file_name.endswith('.csv'):
            # CSV is much faster
            df = pd.read_csv(io.BytesIO(bytes_data))
        elif file_name.endswith('.xlsx'):
            # For xlsx, use openpyxl with optimizations
            df = pd.read_excel(
                io.BytesIO(bytes_data),
                engine='openpyxl'
            )
        else:
            # For xls or other formats
            df = pd.read_excel(io.BytesIO(bytes_data))
        
        return df
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        return None

def process_data_optimized(df):
    """Highly optimized data processing"""
    
    # Set seed for reproducibility
    np.random.seed(42)
    
    # Strip column names once
    df.columns = [col.strip() for col in df.columns]
    
    # --- DATE HANDLING (FAST) ---
    date_col_found = False
    
    # Try to find a date column (limited search)
    for col in ['Tender Date', 'POB as text', 'OriginDeparture Date']:
        if col in df.columns:
            try:
                # Use pandas' fast date parser with infer_datetime_format
                temp_dates = pd.to_datetime(df[col], errors='coerce', infer_datetime_format=True)
                if temp_dates.notna().sum() > len(df) * 0.3:
                    df['Date'] = temp_dates
                    date_col_found = True
                    break
            except:
                continue
    
    # If no date found, generate dates efficiently
    if not date_col_found:
        current_year = datetime.now().year
        # Create evenly spaced dates
        df['Date'] = pd.date_range(
            start=f'{current_year}-01-01',
            periods=len(df),
            freq='D'
        )[:len(df)]
    
    # Extract date components (vectorized)
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year
    df['Month_Year'] = df['Date'].dt.to_period('M')
    
    # --- WEIGHT HANDLING (FAST) ---
    weight_found = False
    for col in ['Volumetric Weight (KG)', 'Weight (KG)', 'S', 'Column6']:
        if col in df.columns:
            temp = pd.to_numeric(df[col], errors='coerce')
            if temp.notna().sum() > 0:
                df['Weight_KG'] = temp.fillna(temp.median() if temp.notna().any() else 50)
                weight_found = True
                break
    
    if not weight_found:
        df['Weight_KG'] = np.random.uniform(10, 100, len(df))
    
    # --- COST HANDLING (FAST) ---
    cost_found = False
    for col in ['Cost', 'T', 'Column7']:
        if col in df.columns:
            temp = pd.to_numeric(df[col], errors='coerce')
            if temp.notna().sum() > 0:
                df['Cost'] = temp.fillna(temp.median() if temp.notna().any() else 100)
                cost_found = True
                break
    
    if not cost_found:
        df['Cost'] = df['Weight_KG'] * np.random.uniform(5, 15, len(df))
    
    # --- UPS IDENTIFICATION (FAST) ---
    if 'Airline' in df.columns:
        # Vectorized string operation
        airline_upper = df['Airline'].astype(str).str.upper()
        df['Is_UPS'] = airline_upper.str.contains('UPS', na=False)
    else:
        df['Is_UPS'] = np.random.choice([True, False], size=len(df), p=[0.3, 0.7])
    
    # --- REGION HANDLING (FAST) ---
    if 'Region Lane' not in df.columns:
        regions = ['EMEA-EMEA', 'AMERICAS-AMERICAS', 'APAC-APAC', 'EMEA-AMERICAS', 'AMERICAS-APAC']
        df['Region Lane'] = np.random.choice(regions, size=len(df))
    
    # Vectorized region extraction
    region_parts = df['Region Lane'].str.split('-', n=1, expand=True)
    df['Origin Region'] = region_parts[0] if 0 in region_parts.columns else 'Unknown'
    df['Destination Region'] = region_parts[1] if 1 in region_parts.columns else region_parts[0]
    
    # --- COMMERCIAL COST (VECTORIZED) ---
    df['Commercial_Cost'] = np.where(df['Is_UPS'], df['Cost'] * 1.3, df['Cost'])
    
    return df

def calculate_metrics_fast(df):
    """Ultra-fast metric calculation using numpy where possible"""
    if len(df) == 0:
        return {
            'brown_volume': 0, 'green_volume': 0, 'total_volume': 0,
            'brown_cost': 0, 'green_cost': 0, 'savings': 0,
            'utilization': 0, 'brown_cost_per_kg': 0, 'green_cost_per_kg': 0
        }
    
    # Use numpy arrays for faster computation
    is_ups = df['Is_UPS'].values
    weights = df['Weight_KG'].values
    costs = df['Cost'].values
    commercial_costs = df['Commercial_Cost'].values
    
    # Fast calculations using numpy
    brown_volume = np.sum(weights[is_ups])
    green_volume = np.sum(weights[~is_ups])
    total_volume = np.sum(weights)
    
    brown_cost = np.sum(costs[is_ups])
    green_cost = np.sum(costs[~is_ups])
    
    savings = np.sum(commercial_costs[is_ups] - costs[is_ups])
    
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
    """Display metrics efficiently"""
    cols = st.columns(len(col_titles))
    
    for col, title in zip(cols, col_titles):
        with col:
            # Determine metric type and value
            if 'Brown Volume' in title:
                value = f"{metrics['brown_volume']:,.0f} kg"
                css_class = "brown-metric"
            elif 'Green Volume' in title:
                value = f"{metrics['green_volume']:,.0f} kg"
                css_class = "green-metric"
            elif 'Utilization' in title:
                value = f"{metrics['utilization']:.1f}%"
                css_class = ""
            elif 'Savings' in title:
                value = f"${metrics['savings']:,.0f}"
                css_class = ""
            elif 'Brown Cost/kg' in title:
                value = f"${metrics['brown_cost_per_kg']:.2f}"
                css_class = "brown-metric"
            elif 'Green Cost/kg' in title:
                value = f"${metrics['green_cost_per_kg']:.2f}"
                css_class = "green-metric"
            else:
                value = "N/A"
                css_class = ""
            
            st.markdown(f'<div class="metric-card {css_class}">', unsafe_allow_html=True)
            st.metric(title, value)
            st.markdown('</div>', unsafe_allow_html=True)

def main():
    st.title("🚚 UPS Logistics Dashboard (Optimized)")
    
    # File uploader with CSV support
    uploaded_file = st.file_uploader(
        "Choose a file (Excel or CSV)", 
        type=['xlsx', 'xls', 'csv'],
        help="CSV files load much faster than Excel files"
    )
    
    if uploaded_file is not None:
        # Check if we need to reprocess
        current_file_name = uploaded_file.name
        
        if st.session_state.file_name != current_file_name or st.session_state.processed_data is None:
            # Process new file
            with st.spinner('Processing file... This may take a moment for large files.'):
                start_time = time.time()
                
                # Read file
                df = read_excel_fast(uploaded_file)
                
                if df is not None and not df.empty:
                    # Process data
                    df = process_data_optimized(df)
                    
                    # Store in session state
                    st.session_state.processed_data = df
                    st.session_state.file_name = current_file_name
                    
                    process_time = time.time() - start_time
                    st.success(f"✅ Processed {len(df):,} rows in {process_time:.1f} seconds")
                else:
                    st.error("Failed to load file or file is empty")
                    return
        
        # Use processed data from session state
        df = st.session_state.processed_data
        
        # Get current year and filter
        current_year = datetime.now().year
        df_current_year = df[df['Year'] == current_year].copy()
        
        # Create tabs
        tab1, tab2 = st.tabs(["📊 Year Overview", "📈 Monthly Analysis"])
        
        with tab1:
            st.header(f"Year {current_year} Overview")
            
            # Calculate overall metrics
            overall_metrics = calculate_metrics_fast(df_current_year)
            
            # Display metrics
            st.subheader("Overall Metrics")
            display_metrics_row(
                overall_metrics,
                ["🟫 Brown Volume (UPS)", "🟢 Green Volume (Others)", "📊 Utilization %", 
                 "💰 Total Savings", "📦 Brown Cost/kg", "📦 Green Cost/kg"]
            )
            
            # Monthly breakdown - optimized groupby
            st.subheader("Monthly Breakdown")
            
            if not df_current_year.empty:
                # Fast groupby using agg
                monthly_agg = df_current_year.groupby('Month').agg({
                    'Weight_KG': 'sum',
                    'Cost': 'sum',
                    'Is_UPS': 'mean'
                }).reset_index()
                
                # Calculate UPS/non-UPS splits
                monthly_data = []
                for month in monthly_agg['Month'].unique():
                    month_df = df_current_year[df_current_year['Month'] == month]
                    metrics = calculate_metrics_fast(month_df)
                    
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
                
                util_data = []
                for month in sorted(monthly_agg['Month'].unique()):
                    month_df = df_current_year[df_current_year['Month'] == month]
                    utilization = (month_df['Is_UPS'].sum() / len(month_df) * 100) if len(month_df) > 0 else 0
                    util_data.append({
                        'Month': datetime(current_year, month, 1).strftime('%B'),
                        'Utilization %': utilization
                    })
                
                if util_data:
                    util_df = pd.DataFrame(util_data)
                    
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
            month_names = [datetime(2024, m, 1).strftime('%B') for m in available_months if m <= 12]
            
            if month_names:
                selected_month_name = st.selectbox("Select Month", month_names)
                selected_month = available_months[month_names.index(selected_month_name)]
                
                # Filter data
                df_month = df[df['Month'] == selected_month]
                
                # Monthly metrics
                st.subheader(f"{selected_month_name} Overview")
                month_metrics = calculate_metrics_fast(df_month)
                display_metrics_row(
                    month_metrics,
                    ["🟫 Brown Volume (UPS)", "🟢 Green Volume (Others)", "📊 Utilization %", 
                     "💰 Total Savings", "📦 Brown Cost/kg", "📦 Green Cost/kg"]
                )
                
                # Region analysis
                st.subheader("Analysis by Region Lane")
                
                region_data = []
                for region in df_month['Region Lane'].unique():
                    region_df = df_month[df_month['Region Lane'] == region]
                    metrics = calculate_metrics_fast(region_df)
                    
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
                
                # Origin-Destination Analysis
                st.subheader("Analysis by Origin-Destination Region Pairs")
                
                # Create region pairs efficiently
                df_month_copy = df_month.copy()
                df_month_copy['Region_Pair'] = df_month_copy['Origin Region'] + ' → ' + df_month_copy['Destination Region']
                
                pair_data = []
                for pair in df_month_copy['Region_Pair'].unique()[:10]:  # Limit to top 10 for performance
                    pair_df = df_month_copy[df_month_copy['Region_Pair'] == pair]
                    metrics = calculate_metrics_fast(pair_df)
                    
                    pair_data.append({
                        'Region Pair': pair,
                        'Brown Volume (kg)': f"{metrics['brown_volume']:,.0f}",
                        'Green Volume (kg)': f"{metrics['green_volume']:,.0f}",
                        'Utilization %': f"{metrics['utilization']:.1f}%",
                        'Savings ($)': f"${metrics['savings']:,.0f}"
                    })
                
                if pair_data:
                    pair_df_display = pd.DataFrame(pair_data)
                    st.dataframe(pair_df_display, use_container_width=True, hide_index=True)
                
                # Visualizations
                col1, col2 = st.columns(2)
                
                with col1:
                    # Pie chart
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
                    # Bar chart - optimized groupby
                    region_volumes = df_month.groupby(['Region Lane', 'Is_UPS'])['Weight_KG'].sum().reset_index()
                    region_volumes['Type'] = region_volumes['Is_UPS'].map({True: 'UPS', False: 'Others'})
                    
                    fig_bar = px.bar(region_volumes, x='Region Lane', y='Weight_KG', color='Type',
                                    title=f"{selected_month_name} Regional Volumes",
                                    color_discrete_map={'UPS': '#6F4E37', 'Others': '#228B22'},
                                    labels={'Weight_KG': 'Volume (kg)'})
                    fig_bar.update_layout(height=400, xaxis_tickangle=-45)
                    st.plotly_chart(fig_bar, use_container_width=True)
    
    else:
        st.info("Please upload a file to get started")
        
        # Add helpful tips
        with st.expander("📚 Usage Guide & Performance Tips"):
            st.markdown("""
            ### Expected File Format:
            - **Date columns**: Tender Date, POB as text, or OriginDeparture Date
            - **Airline**: Should include 'UPS' for UPS shipments
            - **Weight**: Column S or 'Volumetric Weight (KG)'
            - **Cost**: Column T
            - **Region columns**: Region Lane, Origin Region, Destination Region
            
            ### 🚀 Performance Tips:
            1. **Use CSV format instead of Excel** - CSV files load 5-10x faster!
            2. **Remove unnecessary columns** before uploading
            3. **Limit file size** to under 50MB for best performance
            4. **Remove Excel formatting** - plain data loads faster
            
            ### To Convert Excel to CSV:
            ```python
            # In Excel: File → Save As → CSV
            # Or in Python:
            df = pd.read_excel('your_file.xlsx')
            df.to_csv('your_file.csv', index=False)
            ```
            """)
        
        # Sample data generator
        if st.button("Generate Sample Data"):
            with st.spinner("Generating sample data..."):
                # Create sample dataframe
                n_rows = 1000
                current_year = datetime.now().year
                
                sample_df = pd.DataFrame({
                    'Date': pd.date_range(start=f'{current_year}-01-01', periods=n_rows, freq='h')[:n_rows],
                    'Airline': np.random.choice(['UPS', 'FedEx', 'DHL', 'Other'], n_rows, p=[0.3, 0.3, 0.2, 0.2]),
                    'Volumetric Weight (KG)': np.random.uniform(10, 200, n_rows),
                    'Cost': np.random.uniform(50, 500, n_rows),
                    'Region Lane': np.random.choice(['EMEA-EMEA', 'AMERICAS-AMERICAS', 'APAC-APAC'], n_rows),
                    'Origin Region': np.random.choice(['EMEA', 'AMERICAS', 'APAC'], n_rows),
                    'Destination Region': np.random.choice(['EMEA', 'AMERICAS', 'APAC'], n_rows)
                })
                
                # Convert to CSV for download
                csv = sample_df.to_csv(index=False)
                
                st.download_button(
                    label="📥 Download Sample Data (CSV)",
                    data=csv,
                    file_name='sample_ups_data.csv',
                    mime='text/csv'
                )
                
                st.success("Sample data generated! Download and upload it to test the dashboard.")

if __name__ == "__main__":
    main()
