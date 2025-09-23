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
    """Highly optimized data processing with proper date handling and filtering"""
    
    # Set seed for reproducibility
    np.random.seed(42)
    
    # Strip column names once
    df.columns = [col.strip() for col in df.columns]
    
    # --- CRITICAL: Remove rows with empty POB as text or Airline ---
    # First, check if these columns exist
    if 'POB as text' in df.columns:
        # Remove rows where POB as text is empty, NaN, or just whitespace
        df = df[df['POB as text'].notna()]
        df = df[df['POB as text'].astype(str).str.strip() != '']
    
    if 'Airline' in df.columns:
        # Remove rows where Airline is empty, NaN, or just whitespace
        df = df[df['Airline'].notna()]
        df = df[df['Airline'].astype(str).str.strip() != '']
    
    # If dataframe is empty after filtering, return early
    if len(df) == 0:
        st.warning("No valid data found after filtering empty rows")
        return df
    
    # --- DATE HANDLING (FIXED FOR MM/DD/YYYY format) ---
    date_col_found = False
    
    # Priority columns for date detection
    date_columns_priority = ['POB as text', 'Tender Date', 'OriginDeparture Date']
    
    for col in date_columns_priority:
        if col in df.columns:
            try:
                # Try parsing with MM/DD/YYYY format first
                temp_dates = pd.to_datetime(df[col], format='%m/%d/%Y', errors='coerce')
                
                # If that didn't work well, try mixed format
                if temp_dates.isna().sum() > len(df) * 0.7:
                    temp_dates = pd.to_datetime(df[col], format='mixed', errors='coerce', dayfirst=False)
                
                # If we have at least 30% valid dates, use this column
                if temp_dates.notna().sum() > len(df) * 0.3:
                    df['Date'] = temp_dates
                    date_col_found = True
                    break
            except Exception as e:
                st.warning(f"Could not parse dates from {col}: {str(e)}")
                continue
    
    # If no date found, generate dates for 2024
    if not date_col_found:
        st.info("No valid date column found, generating sample dates for 2024")
        # Generate dates for 2024
        df['Date'] = pd.date_range(
            start='2024-01-01',
            periods=len(df),
            freq='D'
        )[:len(df)]
    
    # Remove rows with invalid dates
    df = df[df['Date'].notna()]
    
    if len(df) == 0:
        st.error("No valid dates found in the data")
        return df
    
    # Extract date components (vectorized)
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year
    df['Month_Year'] = df['Date'].dt.to_period('M')
    
    # Validate months (1-12 only)
    df = df[(df['Month'] >= 1) & (df['Month'] <= 12)]
    
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
    
    # --- UPS IDENTIFICATION (FIXED - Only "UPS Airlines") ---
    if 'Airline' in df.columns:
        # Clean the airline column and check for exact match "UPS Airlines"
        df['Airline_Clean'] = df['Airline'].astype(str).str.strip()
        # Mark as UPS only if it's exactly "UPS Airlines"
        df['Is_UPS'] = df['Airline_Clean'] == 'UPS Airlines'
    else:
        # If no Airline column, create sample data
        df['Is_UPS'] = np.random.choice([True, False], size=len(df), p=[0.3, 0.7])
    
    # --- REGION HANDLING (FAST) ---
    if 'Region Lane' not in df.columns:
        regions = ['EMEA-EMEA', 'AMERICAS-AMERICAS', 'APAC-APAC', 'EMEA-AMERICAS', 'AMERICAS-APAC']
        df['Region Lane'] = np.random.choice(regions, size=len(df))
    
    # Handle missing or invalid Region Lanes
    df['Region Lane'] = df['Region Lane'].fillna('Unknown-Unknown')
    
    # Vectorized region extraction with error handling
    try:
        region_parts = df['Region Lane'].str.split('-', n=1, expand=True)
        df['Origin Region'] = region_parts[0] if 0 in region_parts.columns else 'Unknown'
        if 1 in region_parts.columns:
            df['Destination Region'] = region_parts[1]
        else:
            df['Destination Region'] = df['Origin Region']
    except:
        df['Origin Region'] = 'Unknown'
        df['Destination Region'] = 'Unknown'
    
    # --- COMMERCIAL COST (VECTORIZED) ---
    df['Commercial_Cost'] = np.where(df['Is_UPS'], df['Cost'] * 1.3, df['Cost'])
    
    # Final validation - remove any remaining invalid rows
    df = df[df['Weight_KG'] > 0]
    df = df[df['Cost'] > 0]
    
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
    st.title("🚚 UPS Logistics Dashboard")
    
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
                    # Show initial row count
                    initial_rows = len(df)
                    
                    # Process data
                    df = process_data_optimized(df)
                    
                    # Check if we have data after processing
                    if len(df) == 0:
                        st.error("No valid data remaining after processing. Please check your file format.")
                        st.stop()
                    
                    # Store in session state
                    st.session_state.processed_data = df
                    st.session_state.file_name = current_file_name
                    
                    process_time = time.time() - start_time
                    st.success(f"✅ Processed {len(df):,} valid rows (from {initial_rows:,} total) in {process_time:.1f} seconds")
                    
                    # Show data quality info
                    if 'Airline' in df.columns:
                        ups_count = df['Is_UPS'].sum()
                        st.info(f"Found {ups_count:,} UPS Airlines shipments and {len(df) - ups_count:,} other shipments")
                else:
                    st.error("Failed to load file or file is empty")
                    st.stop()
        
        # Use processed data from session state
        df = st.session_state.processed_data
        
        # Check if we have data
        if df is None or len(df) == 0:
            st.error("No data to display")
            st.stop()
        
        # Get years available in data
        available_years = sorted(df['Year'].unique())
        
        # Use 2024 as default if available, otherwise use the most recent year
        if 2024 in available_years:
            current_year = 2024
        elif available_years:
            current_year = max(available_years)
        else:
            st.error("No valid year data found")
            st.stop()
        
        df_current_year = df[df['Year'] == current_year].copy()
        
        if len(df_current_year) == 0:
            st.warning(f"No data found for year {current_year}")
            st.stop()
        
        # Create tabs
        tab1, tab2, tab3 = st.tabs(["📊 Year Overview", "📈 Monthly Analysis", "📋 Data Quality"])
        
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
            
            # Get unique months and validate them
            unique_months = sorted(df_current_year['Month'].unique())
            valid_months = [m for m in unique_months if 1 <= m <= 12]
            
            if valid_months:
                monthly_data = []
                for month in valid_months:
                    month_df = df_current_year[df_current_year['Month'] == month]
                    if len(month_df) > 0:
                        metrics = calculate_metrics_fast(month_df)
                        
                        try:
                            month_name = datetime(current_year, int(month), 1).strftime('%B')
                        except:
                            month_name = f"Month {month}"
                        
                        monthly_data.append({
                            'Month': month_name,
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
                    for month in valid_months:
                        month_df = df_current_year[df_current_year['Month'] == month]
                        if len(month_df) > 0:
                            utilization = (month_df['Is_UPS'].sum() / len(month_df) * 100)
                            try:
                                month_name = datetime(current_year, int(month), 1).strftime('%B')
                            except:
                                month_name = f"Month {month}"
                            
                            util_data.append({
                                'Month': month_name,
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
            else:
                st.warning("No valid month data found")
        
        with tab2:
            st.header("Monthly Analysis")
            
            # Month selector - only show valid months
            available_months = sorted(df['Month'].unique())
            valid_months = [m for m in available_months if 1 <= m <= 12]
            
            if valid_months:
                month_names = []
                for m in valid_months:
                    try:
                        month_names.append(datetime(2024, int(m), 1).strftime('%B'))
                    except:
                        month_names.append(f"Month {m}")
                
                selected_month_name = st.selectbox("Select Month", month_names)
                selected_month = valid_months[month_names.index(selected_month_name)]
                
                # Filter data
                df_month = df[df['Month'] == selected_month]
                
                if len(df_month) > 0:
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
                    unique_regions = df_month['Region Lane'].unique()
                    for region in unique_regions[:20]:  # Limit to 20 regions for performance
                        region_df = df_month[df_month['Region Lane'] == region]
                        if len(region_df) > 0:
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
                    
                    # Visualizations
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Pie chart
                        fig_pie = go.Figure(data=[go.Pie(
                            labels=['UPS Airlines', 'Other Airlines'],
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
                        region_volumes['Type'] = region_volumes['Is_UPS'].map({True: 'UPS Airlines', False: 'Others'})
                        
                        # Limit to top 10 regions by volume for clarity
                        top_regions = region_volumes.groupby('Region Lane')['Weight_KG'].sum().nlargest(10).index
                        region_volumes_filtered = region_volumes[region_volumes['Region Lane'].isin(top_regions)]
                        
                        fig_bar = px.bar(region_volumes_filtered, x='Region Lane', y='Weight_KG', color='Type',
                                        title=f"{selected_month_name} Top 10 Regional Volumes",
                                        color_discrete_map={'UPS Airlines': '#6F4E37', 'Others': '#228B22'},
                                        labels={'Weight_KG': 'Volume (kg)'})
                        fig_bar.update_layout(height=400, xaxis_tickangle=-45)
                        st.plotly_chart(fig_bar, use_container_width=True)
                else:
                    st.warning(f"No data found for {selected_month_name}")
            else:
                st.warning("No valid month data available")
        
        with tab3:
            st.header("Data Quality Check")
            
            # Show data statistics
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Data Summary")
                st.write(f"**Total Rows Processed:** {len(df):,}")
                st.write(f"**Date Range:** {df['Date'].min().strftime('%m/%d/%Y')} to {df['Date'].max().strftime('%m/%d/%Y')}")
                st.write(f"**Years in Data:** {', '.join(map(str, sorted(df['Year'].unique())))}")
                st.write(f"**Months in Data:** {len(df['Month'].unique())}")
                
                if 'Airline' in df.columns:
                    st.write(f"**Unique Airlines:** {df['Airline_Clean'].nunique()}")
                    st.write(f"**UPS Airlines Shipments:** {df['Is_UPS'].sum():,}")
                    st.write(f"**Other Airlines Shipments:** {(~df['Is_UPS']).sum():,}")
            
            with col2:
                st.subheader("Data Validation")
                
                # Check for potential issues
                issues = []
                
                # Check for dates outside expected range
                if df['Year'].min() < 2020 or df['Year'].max() > 2025:
                    issues.append("⚠️ Some dates appear to be outside 2020-2025 range")
                
                # Check for zero weights or costs
                zero_weights = (df['Weight_KG'] == 0).sum()
                if zero_weights > 0:
                    issues.append(f"⚠️ {zero_weights} rows have zero weight")
                
                zero_costs = (df['Cost'] == 0).sum()
                if zero_costs > 0:
                    issues.append(f"⚠️ {zero_costs} rows have zero cost")
                
                # Check for missing regions
                unknown_regions = (df['Region Lane'] == 'Unknown-Unknown').sum()
                if unknown_regions > 0:
                    issues.append(f"⚠️ {unknown_regions} rows have unknown regions")
                
                if issues:
                    for issue in issues:
                        st.warning(issue)
                else:
                    st.success("✅ No data quality issues detected")
            
            # Show sample of processed data
            st.subheader("Sample of Processed Data (First 100 rows)")
            
            display_cols = ['Date', 'Weight_KG', 'Cost', 'Is_UPS', 'Region Lane']
            if 'Airline_Clean' in df.columns:
                display_cols.insert(1, 'Airline_Clean')
            
            available_cols = [col for col in display_cols if col in df.columns]
            
            st.dataframe(
                df[available_cols].head(100),
                use_container_width=True,
                hide_index=True
            )
            
            # Download button for processed data
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download Processed Data as CSV",
                data=csv,
                file_name='ups_processed_data.csv',
                mime='text/csv',
            )
    
    else:
        st.info("Please upload a file to get started")
        
        # Add helpful tips
        with st.expander("📚 Usage Guide & Data Requirements"):
            st.markdown("""
            ### Required Data Format:
            
            #### Critical Columns:
            - **POB as text**: Date column in MM/DD/YYYY format (e.g., 1/11/2024)
            - **Airline**: Must contain "UPS Airlines" exactly for UPS shipments
            - **Rows with empty POB as text or Airline will be excluded**
            
            #### Optional Columns:
            - **Volumetric Weight (KG)** or **Column S**: Weight data
            - **Cost** or **Column T**: Cost data
            - **Region Lane**: Regional information
            
            ### Important Notes:
            - ⚠️ Only shipments with **"UPS Airlines"** (exact match) will be counted as UPS
            - ⚠️ Rows with empty Airline or POB as text fields will be automatically removed
            - ⚠️ Dates should be in MM/DD/YYYY format
            
            ### 🚀 Performance Tips:
            1. **Use CSV format** instead of Excel for faster loading
            2. **Remove unnecessary columns** before uploading
            3. **Ensure date format is MM/DD/YYYY**
            """)
        
        # Sample data generator
        if st.button("Generate Sample Data"):
            with st.spinner("Generating sample data..."):
                # Create sample dataframe with proper format
                n_rows = 1000
                
                # Generate dates for 2024 in MM/DD/YYYY format
                dates = pd.date_range(start='2024-01-01', end='2024-12-31', periods=n_rows)
                
                sample_df = pd.DataFrame({
                    'POB as text': dates.strftime('%m/%d/%Y'),  # MM/DD/YYYY format
                    'Airline': np.random.choice(['UPS Airlines', 'FedEx', 'DHL', 'Other Airlines'], n_rows, p=[0.3, 0.3, 0.2, 0.2]),
                    'Volumetric Weight (KG)': np.random.uniform(10, 200, n_rows).round(2),
                    'Cost': np.random.uniform(50, 500, n_rows).round(2),
                    'Region Lane': np.random.choice(['EMEA-EMEA', 'AMERICAS-AMERICAS', 'APAC-APAC', 'EMEA-AMERICAS'], n_rows),
                })
                
                # Convert to CSV for download
                csv = sample_df.to_csv(index=False)
                
                st.download_button(
                    label="📥 Download Sample Data (CSV)",
                    data=csv,
                    file_name='sample_ups_data.csv',
                    mime='text/csv'
                )
                
                st.success("Sample data generated with correct date format (MM/DD/YYYY)!")

if __name__ == "__main__":
    main()
