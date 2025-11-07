 #--------------------------
# Imports & Page Config
# --------------------------
import streamlit as st
st.set_page_config(page_title="Traffic Violations Dashboard", layout="wide", page_icon="🚦")

import pandas as pd
import folium
from streamlit_folium import st_folium
import plotly.express as px
import os
import re
import html
import requests
from folium.plugins import MarkerCluster, HeatMap
from streamlit_autorefresh import st_autorefresh
from typing import Optional, List, Dict, Any

# --------------------------
# Backend Configuration
# --------------------------
BACKEND_URL = "http://localhost:8000"

# --------------------------
# Helper Functions
# --------------------------
def fetch_violations_from_backend() -> Optional[List[Dict[str, Any]]]:
    """Fetch all violations from backend API"""
    try:
        response = requests.get(f"{BACKEND_URL}/violations", timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"❌ Backend error: {response.status_code}")
            return None
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to backend. Make sure it's running on http://localhost:8000")
        st.info("💡 Start backend with: `python backend_server.py`")
        return None
    except Exception as e:
        st.error(f"❌ Error fetching violations: {e}")
        return None


def violations_to_dataframe(violations: List[Dict[str, Any]]) -> pd.DataFrame:
    """Convert violations JSON to pandas DataFrame"""
    if not violations:
        # Return empty DataFrame with expected columns
        return pd.DataFrame(columns=[
            'id', 'plate', 'type', 'time', 'status', 'location', 
            'lat', 'lon', 'image', 'confidence', 'track_id', 
            'violation_type', 'timestamp_utc', 'media', 'details'
        ])
    
    df = pd.DataFrame(violations)
    
    # Ensure critical columns exist with safe defaults
    if 'plate' not in df.columns:
        df['plate'] = df.get('details', {}).apply(lambda x: x.get('plate', 'N/A') if isinstance(x, dict) else 'N/A')
    if 'type' not in df.columns:
        df['type'] = df.get('violation_type', 'Unknown')
    if 'location' not in df.columns:
        df['location'] = 'Unknown'
    if 'lat' not in df.columns:
        df['lat'] = None
    if 'lon' not in df.columns:
        df['lon'] = None
    if 'image' not in df.columns:
        # Try to extract image from media field
        if 'media' in df.columns:
            df['image'] = df['media'].apply(
                lambda x: x.get('context_img') or x.get('crop_img') if isinstance(x, dict) else None
            )
        else:
            df['image'] = None
    if 'time' not in df.columns:
        df['time'] = df.get('timestamp_utc', 'N/A')
    if 'status' not in df.columns:
        df['status'] = 'Pending'
    
    # Convert types
    df['plate'] = df['plate'].astype(str)
    df['type'] = df['type'].astype(str)
    df['location'] = df['location'].astype(str)
    df['status'] = df['status'].astype(str)
    
    return df


def update_violation_status(violation_id: str, new_status: str) -> bool:
    """Update violation status on backend"""
    try:
        response = requests.patch(
            f"{BACKEND_URL}/violations/{violation_id}",
            json={"status": new_status},
            timeout=5
        )
        if response.status_code == 200:
            return True
        else:
            st.error(f"❌ Failed to update: {response.text}")
            return False
    except Exception as e:
        st.error(f"❌ Error updating status: {e}")
        return False


def highlight_substring(text, query):
    """Highlight search query in text"""
    s = "" if text is None else str(text)
    if not query:
        return html.escape(s)
    pattern = re.compile(re.escape(str(query)), re.IGNORECASE)
    return pattern.sub(lambda m: f"<span class='hl'>{html.escape(m.group(0))}</span>", html.escape(s))


def get_selectbox_index(options, current_value):
    """Return the index of current_value in options, default to 0 if not found."""
    try:
        return options.index(current_value)
    except ValueError:
        return 0


# --------------------------
# Load Data from Backend
# --------------------------
@st.cache_data(ttl=5)  # Cache for 5 seconds to reduce API calls
def load_violations_cached():
    """Cached wrapper for fetching violations"""
    return fetch_violations_from_backend()


# Check if we should force refresh
if "force_refresh" not in st.session_state:
    st.session_state.force_refresh = False

if st.session_state.force_refresh:
    st.cache_data.clear()
    st.session_state.force_refresh = False

violations_data = load_violations_cached()

if violations_data is None:
    st.stop()

df = violations_to_dataframe(violations_data)

# --------------------------
# Refresh controls
# --------------------------
st.sidebar.header("⚙️ Dashboard Settings")
auto_refresh = st.sidebar.checkbox("Auto-refresh", value=False, help="Enable periodic refresh")
refresh_secs = st.sidebar.slider("Refresh interval (seconds)", 5, 120, 15)
if auto_refresh:
    st_autorefresh(interval=refresh_secs * 1000, key="dashboard_refresh")

if st.sidebar.button("🔄 Manual Refresh"):
    st.cache_data.clear()
    st.session_state.force_refresh = True
    st.rerun()

# Backend connection status
with st.sidebar.expander("🔌 Backend Status", expanded=False):
    try:
        health_response = requests.get(f"{BACKEND_URL}/", timeout=2)
        if health_response.status_code == 200:
            health_data = health_response.json()
            st.success("✅ Connected")
            st.caption(f"Violations: {health_data.get('violations_count', 0)}")
        else:
            st.error("❌ Backend error")
    except:
        st.error("❌ Backend offline")

# --------------------------
# CSS Styling
# --------------------------
st.markdown("""
<style>
header, footer {visibility: hidden;}
.main .block-container {padding-top: 2rem;}
.navbar { 
    background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
    padding: 20px; border-radius: 12px; color: white; 
    font-size: 32px; font-weight: 700; text-align: center; margin-bottom: 25px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}
.kpi-card { background: white; padding: 20px; border-radius: 12px; text-align: center; box-shadow: 0 4px 12px rgba(0,0,0,0.08); margin-bottom: 20px; border-left: 5px solid #2E8B57; transition: transform 0.2s ease; }
.kpi-card:hover { transform: translateY(-3px); box-shadow: 0 6px 16px rgba(0,0,0,0.12); }
.kpi-title { font-size: 14px; color: #666; font-weight: 600; text-transform: uppercase; margin-bottom: 8px; }
.kpi-value { font-size: 32px; font-weight: 800; color: #1e3c72; }
.approved { background: linear-gradient(135deg, #28B463 0%, #58D68D 100%) !important; color: white !important; padding: 8px 12px; border-radius: 8px; margin-bottom: 6px; font-weight: 600; box-shadow: 0 2px 4px rgba(40, 180, 99, 0.3); }
.rejected { background: linear-gradient(135deg, #C0392B 0%, #E74C3C 100%) !important; color: white !important; padding: 8px 12px; border-radius: 8px; margin-bottom: 6px; font-weight: 600; box-shadow: 0 2px 4px rgba(192, 57, 43, 0.3); }
.pending { background: linear-gradient(135deg, #1E90FF 0%, #6495ED 100%) !important; color: white !important; padding: 8px 12px; border-radius: 8px; margin-bottom: 6px; font-weight: 600; box-shadow: 0 2px 4px rgba(30, 144, 255, 0.3); }
.section-header { color: #1e3c72; font-size: 24px; font-weight: 700; margin-top: 25px; margin-bottom: 15px; padding-bottom: 8px; border-bottom: 3px solid #2E8B57; }
.hl { background-color: #FFD700; color: #000 !important; padding: 1px 4px; border-radius: 4px; font-weight: 700; }
.chart-container { background: white; padding: 20px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); margin-bottom: 20px; border: 1px solid #eaeaea; }
.stDownloadButton button { background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%) !important; color: white !important; border: none !important; border-radius: 8px !important; padding: 10px 20px !important; font-weight: 600 !important; }
</style>
""", unsafe_allow_html=True)

# --------------------------
# Navbar
# --------------------------
st.markdown('<div class="navbar">🚦 Traffic Violations Dashboard - Hyderabad</div>', unsafe_allow_html=True)

# --------------------------
# Sidebar Filters
# --------------------------
st.sidebar.header("🔍 Filters")

# Define options
type_options = ["All"] + sorted(df["type"].dropna().unique().tolist())
location_options = ["All"] + sorted(df["location"].dropna().unique().tolist())
status_options = ["All"] + sorted(df["status"].dropna().unique().tolist())

# Initialize session state if missing
if 'filter_type' not in st.session_state:
    st.session_state['filter_type'] = "All"
if 'filter_location' not in st.session_state:
    st.session_state['filter_location'] = "All"
if 'filter_status' not in st.session_state:
    st.session_state['filter_status'] = "All"
if 'search_plate' not in st.session_state:
    st.session_state['search_plate'] = ""

# Reset Filters Button
if st.sidebar.button("♻️ Reset Filters"):
    st.session_state['filter_type'] = "All"
    st.session_state['filter_location'] = "All"
    st.session_state['filter_status'] = "All"
    st.session_state['search_plate'] = ""
    st.rerun()

# Selectboxes / text input
filter_type = st.sidebar.selectbox(
    "Violation Type",
    type_options,
    index=get_selectbox_index(type_options, st.session_state['filter_type'])
)
st.session_state['filter_type'] = filter_type

filter_location = st.sidebar.selectbox(
    "Location",
    location_options,
    index=get_selectbox_index(location_options, st.session_state['filter_location'])
)
st.session_state['filter_location'] = filter_location

filter_status = st.sidebar.selectbox(
    "Status",
    status_options,
    index=get_selectbox_index(status_options, st.session_state['filter_status'])
)
st.session_state['filter_status'] = filter_status

search_plate = st.sidebar.text_input(
    "🔎 Search vehicle plate",
    value=st.session_state['search_plate'],
    placeholder="e.g., TS09AB1234"
)
st.session_state['search_plate'] = search_plate

# --------------------------
# Apply Filters to DataFrame
# --------------------------
filtered_df = df.copy()

if st.session_state['filter_type'] != "All":
    filtered_df = filtered_df[filtered_df["type"] == st.session_state['filter_type']]
if st.session_state['filter_location'] != "All":
    filtered_df = filtered_df[filtered_df["location"] == st.session_state['filter_location']]
if st.session_state['filter_status'] != "All":
    filtered_df = filtered_df[filtered_df["status"] == st.session_state['filter_status']]
if st.session_state['search_plate']:
    filtered_df = filtered_df[
        filtered_df['plate'].astype(str).str.contains(st.session_state['search_plate'].strip(), case=False, na=False)
    ]

# --------------------------
# KPI Cards
# --------------------------
total = len(filtered_df)
approved = len(filtered_df[filtered_df["status"]=="Approved"])
rejected = len(filtered_df[filtered_df["status"]=="Rejected"])
pending = total - approved - rejected

col1, col2, col3, col4 = st.columns(4)
col1.markdown(f'<div class="kpi-card"><div class="kpi-title">Total Violations</div><div class="kpi-value">{total}</div></div>', unsafe_allow_html=True)
col2.markdown(f'<div class="kpi-card"><div class="kpi-title">Approved</div><div class="kpi-value">{approved}</div></div>', unsafe_allow_html=True)
col3.markdown(f'<div class="kpi-card"><div class="kpi-title">Rejected</div><div class="kpi-value">{rejected}</div></div>', unsafe_allow_html=True)
col4.markdown(f'<div class="kpi-card"><div class="kpi-title">Pending</div><div class="kpi-value">{pending}</div></div>', unsafe_allow_html=True)

# --------------------------
# Charts
# --------------------------
st.markdown('<div class="section-header">📊 Analytics Overview</div>', unsafe_allow_html=True)
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.markdown("#### 🚨 Top Violation Types")
    if not filtered_df.empty:
        top_types = filtered_df['type'].value_counts().nlargest(5)
        fig1 = px.bar(
            x=top_types.index, 
            y=top_types.values, 
            labels={'x':'Violation Type','y':'Count'}, 
            color=top_types.index, 
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        fig1.update_layout(
            plot_bgcolor='rgba(0,0,0,0)', 
            paper_bgcolor='rgba(0,0,0,0)', 
            showlegend=False, 
            height=350
        )
        st.plotly_chart(fig1, use_container_width=True)
    else:
        st.info("No data for selected filters.")

with chart_col2:
    st.markdown("#### 📈 Status Distribution")
    if not filtered_df.empty:
        status_counts = filtered_df['status'].value_counts()  # FIXED: removed space
        fig2 = px.pie(
            names=status_counts.index, 
            values=status_counts.values, 
            color=status_counts.index, 
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig2.update_traces(textposition='inside', textinfo='percent+label')
        fig2.update_layout(height=350)
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No data for selected filters.")

# --------------------------
# Map Visualization
# --------------------------
st.markdown('<div class="section-header">🗺️ Violation Heatmap</div>', unsafe_allow_html=True)

# Filter out rows with missing lat/lon
map_df = filtered_df.dropna(subset=['lat', 'lon'])

if not map_df.empty:
    # Create base map centered on Hyderabad
    center_lat = map_df['lat'].mean()
    center_lon = map_df['lon'].mean()
    
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=12,
        tiles='OpenStreetMap'
    )
    
    # Add marker cluster
    marker_cluster = MarkerCluster().add_to(m)
    
    # Add markers
    for idx, row in map_df.iterrows():
        # Color code by status
        if row['status'] == 'Approved':
            color = 'green'
        elif row['status'] == 'Rejected':
            color = 'red'
        else:
            color = 'blue'
        
        popup_text = f"""
        <b>Type:</b> {row['type']}<br>
        <b>Plate:</b> {row['plate']}<br>
        <b>Location:</b> {row['location']}<br>
        <b>Time:</b> {row['time']}<br>
        <b>Status:</b> {row['status']}
        """
        
        folium.Marker(
            location=[row['lat'], row['lon']],
            popup=folium.Popup(popup_text, max_width=250),
            tooltip=f"{row['type']} - {row['plate']}",
            icon=folium.Icon(color=color, icon='info-sign')
        ).add_to(marker_cluster)
    
    # Add heatmap layer
    heat_data = [[row['lat'], row['lon']] for idx, row in map_df.iterrows()]
    HeatMap(heat_data, radius=15, blur=25, max_zoom=13).add_to(m)
    
    st_folium(m, width=None, height=500)
else:
    st.info("No location data available for map visualization.")

# --------------------------
# Violations Table with Actions
# --------------------------
st.markdown('<div class="section-header">📋 Violations Records</div>', unsafe_allow_html=True)

if not filtered_df.empty:
    # Download button
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="⬇️ Download CSV",
        data=csv,
        file_name='violations_export.csv',
        mime='text/csv',
    )
    
    st.markdown(f"**Showing {len(filtered_df)} violations**")
    
    # Display violations as expandable cards
    for idx, row in filtered_df.iterrows():
        with st.expander(f"🚗 {row['plate']} - {row['type']} ({row['status']})"):
            col_left, col_right = st.columns([2, 1])
            
            with col_left:
                st.markdown(f"**Violation Type:** {row['type']}")
                st.markdown(f"**Vehicle Plate:** {highlight_substring(row['plate'], search_plate)}", unsafe_allow_html=True)
                st.markdown(f"**Location:** {row['location']}")
                st.markdown(f"**Time:** {row['time']}")
                st.markdown(f"**Status:** <span class='{row['status'].lower()}'>{row['status']}</span>", unsafe_allow_html=True)
                
                # Show additional details if available
                if 'confidence' in row and pd.notna(row['confidence']):
                    st.markdown(f"**Confidence:** {row['confidence']:.2%}")
                if 'track_id' in row and pd.notna(row['track_id']):
                    st.markdown(f"**Track ID:** {row['track_id']}")
            
            with col_right:
                # Display image if available
                if 'image' in row and pd.notna(row['image']) and row['image']:
                    image_path = row['image']
                    if os.path.exists(image_path):
                        st.image(image_path, caption="Evidence", use_container_width=True)
                    else:
                        st.info("Image not found")
                
                # Status update actions
                if 'id' in row:
                    st.markdown("**Update Status:**")
                    new_status = st.selectbox(
                        "Change to:",
                        ["Approved", "Rejected", "Pending"],
                        key=f"status_{idx}",
                        index=["Approved", "Rejected", "Pending"].index(row['status'])
                    )
                    if st.button("✅ Update", key=f"btn_{idx}"):
                        if update_violation_status(str(row['id']), new_status):
                            st.success(f"Updated to {new_status}")
                            st.cache_data.clear()
                            st.rerun()
else:
    st.info("No violations match the current filters.")

# --------------------------
# Footer
# --------------------------
st.markdown("---")
st.markdown(
    '<div style="text-align: center; color: #666; padding: 20px;">'
    '🚦 AI Traffic Violation Detection System | Team 232 | HTF 2025'
    '</div>', 
    unsafe_allow_html=True
)