import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

# --- CRITICAL: Force Light Mode (Enhanced) ---
st.set_page_config(
    page_title="Wind Load Calculator",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Enhanced CSS to force light mode across all browsers and accounts
st.markdown("""
<style>
    /* Force light mode at root level */
    :root {
        color-scheme: light only !important;
    }
    
    html, body, [data-testid="stAppViewContainer"], .main, .stApp {
        color-scheme: light !important;
        background-color: #FFFFFF !important;
        color: #000000 !important;
        forced-color-adjust: none !important;
    }

    /* Override all text colors */
    * {
        color: #000000 !important;
        -webkit-text-fill-color: #000000 !important;
    }
    
    /* Specific overrides for input fields */
    input, textarea, select, button, label, p, span, div {
        color: #000000 !important;
        background-color: #FFFFFF !important;
        -webkit-text-fill-color: #000000 !important;
    }
    
    /* Number input styling - Light gray background with black text */
    input[type="number"] {
        background-color: #F5F5F5 !important;
        color: #000000 !important;
        -webkit-text-fill-color: #000000 !important;
        border: 1px solid #CCCCCC !important;
        border-radius: 4px !important;
        padding: 8px !important;
    }
    
    /* Number input when focused */
    input[type="number"]:focus {
        background-color: #FFFFFF !important;
        border: 2px solid #2196F3 !important;
        outline: none !important;
    }
    
    /* Radio buttons and labels */
    .stRadio label, .stRadio p {
        color: #000000 !important;
        -webkit-text-fill-color: #000000 !important;
    }
    
    /* Plotly charts */
    .stPlotlyChart, .js-plotly-plot, .plotly {
        background-color: #FFFFFF !important;
    }
    
    .stPlotlyChart svg text {
        fill: #000000 !important;
    }

    /* Title */
    h1 {
        color: #1976D2 !important;
        -webkit-text-fill-color: #1976D2 !important;
        font-size: 40px !important;
        text-align: center;
        font-weight: bold;
    }
    
    /* Subheaders */
    h2, h3 {
        color: #000000 !important;
        -webkit-text-fill-color: #000000 !important;
    }
    
    /* Labels */
    .stNumberInput label, .stSelectbox label, .stFileUploader label {
        color: #000000 !important;
        -webkit-text-fill-color: #000000 !important;
        font-weight: bold;
    }   
    
    /* Button styling */
    div.stButton > button:first-child {
        background-color: #2196F3 !important;
        color: white !important;
        -webkit-text-fill-color: white !important;
        border-radius: 8px;
        height: 3em;
        width: 100%;
        font-size: 16px;
        border: none;
    }
    
    div.stButton > button:first-child:hover {
        background-color: #1976D2 !important;
    }
    
    /* Success message */
    .stSuccess {
        background-color: #C8E6C9 !important;
        color: #1B5E20 !important;
        -webkit-text-fill-color: #1B5E20 !important;
    }
    
    /* Info boxes */
    .stInfo {
        background-color: #E3F2FD !important;
        color: #0D47A1 !important;
        -webkit-text-fill-color: #0D47A1 !important;
    }
    
    /* Metrics */
    .stMetric {
        background-color: #F5F5F5 !important;
        padding: 10px;
        border-radius: 8px;
    }
    
    .stMetric label, .stMetric div {
        color: #000000 !important;
        -webkit-text-fill-color: #000000 !important;
    }
    
    /* File uploader */
    .stFileUploader {
        background-color: #F5F5F5 !important;
        border-radius: 8px;
        padding: 8px;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        color: #000000 !important;
        -webkit-text-fill-color: #000000 !important;
        background-color: #F5F5F5 !important;
    }
    
    /* Remove any filters that might invert colors */
    * {
        filter: none !important;
        -webkit-filter: none !important;
    }
</style>
""", unsafe_allow_html=True)


# --- Wind Load Calculation Function ---
def calculate_wind_load(H, omega, g, rho_air, Ax=303.3, Ay=592.5, z0=0.01, c_dir=1, c_season=1, c0=1, cp=1.2):
    z = np.arange(1, H + 1)
    v_b0 = 100 / 3.6
    vb = c_dir * c_season * v_b0
    kr = 0.19 * (z0 / 0.05) ** 0.07
    cr = kr * np.log(z / z0)
    vm = cr * c0 * vb
    vm_max = vm[-1]
    kl = 1
    Iv = kl / (c0 * np.log(z / z0))
    q_p = 0.5 * rho_air * (vm ** 2) * (1 + 7 * Iv)
    Fwy = q_p * cp * Ay / 1e3
    Fwx = q_p * cp * Ax / 1e3
    return {'z': z, 'vm': vm, 'vm_max': vm_max, 'Fwy': Fwy, 'Fwx': Fwx, 'q_p': q_p, 'Iv': Iv}


# --- File Reader ---
def read_parameter_file(uploaded_file):
    try:
        data = np.loadtxt(uploaded_file, delimiter=",")
        if len(data) != 4:
            st.error("❌ File must contain exactly 4 numeric values: H, g, rho_air, omega_rpm")
            return None
        H, g, rho_air, omega_rpm = data
        return float(H), float(g), float(rho_air), float(omega_rpm)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None


# --- Plotly Plots with explicit light theme ---
def create_interactive_plots(results):
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Wind Load Distribution', 'Wind Velocity Profile'),
        horizontal_spacing=0.12
    )
    
    fig.add_trace(
        go.Scatter(
            x=results['Fwy'], y=results['z'], mode='lines', name='Y-direction',
            line=dict(color='#00BCD4', width=3),
            hovertemplate='Load: %{x:.2f} kN<br>Height: %{y} m<extra></extra>'
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=results['Fwx'], y=results['z'], mode='lines', name='X-direction',
            line=dict(color='#FF9800', width=3),
            hovertemplate='Load: %{x:.2f} kN<br>Height: %{y} m<extra></extra>'
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=results['vm'], y=results['z'], mode='lines', name='Mean Wind Velocity',
            line=dict(color='#4CAF50', width=3),
            hovertemplate='Velocity: %{x:.2f} m/s<br>Height: %{y} m<extra></extra>'
        ),
        row=1, col=2
    )

    # Explicit axis styling for light theme
    fig.update_xaxes(
        title_text="Wind Load [kN]", row=1, col=1, 
        gridcolor='#E0E0E0',
        linecolor='#000000',
        title_font=dict(color='#000000'),
        tickfont=dict(color='#000000')
    )
    fig.update_xaxes(
        title_text="Wind Velocity [m/s]", row=1, col=2, 
        gridcolor='#E0E0E0',
        linecolor='#000000',
        title_font=dict(color='#000000'),
        tickfont=dict(color='#000000')
    )
    fig.update_yaxes(
        title_text="Height [m]", row=1, col=1, 
        gridcolor='#E0E0E0',
        linecolor='#000000',
        title_font=dict(color='#000000'),
        tickfont=dict(color='#000000')
    )
    fig.update_yaxes(
        title_text="Height [m]", row=1, col=2, 
        gridcolor='#E0E0E0',
        linecolor='#000000',
        title_font=dict(color='#000000'),
        tickfont=dict(color='#000000')
    )
    
    # Force light theme layout
    fig.update_layout(
        height=600, 
        plot_bgcolor='white', 
        paper_bgcolor='white',
        font=dict(color='black', size=12),
        showlegend=True,
        legend=dict(
            bgcolor='rgba(255,255,255,0.95)', 
            bordercolor='#BDBDBD', 
            borderwidth=1,
            font=dict(color='#000000')
        ),
        hovermode='closest',
        title_font=dict(color='#000000'),
        # Fix hover tooltip colors
        hoverlabel=dict(
            bgcolor="white",
            font_size=14,
            font_family="sans-serif",
            font_color="black",
            bordercolor="#2196F3"
        )
    )
    
    # Update subplot titles color
    for annotation in fig['layout']['annotations']:
        annotation['font'] = dict(color='#000000', size=14)
    
    return fig


# --- App UI ---
st.title("Wind Load Calculator (BS EN 1991.1.4) 🌬️")

mode = st.radio("Select Input Mode", ["Manual Input", "Upload Parameters File"])

if mode == "Manual Input":
    st.subheader("📊 Manual Parameter Input")

    col1, col2 = st.columns(2)
    with col1:
        H = st.number_input("Total Height (H) [m]", value=66.7, min_value=1.0)
        omega_rpm = st.number_input("Angular Velocity (ω) [RPM]", value=2.0, min_value=0.0)
        omega = omega_rpm * 2 * np.pi / 60
        st.info(f"ω = {omega:.4f} rad/s")
    with col2:
        g = st.number_input("Gravity (g) [m/s²]", value=9.81)
        rho_air = st.number_input("Air Density (ρ) [kg/m³]", value=1.225, format="%.3f")

    if st.button("Calculate Wind Load"):
        with st.spinner("Calculating..."):
            results = calculate_wind_load(int(H), omega, g, rho_air)
            st.success("✅ Calculation Complete!")
            col1, col2, col3 = st.columns(3)
            col1.metric("Max Velocity", f"{results['vm_max']:.2f} m/s")
            col2.metric("Max Load (X)", f"{results['Fwx'][-1]:.2f} kN")
            col3.metric("Max Load (Y)", f"{results['Fwy'][-1]:.2f} kN")
            fig = create_interactive_plots(results)
            st.plotly_chart(fig, use_container_width=True)

else:
    st.subheader("📁 Upload Parameter File")
    st.write("File format: `H, g, rho_air, omega_rpm` (comma-separated, one line)")
    uploaded_file = st.file_uploader("Upload your .csv or .txt file", type=["csv", "txt"])
    if uploaded_file:
        params = read_parameter_file(uploaded_file)
        if params:
            H, g, rho_air, omega_rpm = params
            omega = omega_rpm * 2 * np.pi / 60
            st.info(f"✅ Loaded parameters: H={H} m, g={g}, ρ={rho_air}, ω={omega_rpm} RPM")
            if st.button("Calculate Wind Load"):
                with st.spinner("Calculating..."):
                    results = calculate_wind_load(int(H), omega, g, rho_air)
                    st.success("✅ Calculation Complete!")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Max Velocity", f"{results['vm_max']:.2f} m/s")
                    col2.metric("Max Load (X)", f"{results['Fwx'][-1]:.2f} kN")
                    col3.metric("Max Load (Y)", f"{results['Fwy'][-1]:.2f} kN")
                    fig = create_interactive_plots(results)
                    st.plotly_chart(fig, use_container_width=True)


with st.expander("ℹ️ About Input Parameters"):
    st.write("""
    **Input Parameters:**
    - **H**: Total height of the structure [m]
    - **ω (omega)**: Angular velocity in RPM (converted to rad/s)
    - **g**: Gravitational acceleration [m/s²] (standard: 9.81)
    - **ρ (rho_air)**: Air density [kg/m³] (standard at sea level: 1.225)
    
    **About BS EN 1991.1.4:**
    This standard provides methods for calculating wind loads on buildings and structures.
    
    **Terrain Categories (typical z0 values):**
    - Open sea, lakes: z0 = 0.003 m
    - Flat terrain with obstacles: z0 = 0.01 m
    - Suburban/industrial: z0 = 0.3 m
    - Urban areas: z0 = 1.0 m
    """)
