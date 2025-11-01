import streamlit as st
from streamlit_option_menu import option_menu
import solver_ode, home, overview_ode
from pathlib import Path

ROOT = Path(__file__).parent
ICON = ROOT / "icon.png"

assert ICON.exists(), f"Image not found: {ICON}"

st.set_page_config(
    page_title="CalcODE — ODE Solver",
    page_icon=str(ICON),
    layout="wide"
)

def tune_layout(min_sidebar_px=280):
    st.markdown(f"""
    <style>
    .main .block-container {{ max-width: 100% !important; padding: 0 2rem; }}

    /* Sidebar width */
    [data-testid="stSidebar"][aria-expanded="true"] {{
        min-width: {min_sidebar_px}px !important; flex-shrink: 0 !important;
    }}
    [data-testid="stSidebar"][aria-expanded="false"] {{
        width: 0 !important; min-width: 0 !important; transform: translateX(-100%);
    }}

    /* Sidebar content stays in normal flow, reduce the top padding */
    [data-testid="stSidebarContent"] {{
        padding-top: 6px !important;
    }}
    </style>
    """, unsafe_allow_html=True)


def streamlit_menu(example=1):
    if example == 1:
        tune_layout(min_sidebar_px=280)

        with st.sidebar:
            selected = option_menu(
                "Main Menu",
                ["Home", "Overview ODE", "ODE Solver"],
                icons=["house", "book", "calculator"],
                menu_icon="cast",
                default_index=0,
                styles={
                    "nav-link": {"--hover-color": "#E7F1FF"},
                    "nav-link-selected": {"background-color": "#0D6EFD", "color": "white"},
                },
            )

        return selected

selected = streamlit_menu(example=1)

if selected == "Home":
    home.app()
elif selected == "Overview ODE":
    overview_ode.app()
elif selected == "ODE Solver":
    solver_ode.app()
