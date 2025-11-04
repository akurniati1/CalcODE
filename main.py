import streamlit as st
from streamlit_option_menu import option_menu
import solver_ode, home, overview_ode
from pathlib import Path

ROOT = Path(__file__).parent
ICON = ROOT / "icon.png"
st.set_page_config(page_icon=str(ICON), layout="wide")

OPTIONS = ["Home", "Overview ODE", "ODE Solver"]

# --- INIT state
st.session_state.setdefault("active_menu", "Home")
st.session_state.setdefault("_menu_ver", 0)

# --- Tangkap redirect SEBELUM render menu
if "_route_to" in st.session_state:
    st.session_state["active_menu"] = st.session_state.pop("_route_to")
    # paksa widget menu dibuat ulang agar highlight ikut pindah
    st.session_state["_menu_ver"] += 1

def streamlit_menu():
    default_idx = OPTIONS.index(st.session_state["active_menu"])
    # key dinamis: berubah bila ada redirect
    menu_key = f"main_menu_v{st.session_state['_menu_ver']}"
    with st.sidebar:
        selected = option_menu(
            "Main Menu",
            OPTIONS,
            icons=["house", "book", "calculator"],
            menu_icon="cast",
            default_index=default_idx,
            key=menu_key,  # <- dipaksa re-create saat _menu_ver berubah
            styles={
                "nav-link": {"--hover-color": "#E7F1FF"},
                "nav-link-selected": {"background-color": "#0D6EFD", "color": "white"},
            },
        )
    # simpan pilihan user biasa
    st.session_state["active_menu"] = selected
    return selected

selected = streamlit_menu()

if selected == "Home":
    home.app()
elif selected == "Overview ODE":
    overview_ode.app()
elif selected == "ODE Solver":
    solver_ode.app()
