import streamlit as st
from streamlit_option_menu import option_menu
import solver_ode, home, overview_ode
from pathlib import Path
import base64

# Path setup
ROOT = Path(__file__).parent
ICON = ROOT / "icon.png"              
LOGO = ROOT / "logo.png"      
OPTIONS = ["Home", "Overview ODE", "ODE Solver"]

st.set_page_config(page_icon=str(ICON), layout="wide")
st.session_state.setdefault("active_menu", "Home")
st.session_state.setdefault("_menu_ver", 0)

if "_route_to" in st.session_state:
    st.session_state["active_menu"] = st.session_state.pop("_route_to")
    st.session_state["_menu_ver"] += 1

def streamlit_menu():
    default_idx = OPTIONS.index(st.session_state["active_menu"])
    menu_key = f"main_menu_v{st.session_state['_menu_ver']}"

    with open(LOGO, "rb") as f:
        logo_base64 = base64.b64encode(f.read()).decode("utf-8")

    with st.sidebar:
        st.markdown(
            f"""
            <div style="text-align:center; margin-bottom:9px;">
                <img src="data:image/png;base64,{logo_base64}" width="200" style="margin-bottom:9px;">
            </div>
            """,
            unsafe_allow_html=True
        )

        selected = option_menu(
            "Main Menu",
            OPTIONS,
            icons=["house", "book", "calculator"],
            menu_icon="cast",
            default_index=default_idx,
            key=menu_key,
            styles={
                "nav-link": {"--hover-color": "#E7F1FF"},
                "nav-link-selected": {"background-color": "#0D6EFD", "color": "white"},
            },
        )

    st.session_state["active_menu"] = selected
    return selected

selected = streamlit_menu()

if selected == "Home":
    home.app()
elif selected == "Overview ODE":
    overview_ode.app()
elif selected == "ODE Solver":
    solver_ode.app()
