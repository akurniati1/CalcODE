import streamlit as st

def app():
    st.set_page_config(
        page_title="CalcODE — Home", 
        layout="wide")
    
    st.title("Home")
    
    st.write(
        "CalcODE solves ordinary differential equations (ODEs) up to **third order** "
        "for **Initial Value Problems (IVP)** and **Boundary Value Problems (BVP)**. "
        "It’s built for fast iteration: friendly inputs, clear plots, and easy exports."
    )

    if st.button("Go to ODE Solver"):
        # set flag ROUTE, jangan sentuh key widget langsung
        st.session_state["_route_to"] = "ODE Solver"
        st.rerun()

    st.divider()


    # Method
    st.markdown(
        """
        <style>
        .method-card{
            border:1px solid rgba(200,200,200,0.5);
            border-radius:14px;
            padding:14px 14px;
            height:100%;
            display:flex;
            flex-direction:column;
            gap:6px;                 /* space between title, tag, desc */
        }
        .method-title{
            font-weight:700;
            font-size:1.05rem;
            line-height:1.2;         /* compact title line height */
            margin:0;
        }
        .method-tag{
            display:inline-block;
            padding:2px 8px;
            border-radius:999px;
            background:#F1F5F9;
            font-size:0.8rem;
            line-height:1.2;         /* compact tag line height */
        }
        .method-desc{
            font-size:0.95rem;
            line-height:1.45;        /* nicer paragraph line spacing */
            margin:0;
            white-space:normal;
            word-break:break-word;
        }
        .method-desc p{ margin:0; } /* remove extra paragraph margins */
        </style>
        """,
        unsafe_allow_html=True
    )

    st.subheader("Methods Used")
    st.caption("A concise overview of the numerical methods available under the hood.")

    def method_card(title, tag, desc):
        with st.container():
            st.markdown(
                f"""
                <div class="method-card">
                <div class="method-title">{title}</div>
                <div class="method-tag">{tag}</div>
                <div class="method-desc">{desc}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    c1, c2, c3 = st.columns(3)
    with c1:
        method_card(
            "RK45 (Dormand–Prince)",
            "IVP • Non-stiff",
            "Adaptive explicit Runge–Kutta 4(5). Default for smooth, non-stiff problems; good general-purpose integrator."
        )
    with c2:
        method_card(
            "RK23 (Bogacki–Shampine)",
            "IVP • Non-stiff",
            "Lower-order explicit method with tighter error control at low tolerances; useful for cheaper steps or quick previews."
        )
    with c3:
        method_card(
            "Radau",
            "IVP • Stiff",
            "Implicit Runge–Kutta of Radau IIA (order 5). Stable on stiff dynamics; supports dense output and event handling."
        )

    st.markdown("<div class='row-gap'></div>", unsafe_allow_html=True)

    d1, d2, d3 = st.columns(3)
    with d1:
        method_card(
            "BDF",
            "IVP • Stiff",
            "Variable-order backward differentiation formula (Gear-type). Efficient for very stiff or slowly varying solutions."
        )
    with d2:
        method_card(
            "LSODA",
            "IVP • Auto stiff/non-stiff",
            "Automatically switches between Adams (non-stiff) and BDF (stiff) based on stiffness detection; robust all-rounder."
        )
    with d3:
        method_card(
            "Collocation (solve_bvp)",
            "BVP",
            "Finite-difference collocation for boundary value problems with mesh refinement; suitable for steady-state or spatial ODEs."
        )


    # Footer
    st.divider()
    f1, f2, f3 = st.columns([1, 1, 1])  # Tiga kolom dengan lebar sama
    with f1:
        st.markdown(
            """
            <div style="text-align: left;">
                <b>Version:</b> 0.1.0
            </div>
            """,
            unsafe_allow_html=True
        )

    with f2:
        st.markdown(
            """
            <div style="text-align: center;">
                <a href="mailto:alfibellakurniati@gmail.com?subject=%5BCalcODE%5D%20Issue%20Report&body=Hello%20CalcODE%20Team,%0A%0A
    Please%20describe%20the%20issue%20you%20encountered%3A%0A-%20Steps%20to%20reproduce%3A%0A-%20Error%20message%3A%0A-%20Screenshot%20(if%20any)%3A%0A%0AThank%20you."
            >Report an issue</a> 
            </div>
            """,
            unsafe_allow_html=True
        )

    with f3:
        st.markdown(
            """
            <div style="text-align: right; color: gray; font-size: 0.9em;">
                © 2025 CalcODE
            </div>
            """,
            unsafe_allow_html=True
        )

