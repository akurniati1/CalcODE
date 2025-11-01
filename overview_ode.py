import streamlit as st
import sympy as sp
import numpy as np

def app():
    st.title("Overview of Ordinary Differential Equations (ODEs)")

    st.markdown(
        "An **Ordinary Differential Equation (ODE)** is an equation that relates a function "
        "to its derivatives. It describes how a quantity changes with respect to another "
        "variable, such as time or space."
    )

    st.markdown("A general first-order ODE can be written as:")
    st.latex(r"\frac{dy}{dx} = f(x, y)")
    st.markdown(
        "The goal is to find the function $y(x)$ that satisfies this equation along with given conditions."
    )

    # ---------- IVP ----------
    st.subheader("Initial Value Problem (IVP)")
    st.markdown(
        "In an **Initial Value Problem**, the value of the solution is known at one specific point:"
    )
    st.latex(r"\frac{dy}{dx} = f(x, y), \quad y(x_0) = y_0")

    st.markdown("**Example:**")
    st.latex(r"\frac{dy}{dx} = -2y, \quad y(0) = 1")
    st.markdown(
        "This equation models exponential decay, such as cooling or radioactive decay."
    )

    st.markdown("The exact solution is:")
    st.latex(r"y(x) = e^{-2x}")

    st.markdown("_Common numerical methods:_ Euler’s Method, Runge–Kutta Methods.")

    # Small interactive demo for the IVP example
    with st.expander("Show a quick plot of the IVP example $y'=-2y,\\;y(0)=1$"):
        x_end = st.slider("Domain end (x)", min_value=1.0, max_value=10.0, value=5.0, step=0.5)
        x = np.linspace(0.0, x_end, 400)
        y = np.exp(-2.0 * x)
        st.line_chart(
            data={"x": x, "y(x)=e^{-2x}": y},
            x="x",
        )

    # ---------- BVP ----------
    st.subheader("Boundary Value Problem (BVP)")
    st.markdown(
        "A **Boundary Value Problem** involves finding a function that satisfies the ODE "
        "and meets conditions at two or more points:"
    )
    st.latex(r"y'' = f(x, y, y'), \quad y(a) = \alpha, \; y(b) = \beta")

    st.markdown("**Example:**")
    st.latex(r"y'' + y = 0, \quad y(0) = 0, \quad y(\pi) = 0")
    st.markdown(
        "This represents a vibrating string fixed at both ends."
    )

    st.markdown("A general solution form is:")
    st.latex(r"y(x) = A\sin(x)")
    st.markdown("_Common numerical methods:_ Shooting Method, Finite Difference Method.")

    # ---------- About This Calculator ----------
    st.subheader("About This Calculator")
    st.markdown(
        "This website allows users to:\n"
        "1. Choose the problem type (IVP or BVP),\n"
        "2. Enter the differential equation and conditions,\n"
        "3. Select a numerical method,\n"
        "4. Visualize the solution instantly.\n\n"
        "It provides an interactive way to understand how ODEs behave and how numerical methods work in solving real-world problems."
    )