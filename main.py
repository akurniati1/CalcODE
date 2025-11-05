import numpy as np
import streamlit as st
import sympy as sp
import re
from scipy.integrate import solve_ivp, solve_bvp
import plotly.graph_objects as go
from sympy.parsing.sympy_parser import parse_expr
from typing import Callable, List, Dict, Tuple


COMMON_FUNCS = {
    # trig + hyperbolic
    "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
    "asin": sp.asin, "acos": sp.acos, "atan": sp.atan,
    "sinh": sp.sinh, "cosh": sp.cosh, "tanh": sp.tanh,
    # exp/log/sqrt/abs/step
    "exp": sp.exp, "log": sp.log, "sqrt": sp.sqrt,
    "Abs": sp.Abs, "Heaviside": sp.Heaviside,
    # constants
    "pi": sp.pi, "E": sp.E,
}

def _normalize_ops(s: str) -> str:
    """Normalisasi simbol operator non-ASCII → ASCII agar deteksi konsisten."""
    if not isinstance(s, str):
        return s
    return (s
        # minus
        .replace("−", "-")   
        .replace("–", "-")   
        .replace("—", "-")   
        .replace("﹣", "-")   
        .replace("－", "-")  
        # times
        .replace("×", "*")   
        .replace("⋅", "*")   
        .replace("·", "*")   
        .replace("∙", "*")   
        .replace("＊", "*")  
        # divide
        .replace("÷", "/")   
        .replace("／", "/")  
        # power
        .replace("＾", "^")  
        # percent
        .replace("％", "%")  
        # plus
        .replace("＋", "+")  
        # parentheses
        .replace("（", "(")  
        .replace("）", ")")  
    )

def to_pyint(x, default=None):
    """
    Convert to int. If it fails:
    - default is None -> raise ValueError
    - default is number -> return int(default)
    """
    try:
        if isinstance(x, sp.Basic):
            return int(float(x))
        return int(x)
    except Exception:
        if default is None:
            raise ValueError(f"The value '{x}' cannot be converted to int")
        return int(default)

# Convert prime-notation like u''', u'', u' into SymPy Derivative(u, (t, n))
def _prime_to_derivative(expr_str: str, dep_name: str, indep_name: str) -> str:
    """
    Convert prime-notation like u''', u'', u' into SymPy Derivative(u, (t, n))
    Works for any number of primes.
    """
    pattern = rf"\b{re.escape(dep_name)}('+)"
    def repl(m):
        n = len(m.group(1))
        if n == 1:
            return f"Derivative({dep_name}, {indep_name})"
        else:
            return f"Derivative({dep_name}, ({indep_name}, {n}))"
    return re.sub(pattern, repl, expr_str)

# Detect candidate symbolic parameters for UI
def detect_params_for_ui(lhs, rhs, dep, indep):
    """
    Detect candidate symbolic as optional parameters from LHS & RHS,
    excluding the free variable (indep) and dep name (if appears bare).
    """
    try:
        t = sp.Symbol(indep)
        y = sp.Function(dep)(t)
        base_dict = {indep: t, dep: y, "Derivative": sp.Derivative, **COMMON_FUNCS}

        lexpr = parse_expr(
            _prime_to_derivative(lhs, dep, indep),
            local_dict=base_dict,
            evaluate=False
        )
        rexpr = parse_expr(rhs, local_dict=base_dict, evaluate=False)

        syms = (lexpr.free_symbols | rexpr.free_symbols)
        exclude = {t, sp.Symbol(dep)} 
        final = sorted([s for s in syms if s not in exclude], key=lambda s: s.name)
        return [s.name for s in final if s.name != dep]
    except Exception:
        return []
    
# Detect and validate orders greater than 3
def leibniz_to_sympy(expr: str, dep_name: str, indep_name: str) -> str:
        e = expr
        # y''' -> Derivative(y(t),(t,3))
        e = re.sub(
            rf"{re.escape(dep_name)}'''",
            f"Derivative({dep_name}({indep_name}), ({indep_name}, 3))",
            e
        )
        # y''  -> Derivative(y(t),(t,2))
        e = re.sub(
            rf"{re.escape(dep_name)}''",
            f"Derivative({dep_name}({indep_name}), ({indep_name}, 2))",
            e
        )
        # y'   -> Derivative(y(t), t)
        e = re.sub(
            rf"{re.escape(dep_name)}'",
            f"Derivative({dep_name}({indep_name}), {indep_name})",
            e
        )
        return e

# Safe sympify with limited functions & symbols
def _safe_sympify(expr_str: str):
        allowed = {
            "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
            "exp": sp.exp, "log": sp.log, "sqrt": sp.sqrt,
            "pi": sp.pi, "E": sp.E
        }

        param_names = [
            "alpha", "beta", "mu", "kappa", "gamma", "lambda",
            "delta", "sigma", "rho", "omega", "c"
        ]
        allowed.update({name: sp.Symbol(name, real=True) for name in param_names})

        allowed.update({
            "Gamma": sp.gamma,    
            "Lambda": sp.Lambda   
        })

        return sp.sympify(expr_str, locals=allowed)

# Normalize prime variations to ASCII apostrophe (')
def _normalize_primes(s: str) -> str:
            """
            Convert all prime variations to ASCII apostrophe (').
            """
            return (s
                .replace("’", "'")
                .replace("′", "'")
                .replace("″", "''")
                .replace("‴", "'''")
            )

# Find derivative variable names in LHS
def _find_derivative_vars(lhs: str, dep_name: str, indep_name: str):
        """
        Detect all variable names that appear in derivative notation in the given lhs string.
        Return a set of variable names found.
        """
        vars_found = set()
        s = _normalize_primes(lhs)
        
        # Notasi prime: y', y'', y'''
        prime_pat = r"(?<![A-Za-z0-9_])([A-Za-z_]\w*)(?:\s*'{1,3})(?![A-Za-z0-9_])"
        for m in re.finditer(prime_pat, s):
            vars_found.add(m.group(1))
            
        indep_esc = re.escape(indep_name)
        leibniz_pat = (
            rf"d\^?\s*(\d+)?\s*([A-Za-z_]\w*)\s*/\s*d\s*{indep_esc}\^?\s*(\d+)?"
        )
        for m in re.finditer(leibniz_pat, s, flags=re.IGNORECASE):
            var = m.group(2)
            vars_found.add(var)

        return vars_found

# Detect maximum order from text patterns
def _max_order_heuristic(left_side: str, dep_name: str) -> int:
        """
        Detect the maximum order of derivatives from text patterns in the left_side string.
        """
        max_ord = 1
        # Prime notation
        for m in re.finditer(rf"{re.escape(dep_name)}('+)", left_side):
            max_ord = max(max_ord, len(m.group(1)))
        # LaTeX style y^{(n)}
        for m in re.finditer(rf"{re.escape(dep_name)}\s*\^\s*\(\s*(\d+)\s*\)", left_side):
            try:
                max_ord = max(max_ord, int(m.group(1)))
            except Exception:
                pass
        # SymPy style Derivative(y(t),(t,n))
        for m in re.finditer(r"Derivative\([^)]*?,\s*\([^)]*?,\s*(\d+)\)\)", left_side):
            try:
                max_ord = max(max_ord, int(m.group(1)))
            except Exception:
                pass
        return max_ord

def _infer_order_from_lhs(left_side: str, dep_name: str, indep_name: str) -> int:
        try:
            t = sp.symbols(indep_name)
            y = sp.Function(dep_name)(t)
            lhs_conv = leibniz_to_sympy(left_side, dep_name, indep_name)
            lhs_expr = _safe_sympify(lhs_conv)
            orders = []
            for d in lhs_expr.atoms(sp.Derivative):
                if d.expr == y:
                    orders.append(d.variables.count(t))
            detected = max(orders) if orders else 1
            return max(1, min(detected, 3))
        except Exception:
            return 1

# Format number for LaTeX output
def fmt_num_ltx(x, nd: int = 6) -> str:
    try:
        xf = float(x)
        s = f"{xf:.{nd}g}"
        if s.endswith("."):
            s = s[:-1]
        if s == "-0":
            s = "0"
        return s
    except Exception:
        return str(x)

# Detect if RHS has derivative of dep_name w.r.t indep_name
def _rhs_has_derivative(rhs_str: str, dep_name: str, indep_name: str) -> bool:
    if not isinstance(rhs_str, str) or not rhs_str.strip():
        return False

    s = _normalize_primes(rhs_str)
    dep_esc = re.escape(dep_name)
    indep_esc = re.escape(indep_name)

    # Prime notation: u', u'', u'''
    prime_pat = rf"(?<![A-Za-z0-9_]){dep_esc}\s*'{1,}(?![A-Za-z0-9_])"
    if re.search(prime_pat, s):
        return True

    # Leibniz notation: d^n u / d t^n
    leibniz_pat = rf"d\^?\s*\d*\s*{dep_esc}\s*/\s*d\s*{indep_esc}(?:\s*\^\s*\d+)?"
    if re.search(leibniz_pat, s, flags=re.IGNORECASE):
        return True

    # Sympy Derivative notation: Derivative(u(t), t) or Derivative(u(t), (t, n))
    sympy_der_pat = rf"Derivative\s*\(\s*{dep_esc}\s*\(\s*{indep_esc}\s*\)\s*,"
    if re.search(sympy_der_pat, s):
        return True

    # Try parsing with sympy
    try:
        t = sp.Symbol(indep_name)
        y = sp.Function(dep_name)(t)
        base = {indep_name: t, dep_name: y, "Derivative": sp.Derivative, **COMMON_FUNCS}

        rhs_prep = _prime_to_derivative(s, dep_name, indep_name) 
        rexpr = parse_expr(rhs_prep, local_dict=base, evaluate=False)

        for d in rexpr.atoms(sp.Derivative):
            if d.expr == y:
                if any(var == t for var in d.variables):
                    return True
    except Exception:
        pass
    return False



# Sytem for solve_ivp
def build_rhs_from_input(
    left_expr_str,
    right_expr_str,
    params=None,
    indep_name="t",
    dep_name="u",
):
    params = params or {}

    # Parse expressions
    t = sp.Symbol(indep_name)
    y_func = sp.Function(dep_name)(t)

    exprL = _prime_to_derivative(left_expr_str, dep_name, indep_name)
    local_dict = {indep_name: t, dep_name: y_func, "Derivative": sp.Derivative, **COMMON_FUNCS}
    left_expr  = parse_expr(exprL, local_dict=local_dict, evaluate=False)
    right_expr = parse_expr(right_expr_str, local_dict=local_dict, evaluate=False)

    # Detect highest derivative on LHS
    ders = sorted(left_expr.atoms(sp.Derivative),
                  key=lambda d: d.derivative_count if hasattr(d, "derivative_count") else 0)
    if not ders:
        raise ValueError("No derivative detected on the left-hand side. Use u', u'', u''', ...")
    highest_der = ders[-1]
    n = highest_der.derivative_count

    # Solve for highest derivative
    sol_list = sp.solve(sp.Eq(left_expr, right_expr), highest_der, dict=True)
    if not sol_list:
        raise ValueError(f"Cannot solve the ODE for {highest_der}. Make sure the ODE is linear in its highest derivative.")
    rhs_expr = sol_list[0][highest_der]

    ode_latex = sp.latex(sp.Eq(left_expr, right_expr), mode="equation")

    # Substitute derivatives with Y0, Y1, ..., Yn-1
    n_int = max(1, to_pyint(n))
    Y = [sp.Symbol(f"Y{k}") for k in range(n_int)]
    subs_map = {y_func: Y[0]}
    for k in range(1, n_int):
        subs_map[sp.Derivative(y_func, (t, k))] = Y[k]
    rhs_sub = rhs_expr.subs(subs_map)

    # Identify parameter symbols
    free_syms = rhs_sub.free_symbols.union(left_expr.free_symbols).union(right_expr.free_symbols)
    param_syms = sorted([s for s in free_syms if s != t and s not in Y], key=lambda s: s.name)
    param_names = tuple(s.name for s in param_syms)

    # Check parameter values
    missing = [s.name for s in param_syms if s.name not in params]
    if missing:
        raise ValueError("Missing parameter values: " + ", ".join(missing))

    param_vals = [float(params[s.name]) for s in param_syms]

    # Lambdify RHS
    lamb_args = (t, *Y, *param_syms)
    g_num = sp.lambdify(lamb_args, rhs_sub, "numpy")

    # Define ODE system for solve_ivp
    def f(t_val, yvec):
        dydt = np.zeros_like(yvec, dtype=float)
        for k in range(n_int - 1):
            dydt[k] = yvec[k + 1]
        dydt[n_int - 1] = g_num(t_val, *yvec, *param_vals)
        return dydt

    return f, n, param_names, ode_latex


# System for solve_bvp
def make_bvp_system(order: int, g: Callable):
    """
    Build first-order system Y' = F(x, Y) for an nth-order ODE y^(n) = g(x, y, y', ..., y^(n-1)).
    Y = [y, y', ..., y^(n-1)]^T
    """
    assert order in (1, 2, 3), "Only orders 1–3 are supported"
    def fun(x, Y):
        dY = np.empty_like(Y)
        
        if order > 1:
            dY[:-1] = Y[1:]
        
        m = Y.shape[1]
        last = np.empty(m)
        if order == 1:
            for j in range(m):
                last[j] = g(x[j], Y[0, j])
        elif order == 2:
            for j in range(m):
                last[j] = g(x[j], Y[0, j], Y[1, j])
        else:  # order == 3
            for j in range(m):
                last[j] = g(x[j], Y[0, j], Y[1, j], Y[2, j])
        dY[-1] = last
        return dY
    return fun

def make_bc_dirichlet_neumann(order: int, conditions: List[Dict]):
    """
    conditions: list of dicts length==order, each like:
      {"at": "left"/"right", "kind": "dirichlet"/"neumann", "der": k, "value": float}
    We allow der in [0, order-1], regardless of "kind" wording (dirichlet/neumann are labels).
    Residual is Y[der] - value at the chosen boundary.
    """
    assert len(conditions) == order, "Number of boundary conditions must equal the ODE order"

    def pick(side, Ya, Yb):
        return Ya if side.lower() == "left" else Yb

    def bc(Ya, Yb):
        res = []
        for c in conditions:
            side = c["at"]
            der  = int(c.get("der", 0))
            val  = float(c["value"])
            if der < 0 or der > order - 1:
                raise ValueError(f"der={der} out of range [0,{order-1}]")
            Y = pick(side, Ya, Yb)
            res.append(Y[der] - val)
        return np.array(res, dtype=float)

    return bc

def initial_guess_mesh(a: float, b: float, order: int, m: int = 400, guess_y: Callable = None):
    """
    Build an initial mesh and guess profile Y0 for solve_bvp.
    guess_y(x) returns y(x) initial guess; higher derivatives start at 0.
    """
    x = np.linspace(a, b, m)
    y0 = np.zeros_like(x) if guess_y is None else guess_y(x)
    Y0 = np.vstack([y0] + [np.zeros_like(x) for _ in range(order - 1)])
    return x, Y0

def solve_bvp_general_dn(order: int,
                         g: Callable,
                         domain: Tuple[float, float],
                         conditions: List[Dict],
                         guess_y: Callable = None,
                         m: int = 400,
                         tol: float = 1e-6,
                         max_nodes: int = 100000):
    """
    Generic wrapper over scipy.integrate.solve_bvp for nth-order ODEs with mixed
    Dirichlet/Neumann-like conditions expressed as Y[der]=value at left/right.
    """
    a, b = map(float, domain)
    fun = make_bvp_system(order, g)
    bc  = make_bc_dirichlet_neumann(order, conditions)
    x, Y0 = initial_guess_mesh(a, b, order, m=m, guess_y=guess_y)
    sol = solve_bvp(fun, bc, x, Y0, tol=tol, max_nodes=max_nodes)
    return sol


# Streamlit app
def app():
    st.set_page_config(
        page_title="CalcODE — ODE Solver",
        layout="wide"
    )

    st.title("ODE Solver")

    # CSS
    st.markdown("""
    <style>
    /* --- Sidebar title --- */
    .sidebar-title { 
        font-weight:700; 
        font-size:1.05rem; 
        margin:.25rem 0 .5rem 0; 
    }

    /* Style untuk tombol utama (Solve) */
    div.stButton > button:first-child {
        background-color: #0D6EFD;   /* biru */
        color: white;
        border: 2px solid #0D6EFD;
        border-radius: 8px;
        padding: 0.5em 1em;
        font-weight: 600;
    }

    /* Efek hover */
    div.stButton > button:first-child:hover {
        background-color: #E7F1FF;   /* biru tua */
        color: black;
        border-color: #E7F1FF;
    }

    /* Saat tombol disabled */
    div.stButton > button:disabled {
        background-color: #D3D3D3 !important;  /* Light Gray */
        color: #FFFFFF !important;
        border: 1px solid #C0C0C0 !important;
    }
    </style>
    """, unsafe_allow_html=True)


    st.subheader("Define the ODE")

    # Dependent and independent variable names
    c1, c2 = st.columns([1, 1])
    with c1:
        dep_name   = st.text_input("Dependent variable", 
                                   value="u", 
                                   key="ode_dep",
                                   help="The function being solved for, e.g., u(t), y(x).")
        
        # Validate dependent variable name
        if not dep_name.strip():
            st.error("Dependent variable cannot be blank. Please enter a valid name (e.g., u, y).")
            st.stop()
        if not re.fullmatch(r"[A-Za-z]+", dep_name.strip()):
            st.error("Dependent variable must contain letters only (e.g., 'u', 'y'). No numbers, spaces, or symbols.")
            st.stop()

    with c2:
        indep_name = st.text_input(
            "Independent variable", 
            value="t", 
            key="ode_indep",
            help="The variable with respect to which differentiation is done, e.g., t, x. Note: must be different from the dependent variable."
            )

        # Validate independent variable name
        if not indep_name.strip():
            st.error("Independent variable cannot be blank. Please enter a valid name (e.g., t, x).")
            st.stop()
        if not re.fullmatch(r"[A-Za-z]+", indep_name.strip()):
            st.error("Independent variable must contain letters only (e.g., 't', 'x'). No numbers, spaces, or symbols.")
            st.stop()

    # Validate variable names between dep and indep    
    _dep = dep_name.strip()
    _ind = indep_name.strip()

    if _dep.lower() == _ind.lower():
        st.error(
            f"Dependent and independent variables must be different. "
            f"Change one of them (e.g., {_dep or 'y'} vs {_ind or 't'})."
        )
        st.stop()
    

    # LHS and RHS of the ODE
    c1, c2, c3 = st.columns([4, 1, 4])
    with c1:
        left_side  = st.text_input("Left-hand side (LHS)", 
                                   value="u'' - u' + u",
                                   help="Enter the left-hand side of the ODE, e.g., u'' + 2*u' + u.")

        # Validate LHS derivatives
        _lhs = left_side.strip()

        if _lhs == "":
            st.error("LHS cannot be blank. Please enter a valid expression.")
            st.stop()
        if _lhs and re.fullmatch(r"[+\-*/%^()\s\.]+", _lhs):
                st.error("LHS cannot contain only operators or parentheses.")
                st.stop()
        
        # Missing operator detection (e.g., 2t, u'u)
        _func_names = "|".join(map(re.escape, COMMON_FUNCS.keys())) or "___NONE___"

        missing_op_pattern = re.compile(
            rf"""
            (?:
                # 1) Number diikuti huruf atau '('  → 2t, 3(x+1)
                \d(?:\.\d+)?\s*(?=[A-Za-z(])

            | # 2) Var (bukan fungsi terdaftar) diikuti '('  → t(x+1)
                \b(?!(?:{_func_names})\b)[A-Za-z]\w*\s*(?=\()

            | # 3) Var diikuti var dengan spasi  → u v
                \b[A-Za-z]\w*\s+(?=[A-Za-z]\w*)

            | # 4) ')' diikuti huruf/angka/'('  → )(, )x, )2, )(
                \)\s*(?=[A-Za-z0-9(])

            | # 5) Turunan prime diikuti huruf/'('  → u'v, u'(x)
                {re.escape(_dep)}\s*'+\s*(?=[A-Za-z(])
            )
            """,
            re.VERBOSE,
        )

        if _lhs and re.search(missing_op_pattern, _lhs):
            st.error(
                "It seems there might be missing operators between terms in the LHS. "
                "For example, use '2*t' instead of '2t', or 'u'*u' instead of 'u'u'."
            )
            st.stop()
        
        lhs_norm = _normalize_ops(_lhs)
        if "_" in lhs_norm:
            st.error(
                "LHS contains unsupported symbols: _ "
            )
            st.stop()
        if "," in lhs_norm:
            st.error(
                "LHS contains unsupported symbols: , "
            )
            st.stop()

        _allowed_lhs = r"[A-Za-z0-9_+\-*/%^(),.'\s]*"

        if not re.fullmatch(_allowed_lhs, lhs_norm):
            bad_chars = sorted(set(re.findall(r"[^A-Za-z0-9_+\-*/%^(),.'\s]", lhs_norm)))
            st.error(
                "LHS contains unsupported symbols: "
                + " ".join(f"**{c}**" for c in bad_chars)
            )
            st.stop()
        
        def _paren_balance_ok(s: str) -> bool:
            """True jika kurung '(' dan ')' seimbang dan urutannya valid."""
            bal = 0
            for ch in s:
                if ch == "(":
                    bal += 1
                elif ch == ")":
                    bal -= 1
                    if bal < 0:  # ada ')' lebih dulu tanpa '(' pembuka
                        return False
            return bal == 0  # harus habis
        
        if not _paren_balance_ok(lhs_norm):
            st.error(
                "Unmatched parentheses on RHS. Make sure every '(' has a matching ')', "
                "and that ')' does not appear before its matching '('."
            )
            st.stop()
        if re.search(r"\(\s*\)", lhs_norm):
            st.error("Empty parentheses '()' are not allowed on LHS. Put a value inside or remove the parentheses.")
            st.stop()
        if re.search(r"[+\-*/%^]\s*(?:$|\)|,)", lhs_norm):
            st.error(
                "LHS ends with an operator or has an operator without a following term. "
                "Please complete the expression."
            )
            st.stop()
        
        # Validate all derivatives use the same dependent variable
        deriv_vars = _find_derivative_vars(left_side, dep_name, indep_name)
        bad_vars = {v for v in deriv_vars if v != dep_name}

        if bad_vars:
            st.error(
                "Detected derivative(s) that do not match the dependent variable.\n\n"
                f"- Dependent variable set to: **{dep_name}**\n"
                f"- Found derivative(s) of: **{', '.join(sorted(bad_vars))}**\n\n"
                "Please rewrite the LHS so that all derivatives use the same dependent variable, "
                f"e.g., {dep_name}', {dep_name}'' or Leibniz form d^n {dep_name}/d{indep_name}^n."
            )
            st.stop()
        
        # Validate at least one derivative of the dependent variable is present
        if dep_name not in deriv_vars:
            st.error(
                "No derivatives of the dependent variable were detected on the LHS.\n"
                f"Please include at least one derivative of **{dep_name}** (e.g., {dep_name}', {dep_name}'', etc.) "
            )
            st.stop()

        # Detect and validate orders greater than 3
        detected_order_sympy = _infer_order_from_lhs(left_side, dep_name, indep_name)
        detected_order_text  = _max_order_heuristic(left_side, dep_name)
        detected_order_raw   = max(detected_order_sympy, detected_order_text)

        if detected_order_raw > 3:
            st.error(
                f"Detected a derivative of order {detected_order_raw}, "
                f"but the current application only supports up to third-order derivatives.\n\n"
                f"Please reduce the equation’s order."
            )
            st.stop()

        detected_order = min(max(detected_order_raw, 1), 3)
        st.caption(f"Highest detected ODE order from left-hand side: **{detected_order}**")

    with c2:
        st.markdown("<div style='text-align:center;font-size:24px;margin-top:26px;'>=</div>", unsafe_allow_html=True)
    with c3:
        right_side = st.text_input("Right-hand side (RHS)", 
                                   value="cos(t)*omega",
                                   help="The right-hand side of the ODE, e.g., sin(t) + alpha.")

        if _rhs_has_derivative(right_side, dep_name.strip(), indep_name.strip()):
            st.error(
                "The RHS contains derivatives of "
                f"**{dep_name}**. This app requires all derivatives to appear on the LHS. "
                f"Please move any derivative terms (e.g., {dep_name}', {dep_name}'', ...) to the left-hand side."
            )
            st.stop()
        
        # Validate RHS expression
        _rhs = right_side.strip()

        if _rhs == "":
            st.error("RHS cannot be blank. Please enter a valid expression.")
            st.stop()
        if _rhs and re.fullmatch(r"[+\-*/%^()\s\.]+", _rhs):
                st.error("RHS cannot contain only operators or parentheses.")
                st.stop()
        if _rhs and re.search(missing_op_pattern, _rhs):
            st.error(
                "It seems there might be missing operators between terms in the RHS. "
                "For example, use '2*t' instead of '2t', or 'u'*u' instead of 'u'u'."
            )
            st.stop()
        
        rhs_norm = _normalize_ops(_rhs)

        if "_" in rhs_norm:
            st.error(
                "RHS contains unsupported symbols: _ "
            )
            st.stop()
        if "," in rhs_norm:
            st.error(
                "RHS contains unsupported symbols: , "
            )
            st.stop()
        if not re.fullmatch(r"[A-Za-z0-9_+\-*/%^().\s]*", rhs_norm):
            bad_chars = sorted(set(re.findall(r"[^A-Za-z0-9_+\-*/%^(),.\s]", rhs_norm)))
            st.error(
                "RHS contains unsupported symbols: "
                + " ".join(f"**{c}**" for c in bad_chars)
            )
            st.stop()
        if not _paren_balance_ok(rhs_norm):
            st.error(
                "Unmatched parentheses on RHS. Make sure every '(' has a matching ')', "
                "and that ')' does not appear before its matching '('."
            )
            st.stop()
        if re.search(r"\(\s*\)", rhs_norm):
            st.error("Empty parentheses '()' are not allowed on RHS. Put a value inside or remove the parentheses.")
            st.stop()
        if re.search(r"[+\-*/%^]\s*(?:$|\)|,)", rhs_norm):
            st.error(
                "LHS ends with an operator or has an operator without a following term. "
                "Please complete the expression."
            )
            st.stop()



    

    # Optional Parameter values
    cand_params = detect_params_for_ui(left_side, right_side, dep_name, indep_name)

    with st.expander("Optional Parameters", expanded=True):
        st.caption("Parameters auto-detected from your equation. Adjust values as needed. A maximum of 10 optional parameters is supported.")
        param_values = {}
        if cand_params:
            cols = st.columns(min(4, max(1, len(cand_params))))
            for i, pname in enumerate(cand_params):
                val = cols[i % 4].number_input(f"{pname} =", value=5.0, step=0.01, format="%.2f", key=f"param_{pname}")
                param_values[pname] = float(val)
        else:
            st.info("No parameters detected.")

    # Domain inputs
    with st.expander("Domain", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            t0 = st.number_input(f"{indep_name} start", value=0.0, step=0.01, format="%.2f")
        with c2:
            t1 = st.number_input(f"{indep_name} end", value=20.0, step=0.01, format="%.2f")

        if t1 <= t0:
            st.error(f"Invalid domain: end value ({t1}) must be greater than start value ({t0}).")
            st.stop()


    # Try to build to know order and LaTeX
    build_error = None
    n_order = None
    ode_preview = None
    param_order = ()
    f_callable = None

    try:
        f_callable, n_order, param_order, ode_preview = build_rhs_from_input(
            left_side, right_side,
            params=param_values,
            indep_name=indep_name,
            dep_name=dep_name,
        )
        
    except Exception as e:
        build_error = str(e)

    # Problem Type and Condition 
    with st.expander("Problem Type and Conditions", expanded=True):
        # Header badges
        st.markdown(
            f"""
            <div style="display:flex;gap:.5rem;flex-wrap:wrap;margin:.25rem 0 1rem 0">
            <span style="padding:.2rem .5rem;border-radius:999px;background:#eef;white-space:nowrap">ODE order: {max(1, to_pyint(n_order))} </span>
            <span style="padding:.2rem .5rem;border-radius:999px;background:#efe;white-space:nowrap">Dependent variable: {dep_name}</span>
            <span style="padding:.2rem .5rem;border-radius:999px;background:#ffe;white-space:nowrap">Independent variable: {indep_name}</span>
            <span style="padding:.2rem .5rem;border-radius:999px;background:#f7f7f7;white-space:nowrap">Domain: {indep_name}\u2208[{t0}, {t1}]</span>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Block IVP when start of domain < 0
        if t0 < 0:
            st.warning(
                f"Initial Value Problem (IVP) is disabled because the domain starts at "
                f"{indep_name} = {t0} < 0. Based on your domain, the problem type is Boundary Value Problem (BVP)."
            )

            cond_options = ["Boundary Value Problem (BVP)"]
            if st.session_state.get("cond_type_choice") == "Initial Value Problem (IVP)":
                st.session_state["cond_type_choice"] = "Boundary Value Problem (BVP)"
        else:   
            cond_options = ["Initial Value Problem (IVP)", "Boundary Value Problem (BVP)"]  

        # Select condition type first (IVP or BVP)
        cond_type = st.radio(
            "**Choose problem type**",
            options=["Initial Value Problem (IVP)", "Boundary Value Problem (BVP)"],
            horizontal=True,
            key="cond_type_choice",
            help="Choose Initial Value Problem (IVP) or Boundary Value Problem (BVP) to configure conditions."
        )
        
        # Input IVP or BVP conditions
        if cond_type == "Initial Value Problem (IVP)":
            # Set initial point t0
            seed_ic = float(st.session_state.get("t_ic", t0))
            seed_ic = max(float(t0), min(float(t1), seed_ic))

            t_ic = st.number_input(
                f"Initial point {indep_name}\u2080",
                value=seed_ic,
                min_value=float(t0),
                max_value=float(t1),
                step=0.01, format="%.2f", key="t_ic"
            )
                
            if build_error is None and n_order is not None:
                int_n = max(1, to_pyint(n_order))

            st.markdown("**Initial conditions:**")
            ic_values = []
            cols = st.columns(int_n)
            prime_map = ["", "′", "″", "‴"]

            def _fmt_num(x):
                try:
                    xf = float(x)
                    return str(int(xf)) if xf.is_integer() else f"{xf:g}"
                except Exception:
                    return str(x)
            
            t0_eq = f"{indep_name}₀={_fmt_num(t_ic)}"
                
            for k in range(int_n):
                # Use primes up to 3rd derivative, then fallback to u^(k)
                if k <= 3:
                    ypart = f"{dep_name}{prime_map[k]}"
                else:
                    ypart = f"{dep_name}^({k})"

                lab = f"{ypart}({t0_eq})"


                ic_values.append(
                    cols[k % 4].number_input(
                        lab,
                        value=0.0,
                        step=0.01,
                        format="%.2f",
                        key=f"ic_{k}",
                    )
                )
        # BVP: show possible options
        else:  # BVP
            prime_map = ["", "′", "″", "‴"]
            def _prime_str(k: int) -> str:
                return f"{dep_name}{prime_map[k]}" if k <= 3 else f"{dep_name}^({k})"

            # Build options
            n_int = max(1, to_pyint(n_order))
            all_opts = []
            for k in range(n_int):
                all_opts.append({"id": f"y{k}_a", "order": k, "kind": "bvp", "where": "a"})
                all_opts.append({"id": f"y{k}_b", "order": k, "kind": "bvp", "where": "b"})
            id2opt = {o["id"]: o for o in all_opts}
            valid_ids = sorted(id2opt.keys())

            def _label_for(id_str: str) -> str:
                o = id2opt[id_str]
                at = f"{indep_name}={t0}" if o["where"] == "a" else f"{indep_name}={t1}"
                return f"{_prime_str(o['order'])}({at})"

            # Persist previous picks
            prev_selected = st.session_state.get("cond_multi_bvp", [])
            prev_selected = [sid for sid in prev_selected if sid in valid_ids]

            # Mutate options to prevent removing when max reached
            options_now = valid_ids if len(prev_selected) < n_int else prev_selected

            st.markdown("**Select boundary conditions (choose exactly as many as the ODE order):**")
            selected_ids = st.multiselect(
                "Boundary conditions",
                options=options_now,
                default=prev_selected,
                key="cond_multi_bvp",
                format_func=_label_for,
                help=("Pick exactly "
                      f"{n_int} condition(s). "
                      + ("Limit reached. Remove one to change selection."
                         if len(prev_selected) >= n_int else "Select the remaining conditions."))
            )

            # Enforce max n_int
            if len(selected_ids) > n_int:
                selected_ids = selected_ids[:n_int]
                st.session_state["cond_multi_bvp"] = selected_ids

            # Feedback on remaining needed
            remaining = max(0, n_int - len(selected_ids))
            if len(selected_ids) < n_order:
                st.warning(f"Please select {remaining} more condition(s).")
            
            #st.caption(f"Possible conditions left: **{remaining}**")
            #if remaining == 0:
            #    st.success("Boundary condition count is valid.")
            #else:
            #    st.warning(f"Please select {remaining} more condition(s).")

            # separate selected at a and b
            selected_a = sorted([sid for sid in selected_ids if id2opt[sid]["where"] == "a"],
                                key=lambda s: id2opt[s]["order"])
            selected_b = sorted([sid for sid in selected_ids if id2opt[sid]["where"] == "b"],
                                key=lambda s: id2opt[s]["order"])

            # Input values for selected conditions
            store = st.session_state.setdefault("bvp_store", {})
            cond_spec = {"kind": "bvp", "items": []}

            col_a, col_b = st.columns(2, gap="medium")

            with col_a:
                st.markdown(f"**At {indep_name} = {t0}**")
                if not selected_a:
                    st.caption("No conditions at the start boundary.")
                for sid in selected_a:
                    o = id2opt[sid]
                    label = f"{_prime_str(o['order'])}({indep_name}={t0}) ="
                    seed = float(store.get(sid, {}).get("value", 0.0))
                    val = st.number_input(label, value=seed, key=f"bvp_val_{sid}")
                    store[sid] = {"value": float(val), "opt": o}
                    cond_spec["items"].append({
                        "kind":  "bvp",
                        "order": o["order"],
                        "where": "a",
                        "value": float(val),
                    })

            with col_b:
                st.markdown(f"**At {indep_name} = {t1}**")
                if not selected_b:
                    st.caption("No conditions at the end boundary.")
                for sid in selected_b:
                    o = id2opt[sid]
                    label = f"{_prime_str(o['order'])}({indep_name}={t1}) ="
                    seed = float(store.get(sid, {}).get("value", 0.0))
                    val = st.number_input(label, value=seed, key=f"bvp_val_{sid}")
                    store[sid] = {"value": float(val), "opt": o}
                    cond_spec["items"].append({
                        "kind":  "bvp",
                        "order": o["order"],
                        "where": "b",
                        "value": float(val),
                    })

            st.session_state["cond_spec"] = cond_spec




    # Preview (LaTeX)
    st.subheader("LaTeX Preview")
    with st.expander("Show details", expanded=True):
        if build_error:
            st.error(build_error)
        else:
            if ode_preview:
                st.markdown("**Normalized ODE (LaTeX):**")
                st.latex(ode_preview)
                
            if param_values:
                st.markdown("**Parameters:**") 
                names = list(param_order) if param_order else sorted(param_values.keys(), key=str.lower)
                items = [f"{sp.latex(sp.Symbol(k))} = {fmt_num_ltx(param_values[k], nd=6)}" for k in names]
                per_row = 4  
                rows = ["\\,\\;,\\; ".join(items[i:i+per_row]) for i in range(0, len(items), per_row)]
                latex_block = r"\begin{aligned}" + r"\\ ".join(rows) + r"\end{aligned}"
                st.latex(latex_block)

            st.markdown("**Domain:**")
            st.latex(rf"{indep_name}\in\left[{t0},\,{t1}\right]")

        # Conditions preview    
        try:
            int_n = max(1, to_pyint(n_order))
            cond_spec_ivp = {"kind": "ivp", "items": []}
            for k in range(int_n):
                cond_spec_ivp["items"].append({
                    "kind":  "ivp",
                    "order": k,
                    "t0":    float(t_ic),          
                    "value": float(ic_values[k]),  
                })
            st.session_state["cond_spec"] = cond_spec_ivp
        except Exception:
            pass

        def _fmt_num_clean(x, nd=6):
            try:
                xf = float(x)
                s = f"{xf:.{nd}g}"
                if s.endswith("."): s = s[:-1]
                if s == "-0": s = "0"
                return s
            except Exception:
                return str(x)

        try:
            cond_spec_prev = st.session_state.get("cond_spec", {})
            items = cond_spec_prev.get("items", [])

            st.markdown("**Conditions:**")
            if items:
                def _fmt_deriv_name(dep: str, order: int) -> str:
                    base = sp.latex(sp.Symbol(dep))
                    if order == 0: return base
                    if order == 1: return base + r"^{\prime}"
                    if order == 2: return base + r"^{\prime\prime}"
                    if order == 3: return base + r"^{\prime\prime\prime}"
                    return base + rf"^{{({order})}}"

                def _fmt_arg(it: dict) -> str:
                    if it.get("kind") == "ivp":
                        # prefers numeric t0 saved in the item; falls back to current t0
                        return _fmt_num_clean(it.get("t0", t0), nd=6)
                    return _fmt_num_clean(t0 if it.get("where") == "a" else t1, nd=6)

                lines = []
                for it in sorted(items, key=lambda z: z.get("order", 0)):
                    lhs = f"{_fmt_deriv_name(dep_name, it.get('order', 0))}({_fmt_arg(it)})"
                    rhs_s = _fmt_num_clean(it.get("value", 0.0), nd=6)
                    lines.append(f"{lhs} = {rhs_s}")

                st.latex(r"\begin{aligned}" + r"\\ ".join(lines) + r"\end{aligned}")
            else:
                st.caption("No conditions selected.")
        except Exception:
            pass


    
    #Solution Settings

    # Catalogs
    IVP_METHODS = [
        ("RK45 (Dormand–Prince)", "RK45"),
        ("RK23 (Bogacki–Shampine)", "RK23"),
        #("DOP853 (explicit high-order)", "DOP853"),
        ("Radau (implicit, stiff)", "Radau"),
        ("BDF (implicit, stiff)", "BDF"),
        ("LSODA (auto stiff/nonstiff)", "LSODA"),
    ]

    if cond_type == "Initial Value Problem (IVP)":
        st.subheader("Solver Method and Settings")
        
        method_options = IVP_METHODS
        method_help = "Choose a solver method suitable for IVP. RK45 is a good general-purpose choice."
        
        DEFAULTS = {
                        "method": "RK45",
                        "n_points": 400,
                        "rtol": 1e-7,
                        "atol": 1e-9,
                        "max_step": np.inf
                    }   

        chosen_method = st.selectbox(
            "IVP method",
            options=[m[1] for m in method_options],
            format_func=lambda v: next((n for n, val in method_options if val == v), v),
            index=next((i for i, m in enumerate(method_options) if m[1] == DEFAULTS["method"]), 0),
            help=method_help,
            key="ivp_method_choice"
        )
        
        show_extra = st.checkbox("Customize method parameters (advanced)", 
                                 help="Leave unchecked to use recommended defaults for this method. Check to override.",
                                 value=False)    

    

        # Advanced Settings
        if show_extra:
                with st.expander("Method Parameters", expanded=True):
                    n_points = st.number_input("Evaluation points", 
                                             key="n_points",
                                             min_value=50, 
                                             max_value=5000, 
                                             value=DEFAULTS["n_points"], 
                                             step=50
                                             )
                    
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        rtol = st.number_input("Relative tolerance", 
                                               key="rtol",
                                               value=DEFAULTS["rtol"], 
                                               format="%.1e",
                                               help="Typical: 1e-6 to 1e-9 (smaller = more accurate, slower)."
                                               )
                    with c2:
                        atol = st.number_input("Absolute tolerance", 
                                               key="atol",
                                               value=DEFAULTS["atol"], 
                                               format="%.1e",
                                               help="Typical: 1e-8 to 1e-12."
                                               )
                    with c3:
                        max_step = st.number_input("Max step size (0 = no limit)", 
                                                   key="max_step",
                                                   min_value=0.00, 
                                                   value=0.00, 
                                                   format="%.2f",
                                                   help="Limit the maximum step size taken by the solver. \n"
                                                        "Leave 0 for no limit (default)."
                                                   )
                        if max_step == 0.0:
                            max_step = DEFAULTS["max_step"]
                    
                    # Check if anything changed from defaults
                    has_changed = (
                        st.session_state["n_points"] != DEFAULTS["n_points"] or
                        st.session_state["rtol"] != DEFAULTS["rtol"] or
                        st.session_state["atol"] != DEFAULTS["atol"] or
                        (max_step != DEFAULTS["max_step"])
                    )
                    
                    # Reset button
                    st.button(
                        "Reset to recommended defaults", 
                        disabled=not has_changed,
                        on_click=lambda: (
                            st.session_state.update({
                                "n_points": DEFAULTS["n_points"],
                                "rtol": DEFAULTS["rtol"],
                                "atol": DEFAULTS["atol"],
                                "max_step": 0.0
                            })
                        )
                    )

                    # IVP params
                    method_params = {
                        "method": chosen_method,
                        "n_points": st.session_state["n_points"],
                        "rtol": st.session_state["rtol"],
                        "atol": st.session_state["atol"],
                        "max_step": max_step
                    }
        else:
            method_params = DEFAULTS.copy()
            method_params["method"] = chosen_method



    # Solve button
    btn_solve = st.button("Solve", use_container_width=True, disabled=bool(build_error))

    def _ivp_kwargs(params):
        # Selalu kembalikan dict
        if not isinstance(params, dict):
            params = {}

        kw = {}

        # method (opsional, biar bisa pakai default SciPy kalau tak di-set)
        method = params.get("method")
        if method is not None:
            kw["method"] = str(method)

        # rtol/atol (opsional)
        rtol = params.get("rtol")
        if rtol is not None:
            kw["rtol"] = float(rtol)

        atol = params.get("atol")
        if atol is not None:
            kw["atol"] = float(atol)

        # max_step: kirim hanya jika finite dan > 0
        max_step = params.get("max_step")
        if max_step is not None:
            kw["max_step"] = float(max_step)

        return kw

    def _alloc_eval_counts(a, b, t0, n_total):
        length = b - a
        if length <= 0:
            return 0, 0
        left_len  = max(0.0, t0 - a)
        right_len = max(0.0, b - t0)
        # Pastikan minimal 2 titik per sisi jika sisi itu ada panjangnya
        if left_len == 0 and right_len == 0:
            return n_total, 0
        if left_len == 0:
            return 0, max(2, n_total)
        if right_len == 0:
            return max(2, n_total), 0
        n_left  = int(round(n_total * (left_len / length)))
        n_right = n_total - n_left
        if n_left < 2:  n_left = 2
        if n_right < 2: n_right = 2
        return n_left, n_right

    if btn_solve and f_callable is not None:
        if cond_type == "Initial Value Problem (IVP)":
            with st.spinner(f"Processing..."):
                try:
                    int_n = max(1, to_pyint(n_order))
                    # Validasi IC
                    y0 = [float(v) for v in ic_values] if ic_values else None
                    if y0 is None or len(y0) != int_n:
                        st.error(f"Initial conditions must be exactly {int_n} values.")
                    else:
                        
                        n_left, n_right = _alloc_eval_counts(float(t0), float(t1), float(t_ic), int(method_params.get("n_points")))
                        
                        ivp_kwargs = _ivp_kwargs(method_params)

                        # Integration to the left (t_ic -> t0), only if t_ic > t0
                        T_left = np.array([])
                        Y_left = None
                        if t_ic > t0 and n_left > 0:
                            t_eval_left = np.linspace(float(t_ic), float(t0), n_left)
                            sol_left = solve_ivp(
                                f_callable,
                                (float(t_ic), float(t0)),
                                y0,
                                t_eval=t_eval_left,
                                **ivp_kwargs
                            )
                            if not sol_left.success:
                                st.warning(f"Solve status (left): {sol_left.message}")
                            T_left = sol_left.t[::-1]  
                            Y_left = sol_left.y[:, ::-1]

                        # Integration to the right (t_ic -> t1), only if t_ic < t1
                        T_right = np.array([])
                        Y_right = None
                        if t_ic < t1 and n_right > 0:
                            t_eval_right = np.linspace(float(t_ic), float(t1), n_right)
                            sol_right = solve_ivp(
                                f_callable,
                                (float(t_ic), float(t1)),
                                y0,
                                t_eval=t_eval_right,
                                **ivp_kwargs
                            )
                            if not sol_right.success:
                                st.warning(f"Solve status (right): {sol_right.message}")
                            T_right = sol_right.t
                            Y_right = sol_right.y

                            ok = sol_right.success
                            if ok:
                                st.success("Solve status: SUCCESS")
                            else:
                                st.error("Solve status: FAILED")
                            st.caption(sol_right.message)

                            
                            
                        # Combine left and right results
                        if T_left.size > 0 and T_right.size > 0:
                            T = np.concatenate([T_left, T_right[1:]])
                            Y = np.concatenate([Y_left, Y_right[:, 1:]], axis=1)
                        elif T_left.size > 0:
                            T, Y = T_left, Y_left
                        elif T_right.size > 0:
                            T, Y = T_right, Y_right
                        else:
                            st.error("No points to evaluate. Increase evaluation points or adjust domain.")
                            st.stop()


                        # Plot
                        st.markdown("#### Solution Plot")

                        fig = go.Figure()
                        for k in range(int_n):
                            label = f"{dep_name}" if k == 0 else rf"{dep_name}{prime_map[k]}"
                            fig.add_trace(
                                go.Scatter(
                                    x=T, 
                                    y=Y[k],
                                    mode="lines",
                                    name=label,
                                    hovertemplate=f"{indep_name}=%{{x}}<br>Value=%{{y}}<extra>{label}</extra>",
                                )
                            )

                        fig.update_layout(
                            xaxis_title=indep_name,
                            yaxis_title="Value",
                            template="plotly_white",
                            legend_title_text="Components",
                            margin=dict(l=40, r=20, t=60, b=40),
                        )

                        fig.update_xaxes(showgrid=True)
                        fig.update_yaxes(showgrid=True)

                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Data sample
                        st.markdown("#### Sample of solution values")
                        show_n = min(10, len(T))
                        table = {indep_name: T[:show_n]}
                        for i in range(int_n):
                            key = f"{dep_name}" if i == 0 else f"{dep_name}{prime_map[i]}"
                            table[key] = Y[i, :show_n]
                        st.dataframe(table)

                except Exception as e:
                    st.error(f"Solver error: {e}")

        
        elif cond_type == "Boundary Value Problem (BVP)":
            with st.spinner("Processing..."):
                try:
                    cond_spec = st.session_state.get("cond_spec", {})
                    items = cond_spec.get("items", [])
                    int_n = max(1, to_pyint(n_order))

                    if len(items) != int_n:
                        st.error(
                            f"You must select exactly {int_n} boundary condition(s). "
                            f"Currently selected: {len(items)}."
                        )
                        st.stop()

                    t = sp.Symbol(indep_name)
                    y_func = sp.Function(dep_name)(t)

                    exprL = _prime_to_derivative(left_side, dep_name, indep_name)
                    local_dict = {
                        indep_name: t,
                        dep_name: y_func,
                        "Derivative": sp.Derivative,
                        **COMMON_FUNCS
                    }
                    left_expr  = parse_expr(exprL, local_dict=local_dict, evaluate=False)
                    right_expr = parse_expr(right_side, local_dict=local_dict, evaluate=False)

                    ders = sorted(
                        left_expr.atoms(sp.Derivative),
                        key=lambda d: d.derivative_count if hasattr(d, "derivative_count") else 0
                    )
                    if not ders:
                        st.error("No derivative detected on the LHS for BVP.")
                        st.stop()

                    highest_der = ders[-1]
                    n = highest_der.derivative_count  

                    if int(n) != int_n:
                        n = int_n

                    sol_list = sp.solve(sp.Eq(left_expr, right_expr), highest_der, dict=True)
                    if not sol_list:
                        st.error(
                            f"Cannot solve the ODE for {highest_der}. "
                            "Ensure the ODE is solvable for the highest derivative."
                        )
                        st.stop()

                    rhs_expr = sol_list[0][highest_der]

                    
                    Y_syms = [sp.Symbol(f"Y{k}") for k in range(n)]
                    subs_map = {y_func: Y_syms[0]}
                    for k in range(1, n):
                        subs_map[sp.Derivative(y_func, (t, k))] = Y_syms[k]
                    rhs_sub = rhs_expr.subs(subs_map)

                    
                    free_syms = rhs_sub.free_symbols.union(left_expr.free_symbols).union(right_expr.free_symbols)
                    param_syms = sorted([s for s in free_syms if s != t and s not in Y_syms], key=lambda s: s.name)
                    missing = [s.name for s in param_syms if s.name not in param_values]
                    if missing:
                        st.error("Missing parameter values: " + ", ".join(missing))
                        st.stop()

                    param_vals = [float(param_values[s.name]) for s in param_syms]

                    lamb_args = (t, *Y_syms, *param_syms)
                    g_num = sp.lambdify(lamb_args, rhs_sub, "numpy")

                    
                    def g_for_bvp(x, *Yargs):
                        return g_num(x, *Yargs, *param_vals)

                    conditions = []
                    for it in items:
                        k = int(it.get("order", 0))
                        side = "left" if it.get("where") == "a" else "right"
                        val = float(it.get("value", 0.0))

                        if k == 0:
                            kind = "dirichlet"
                        elif k == 1:
                            kind = "neumann"
                        else:
                            kind = "dirichlet"  

                        conditions.append({
                            "at": side,
                            "kind": kind,
                            "der": k,     
                            "value": val,
                        })
                    
                    # Solve BVP
                    a, b = float(t0), float(t1)
                    m_pts   = 400     
                    tol_val = 1e-6
                    max_nds = 100000

                    sol = solve_bvp_general_dn(
                        order=int_n,
                        g=g_for_bvp,
                        domain=(a, b),
                        conditions=conditions,
                        guess_y=None,       
                        m=m_pts,
                        tol=tol_val,
                        max_nodes=max_nds
                    )

                    ok = bool(sol.success)
                    if ok:
                        st.success("Solve status: SUCCESS")
                    else:
                        st.error("Solve status: FAILED")
                    st.caption(sol.message)

                    # Plotting
                    prime_map = ["", "′", "″", "‴"]

                    X = sol.x
                    Y = sol.y  

                    st.markdown("#### Solution Plot")
                    fig = go.Figure()
                    for k in range(int_n):
                        label = f"{dep_name}" if k == 0 else rf"{dep_name}{prime_map[k] if k<=3 else f'^({k})'}"
                        fig.add_trace(
                            go.Scatter(
                                x=X, y=Y[k],
                                mode="lines",
                                name=label,
                                hovertemplate=f"{indep_name}=%{{x}}<br>Value=%{{y}}<extra>{label}</extra>",
                            )
                        )
                    fig.update_layout(
                        xaxis_title=indep_name,
                        yaxis_title="Value",
                        template="plotly_white",
                        legend_title_text="Components",
                        margin=dict(l=40, r=20, t=60, b=40),
                    )
                    fig.update_xaxes(showgrid=True)
                    fig.update_yaxes(showgrid=True)
                    st.plotly_chart(fig, use_container_width=True)

                    st.markdown("#### Sample of solution values")
                    show_n = min(10, len(X))
                    table = {indep_name: X[:show_n]}
                    for i in range(int_n):
                        key = f"{dep_name}" if i == 0 else f"{dep_name}{prime_map[i] if i<=3 else f'^({i})'}"
                        table[key] = Y[i, :show_n]
                    st.dataframe(table)

                except Exception as e:
                    st.error(f"BVP solver error: {e}")


if __name__ == "__main__":
    app()
