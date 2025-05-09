import streamlit as st
import numpy as np
import streamlit as st

def get_custom_signal_function(expr, TE, TR, alpha_deg):
    def custom_fn(T1, T2, rho):
        alpha_rad = alpha_deg * np.pi / 180
        local_vars = {
            "T1": T1, "T2": T2, "rho": rho,
            "TE": TE, "TR": TR, "alpha": alpha_deg,
            "alpha_rad": alpha_rad,
            "np": np, "exp": np.exp, "sin": np.sin, "cos": np.cos
        }
        try:
            formula = eval(expr, {}, local_vars)
            return formula
        except Exception as e:
            st.error(f"⚠️ Error evaluating expression: {e}")
            return 0
    return custom_fn

def add_custom_sequence():
    st.write("#### 📦 Available MRI Sequences")

    # 一行两张卡片
    col1, col2 = st.columns(2)

    with col1:
        with st.expander("🧠 VIBE (in-phase)", expanded=False):
            st.markdown("**TE**: 4.54 ms  **TR**: 7.25 ms  **α**: 10°")
            st.markdown("Signal ∝ ρ · sin(α) · (1 - exp(-TR/T1)) · exp(-TE/T2) / (1 - cos(α)·exp(-TR/T1))")
            st.caption("Commonly used for dynamic contrast-enhanced MRI")

    with col2:
        with st.expander("🧠 VIBE (opposed-phase)", expanded=False):
            st.markdown("**TE**: 2.30 ms  **TR**: 7.25 ms  **α**: 10°")
            st.markdown("Same formula, earlier TE for fat-water separation")

    col3, col4 = st.columns(2)

    with col3:
        with st.expander("🧠 GRE (T1-weighted Gradient Echo)", expanded=False):
            st.markdown("**TE**: 4.00 ms  **TR**: 7.00 ms  **α**: 10°")
            st.markdown(r"""Signal ∝ ρ · sin(α) · (1 - exp(-TR/T1)) · exp(-TE/T2*) / (1 - cos(α)·exp(-TR/T1))""")
            st.caption("T1-weighted signal used in gradient echo imaging")

    with col4:
        with st.expander("🧠 T2 SPACE / FSE", expanded=False):
            st.markdown("**TE**: 150 ms  **TR**: 2000 ms")
            st.markdown(r"Signal ∝ ρ · (1 - exp(-TR/T1)) · exp(-TE/T2)")
            st.caption("T2-weighted sequence used in high-resolution imaging")

    st.write("#### ➕ Define Custom MRI Sequence")

    with st.expander("🎨 Create your own sequence"):
        custom_name = st.text_input("Name", "CustomSequence1")
        expr = st.text_area("Expression (T1, T2, rho)", "rho * np.sin(alpha_rad) * (1 - np.exp(-TR / T1)) / (1 - np.cos(alpha_rad) * np.exp(-TR / T1)) * np.exp(-TE / T2)")
        TE = st.number_input("TE (ms)", 4.0)
        TR = st.number_input("TR (ms)", 7.0)
        alpha = st.number_input("Flip Angle (°)", 10.0)
        
        if st.button("✅ Add sequence"):
            fn = get_custom_signal_function(expr, TE, TR, alpha)
            st.session_state["custom_sequences"] = st.session_state.get("custom_sequences", {})
            st.session_state["custom_sequences"][custom_name] = {
                "fn": fn,
                "expr": expr,
                "TE": TE,
                "TR": TR,
                "alpha": alpha
            }
            st.success(f"Custom sequence '{custom_name}' added!")
