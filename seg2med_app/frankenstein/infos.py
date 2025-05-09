import streamlit as st
def modes_info():
    with st.expander("🧬 Load Modes Description (Click to expand)"):
        st.markdown("""
    <style>
    .franken-hint {
        background-color: #f0f8ff;
        padding: 12px;
        border-left: 5px solid #0077cc;
        border-radius: 6px;
        font-size: 15px;
    }
    .franken-warning {
        background-color: #fff3cd;
        padding: 12px;
        border-left: 5px solid #ffa500;
        border-radius: 6px;
        font-size: 15px;
    }
    .franken-danger {
        background-color: #f8d7da;
        padding: 12px;
        border-left: 5px solid #dc3545;
        border-radius: 6px;
        font-size: 15px;
    }
    </style>

    <div class='franken-hint'>
    <b>🧍 Default Mode:</b><br>
    All anatomical parts are loaded from the same patient. Suitable for standard use cases.
    </div>
    <br>

    <div class='franken-warning'>
    <b>🧪 Semi-Chaotic Mode:</b><br>
    Contour, tissue segmentation, and fixed organs are loaded from a single patient, while editable organs are randomly mixed from different patients.<br>
    ✅ Recommended for stable Frankenstein experiments.
    </div>
    <br>

    <div class='franken-danger'>
    <b>💥 Fully-Chaotic Mode:</b><br>
    Every part (contour, tissues, and organs) is randomly loaded from different patients.<br>
    ⚠️ Extremely unstable — proceed with caution!
    </div>
    """, unsafe_allow_html=True)
