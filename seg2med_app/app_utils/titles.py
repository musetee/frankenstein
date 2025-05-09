import streamlit as st

title_type = 'stwrite'
if title_type == 'stwrite':
    step1_frankenstein = "### 🧩 Step 1: Select **editable organs** to be transformed on the canvas"
    step2_frankenstein = "### 🧬 Step 2: Choose a **loading mode** for body, tissue, and organs\nExpand the description to understand the differences between modes"
    step3_frankenstein = "### 🖼️ Step 3: Select **slices** for constructing your Frankenstein composite"
    step4_frankenstein = "### 🧠 Step 4: Begin the **Frankenstein operation** on the organ canvas"
    step5_frankenstein = "### ⚙️ Step 5: Set **output parameters** (For quickstart, just change the output modality)"
    step6_frankenstein = "### 🔬 Step 6: Run the **simulation** to generate a prior image for model inference"
    step7_frankenstein = "### 🤖 Step 7: Run the **model inference** using the simulated prior image"
    step8_frankenstein = "### 💾 Step 8: **Download your Frankenstein!**"

elif title_type == 'markdown':
    step_begin = """
    <div class="franken-step blue-box">
    <b>🎨 Begin:</b> Choose a <b>colormap</b> to visualize different tissues.
    </div>
    """

    step1_frankenstein = """
    <div class="franken-step green-box">
    <b>🧩 Step 1:</b> Select <b>editable organs</b> to be transformed on the canvas.
    </div>
    """

    step2_frankenstein = """
    <div class="franken-step blue-box">
    <b>🧬 Step 2:</b> Choose a <b>loading mode</b> for body, tissue, and organs.<br>
    <span style="color:#666;">(Expand the description to understand the differences between modes)</span>
    </div>
    """

    step3_frankenstein = """
    <div class="franken-step orange-box">
    <b>🖼️ Step 3:</b> Select <b>slices</b> for constructing your Frankenstein composite.
    </div>
    """

    step4_frankenstein = """
    <div class="franken-step green-box">
    <b>🧠 Step 4:</b> Begin the <b>Frankenstein operation</b> on the organ canvas.
    </div>
    """

    step5_frankenstein = """
    <div class="franken-step blue-box">
    <b>⚙️ Step 5:</b> Set <b>output parameters</b>. For quickstart, just change the output modality.
    </div>
    """

    step6_frankenstein = """
    <div class="franken-step orange-box">
    <b>🔬 Step 6:</b> Run the <b>simulation</b> to generate a prior image for model inference.
    </div>
    """

    step7_frankenstein = """
    <div class="franken-step red-box">
    <b>🤖 Step 7:</b> Run the <b>model inference</b> using the simulated prior image.
    </div>
    """

    step8_frankenstein = """
    <div class="franken-step green-box">
    <b>💾 Step 8:</b> <b>Download</b> your Frankenstein result.
    </div>
    """

    # 样式初始化（只需写一次）
    style_block = """
    <style>
    .franken-step {
        padding: 12px;
        border-radius: 6px;
        font-size: 15px;
        margin-bottom: 10px;
    }
    .blue-box {
        background-color: #e8f1fa;
        border-left: 5px solid #1f77b4;
    }
    .orange-box {
        background-color: #fff4e5;
        border-left: 5px solid #ff9900;
    }
    .red-box {
        background-color: #fdecea;
        border-left: 5px solid #e74c3c;
    }
    .green-box {
        background-color: #e8f5e9;
        border-left: 5px solid #2e7d32;
    }
    </style>
    """

    # 页面最开始运行一次
    st.markdown(style_block, unsafe_allow_html=True)
    
    
def make_step_renderer(text):
    if title_type == 'stwrite':
        st.write(text)
    elif title_type == 'markdown':
        st.markdown(text, unsafe_allow_html=True)
    
# 使用示例
'''st.markdown(step1, unsafe_allow_html=True)'''
