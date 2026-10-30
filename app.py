import streamlit as st
from PIL import Image
from detector import predict_image


# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Truth vs. Illusion",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================================
# PREMIUM DARK THEME
# ==========================================================

st.markdown("""
<style>

/* ================================
   MAIN BACKGROUND
================================ */

.stApp {
    background-color: #0b0d10;
    color: #f5f7fa;
}


/* ================================
   MAIN CONTENT WIDTH
================================ */

.block-container {
    max-width: 1050px;
    padding-top: 40px;
    padding-bottom: 150px;
}


/* ================================
   SIDEBAR
================================ */

[data-testid="stSidebar"] {
    background-color: #080a0d;
    border-right: 1px solid rgba(255,255,255,0.08);
}

[data-testid="stSidebar"] h2 {
    color: #ffffff;
}


/* ================================
   HERO
================================ */

.hero-space {
    height: 100px;
}

.hero-title {
    text-align: center;
    font-size: 48px;
    font-weight: 700;
    letter-spacing: -2px;
    color: #ffffff;
}

.hero-subtitle {
    text-align: center;
    font-size: 18px;
    color: #9ca3af;
    margin-top: 10px;
    margin-bottom: 45px;
}


/* ================================
   FEATURE CARDS
================================ */

.feature-box {
    background-color: #111419;
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 18px;
    padding: 25px;
    min-height: 150px;
}

.feature-heading {
    color: #ffffff;
    font-size: 18px;
    font-weight: 600;
}

.feature-text {
    color: #9ca3af;
    font-size: 14px;
    line-height: 1.6;
}


/* ================================
   CHAT INPUT
================================ */

[data-testid="stChatInput"] {
    background-color: #15181d;
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 18px;
}


/* ================================
   BUTTONS
================================ */

.stButton > button {
    width: 100%;
    border-radius: 10px;
    background-color: #15181d;
    border: 1px solid rgba(255,255,255,0.1);
    color: white;
}

.stButton > button:hover {
    border-color: rgba(255,255,255,0.25);
}


/* ================================
   FOOTER
================================ */

.footer {
    text-align: center;
    color: #6b7280;
    font-size: 12px;
    margin-top: 70px;
}

</style>
""", unsafe_allow_html=True)


# ==========================================================
# SESSION STATE
# ==========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# ==========================================================
# SIDEBAR
# ==========================================================

with st.sidebar:

    st.markdown("## ◈ Truth vs. Illusion")

    st.caption("AI Authenticity Intelligence")

    st.divider()

    if st.button("＋ New Analysis"):

        st.session_state.messages = []

        st.rerun()

    st.button("⌕ Analysis History")

    st.divider()

    st.markdown("### Supported Content")

    st.markdown("""
    ✅ Text

    ✅ JPG

    ✅ JPEG

    ✅ PNG

    ✅ WEBP
    """)

    st.markdown("### Not Supported")

    st.markdown("""
    ❌ Video

    ❌ Audio
    """)

    st.divider()

    st.caption(
        "The system uses machine-learning evidence and "
        "will indicate uncertainty when confidence is low."
    )


# ==========================================================
# HOME SCREEN
# ==========================================================

if len(st.session_state.messages) == 0:

    st.markdown(
        '<div class="hero-space"></div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="hero-title">Truth vs. Illusion</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="hero-subtitle">'
        'Detect what looks real. Verify what matters.'
        '</div>',
        unsafe_allow_html=True
    )


    # ======================================================
    # FEATURE CARDS
    # ======================================================

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            '<div class="feature-box">',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="feature-heading">'
            '🖼️ Image Detection'
            '</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="feature-text">'
            'Upload an image and analyze visual patterns '
            'associated with AI-generated or authentic content.'
            '</div>',
            unsafe_allow_html=True
        )

        st.markdown("</div>", unsafe_allow_html=True)


    with col2:

        st.markdown(
            '<div class="feature-box">',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="feature-heading">'
            '✦ Text Detection'
            '</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="feature-text">'
            'Paste text into the input bar and analyze '
            'patterns associated with AI-generated or '
            'human-written content.'
            '</div>',
            unsafe_allow_html=True
        )

        st.markdown("</div>", unsafe_allow_html=True)


# ==========================================================
# SHOW PREVIOUS MESSAGES
# ==========================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        if message["text"]:
            st.markdown(message["text"])

        if message["image"] is not None:
            st.image(
                message["image"],
                width=450
            )


# ==========================================================
# CHAT INPUT
# ==========================================================

prompt = st.chat_input(
    "Ask about this content...",
    accept_file=True,
    file_type=[
        "jpg",
        "jpeg",
        "png",
        "webp"
    ]
)


# ==========================================================
# USER INPUT
# ==========================================================

if prompt:

    user_text = prompt.text.strip()

    uploaded_image = None


    # ======================================================
    # READ IMAGE
    # ======================================================

    if prompt.files:

        uploaded_file = prompt.files[0]

        try:

            uploaded_image = Image.open(
                uploaded_file
            ).convert("RGB")

        except Exception:

            st.error(
                "Please upload a valid JPG, JPEG, PNG or WEBP image."
            )

            st.stop()


    # ======================================================
    # SHOW USER MESSAGE
    # ======================================================

    with st.chat_message("user"):

        if user_text:

            st.markdown(user_text)

        if uploaded_image:

            st.image(
                uploaded_image,
                width=450
            )


    # ======================================================
    # SAVE MESSAGE
    # ======================================================

    st.session_state.messages.append(
        {
            "role": "user",
            "text": user_text,
            "image": uploaded_image
        }
    )



# ======================================================
# AI DETECTION RESPONSE
# ======================================================

with st.chat_message("assistant"):

   if prompt and uploaded_image is not None:

        with st.spinner("Analyzing image..."):

            prediction = predict_image(uploaded_image)

        st.markdown(
            f"### {prediction['result']}"
)

        st.write(
            f"**Confidence:** {prediction['confidence']:.2f}%"
        )

        st.markdown(
            "The result is based on visual patterns detected "
            "by the trained image classification model."
        )
# ==========================================================
# FOOTER
# ==========================================================

st.markdown(
    '<div class="footer">'
    'Truth vs. Illusion · AI Authenticity Detection'
    '</div>',
    unsafe_allow_html=True
)