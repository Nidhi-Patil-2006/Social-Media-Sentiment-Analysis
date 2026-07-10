import streamlit as st

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Social Media Sentiment Analysis",
    page_icon="💬",
    layout="wide"
)

# --------------------------------------------------
# Sidebar Styling
# --------------------------------------------------

st.markdown(
    """
    <style>

    section[data-testid="stSidebar"] {
        width: 270px;
    }

    section[data-testid="stSidebar"] * {
        font-size: 18px !important;
    }

    section[data-testid="stSidebarNav"] li {
        padding: 6px 0px;
    }

    section[data-testid="stSidebarNav"] a {
        font-weight: 600;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# --------------------------------------------------
# Pages
# --------------------------------------------------

home = st.Page(
    "pages/home.py",
    title="🏠 Home"
)

eda = st.Page(
    "pages/eda.py",
    title="📊 Exploratory Data Analysis"
)

svm = st.Page(
    "pages/svm.py",
    title="🧠 SVM Predictor"
)

random_forest = st.Page(
    "pages/randomforest.py",
    title="🌳 Random Forest Predictor"
)

comparison = st.Page(
    "pages/comparison.py",
    title="📈 Model Comparison"
)

# --------------------------------------------------
# Navigation
# --------------------------------------------------

with st.sidebar:

    st.title("💬 Sentiment Analysis")

    st.markdown("---")

    pg = st.navigation(
        [
            home,
            eda,
            svm,
            random_forest,
            comparison
        ],
        position="sidebar"
    )

    st.markdown("---")

    st.caption("Machine Learning Lab Micro-Project")

# --------------------------------------------------
# Run Selected Page
# --------------------------------------------------

pg.run()