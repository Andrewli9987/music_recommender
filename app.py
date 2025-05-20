import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ── Page config & CSS ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Music Popularity Explorer",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown("""
  <style>
    .streamlit-spinner {
      transform: scale(1.5);
      margin: 2rem auto;
    }
    .stButton>button {
      background-color: #13823B;
      color: white;
      border-radius: 999px;
      padding: 0.6em 1.2em;
    }
    .stButton>button:hover {
      background-color: #1DB954;
    }
  </style>
""", unsafe_allow_html=True)

# ── Sidebar ─────────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Configuration")

with st.sidebar.expander("📖 Feature Descriptions"):
    st.markdown("""
- **danceability** (0.0–1.0): How suitable a track is for dancing.
- **energy** (0.0–1.0): Perceptual measure of intensity and activity.
- **loudness** (dB): Overall loudness; typical values range –60 to 0 dB.
- **speechiness** (0.0–1.0): Detects presence of spoken words.
- **acousticness** (0.0–1.0): Confidence measure the track is acoustic.
- **instrumentalness** (0.0–1.0): Likelihood track has no vocals.
- **liveness** (0.0–1.0): Detects presence of an audience.
- **valence** (0.0–1.0): Musical positiveness (e.g., happy vs. sad).
- **tempo** (BPM): Beats per minute.
    """)

selected = st.sidebar.multiselect(
    "Select features",
    ['danceability','energy','loudness','speechiness','acousticness',
     'instrumentalness','liveness','valence','tempo'],
    default=['danceability','energy','loudness','speechiness'],
    help="Pick which features to include as inputs to the model."
)
normalize = st.sidebar.checkbox(
    "Normalize features", True,
    help="Scale selected features to have zero mean and unit variance."
)
test_size = st.sidebar.slider(
    "Test set %", 10, 50, 20,
    help="Percentage of data held out for evaluating the model."
) / 100
model_choice = st.sidebar.selectbox(
    "Choose model",
    ["Linear Regression", "Random Forest"],
    help="Select which algorithm to train."
)
run = st.sidebar.button(
    "Train & Evaluate",
    help="Click to train the model on your selected features and show metrics."
)

# ── Header ──────────────────────────────────────────────────────────────────
col1, col2 = st.columns([1, 6])
with col1:
    # Optional: drop in a 'spotify_logo.png' for branding
    # st.image("spotify_logo.png", width=60)
    pass
with col2:
    st.markdown("""
      <h1 style="color: #13823B; margin: 0">Music Popularity Explorer</h1>
      <p style="color: #B3B3B3; margin: 0">Powered by scikit-learn & Streamlit</p>
    """, unsafe_allow_html=True)
st.markdown("---")

# ── Data & Constants ─────────────────────────────────────────────────────────
FEATURES = selected
TARGET = 'popularity'

@st.cache_data
def load_data():
    df = pd.read_csv("tracks.csv")
    return df.dropna(subset=FEATURES + [TARGET]).reset_index(drop=True)

df = load_data()

# ── Main Tabs ────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔍 EDA", "⚙️ Train & Evaluate", "Predict"])

with tab1:
    st.subheader("Exploratory Data Analysis")
    corr = df[FEATURES + [TARGET]].corr()
    spotify_cmap = LinearSegmentedColormap.from_list(
        "spotify_theme", ["#13823B", "#FFD700", "#FF69B4"]
    )
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap=spotify_cmap, ax=ax)
    st.pyplot(fig)

with tab2:
    st.subheader("Model Training & Evaluation")
    placeholder = st.empty()
    if run:
        with placeholder.container():
            with st.spinner("Training model..."):
                X = df[FEATURES].copy()
                scaler = None
                if normalize:
                    scaler = StandardScaler()
                    X = pd.DataFrame(scaler.fit_transform(X), columns=FEATURES)
                y = df[TARGET]
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42
                )
                model = (
                    RandomForestRegressor(random_state=42)
                    if model_choice == "Random Forest"
                    else LinearRegression()
                )
                model.fit(X_train, y_train)
                st.session_state.model = model
                st.session_state.scaler = scaler
                st.session_state.X_test = X_test
                st.session_state.y_test = y_test
        placeholder.empty()

        preds = st.session_state.model.predict(st.session_state.X_test)
        rmse = root_mean_squared_error(st.session_state.y_test, preds)
        mae  = mean_absolute_error(st.session_state.y_test, preds)
        r2   = r2_score(st.session_state.y_test, preds)
        c1, c2, c3 = st.columns(3)
        c1.metric("RMSE", f"{rmse:.2f}", help="Root Mean Squared Error: sqrt of average squared error")
        c2.metric("MAE",  f"{mae:.2f}", help="Mean Absolute Error: average absolute error")
        c3.metric("R²",   f"{r2:.2f}", help="R² (explained variance): closer to 1 is better")

        fig2 = px.scatter(
            x=st.session_state.y_test,
            y=preds,
            color=preds,
            labels={"x": "Actual Popularity", "y": "Predicted Popularity"},
            title=model_choice,
            template="plotly_dark",
            color_continuous_scale=["#13823B", "#FFD700", "#FF69B4"]
        )
        fig2.update_traces(marker=dict(size=8))
        st.plotly_chart(fig2, use_container_width=True)

        sample = pd.DataFrame({
            "track":     df.loc[st.session_state.y_test.index, "track_name"].values,
            "actual":    st.session_state.y_test.values,
            "predicted": np.round(preds, 1)
        }).head(10)
        st.subheader("Sample Predictions")
        st.dataframe(sample, use_container_width=True)
    else:
        st.info("Click 'Train & Evaluate' to start training.")

with tab3:
    st.subheader("Single-Track Prediction")
    if 'model' not in st.session_state:
        st.info("Train the model first under 'Model Training & Evaluate'.")
    else:
        inputs = {}
        cols = st.columns(len(st.session_state.X_test.columns))
        for i, feat in enumerate(st.session_state.X_test.columns):
            inputs[feat] = cols[i].slider(
                feat,
                float(df[feat].min()), float(df[feat].max()),
                float(df[feat].mean())
            )
        if st.button("Predict Single Track"):
            x_df = pd.DataFrame([inputs])[st.session_state.X_test.columns]
            if st.session_state.scaler:
                x_scaled = st.session_state.scaler.transform(x_df)
            else:
                x_scaled = x_df.values
            pop = st.session_state.model.predict(x_scaled)[0]
            st.metric("Predicted Popularity", f"{pop:.1f}")