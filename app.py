import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model

st.set_page_config(page_title="CSV Line-by-Line Plotter", layout="centered")

# Load model once
@st.cache_resource
def load_cnn_model():
    return load_model("1D_CNN.keras")

model = load_cnn_model()

st.title("CSV Line-by-Line Plotter")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
if uploaded_file:
    if "line_index" not in st.session_state:
        st.session_state.line_index = 0

    # Use chunksize=1 and track current index
    chunk_size = 1
    csv_iterator = pd.read_csv(uploaded_file, chunksize=chunk_size)

    # Rewind to current index
    for _ in range(st.session_state.line_index):
        try:
            next(csv_iterator)
        except StopIteration:
            st.warning("End of CSV reached.")
            st.stop()

    try:
        chunk = next(csv_iterator)
        values = chunk.iloc[0].values.astype(float)
        
        # Plot
        fig, ax = plt.subplots()
        ax.plot(values, marker='o')
        ax.set_title(f"Line {st.session_state.line_index + 1}")
        st.pyplot(fig)

        # Predict
        input_array = np.array(values).reshape(1, 187, 1)
        prediction = model.predict(input_array, verbose=0)
        st.markdown(f"### Model Output: `{prediction[0][0]:.4f}`")

        if st.button("Next Line"):
            st.session_state.line_index += 1
            st.rerun()

    except StopIteration:
        st.warning("End of CSV reached.")
