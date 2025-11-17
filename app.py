import streamlit as st
from numpy.random import default_rng as rng
import pandas as pd



st.write("# Welcome to the Streamlit App!")

col1, col2 = st.columns(2)

with col1:
    x = st.slider("Select a value", 1, 10)
with col2:
    st.write(f"You :blue[***selected***]: {x}")


df = pd.DataFrame(rng(0).standard_normal((20, 3)), columns=["a", "b", "c"])

st.area_chart(df)