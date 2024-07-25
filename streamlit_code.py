import streamlit as st
gradio_url = "https://9532651235ce28cfca.gradio.live/"  
st.markdown(
    f'<iframe src="{gradio_url}" width="100%" height="800px" style="border:none;"></iframe>',
    unsafe_allow_html=True,
)
