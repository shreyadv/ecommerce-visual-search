import os
# This line fixes the OpenMP libomp.dylib collision error on Mac
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
from PIL import Image
from extract_embeddings import FeatureExtractor
from database import VectorDatabase

@st.cache_resource
def load_system():
    extractor = FeatureExtractor()
    db = VectorDatabase()
    if not db.load():
        st.error("Database not found! Run build_index.py first.")
    return extractor, db

st.title("🛍️ E-Commerce Visual Product Search")
st.write("Upload an image of a product to find visually similar items in the database.")

extractor, db = load_system()

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.subheader("Your Query Image")
    st.image(uploaded_file, width=300)
        
    st.write("🔍 Searching database for similar items...")
    
    temp_path = "temp_query.png"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    query_vector = extractor.extract(temp_path)
    
    if query_vector is not None:
        results = db.search(query_vector, top_k=5)
        
        if results:
            st.subheader("Top Matches")
            cols = st.columns(len(results))
            for i, img_path in enumerate(results):
                with cols[i]:
                    match_img = Image.open(img_path)
                    st.image(match_img, use_container_width=True)
        else:
            st.warning("No matches found.")
            
    if os.path.exists(temp_path):
        os.remove(temp_path)