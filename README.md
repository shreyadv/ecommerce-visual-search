# 🛍️ E-Commerce Visual Product Search Engine

An end-to-end Machine Learning pipeline that allows users to upload an image of a product and instantly retrieve visually similar items from a database. This project was built to demonstrate scalable image retrieval techniques using Deep Learning and Vector Similarity Search.

## 🧠 System Architecture

1. **Feature Extraction (PyTorch):** Uses a pre-trained **ResNet-50** Convolutional Neural Network (CNN) with the final classification layer removed to extract a 2048-dimensional feature embedding from the input image.
2. **Vector Database (FAISS):** The embeddings are indexed using Facebook AI Similarity Search (FAISS) for lightning-fast nearest-neighbor retrieval (L2 distance).
3. **Web Interface (Streamlit):** An interactive web app allowing users to upload query images and view the top-K similar product matches in real-time.

## 🛠️ Tech Stack
* **Deep Learning Framework:** PyTorch, Torchvision
* **Vector Search:** FAISS (Facebook AI Similarity Search)
* **Frontend UI:** Streamlit
* **Data Processing:** Pandas, NumPy, Pillow (PIL)

## 🚀 How to Run Locally

### 1. Setup Environment
Clone the repository and install the required dependencies:
```bash
git clone [https://github.com/shreyadv/ecommerce-visual-search.git](https://github.com/shreyadv/ecommerce-visual-search.git)
cd ecommerce-visual-search
python3 -m venv venv
source venv/bin/activate
pip install torch torchvision faiss-cpu streamlit pandas pillow tqdm
