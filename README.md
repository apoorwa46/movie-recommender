
# 🎬 Movie Recommendation System

This is a **Movie Recommendation System** built with **Streamlit** using **item-based KNN collaborative filtering**.  

It recommends movies based on the ones you like and displays **posters** and **genre insights**.

---

## 🗂 Project Structure

```

movie_recommender/
│
├── app.py                 # Main Streamlit app
├── requirements.txt       # Python dependencies
├── models/                # Folder for model files (.pkl, .npz)
│   └── .gitkeep           # Placeholder to track empty folder
├── data/                  # Folder for dataset files (CSV, JSON)
│   └── .gitkeep           # Placeholder to track empty folder
└── README.md

````

> **Note:** The `models/` and `data/` folders are empty in the repo.  
> You must manually add the required model and dataset files before running the app.

---

## ⚡ Installation

1. Clone the repository:

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd movie_recommender
````

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Add your **model files** in the `models/` folder:

   * `user_movie_matrix_sparse.npz`
   * `movie_id_map.pkl`
   * `movies.pkl`

4. Add your **dataset files** in the `data/` folder if needed.

---

## 🚀 Run the App

```bash
streamlit run app.py
```

* Open the URL shown in the terminal to interact with the app.
* Select movies you like and get personalized recommendations with posters.

---

## 🖼 Poster Images

Movie posters are fetched dynamically from the **TMDb API**.
If posters are missing, a **placeholder image** will be shown.

---

## ⚠️ Notes

* Do **not push large model files** to GitHub; keep the repo lightweight.
* Use the `.gitignore` to ignore `models/*.pkl`, `models/*.npz`, and `data/*.csv`.
* You can manually upload models when deploying to **Streamlit Cloud** or host them online for automatic download.

---

## 📌 Dependencies

* streamlit
* pandas
* numpy
* scikit-learn
* scipy
* matplotlib
* requests

---

Enjoy discovering movies! 🎬

```

Do you want me to do that?
```
