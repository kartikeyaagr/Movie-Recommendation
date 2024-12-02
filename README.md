# 🎥 **Movie Recommendation System**  
_A powerful machine learning-powered recommendation engine tailored to deliver personalized movie suggestions._

![Project Banner](https://img.shields.io/badge/Machine_Learning-Scikit_Learn-orange) ![Project Banner](https://img.shields.io/badge/Clustering-KMeans-blue) 

---

##  **About the Project**
This project is a **Movie Recommendation System** designed to cluster and analyze movies using advanced machine learning techniques. By leveraging a combination of content-based filtering, clustering algorithms, and dimensionality reduction, the system offers personalized recommendations tailored to user preferences.

### **Key Features**
-  **Multi-modal Feature Integration**: Combines textual, numerical, and metadata features.
-  **Cluster-based Recommendations**: Groups similar movies using **K-Means Clustering**.
-  **Dimensionality Reduction**: Applies **PCA** for efficient computation and visualization.

---

## 📂 **Dataset**
- Dataset sourced from **[Kaggle]([https://www.kaggle.com/datasets](https://www.kaggle.com/datasets/asaniczka/tmdb-movies-dataset-2023-930k-movies))**, containing:
  - Metadata: Genres, Languages, Keywords, Production Companies
  - Numerical Features: IMDB Ratings, Release Years
- Extensive data cleaning and preprocessing ensure high-quality inputs.

---

## 🛠️ **Technologies Used**
- **Python** 
- **Scikit-learn** for ML algorithms  
- **Pandas** for data wrangling   
- **TF-IDF Vectorizer** for text processing  
- **StandardScaler** for feature normalization  

## 🧠 **How It Works**

1. **Data Preprocessing**:  
   - Clean and process the Kaggle dataset.
   - Transform textual data into numerical matrices using `CountVectorizer` and `TF-IDF`.

2. **Feature Engineering**:  
   - Merge vectorized features with numerical attributes to create a unified feature matrix.  

3. **Clustering**:  
   - Apply **K-Means Clustering** to segment movies into meaningful groups.  

4. **Dimensionality Reduction**:  
   - Use **PCA** to retain 70% variance for computational efficiency and cluster visualization.

---

## 🏆 **Future Enhancements**
- Implement **image similarity analysis** using movie posters.
- Introduce **hybrid recommendation models** by combining collaborative and content-based filtering.
- Enable real-time scalability for handling larger datasets.
