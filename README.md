# Customer Segmentation Using Unsupervised Learning

Clustering retail customers into distinct segments based on demographics and purchasing behavior, using multiple unsupervised learning techniques.

---

## Overview

A retail company wants to understand its customer base. Given data on 200 customers with no predefined labels, the goal is to **discover natural customer segments** that can inform marketing strategy and business decisions.

---

## Dataset

| Feature | Description |
|---------|-------------|
| `age` | Customer age |
| `income` | Annual income ($) |
| `spending_score` | Spending intensity (1-100) |
| `visits_per_month` | Store visit frequency |
| `online_ratio` | Share of purchases made online (0-1) |

200 customers · 5 features · No labels

---

## Approach

1. **Exploratory analysis** — inspected distributions and feature scales
2. **Scaling** — StandardScaler to normalize features for distance-based algorithms
3. **Optimal k selection** — Elbow method + Silhouette score (k=3 selected, silhouette=0.506)
4. **KMeans clustering** — segmented customers into 3 groups
5. **Hierarchical clustering** — dendrogram analysis for comparison
6. **t-SNE visualization** — confirmed cluster separation in 2D
7. **PCA** — found 3 components capture 92% of variance; pipeline with PCA+KMeans achieved silhouette=0.589

---

## Results

### Customer Segments Discovered

| Segment | Age | Income | Spending | Visits/Month | Online Ratio | Profile |
|---------|-----|--------|----------|--------------|--------------|---------|
| 0 | ~45 | ~$88k | Medium | ~4 | Low (0.30) | Wealthy in-store shoppers |
| 1 | ~24 | ~$25k | High | ~12 | High (0.80) | Young online spenders |
| 2 | ~60 | ~$50k | Low | ~7 | Medium (0.49) | Older conservative shoppers |

### Key Finding

Applying PCA (5→3 dimensions) before KMeans **improved** clustering quality (silhouette 0.506 → 0.589) by removing noise from less informative features.

---

## Tools & Libraries

- Python 3
- scikit-learn (KMeans, PCA, StandardScaler, t-SNE, silhouette_score)
- scipy (hierarchical clustering, dendrograms)
- pandas, matplotlib

---

## How to Run

```bash
git clone https://github.com/MohamedChaker-bha/customer-segmentation.git
cd customer-segmentation
pip install pandas scikit-learn matplotlib scipy
python customer_segmentation.py
```

---

## Author

**Mohamed Chaker Iben Hadj Amor**  
Data Engineering Student · FSB, Tunisia  
[LinkedIn](https://www.linkedin.com/in/mohamed-chaker-iben-hadj-amor/) · [GitHub](https://github.com/MohamedChaker-bha)
