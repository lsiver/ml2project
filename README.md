# Unsupervised Genre Clustering from Song Lyrics

Notebook for a **CSCA 5632 (Unsupervised Learning)** project exploring how well **unsupervised methods** can cluster **music genres** using only **song lyrics** (bag‑of‑words).

Primary work lives in: **`ML2 Project.ipynb`**

Project repo (as referenced in the notebook): https://github.com/lsiver/ml2project

---

## What this does

- Builds a lyrics dataset by joining:
  - `msd_tagtraum_cd2c.cls` (genre labels / annotations)
  - `mxm_dataset.db` (Musixmatch lyrics bag‑of‑words)
- Cleans + preprocesses lyrics text (stemming, filtering problematic characters)
- Vectorizes text with **CountVectorizer → TF‑IDF**
- Trains and evaluates several **unsupervised** approaches:
  - **NMF** (topic/cluster discovery on TF‑IDF)
  - **KMeans** (hard clustering)
  - **LDA** (soft clustering / topic model)
- Includes a **Logistic Regression** supervised baseline for comparison

> Even though the models are unsupervised, the notebook uses the known genre labels **only for evaluation** (validation). Cluster IDs are mapped to genres using majority vote within each cluster/topic.

---

## Data files expected

The notebook references these local files:

- `msd_tagtraum_cd2c.cls`
- `mxm_dataset.db`
- `stemmed_words.txt` (precomputed word→stem mapping used to normalize lyric tokens)
- `df_final.csv` (cached merged/cleaned dataset produced by the notebook to avoid re‑reading the DB)

Because of dataset licensing/terms, these files may not be included in the repo by default.

---

## Methods at a glance

### Preprocessing
- Merge genre labels with lyrics counts by track id
- Remove nulls and clean malformed / non‑standard characters
- Normalize tokens using a precomputed stemming map (`stemmed_words.txt`)

### Vectorization
- `CountVectorizer(stop_words="english", max_features=1000)`
- `TfidfTransformer(sublinear_tf=True, norm="l2")`

### Model evaluation
- For unsupervised outputs, clusters/topics are **assigned a genre label** using the **most common true genre** within that cluster (majority vote).
- Reports include accuracy, per‑class precision/recall/F1, and confusion matrices.

---

## Results (from notebook runs)

### All genres (13 classes)

| Method | Accuracy |
|---|---:|
| NMF | 0.324 |
| KMeans | 0.357 |
| LDA | 0.319 |
| Logistic Regression (supervised baseline) | 0.510 |

### Reduced genres (Latin, Metal, Pop, Rap)

| Method | Accuracy |
|---|---:|
| NMF | 0.801 |
| KMeans | 0.811 |
| LDA | 0.805 |
| Logistic Regression (supervised baseline) | 0.875 |

Interpretation:
- With **all 13 genres**, the unsupervised methods land around **~0.32–0.36** accuracy.
- With a **smaller set of 4 more distinct genres**, unsupervised performance improves to **~0.80–0.81**.
- The supervised baseline is higher (especially for 4 genres), which helps contextualize the ceiling with this feature representation.

---

## Getting started

### 1) Create an environment (recommended)
```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate  # Windows PowerShell
```

### 2) Install dependencies
```bash
pip install -U pip
pip install numpy pandas scipy scikit-learn matplotlib seaborn jupyter
```

### 3) Place the dataset files
Put `msd_tagtraum_cd2c.cls`, `mxm_dataset.db`, and `stemmed_words.txt` where the notebook expects them (or update the paths in the notebook).

### 4) Run
```bash
jupyter lab
```
Open **`ML2 Project.ipynb`** and run top‑to‑bottom.

---

## Notes / limitations

- Lyrics are **bag‑of‑words** (no word order), which limits expressive power.
- Some genres naturally overlap (hybrid styles, shared vocab), making “clean” clustering difficult.
- Better results may require richer text representations (e.g., full lyrics, embeddings, BERTopic/HDBSCAN, or stronger feature engineering).

---

## Citation / credit

The notebook cites the genre/lyrics sources (Tagtraum genre annotations + Musixmatch dataset used with the Million Song Dataset).  
Please follow the original dataset licenses/terms when downloading and sharing data.
