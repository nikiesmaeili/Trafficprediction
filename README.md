
#  Machine Learning for Traffic Prediction in SDN

This project explores how machine learning (ML) can enhance Software-Defined Networking (SDN) by improving traffic classification and network optimization. Conducted as part of the ENGG*6600 course at the University of Guelph, the study evaluates both traditional and deep learning approaches to identify traffic patterns and improve network management.

## Overview

Software-Defined Networking separates the control plane from the data plane, enabling more flexible and programmable network infrastructures. By integrating ML techniques, we aim to intelligently classify and manage traffic, especially under encrypted and dynamic network conditions.

## Dataset

- **Source**: Kaggle (CSV format)
- **Size**: ~21,000 entries
- **Attributes**: IP addresses, number of flows, dates, Autonomous System Numbers (ASNs)

##  Methodology

1. **Data Preprocessing**
   - Handled missing values with column means
   - Standardized numerical features
   - Encoded categorical values
   - Detected and handled outliers
   - Dataset split into training and test sets

2. **Data Visualization**
   - Scatter plots, line plots, bar charts, and heatmaps
   - Visual exploration of destination IP distribution, flow duration, and classification matrices

3. **Clustering**
   - Applied KMeans for unsupervised grouping of traffic into clusters

4. **Classification Models**
   - **K-Nearest Neighbors (KNN)**: Simple, interpretable, and effective
   - **Gated Recurrent Unit (GRU)**: RNN variant used for sequential pattern analysis

## Results

| Model | Accuracy  | Recall    | Precision |
|-------|-----------|-----------|-----------|
| KNN   | 97.91%    | 99.36%    | 98.01%    |
| GRU   | **99.65%**| **99.82%**| **99.75%**|

##  Libraries Used

- Python
- NumPy, Pandas
- TensorFlow / Keras
- Scikit-learn
- Matplotlib, Seaborn

## Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/traffic-prediction-ml.git
   cd traffic-prediction-ml
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the models:
   ```bash
   python knn_classifier.py
   python gru_model.py
   ```

##  Future Work

- Integrate real-time traffic streaming support
- Apply models to larger-scale enterprise-level datasets
- Explore explainable AI (XAI) for network anomaly interpretation

##  Contributors

- **Niki Esmaeili** – School of Engineering, University of Guelph  
- **Janice Austin** – School of Engineering, University of Guelph

##  License

This project is for educational use. Please credit the authors if used in academic or research settings.
