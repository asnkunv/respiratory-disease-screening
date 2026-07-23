# 🫁 AI-Based Respiratory Disease Screening Using Cough Audio

An end-to-end machine learning and MLOps project that analyzes cough audio recordings to estimate respiratory disease risk indicators. The project combines classical machine learning, deep learning, and production software engineering practices to build a scalable web application where users can record or upload cough sounds and receive AI-generated screening predictions.

> ⚠️ **Medical Disclaimer:**  
> This project is developed for research and educational purposes only. The model predictions are not medical diagnoses and should not replace professional healthcare evaluation.

---

# 📖 Project Overview

Respiratory diseases such as COVID-19, tuberculosis, pneumonia, and bronchitis affect millions of people worldwide. Traditional diagnosis often requires clinical visits, specialized equipment, and laboratory testing. Recent advances in artificial intelligence have shown that cough sounds contain acoustic patterns that may provide useful information for respiratory screening.

This project explores whether machine learning and deep learning models can identify respiratory risk indicators from cough audio recordings.

The final goal is to develop a production-ready AI application where users can:

1. Record a cough using a web application.
2. Upload the audio sample.
3. Automatically preprocess and analyze the recording.
4. Run machine learning inference.
5. Receive an estimated respiratory risk prediction.

The project focuses on the complete AI lifecycle:

- Data collection and preprocessing
- Exploratory data analysis
- Audio feature engineering
- Machine learning experimentation
- Deep learning development
- Model evaluation
- Backend API development
- ML deployment architecture
- Future MLOps implementation

---

# 🎯 Objectives and Goals

## Machine Learning Goals

- Develop AI models capable of extracting meaningful information from cough recordings.
- Compare classical machine learning algorithms with deep learning approaches.
- Build robust preprocessing pipelines for noisy real-world audio data.
- Evaluate model performance using appropriate classification metrics.
- Investigate generalization across multiple respiratory datasets.

## Software Engineering Goals

- Design a scalable full-stack AI application.
- Create an API-based inference system.
- Separate frontend, backend, and ML services.
- Containerize services for deployment.
- Apply production software engineering practices.

## MLOps Goals

- Create reproducible training pipelines.
- Enable model versioning.
- Monitor model performance.
- Prepare the system for cloud deployment.

---

# 🏗 System Architecture (Planned)

The final application architecture follows a production ML system design:

```text
                         User
                           |
                           v
              +---------------------------+
              |   React Web Application   |
              |                           |
              | - Record cough audio      |
              | - Upload audio files      |
              | - Display predictions     |
              +---------------------------+
                           |
                           v
              +---------------------------+
              |     FastAPI Backend       |
              |                           |
              | - REST API endpoints      |
              | - Authentication          |
              | - Request validation      |
              | - Manage inference jobs   |
              +---------------------------+
                           |
                           v
              +---------------------------+
              |  Audio Processing Service |
              |                           |
              | - Audio validation        |
              | - Resampling              |
              | - Noise reduction         |
              | - Segmentation            |
              | - Feature extraction      |
              | - Mel-spectrogram         |
              +---------------------------+
                           |
                           v
              +---------------------------+
              |   ML Inference Service    |
              |                           |
              | - Load trained model      |
              | - Run prediction          |
              | - Generate confidence     |
              | - Return classification   |
              +---------------------------+
                           |
                           |
              +------------+------------+
              |                         |
              v                         v

+---------------------+       +----------------------+
|   Model Storage     |       |  Prediction Storage  |
|                     |       |                      |
| - Trained models    |       | - User requests      |
| - Model versions    |       | - Results            |
| - Feature configs   |       | - Metadata           |
+---------------------+       +----------------------+

                           |
                           v

              +---------------------------+
              |   MLOps Pipeline          |
              |                           |
              | - Data ingestion          |
              | - Data validation         |
              | - Model training          |
              | - Model evaluation        |
              | - Model registry          |
              | - CI/CD deployment        |
              | - Model monitoring        |
              +---------------------------+
                           |
                           v

              +---------------------------+
              | Cloud Infrastructure      |
              |                           |
              | - Docker containers       |
              | - AWS/GCP/Azure           |
              | - Load balancing          |
              | - Logging                 |
              | - Monitoring              |
              +---------------------------+
```

---

# 🛠 Technology Stack

## Machine Learning

- Python
- Scikit-learn
- PyTorch
- XGBoost
- LightGBM
- CatBoost

## Audio Processing

- Librosa
- NumPy
- SciPy

## Data Analysis

- Pandas
- Matplotlib
- Seaborn

## Backend (Planned)

- FastAPI
- REST API
- Pydantic

## Frontend (Planned)

- React
- TypeScript

## Deployment / MLOps (Planned)

- Docker
- CI/CD pipelines
- Model registry
- Monitoring
- Cloud deployment

---

# 📂 Datasets

This project uses multiple publicly available respiratory audio datasets to improve model robustness and evaluate performance across different recording conditions.

Because respiratory audio data varies significantly depending on:

- Recording devices
- Patient demographics
- Background noise
- Disease type
- Recording environment

multiple datasets are used to create a more realistic machine learning pipeline.

---

# 1. COUGHVID V3 Dataset

Source:

https://www.kaggle.com/datasets/orvile/coughvid-v3

The COUGHVID dataset is a large-scale crowdsourced cough audio dataset containing recordings collected from individuals worldwide.

### Usage:

- Primary cough classification dataset
- Feature engineering experiments
- Classical ML model training
- Deep learning experiments

### Processing:

- Audio cleaning
- Label filtering
- Feature extraction
- Data normalization

---

# 2. CODA TB DREAM Challenge Dataset

Source:

https://www.synapse.org/Synapse:syn31472953/wiki/619711

The CODA TB DREAM Challenge dataset contains cough recordings collected for tuberculosis screening research.

### Usage:

- Testing model robustness
- Studying real-world cough recordings
- Evaluating domain differences

### Challenges:

- Different recording devices
- Environmental noise
- Real-world audio quality variation

---

# 3. Pediatric Bronchitis and Pneumonia Cough Dataset

Source:

https://figshare.com/articles/dataset/Data_from_A_classification_framework_for_identifying_bronchitis_and_pneumonia_in_children_based_on_a_small-scale_cough_sounds_dataset/21176197

This dataset contains cough recordings from pediatric patients with respiratory conditions including bronchitis and pneumonia.

### Usage:

- Disease-specific cough analysis
- Model validation
- Studying respiratory disease patterns

---

# 🔬 Machine Learning Methodology

## 1. Exploratory Data Analysis

Performed analysis on:

- Dataset distribution
- Class imbalance
- Audio duration
- Recording quality
- Feature relationships

Visualizations include:

- Class distribution plots
- Audio waveform analysis
- Spectrogram visualization
- Feature distributions

---

# 2. Audio Preprocessing

The preprocessing pipeline includes:

- Audio loading
- Resampling
- Noise analysis
- Normalization
- Segmentation
- Feature extraction

Real-world audio challenges considered:

- Background noise
- Compression artifacts
- Different microphones
- Recording environments

---

# 3. Feature Engineering

Extracted audio-based features including:

- MFCCs
- Spectral centroid
- Spectral bandwidth
- Spectral rolloff
- Zero crossing rate
- Other acoustic characteristics

These features are used for classical machine learning models.

---

# 🤖 Classical Machine Learning Models

The following models were implemented and evaluated:

## Individual Models

- Logistic Regression
- Random Forest
- Support Vector Machine (RBF)
- XGBoost
- LightGBM
- CatBoost
- Gradient Boosting

## Ensemble Model

A Voting Classifier was developed using:

```python
[
 GradientBoostingClassifier(random_state=42),

 RandomForestClassifier(
     n_estimators=300,
     class_weight="balanced",
     random_state=42
 ),

 SVC(
     kernel="rbf",
     class_weight="balanced",
     probability=True,
     random_state=42
 ),

 CatBoostClassifier(
     iterations=300,
     random_state=42,
     verbose=0
 )
]
    
