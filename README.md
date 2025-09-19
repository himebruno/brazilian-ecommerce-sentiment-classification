# E-commerce Customer Review Sentiment Prediction

## Project Description
This project aims to build a classification model to predict whether a product will receive a positive or negative review on an e-commerce platform. The model leverages various transactional and product-related features, along with textual review content, to understand the drivers of customer satisfaction. The primary goal is to provide actionable insights for improving customer experience and product offerings.

## Dataset
The project utilizes the **Brazilian E-commerce Public Dataset by Olist**, available on Kaggle. This dataset contains information on 100k orders made at Olist Store, including customer, order, product, seller, and review data.

The following datasets are used:
- `olist_order_items_dataset.csv`: Data about items purchased within each order.
- `olist_order_reviews_dataset.csv`: Data about customer reviews.
- `olist_orders_dataset.csv`: Order status and purchase timestamps.
- `olist_products_dataset.csv`: Product information (e.g., name, category, description length, number of photos).

## Data Preprocessing and Feature Engineering
The data preprocessing phase involves cleaning and merging various datasets to create a unified view for each product purchase. Key features engineered include:

-   `delivery_time`: Time taken for order delivery.
-   `delivery_delay`: Difference between estimated and actual delivery dates.
-   `review_score`: Customer rating (1-5), which is later transformed into the target variable.
-   `combined_text`: Concatenated review titles and messages, lemmatized and accent-removed.
-   `price`: Product unit price.
-   `freight_value`: Shipping cost per item.
-   `product_category_name`: Product category.
-   `product_description_length`: Length of product description.
-   `product_photos_qty`: Number of product photos.
-   `seller_historical_score`: Seller's average rating over time, calculated to avoid temporal leakage.
-   `seller_total_sales`: Number of the seller's past sales.

Temporal leakage is carefully avoided by calculating `seller_historical_score` based only on past sales relative to the current order's purchase date.

## Target Variable
The target variable is derived from the `review_score`. To minimize ambiguity, only "safe scores" are used for initial training:
-   **Positive (1):** Review scores 4-5
-   **Negative (0):** Review scores 1-2
-   Mid-range scores (3) are held out for separate validation to analyze inconsistent sentiment.

## Model Pipeline
A scikit-learn pipeline is constructed to organize the training process. It includes:

1.  **Numeric Features Pipeline:**
    -   `SimpleImputer(strategy='mean')` for missing values.
    -   `StandardScaler()` for normalization.
2.  **Categorical Features Pipeline:**
    -   `SimpleImputer(strategy='most_frequent')` for missing values.
    -   `CatBoostEncoder()` for target-based encoding of `product_category_name` to handle high cardinality and prevent data leakage.
3.  **Text Features Pipeline:**
    -   `TfidfVectorizer()` to convert `combined_text` into numerical features, capturing unigrams and bigrams, with sublinear term frequency scaling and filtering of rare/frequent terms.
    -   Lemmatization and accent removal are applied to text features using `spaCy` (Portuguese model) to normalize words.

These pipelines are combined using a `ColumnTransformer`.

## Train-Test Split (Temporal)
Given the e-commerce context, a **temporal train-test split** is used. The dataset is sorted by `order_purchase_timestamp`, and an 80-20 split ensures the model learns from past data and is evaluated on future data, simulating a real-world scenario and preventing temporal leakage.

## Model Training and Hyperparameter Optimization
Three classification algorithms were considered:
-   **LightGBM**
-   **XGBoost**
-   **Logistic Regression**

Hyperparameter optimization is performed using **Hyperopt**, a Bayesian optimization technique. Time-series cross-validation is employed during optimization to respect the temporal order of the data and minimize error by averaging performance across folds.

## Model Evaluation
The final model is evaluated on the test set using:
-   **Classification Report:** Precision, recall, f1-score, and support.
-   **Confusion Matrix:** Visualizing true positives, true negatives, false positives, and false negatives.
-   **ROC Curve and AUC Score:** Assessing the model's ability to discriminate between positive and negative classes.

## Mid-range Score Validation
Mid-range scores (reviews with a score of 3) are inherently ambiguous. To validate the model's performance on these cases, sentence embeddings and cosine similarity are used to classify the sentiment of review texts in the holdout set. This independent classification serves as a ground truth for evaluating how the trained model predicts these nuanced reviews without introducing bias from training on derived labels.

## Model Interpretation with SHAP
**SHAP (SHapley Additive exPlanations)** is used for model interpretability, providing both global and local insights:

-   **Global Interpretation:** Identifies features with the largest overall impact on predictions across the entire dataset. Key drivers of customer satisfaction, such as positive/negative language, delivery time, and seller reputation, are highlighted.
-   **Local Interpretation:** Explains individual predictions, detailing how each feature contributes to a specific review's sentiment prediction. Waterfall plots illustrate the additive contributions of features.

## Instructions to Run the Notebook
1.  **Clone the repository**

2.  **Create a Python virtual environment and activate it:**
    ```bash
    python -m venv venv
    # On Windows:
    .\venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```
3.  **Install the required dependencies:**
    The dependencies are listed in the `requirements.txt` file. You can install them by running:
    ```bash
    pip install -r requirements.txt
    ```
    _Note: The `sentence-transformers` library and the `pt_core_news_sm` spaCy model are explicitly downloaded in the notebook, but it's good practice to ensure all dependencies are installed upfront._

4.  **Download the dataset:**
    The notebook automatically downloads the `brazilian-ecommerce` dataset from Kaggle using `kagglehub`.

5.  **Run the `main.ipynb` notebook:**
    Open the `main.ipynb` file in a Jupyter-compatible environment (e.g., Jupyter Lab, VS Code with Jupyter extension) and run all cells.
    The notebook will:
    -   Download the necessary data.
    -   Perform data preprocessing and feature engineering.
    -   Train and optimize machine learning models.
    -   Evaluate model performance.
    -   Provide insights through SHAP analysis.

    _Note: Hyperparameter optimization can take a significant amount of time (up to 50 minutes for all three models, or 8 minutes for LightGBM). You can comment out models you don't wish to train in Cell 69 to speed up the process._
