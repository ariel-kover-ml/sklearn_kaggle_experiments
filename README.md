# Project Overview

This project includes Jupyter Notebooks that combine Kaggle datasets with the power of Scikit-learn and PyCaret to perform classification and regression analysis.

# Kaggle Datasets

* **New York Taxi Dataset**: Used for predicting taxi fares or other relevant metrics within the new_york_taxi.ipynb and new_york_taxi_pycaret.ipynb notebooks.
* **Rain in Australia Dataset**: Utilized to develop prediction models related to rainfall in the rain_in_australia.ipynb and rain_in_australia_pycaret.ipynb notebooks.
* **Rossmann Dataset**: Employed for tasks such as sales forecasting rossmann.ipynb notebook.

# Notebooks

* **new_york_taxi.ipynb**:
Utilizes Scikit-learn for traditional machine learning approaches to analyze and model the New York Taxi Dataset.
* **new_york_taxi_pycaret.ipynb**:
Leverages PyCaret to efficiently explore, preprocess, and model the New York Taxi Dataset, streamlining the machine learning process.
* **rain_in_australia.ipynb**:
Employs Scikit-learn to create machine learning models focused on the Rain in Australia Dataset.
* **rain_in_australia_pycaret.ipynb**:
Streamlines the machine learning workflow for the Rain in Australia Dataset using PyCaret.
* **rossmann.ipynb**:
Applies Scikit-learn to address challenges pertaining to the Rossmann Store Sales dataset.

# Dependencies

* Python (version 3.10 or later)
* Scikit-learn
* PyCaret
* NumPy
* Pandas
* Matplotlib
* Seaborn
* ...

# Installation

* Use pip or conda to install the required packages (consider setting up a virtual environment):

    ```Bash
    pip install -r requirements.txt
    ```

* Rename .env_sample to .env and populate with your Dagshub credentials.

* Open the desired notebook (.ipynb).
Execute the cells in sequential order.
