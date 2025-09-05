# Predicting Corporate Bankruptcies using Financial Statement Ratios and Machine Learning

**Overview:**

This project aims to develop a predictive model for corporate bankruptcies using machine learning techniques and financial statement data.  The analysis leverages key financial ratios derived from balance sheets and income statements to identify companies at a high risk of bankruptcy. This allows for proactive risk management strategies for lenders and investors.  The model's performance is evaluated using appropriate metrics, and the results are presented through both printed analysis and visualizations.

**Technologies Used:**

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn

**How to Run:**

1. **Install Dependencies:**  Ensure you have Python 3 installed.  Then, install the required Python libraries listed in `requirements.txt` using the following command:

   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Main Script:** Execute the main script using:

   ```bash
   python main.py
   ```

**Example Output:**

The script will print key performance indicators and model evaluation metrics to the console.  Additionally, it will generate several visualization files (e.g., plots showing the distribution of key financial ratios, model performance curves) in the `output` directory.  These files will provide a visual representation of the data analysis and model performance.  The specific output files generated may vary depending on the model and analysis performed.  Examples include:

* Console output detailing model accuracy, precision, recall, F1-score etc.
* `output/ratio_distribution.png` (Example visualization - actual filename may differ)


**Data:**

The project utilizes financial statement data.  While the specific dataset used may not be included in this repository for confidentiality reasons, the code is structured to be easily adaptable to other similar datasets.  Detailed information on data pre-processing and feature engineering is included in the code comments and documentation.

**Further Development:**

Future improvements could include exploring different machine learning algorithms, incorporating additional financial indicators, and refining feature engineering techniques to enhance model accuracy.  Hyperparameter tuning and cross-validation could also be further explored to optimize model performance.