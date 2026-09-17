# MLDiabetes

This project is a machine learning pipeline that uses a Support Vector Machine (SVM) to predict diabetes based on the PIMA Diabetes Dataset.  

- The notebook imports numpy and pandas for data manipulation, alongside scikit-learn modules for preprocessing, splitting, modeling, and evaluation.
- It loads a dataset of 768 patient records with 8 medical predictor variables (such as Pregnancies, Glucose, BloodPressure, BMI, and Age) and 1 target variable (Outcome). The data shows a class imbalance, containing 500 non-diabetic cases and 268 diabetic cases. Grouping the data by outcome reveals that individuals with diabetes have higher average values across features like Glucose (141.25 vs 109.98) and Age (37.06 vs 31.19).
- The predictor variables are separated from the target labels. Because SVMs are sensitive to the scale of input features, the project applies StandardScaler to standardize the dataset.
- The standardized data is split into an 80% training set (614 records) and a 20% test set (154 records). Stratified sampling is used to maintain the ratio of diabetic to non-diabetic cases in both splits. The model utilized is an SVM classifier configured with a linear kernel.  
- The trained model achieves an accuracy of approximately 78.66% on the training data and 77.27% on the testing data, indicating a consistent fit without severe overfitting.
- The notebook concludes by building an inference block that accepts a raw tuple of patient data, reshapes it as a NumPy array, scales it using the previously fitted StandardScaler, and outputs a classification of "diabetic" or "not diabetic".
