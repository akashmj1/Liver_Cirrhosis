from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

# Create a Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
df = pd.read_csv('cirrhosis.csv')
missing_columns = ['Drug', 'Ascites', 'Hepatomegaly',
                   'Spiders', 'Cholesterol', 'Copper',
                   'Alk_Phos', 'SGOT', 'Tryglicerides',
                   'Platelets', 'Prothrombin', 'Stage']

# Drop missing values from affected columns
df = df.dropna(subset=missing_columns)

columns = ['Status', 'Drug', 'Sex', 'Ascites', 'Hepatomegaly', 'Spiders', 'Edema', 'Stage']

for column in columns:
    print(f'The unique values of {column}: {df[column].unique()}')
df['Age'] = df['Age'] / 365.25
df['Age'] = df['Age'].astype(int)
df['Stage'] = df['Stage'].astype(int)
df['Stage'] = df['Stage'].astype(str)

df.rename(columns={'Tryglicerides': 'Triglycerides', 'Alk_Phos': 'ALP', 'SGOT': 'AST'}, inplace=True)
duplicated_rows = df[df.duplicated()]

# Group by the duplicated rows and calculate their sum
sum_of_duplicated_rows = duplicated_rows.groupby(duplicated_rows.columns.tolist()).size().reset_index(name='count')

# Define X (features) and y (target)
X = df[['Stage', 'Age', 'Bilirubin', 'Cholesterol', 'Albumin', 'Copper', 'ALP', 'AST', 'Triglycerides', 'Platelets',
        'Prothrombin']]
y = df['Status']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ----------------------------------Logistic Regression--------------------------------------
# Create a Logistic Regression classifier
lr = LogisticRegression()

# Train the classifier on the training data
lr.fit(X_train, y_train)

# Make predictions on the test data
y_pred = lr.predict(X_test)

# Evaluate the classifier's performance
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Print the evaluation results
print(f"Accuracy: {accuracy:.2f}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)

# -------------------------------------------Hyperparameter Tuning----------------------------------
# Define a grid of hyperparameters to search
param_grid = {
    'penalty': ['l1', 'l2'],  # Regularization penalty
    'C': [0.001, 0.01, 0.1, 1, 10],  # Inverse of regularization strength
    'solver': ['liblinear'],  # Solver for L1 penalty
}

# Create a grid search cross-validation object
grid_search = GridSearchCV(estimator=lr, param_grid=param_grid, cv=5)

# Fit the grid search to your data
grid_search.fit(X, y)

# Print the best hyperparameters found
print("Best Hyperparameters: ", grid_search.best_params_)

# Print the best cross-validation score
print("Best Cross-Validation Score: {:.2f}".format(grid_search.best_score_))

# You can also access the best trained model using grid_search.best_estimator_
best_logistic_regression = grid_search.best_estimator_

# Fit the model to your training data
best_logistic_regression.fit(X_train, y_train)

# Make predictions on your test data
y_pred = best_logistic_regression.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Confusion Matrix:\n", confusion)
print("Classification Report:\n", classification_rep)

# -----------------------------------------------Gradient Boosting---------------------------------------
from sklearn.ensemble import GradientBoostingClassifier

# Create a Gradient Boosting classifier
gb_classifier = GradientBoostingClassifier(n_estimators=100, random_state=42)

# Train the classifier on the training data
gb_classifier.fit(X_train, y_train)

# Make predictions on the test data
y_pred = gb_classifier.predict(X_test)

# Evaluate the classifier's performance
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {:.2f}".format(accuracy))

# ------------------------------------------------AdaBoostClassifier---------------------------------------
from sklearn.ensemble import AdaBoostClassifier
# Random Forest Classifier
rfo_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rfo_classifier.fit(X_train, y_train)
rfo_predictions = rfo_classifier.predict(X_test)
rfo_accuracy = accuracy_score(y_test, rfo_predictions)
print("Random Forest Accuracy: {:.2f}".format(rfo_accuracy))

# AdaBoost Classifier
adaboost_classifier = AdaBoostClassifier(n_estimators=50, random_state=42)
adaboost_classifier.fit(X_train, y_train)
adaboost_predictions = adaboost_classifier.predict(X_test)
adaboost_accuracy = accuracy_score(y_test, adaboost_predictions)
print("AdaBoost Accuracy: {:.2f}".format(adaboost_accuracy))
