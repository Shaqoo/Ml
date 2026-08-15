from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from typing import Any
import joblib

import numpy as np
import pandas as pd

# X = np.array([[10], [20], [30], [40]])

# y = np.array([100, 200, 300, 400])

# model = LinearRegression()

# model.fit(X, y)

# match = model.predict(np.array([[50]]))

# print(match)

# scalar = StandardScaler()

# scalar.fit(X)

# print(scalar.mean_)
# print(scalar.scale_)

# print(scalar.transform(np.array([[50]])))
#50 sits [[2.23606798]] standard deviations above the mean of the training data(25). The mean of the training data is 25, and the standard deviation is 11.18033989. Therefore, (50 - 25) / 11.18033989 = 2.23606798.


# X = np.array(range(1, 101))

# X_train_man = X[:81]
# X_test_man = X[80:]

# X_train, X_test = train_test_split(X, test_size=0.2, shuffle=False)

# print(X_train)
# print(X_test)

# print(X_train_man)
# print(X_test_man)

# X_train = np.array([[1], [2], [3], [4], [5], [6]])
# y_train = np.array([0, 0, 0, 1, 1, 1])

# pipeline = make_pipeline(StandardScaler(), LinearRegression())

# pipeline.fit(X_train, y_train)

# print(pipeline.predict(np.array([[7]])))
# print(pipeline.predict(np.array([[3.5]])))


# X_train = np.array([[2], [4], [6], [8], [10], [12]])
# y_train = np.array([1, 2, 2, 3, 4, 5])

# pipeline = make_pipeline(StandardScaler(), LinearRegression())

# pipeline.fit(X_train, y_train)

# print(pipeline.predict(np.array([[14]])))
# print(pipeline.predict(np.array([[7]])))

# X_train = np.array([[5], [10], [15], [20], [25], [30]])
# y_train = np.array([3, 5, 4, 7, 8, 10])

# pipeline = make_pipeline(StandardScaler(), LinearRegression())

# pipeline.fit(X_train, y_train)

# print(pipeline.predict(np.array([[35]])))
# print(pipeline.predict(np.array([[18]])))


# df = pd.DataFrame({"Color": ["Red", "Green", "Blue", "Orange", "Purple"]})

# encoder = OneHotEncoder(sparse_output=True)
# encoded = encoder.fit_transform(df[["Color"]])

# print(encoder.categories_)
# print(encoded)
# print(encoded.tolist())


# data = fetch_california_housing()
# X, y = fetch_california_housing(return_X_y=True, as_frame=True)

# pipeline = make_pipeline(StandardScaler(), LinearRegression())

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# pipeline.fit(X_train, y_train)

# y_Pred = pipeline.predict(X_test)


# print(f"R2 Score on Training Set: {pipeline.score(X_train, y_train):.4f}")
# print('R2 score on test set:', pipeline.score(X_test, y_test))
# print('Mean Squared Error on test set:', mean_squared_error(y_test, y_Pred))
# print('Mean Absolute Error on test set:', mean_absolute_error(y_test, y_Pred))


# pipe = make_pipeline(StandardScaler(), HistGradientBoostingRegressor(random_state=42))
# pipe.fit(X_train, y_train)

# # Predictions
# y_prediction = pipe.predict(X_test)

# # Calculate new metrics
# train_r2 = pipe.score(X_train, y_train)
# test_r2 = pipe.score(X_test, y_test)
# mse = mean_squared_error(y_test, y_prediction)
# mae = mean_absolute_error(y_test, y_prediction)

# # Print results
# print(f"New R2 Score on Training Set: {train_r2:.4f}")
# print(f"New R2 Score on Test Set: {test_r2:.4f}")
# print(f"New Mean Squared Error (MSE): {mse:.4f}")
# print(f"New Mean Absolute Error (MAE): {mae:.4f}")

# import numpy as np

# from sklearn.model_selection import train_test_split
# from sklearn.pipeline import make_pipeline
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LinearRegression

# # ----------------------------------
# # 1. Create the dataset
# # ----------------------------------
# # Features (House Size in square meters)
# X = np.array([
#     [50],
#     [60],
#     [70],
#     [80],
#     [90],
#     [100],
#     [110],
#     [120],
#     [130],
#     [140]
# ])

# # Target (House Price in thousands of dollars)
# y = np.array([100, 120, 140, 160, 180, 200, 220, 240, 260, 280])

# # ----------------------------------
# # 2. Split into Training and Testing
# # ----------------------------------
# X_train, X_test, y_train, y_test = train_test_split(
#     X,
#     y,
#     test_size=0.2,
#     random_state=42
# )

# # ----------------------------------
# # 3. Build a Pipeline
# # Step 1 -> Scale the data
# # Step 2 -> Train Linear Regression
# # ----------------------------------
# pipe = make_pipeline(
#     StandardScaler(),
#     LinearRegression()
# )

# # ----------------------------------
# # 4. Train the pipeline
# # ----------------------------------
# pipe.fit(X_train, y_train)

# print(f"Model trained. Coefficients: {pipe.named_steps['linearregression'].coef_}, Intercept: {pipe.named_steps['linearregression'].intercept_}")


# # ----------------------------------
# # 5. Make predictions
# # ----------------------------------
# prediction = pipe.predict([[95]])

# print("Predicted price:", prediction)

# # ----------------------------------
# # 6. Evaluate the model
# # ----------------------------------
# score = pipe.score(X_test, y_test)

# print("R² Score:", score)

# X = np.array([[1], [2], [3], [4], [5], [6]])
# y = np.array([0, 0, 0, 1, 1, 1])

# model = LinearRegression()

# model.fit(X, y)

# print(f"Model trained. Coefficients: {model.coef_}, Intercept: {model.intercept_}")

# match = model.predict(np.array([[7]]))

# print(match)

 
# X = np.array([[0], [1], [2]])
# y = np.array([0, 1, 4])
 
# pip_deg_One = make_pipeline(PolynomialFeatures(degree=1), LinearRegression())

# pip_deg_One.fit(X, y)

# print("R2 Score: ",r2_score(y, pip_deg_One.predict(X)))
# print("Mean Squared Error: ",mean_squared_error(y, pip_deg_One.predict(X)))
# print("Mean Absolute Error: ",mean_absolute_error(y, pip_deg_One.predict(X)))


# pip_deg_Two = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())

# pip_deg_Two.fit(X, y)

# print("R2 Score: ",r2_score(y, pip_deg_Two.predict(X)))
# print("Mean Squared Error: ",np.mean((y - pip_deg_Two.predict(X)) ** 2))
# print("Mean Absolute Error: ",np.mean(np.abs(y - pip_deg_Two.predict(X))))


# cal_data: Any = fetch_california_housing()
# housing_dataframe = pd.DataFrame(data=cal_data.data, columns=cal_data.feature_names)
# housing_dataframe['MedHouseVal'] = cal_data.target

# X_train, X_test, y_train, y_test = train_test_split(
#     housing_dataframe[["MedInc"]], 
#     housing_dataframe['MedHouseVal'], 
#     test_size=0.1, 
#     random_state=42
# )


# lin = make_pipeline(StandardScaler(), LinearRegression())
# lin.fit(X_train, y_train)
# joblib.dump(lin, 'linear_regression_model.pkl')

# loaded_lin_Model = joblib.load('linear_regression_model.pkl')

# lin_pred = loaded_lin_Model.predict(X_test)

# print("--- Linear Regression (Degree 1) ---")
# print("R2 Score: ", r2_score(y_test, lin_pred))
# print("Mean Absolute Error: ", mean_absolute_error(y_test, lin_pred))
# print("Mean Squared Error: ", mean_squared_error(y_test, lin_pred))
# print("Root Mean Squared Error: ", np.sqrt(mean_squared_error(y_test, lin_pred)))



# poly = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
# poly.fit(X_train, y_train)
# poly_pred = poly.predict(X_test)

# print("\n--- Polynomial Regression (Degree 2) ---")

# print("R2 Score: ", poly.score(X_test, y_test))

# print("Mean Absolute Error: ", mean_absolute_error(y_test, poly_pred))
# print("Mean Squared Error: ", mean_squared_error(y_test, poly_pred))
# print("Root Mean Squared Error: ", np.sqrt(mean_squared_error(y_test, poly_pred)))

# plt.scatter(X_test, y_test, color='blue', label='Actual Data')
# plt.scatter(X_test, lin_pred, color='red', label='Linear Regression Predictions')
# plt.scatter(X_test, poly_pred, color='green', label='Polynomial Regression Predictions')
# plt.xlabel('MedInc')
# plt.ylabel('MedHouseVal')
# plt.legend()
# plt.show()


# X = np.array([[0], [10], [3], [8]])
# y = np.array([0, 1, 0, 1]) 

# # model = LogisticRegression()

# # model.fit(X,y)

# # print(model.predict_proba(np.array([[5]])))
# # print(model.predict(np.array([[5]])))

# tree_clf = DecisionTreeClassifier(max_depth=2)

# tree_clf.fit(X, y)

# plt.figure(figsize=(8, 6))
# plot_tree(tree_clf, filled=True, feature_names=["Feature"], class_names=["Class 0", "Class 1"])
# plt.show()


# df = pd.DataFrame({
#     "Hours_Studied": [1,2,3,4,5,6,7,8,2,5,7,9,4,6,3,8,5,9,1,10],
#     "Attendance":    [55,60,65,70,80,85,90,95,50,78,88,96,72,82,68,91,76,98,58,99],
#     "Assignments":   [1,2,2,3,4,5,5,6,1,4,5,6,3,5,2,6,4,6,1,6],
#     "Pass":          [0,0,0,0,1,1,1,1,0,1,1,1,0,1,0,1,1,1,0,1]
# })

# print(df)

# X = df[["Hours_Studied","Attendance","Assignments"]]
# y = df["Pass"]

# X_train, Xtest, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

# decision_tree = DecisionTreeClassifier(max_depth=3)

# decision_tree.fit(X_train,y_train)

# y_pred = decision_tree.predict(Xtest)

# acc_score = accuracy_score(y_test,y_pred)

# print("Accuracy: ",acc_score)
# print("Predictions: ",y_pred)
# print("Actual Values: ",y_test)
# print("Feature Used:", decision_tree.tree_.feature)
# print("Thresholds:", decision_tree.tree_.threshold)
# print("Number of Nodes:", decision_tree.tree_.node_count)
# print("Max Depth:", decision_tree.tree_.max_depth)

# plt.figure(figsize=(15,8))

# plot_tree(decision_tree, filled=True, feature_names=X.columns.tolist(), class_names=["Fail", "Pass"])

# plt.show()

# new_student = pd.DataFrame({
#     "Hours_Studied": [6],
#     "Attendance": [84],
#     "Assignments": [5]
# })

# print(decision_tree.predict(new_student))
# print(decision_tree.predict_proba(new_student))




from sklearn.datasets import make_classification

# X, y = make_classification(
#     n_samples=5000,
#     n_features=6,
#     n_informative=5,
#     n_redundant=1,
#     n_classes=2,
#     random_state=42
# )

# df = pd.DataFrame(X, columns=[
#     "Age",
#     "Income",
#     "CreditScore",
#     "PreviousPurchases",
#     "TimeOnWebsite",
#     "PagesVisited"
# ])

# df.info()
# print(df.head())

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# scalar = StandardScaler()
# X_train = scalar.fit_transform(X_train)
# X_test = scalar.transform(X_test)

# model = LogisticRegression()
# model.fit(X_train, y_train)

# y_pred = model.predict(X_test)
# probabilities = model.predict_proba(X_test)

# print("Accuracy Logistic Regression:", accuracy_score(y_test, y_pred))
# print("Precision Logistic Regression:", precision_score(y_test, y_pred))
# print("Recall Logistic Regression:", recall_score(y_test, y_pred))
# print("F1 Score Logistic Regression:", f1_score(y_test, y_pred))
# print("Probabilities Logistic Regression:", probabilities)
# print("Predictions Logistic Regression:", y_pred)
# print("Actual Values Logistic Regression:", y_test)
# print("--- Confusion Matrix Logistic Regression ---")
# print(confusion_matrix(y_test, y_pred))

# print("\n--- Classification Report Logistic Regression ---")
# print(classification_report(y_test, y_pred))


# tree = DecisionTreeClassifier()
# tree.fit(X_train, y_train)

# y_pred = tree.predict(X_test)
# probabilities = tree.predict_proba(X_test)

# print("Accuracy Decision Tree:", accuracy_score(y_test, y_pred))
# print("Precision Decision Tree:", precision_score(y_test, y_pred))
# print("Recall Decision Tree:", recall_score(y_test, y_pred))
# print("F1 Score Decision Tree:", f1_score(y_test, y_pred))
# print("Probabilities Decision Tree:", probabilities)
# print("Predictions Decision Tree:", y_pred)
# print("Actual Values Decision Tree:", y_test)
# print("--- Confusion Matrix Decision Tree ---")
# print(confusion_matrix(y_test, y_pred))

# print("\n--- Classification Report Decision Tree ---")
# print(classification_report(y_test, y_pred))

# shallow_tree = DecisionTreeClassifier(max_depth=3)
# shallow_tree.fit(X_train, y_train)

# y_pred = shallow_tree.predict(X_test)
# probabilities = shallow_tree.predict_proba(X_test)  

# print("Accuracy Shallow Decision Tree:", accuracy_score(y_test, y_pred))
# print("Precision Shallow Decision Tree:", precision_score(y_test, y_pred))
# print("Recall Shallow Decision Tree:", recall_score(y_test, y_pred))
# print("F1 Score Shallow Decision Tree:", f1_score(y_test, y_pred))
# print("Probabilities Shallow Decision Tree:", probabilities)
# print("Predictions Shallow Decision Tree    :", y_pred)
# print("Actual Values Shallow Decision Tree   :", y_test)    

# print("--- Confusion Matrix Shallow Decision Tree ---")
# print(confusion_matrix(y_test, y_pred))
# print("\n--- Classification Report Shallow Decision Tree ---")
# print(classification_report(y_test, y_pred))

# forest = RandomForestClassifier(n_estimators=300,
#     random_state=42, max_depth=None)
# forest.fit(X_train, y_train)

# y_pred = forest.predict(X_test)
# probabilities = forest.predict_proba(X_test)

# print("Accuracy Random Forest:", accuracy_score(y_test, y_pred))
# print("Precision Random Forest:", precision_score(y_test, y_pred))
# print("Recall Random Forest:", recall_score(y_test, y_pred))
# print("F1 Score Random Forest:", f1_score(y_test, y_pred))
# print("Probabilities Random Forest:", probabilities)
# print("Predictions Random Forest:", y_pred)
# print("Actual Values Random Forest:", y_test)
# print("--- Confusion Matrix Random Forest ---")
# print(confusion_matrix(y_test, y_pred))
# print("\n--- Classification Report Random Forest ---")
# print(classification_report(y_test, y_pred))

# shallow_forest = RandomForestClassifier(n_estimators=300,
#     random_state=42, max_depth=3)
# shallow_forest.fit(X_train, y_train)

# y_pred = shallow_forest.predict(X_test)
# probabilities = shallow_forest.predict_proba(X_test)

# print("Accuracy Shallow Random Forest:", accuracy_score(y_test, y_pred))
# print("Precision Shallow Random Forest:", precision_score(y_test, y_pred))  
# print("Recall Shallow Random Forest:", recall_score(y_test, y_pred))
# print("F1 Score Shallow Random Forest:", f1_score(y_test, y_pred))
# print("Probabilities Shallow Random Forest:", probabilities)
# print("Predictions Shallow Random Forest:", y_pred)
# print("Actual Values Shallow Random Forest:", y_test)
# print("--- Confusion Matrix Shallow Random Forest ---")
# print(confusion_matrix(y_test, y_pred))
# print("\n--- Classification Report Shallow Random Forest ---")
# print(classification_report(y_test, y_pred))

# gradient_boost = GradientBoostingClassifier(n_estimators=300,
#     random_state=42, max_depth=None)
# gradient_boost.fit(X_train, y_train)    
# y_pred = gradient_boost.predict(X_test)
# probabilities = gradient_boost.predict_proba(X_test)

# print("Accuracy Gradient Boosting:", accuracy_score(y_test, y_pred))
# print("Precision Gradient Boosting:", precision_score(y_test, y_pred))
# print("Recall Gradient Boosting:", recall_score(y_test, y_pred))
# print("F1 Score Gradient Boosting:", f1_score(y_test, y_pred))
# print("Probabilities Gradient Boosting:", probabilities)
# print("Predictions Gradient Boosting:", y_pred)
# print("Actual Values Gradient Boosting:", y_test)
# print("--- Confusion Matrix Gradient Boosting ---")
# print(confusion_matrix(y_test, y_pred))  
# print("\n--- Classification Report Gradient Boosting ---")
# print(classification_report(y_test, y_pred))


# df1 = pd.DataFrame([{
#     'Age': 25,
#     'Income': 45000,
#     'CreditScore': 650,
#     'PreviousPurchases': 4,
#     'TimeOnWebsite': 12,
#     'PagesVisited': 18
# }])

# # DataFrame 2: Second user profile
# df2 = pd.DataFrame([{
#     'Age': 48,
#     'Income': 90000,
#     'CreditScore': 790,
#     'PreviousPurchases': 10,
#     'TimeOnWebsite': 35,
#     'PagesVisited': 42
# }])

# # DataFrame 3: Third user profile
# df3 = pd.DataFrame([{
#     'Age': 19,
#     'Income': 18000,
#     'CreditScore': 520,
#     'PreviousPurchases': 0,
#     'TimeOnWebsite': 2,
#     'PagesVisited': 3
# }])

# logistic_pred1 = model.predict(df1)
# logistic_pred2 = model.predict(df2)
# logistic_pred3 = model.predict(df3)

# tree_pred1 = tree.predict(df1)
# tree_pred2 = tree.predict(df2)
# tree_pred3 = tree.predict(df3)

# shallow_tree_pred1 = shallow_tree.predict(df1)
# shallow_tree_pred2 = shallow_tree.predict(df2)  
# shallow_tree_pred3 = shallow_tree.predict(df3)

# forest_pred1 = forest.predict(df1)
# forest_pred2 = forest.predict(df2)
# forest_pred3 = forest.predict(df3)

# shallow_forest_pred1 = shallow_forest.predict(df1)
# shallow_forest_pred2 = shallow_forest.predict(df2)
# shallow_forest_pred3 = shallow_forest.predict(df3)

# gradient_boost_pred1 = gradient_boost.predict(df1)
# gradient_boost_pred2 = gradient_boost.predict(df2)
# gradient_boost_pred3 = gradient_boost.predict(df3)

# print("Logistic Regression Predictions:", logistic_pred1, logistic_pred2, logistic_pred3)
# print("Decision Tree Predictions:", tree_pred1, tree_pred2, tree_pred3)
# print("Shallow Decision Tree Predictions:", shallow_tree_pred1, shallow_tree_pred2, shallow_tree_pred3)
# print("Random Forest Predictions:", forest_pred1, forest_pred2, forest_pred3)
# print("Shallow Random Forest Predictions:", shallow_forest_pred1, shallow_forest_pred2, shallow_forest_pred3)
# print("Gradient Boosting Predictions:", gradient_boost_pred1, gradient_boost_pred2, gradient_boost_pred3)

# logistic_pred1_proba = model.predict_proba(df1)
# logistic_pred2_proba = model.predict_proba(df2)
# logistic_pred3_proba = model.predict_proba(df3)

# tree_pred1_proba = tree.predict_proba(df1)
# tree_pred2_proba = tree.predict_proba(df2)
# tree_pred3_proba = tree.predict_proba(df3)

# shallow_tree_pred1_proba = shallow_tree.predict_proba(df1)
# shallow_tree_pred2_proba = shallow_tree.predict_proba(df2)
# shallow_tree_pred3_proba = shallow_tree.predict_proba(df3)

# forest_pred1_proba = forest.predict_proba(df1)
# forest_pred2_proba = forest.predict_proba(df2)
# forest_pred3_proba = forest.predict_proba(df3)

# shallow_forest_pred1_proba = shallow_forest.predict_proba(df1)
# shallow_forest_pred2_proba = shallow_forest.predict_proba(df2)
# shallow_forest_pred3_proba = shallow_forest.predict_proba(df3)

# gradient_boost_pred1_proba = gradient_boost.predict_proba(df1)
# gradient_boost_pred2_proba = gradient_boost.predict_proba(df2)
# gradient_boost_pred3_proba = gradient_boost.predict_proba(df3)

# print("Logistic Regression Probabilities:", logistic_pred1_proba, logistic_pred2_proba, logistic_pred3_proba)
# print("Decision Tree Probabilities:", tree_pred1_proba, tree_pred2_proba, tree_pred3_proba)
# print("Shallow Decision Tree Probabilities:", shallow_tree_pred1_proba, shallow_tree_pred2_proba, shallow_tree_pred3_proba)
# print("Random Forest Probabilities:", forest_pred1_proba, forest_pred2_proba, forest_pred3_proba)
# print("Shallow Random Forest Probabilities:", shallow_forest_pred1_proba, shallow_forest_pred2_proba, shallow_forest_pred3_proba)
# print("Gradient Boosting Probabilities:", gradient_boost_pred1_proba, gradient_boost_pred2_proba, gradient_boost_pred3_proba)

