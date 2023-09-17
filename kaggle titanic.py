import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load the data
data_1 = pd.read_csv("train.csv")
data_2 = pd.read_csv("test.csv")

# Separate features (X) and target variable (y)
X_train = data_1.dropna().drop_duplicates()  # Assuming you want to drop rows with missing values
X_test = data_2.dropna().drop_duplicates()
y_train = X_train['Transported']  # Replace 'target_column' with the actual name of your target variable

features = ['PassengerId', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'CryoSleep', 'VIP']

# Create a validation set
X_train, X_val, y_train, y_val = train_test_split(X_train[features], y_train, test_size=0.2, random_state=1)

# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the validation set
predictions_val = model.predict(X_val)

# Calculate accuracy on the validation set
accuracy = accuracy_score(y_val, predictions_val)
print("Accuracy on validation data:", accuracy)

# Generate a classification report
class_report = classification_report(y_val, predictions_val)
print("Classification Report on validation data:\n", class_report)

# Predict on the test data
predictions_test = model.predict(X_test[features])

# Now, create a DataFrame with PassengerId and predictions
passenger_predictions = pd.DataFrame({'PassengerId': X_test['PassengerId'], 'Predicted_Transported': predictions_test})

# Now, predictions_test contains the predicted values for the test data
print("Predictions for test data:", predictions_test)

# Save passenger_predictions to a CSV file
passenger_predictions.to_csv('submition.csv', index=False)