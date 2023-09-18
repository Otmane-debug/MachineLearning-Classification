import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Load the data
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

print(train_data.info())
print(train_data.descibe()
print(train.isnull().sum())
print("")
      
train = train_data.drop(['PassengerId', 'Cabin', 'Name'], axis=1)
test = test_data.drop(['PassengerId', 'Cabin', 'Name'], axis=1)

print(test_data.info())
print(test_data.descibe()
print(test.isnull().sum())

train.fillna(train[['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].mean(), inplace=True)
test.fillna(test[['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].mean(), inplace=True)

cols = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Transported']
Cols = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP']

le=LabelEncoder()
for col in cols:
    train[col] = le.fit_transform(train[col])
for col in Cols:
    test[col] = le.transform(test[col])

train = train.ffill()
test = test.bfill()

print(train.isnull().sum())
print("")
print(test.isnull().sum())

X = train.drop('Transported', axis=1)
y = train['Transported']

# Create a validation set
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=1)

pipe = make_pipeline(StandardScaler(), LogisticRegression())

pipe.fit(X_train, y_train)  # apply scaling on training data

pipe.score(X_val, y_val)  # apply scaling on testing data, without leaking training data.

# Make predictions on the validation set
predictions_val = pipe.predict(X_val)

# Calculate accuracy on the validation set
accuracy = accuracy_score(y_val, predictions_val)
print("Accuracy on validation data:", accuracy)

# Generate a classification reportAccuracy on validation data: 0.7860839562967222
class_report = classification_report(y_val, predictions_val)
print("Classification Report on validation data:\n", class_report)

# Predict on the test data
predictions_test = pipe.predict(test)

predictions_test = [True if pred ==  1 else False for pred in predictions_test]

passenger_predictions = pd.DataFrame({'PassengerId': test_data['PassengerId'], 'Transported': predictions_test})

# Now, predictions_test contains the predicted values for the test data
print("Predictions for test data:", passenger_predictions)

# Save passenger_predictions to a CSV file
passenger_predictions.to_csv('C:\\Users\\pc\\Desktop\\submission.csv', index=False)
