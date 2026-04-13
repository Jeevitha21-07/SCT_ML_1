import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Load Dataset

data = pd.read_csv("data/train.csv")


# 2. Select Important Features

data = data[['GrLivArea', 'BedroomAbvGr', 'FullBath', 
             'OverallQual', 'GarageCars', 'SalePrice']]

# Remove missing values
data = data.dropna()


# 3. Define Features & Target

X = data[['GrLivArea', 'BedroomAbvGr', 'FullBath', 
          'OverallQual', 'GarageCars']]

y = data['SalePrice']

# 4. Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# 6. Predictions

predictions = model.predict(X_test)

# 7. Evaluation

mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("Mean Absolute Error:", mae)
print("R2 Score:", r2)


# 8. Example Prediction (NO WARNING)

sample_house = pd.DataFrame([[2000, 3, 2, 7, 2]],
                            columns=['GrLivArea', 'BedroomAbvGr', 'FullBath',
                                     'OverallQual', 'GarageCars'])

predicted_price = model.predict(sample_house)

print("Predicted Price:", predicted_price[0])

# 9. Visualization

plt.scatter(y_test, predictions)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Prices")
plt.show()