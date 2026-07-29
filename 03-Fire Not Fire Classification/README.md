- Operation Performed

    1. Data Cleaning 
    2. EDA
    3. Feature Engineering & Model Training 
    4. Tuning

---
#### Linear Regression Model

```
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

linreg = LinearRegression()                         # Creates Linreg Model
linreg.fit(X_train_scaled, y_train)                 # Using {p1} to train with {p2} target val. .fir is used to map r/s
y_pred = linreg.predict(X_test_scaled)              # {p1} is new feat for model to predict saved in y_pred

mae = mean_absolute_error(y_test, y_pred)           # {p1} is actual val, {p2} is predicted val, mae will calc mae
score = r2_score(y_test, y_pred)                    #same, r2 calc variance

print("Mean Square Error", mae)                     # The Lower the better, 0 best
print("R2 Score", score)                            # The higher the better, 1 best

plt.scatter(y_test, y_pred)
```