
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load Dataset
df = pd.read_csv("Superstore Sales.csv", encoding="latin1")

# Convert Date Column
df["Order Date"] = pd.to_datetime(df["Order Date"])

# Sort Dates
df = df.sort_values("Order Date")

# Monthly Sales Aggregation
monthly_sales = df.groupby(
    pd.Grouper(key="Order Date", freq="ME")
)["Sales"].sum().reset_index()

# Create Numeric Index
monthly_sales["Month_No"] = np.arange(len(monthly_sales))

# Training Data
X = monthly_sales[["Month_No"]]
y = monthly_sales["Sales"]

# Train Model
model = LinearRegression()
model.fit(X, y)

# Predict Next 12 Months
future_months = 12

future_index = pd.DataFrame({
    "Month_No": np.arange(
        len(monthly_sales),
        len(monthly_sales) + future_months
    )
})

future_predictions = model.predict(future_index)

# Future Dates
future_dates = pd.date_range(
    start=monthly_sales["Order Date"].max() + pd.offsets.MonthEnd(1),
    periods=future_months,
    freq="ME"
)

# Save Predictions
forecast_df = pd.DataFrame({
    "Date": future_dates,
    "Predicted Sales": np.round(future_predictions, 2)
})

forecast_df.to_csv(
    "Future_Sales_Predictions.csv",
    index=False
)

print("\nForecast saved successfully!\n")
print(forecast_df)

# Plot Graph
plt.figure(figsize=(14, 7))

plt.plot(
    monthly_sales["Order Date"],
    monthly_sales["Sales"],
    marker="o",
    linewidth=2,
    label="Historical Sales"
)

plt.plot(
    future_dates,
    future_predictions,
    marker="o",
    linestyle="--",
    linewidth=2,
    label="Forecast"
)

plt.title("Sales Forecasting", fontsize=18)
plt.xlabel("Date", fontsize=14)
plt.ylabel("Sales", fontsize=14)
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
