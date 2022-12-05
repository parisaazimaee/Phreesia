import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
from scipy.stats import mode
import numpy as np

# Load the data
screening_data = pd.read_csv("data/screening_data.csv")
visit_data = pd.read_csv("data/visit_data.csv")
# clean the data
screening_data["Date"] = pd.to_datetime(screening_data["Date"])
visit_data["Date"] = pd.to_datetime(visit_data["Date"])
# join the data
joined_df = screening_data.merge(
    visit_data, on="Date", suffixes=["_screening", "_visit"]
)
# clean the data
joined_df.isna().sum()
joined_df.info()
# visualize the data
# joined_df["month"] = joined_df.Date.dt.month
# joined_df["day"] = joined_df.Date.dt.day
# joined_df["weekend_ind"] = np.where(joined_df.Date.dt.day >= 5, 1, 0)
# fig, axes = plt.subplots(nrows=3, ncols=1, dpi=150, figsize=(10, 30))
# joined_df.groupby(["month"]).mean().plot.bar(ax=axes[0], legend=True)
# joined_df.groupby(["day"]).mean().plot.bar(ax=axes[1], legend=True)
# joined_df.groupby(["weekend_ind"]).mean().plot.bar(ax=axes[2], legend=True)
# fig.savefig("month_day.png")

plt.figure()
title = "Number of Daily Visits vs Number of Daily Covid Screening"
ax = joined_df["Counts_visit"].plot(legend=True, title=title)
ax.autoscale(axis="x", tight=True)
ax.set(xlabel="Day", ylabel="Count")
joined_df["Counts_screening"].plot(legend=True)
plt.savefig("daily_visit_screen_timeseries.png")

# Assuming stationarity
# In aperfect world we test stainarity using Augmented Dickey Fuller Test or Johansen’s test and if the data be non-stationary
# we have to make it stationary
# Create a date-time index
joined_df.set_index("Date", inplace=True)
# Out of Time Train Test Split (Leaving the last 2 weeks for testing)
train = joined_df.iloc[:-14]
train.index = pd.DatetimeIndex(train.index.values, freq="D")
test = joined_df.iloc[-14:]
# Initilizing a Vector AutoRegression Model
var_model = VAR(train)
# fitting the model with lar_oder being the order with minimum aic
fitted_model = var_model.fit(maxlags=20, ic="aic")
# summary of the out put
print(fitted_model.summary())
# predict screening and visits forecast for the test days to validate the model
lag_order = fitted_model.k_ar
predictions = fitted_model.forecast(y=train.values[-lag_order:], steps=14)
test[test.columns[0] + "_pred"], test[test.columns[1] + "_pred"] = (
    predictions[:, 0],
    predictions[:, 1],
)
# plotting the test results
plt.figure()
fitted_model.plot_forecast(14)
plt.savefig("test.pred.png")

plt.figure()
fitted_model.plot_acorr()
plt.savefig("autocorr.png")
# predict screening forecast for the July and test if it passes the threshold
screening_threshold = 300000
july_predictions = fitted_model.forecast(y=train.values[-lag_order:], steps=31)
if any(screening_cnt > screening_threshold for screening_cnt in july_predictions[:, 0]):
    print(True)
else:
    print(False)

# validation
# adding july data to the original data to take a look at predictions trend
july_predictions_df = pd.DataFrame(
    {"Counts_screening": july_predictions[:, 0], "Counts_visit": july_predictions[:, 1]}
)
joined_df2 = pd.concat([joined_df, july_predictions_df])
joined_df2.index = range(1, joined_df2.shape[0] + 1)
plt.figure()
title = "Number of Daily Visits vs Number of Daily Covid Screening"
plt.plot(joined_df2.index, joined_df2["Counts_screening"], label="Screening")
plt.plot(joined_df2.index, joined_df2["Counts_visit"], label="Visit")
plt.legend(loc="best")
plt.title(title)
plt.savefig("daily_visit_screen_timeseries2.png")

# plotting the actual versus prediction for the last 14 days
fig, axes = plt.subplots(nrows=2, ncols=1, dpi=150, figsize=(10, 10))
axes[0].plot(test.index, test["Counts_screening"], label = "screening_actual")
axes[0].plot(test.index, test["Counts_screening_pred"], label = "screening_forecast")
axes[0].legend(loc="best")
axes[1].plot(test.index, test["Counts_visit"], label = "visit_actual")
axes[1].plot(test.index, test["Counts_visit_pred"], label = "visit_forecast")
axes[1].legend(loc="best")
# test["Counts_screening"].plot(legend=True, ax=axes[0]).autoscale(axis="x", tight=True)
# test["Counts_screening_pred"].plot(legend=True, ax=axes[0]).autoscale(
#     axis="x", tight=True
# )
# test["Counts_visit"].plot(legend=True, ax=axes[1]).autoscale(axis="x", tight=True)
# test["Counts_visit_pred"].plot(legend=True, ax=axes[1]).autoscale(axis="x", tight=True)
fig.savefig("test_pred_actual2.png")

# rmse and corr
from sklearn.metrics import mean_squared_error

rmse = np.sqrt(
    mean_squared_error(test["Counts_screening"], test["Counts_screening_pred"])
)
corr = test["Counts_screening"].corr(test["Counts_screening_pred"])
print(rmse, corr)
