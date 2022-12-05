import pandas as pd
from statsmodels.tsa.api import VAR
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
import os
import logging


class ForecastScreeningCapcity:
    """
    A class to forecast screening capacity
    ....
    Attributes
    ----------
    screening_volume_df: pd.DataFrame
    visit_volume_df: pd.DataFrame
    screening_threshold: int
    path_to_save: str

    Methods
    -------
    prep_data()
        prepares the data, joins screening and visit volume data frames and split the data into train and test
    train_var_model()
        trains vector autoregression model
    forecast_capacity(n_forecast_days)
        predicts the visit and screening capacity for the upcoming 'n_forecast_days'
    generate_increase_capacity_warning(n_days)
        returns True if the forecast of screening volume for upcoming n_days pass the screening_threshold and False otherwise
    visualiz_inputs(plot_name):
        plots the inputs and save it in the directory with the passed plot name
    visualiz_outputs(plot_name):
        plots the actuals versus forecasts and save it in the directory with the passed plot name
    validate_model:
        visualize the output and returns rmse and correlation for screening capacity prediction and actual
    """

    def __init__(
        self,
        screening_volume_df,
        visit_volume_df,
        screening_threshold,
        path_to_save=os.path.join(os.path.dirname(__file__), "output"),
    ):
        """
        Parameters
        ----------
        screening_volume_df: pd.DataFrame
            dataframe with 'screening_counts' and 'date' columns
        visit_volume_df: pd.DataFrame
            dataframe with 'visit_counts' and 'date' columns
        screening_threshold: int
            the screening capacity threshold
        path_to_save: str
            directory to save the output plots and validation results
        """
        self.screening_volume_df = screening_volume_df
        self.visit_volume_df = visit_volume_df
        self.screening_threshold = screening_threshold
        self.path_to_save = path_to_save
        if not os.path.exists(self.path_to_save):
            os.makedirs(self.path_to_save)
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("forecast screening capacity")

    def prep_data(self):
        # cleaning the column names
        self.screening_volume_df.columns = self.screening_volume_df.columns.str.lower()
        self.visit_volume_df.columns = self.visit_volume_df.columns.str.lower()
        # cleaning date columns
        self.screening_volume_df.date = pd.to_datetime(self.screening_volume_df.date)
        self.visit_volume_df.date = pd.to_datetime(self.visit_volume_df.date)
        # joining screening and visit data frames
        self.input_df = self.screening_volume_df.merge(
            self.visit_volume_df, on="date", suffixes=["_screening", "_visit"]
        )
        # filling the missing values with the previous value
        self.input_df.fillna(method="ffill", inplace=True)
        # creating a date-time index with day frequency
        self.input_df.set_index("date", inplace=True)
        # Out of Time Train Test Split (Leaving the last 2 weeks for testing)
        self.train_set = self.input_df.iloc[:-14]
        self.train_set.index = pd.DatetimeIndex(self.train_set.index.values, freq="D")
        self.test_set = self.input_df.iloc[-14:]
        self.logger.info("TRAIN AND TEST SETS ARE PREPARED...")

    def train_var_model(self):
        self.prep_data()
        # Initilizing a Vector AutoRegression Model
        var_model = VAR(self.train_set)
        # fitting the model with lar_oder being the order with minimum metrics
        self.fitted_model = var_model.fit(maxlags=20)
        self.logger.info("MODEL IS TRAINED...")
        # summary of the out put
        self.model_summary = self.fitted_model.summary()

    def forecast_capacity(self, n_forecast_days: int):
        self.train_var_model()
        lag_order = self.fitted_model.k_ar
        self.logger.info(f"MODEL IS FORECASTING FOR NEXT {n_forecast_days} DAYS...")
        return self.fitted_model.forecast(
            y=self.train_set.values[-lag_order:], steps=n_forecast_days
        )

    def generate_increase_capacity_warning(self, n_days: int):
        if any(
            screening_cnt > self.screening_threshold
            for screening_cnt in self.forecast_capacity(n_forecast_days=n_days)[:, 0]
        ):
            self.logger.warning(
                f"WE EXPECT TO CROSS {self.screening_threshold} THRESHOLD IN THE NEXT {n_days} DAYS!!!"
            )
            return True
        else:
            self.logger.info(
                f"WE EXPECT TO NOT CROSS {self.screening_threshold} THRESHOLD IN THE NEXT {n_days} DAYS."
            )
            return False

    def visualiz_inputs(self, plot_name: str):
        plt.figure(figsize=(30, 30))
        plt.plot(
            self.input_df.index,
            self.input_df.iloc[:, 0],
            label=self.input_df.columns[0],
        )
        plt.plot(
            self.input_df.index,
            self.input_df.iloc[:, 1],
            label=self.input_df.columns[1],
        )
        plt.legend(loc="best")
        plt.savefig(os.path.join(self.path_to_save, plot_name + ".png"))
        self.logger.info(
            f"INPUT VOLUMNS ARE PLOTTED AND SAVED AT {self.path_to_save}..."
        )

    def visualize_output(self, plot_name):
        fig, axes = plt.subplots(nrows=2, ncols=1, dpi=150, figsize=(10, 10))
        self.test_forecast = self.forecast_capacity(
            n_forecast_days=len(self.test_set.index)
        )
        axes[0].plot(
            self.test_set.index,
            self.test_set.iloc[:, 0],
            label=self.test_set.columns[0],
        )
        axes[0].plot(
            self.test_set.index,
            self.test_forecast[:, 0],
            label=self.test_set.columns[0] + "_forecast",
        )
        axes[0].legend(loc="best")
        axes[1].plot(
            self.test_set.index,
            self.test_set.iloc[:, 1],
            label=self.test_set.columns[1],
        )
        axes[1].plot(
            self.test_set.index,
            self.test_forecast[:, 1],
            label=self.test_set.columns[1] + "_forecast",
        )
        axes[1].legend(loc="best")
        fig.savefig(os.path.join(self.path_to_save, plot_name + ".png"))
        self.logger.info(
            f"OUTPUT FORECAST VERSUS ACTUAL VOLUMS ARE PLOTTED AND SAVED AT {self.path_to_save} .."
        )

    def validate_model(self):
        self.visualize_output("forecast_vs_actual")
        rmse = np.sqrt(
            mean_squared_error(self.test_set.iloc[:, 0], self.test_forecast[:, 0])
        )
        corr = np.corrcoef(self.test_set.iloc[:, 0], self.test_forecast[:, 0])[0][1]
        validation_metrics = pd.DataFrame(
            {"screening_cnt_rmse": [rmse], "screening_cnt_pred_actual_corr": [corr]}
        )
        validation_metrics.to_csv(
            os.path.join(self.path_to_save, "validation_metrics.csv"), index=False
        )
        self.logger.info(f"VALIDATION OUTPUTS ARE SAVED AT {self.path_to_save}")
        self.logger.info(f"VALIDATION METRIC : {validation_metrics}")


if __name__ == "__main__":
    # Load the data
    screening_data = pd.read_csv("data/screening_data.csv")
    visit_data = pd.read_csv("data/visit_data.csv")
    forecast_capacity_class = ForecastScreeningCapcity(
        screening_volume_df=screening_data,
        visit_volume_df=visit_data,
        screening_threshold=300000,
    )
    # Quick EDA
    forecast_capacity_class.prep_data()
    input_df = forecast_capacity_class.input_df
    input_df["month"] = input_df.index.month
    input_df["day"] = input_df.index.day
    input_df["weekend_ind"] = np.where(input_df.index.day >= 5, 1, 0)

    fig, axes = plt.subplots(nrows=3, ncols=1, dpi=150, figsize=(30, 30))
    input_df.groupby(["month"]).mean()[["counts_screening", "counts_visit"]].plot.bar(
        ax=axes[0], legend=True
    )
    input_df.groupby(["day"]).mean()[["counts_screening", "counts_visit"]].plot.bar(
        ax=axes[1], legend=True
    )
    input_df.groupby(["weekend_ind"]).mean().plot.bar(ax=axes[2], legend=False)
    fig.savefig("output/month_day_weekend_trend.png")

    forecast_capacity_class.visualiz_inputs("actual_screening_vs_visits")
    # end of EDA
    # Validate the model
    forecast_capacity_class.validate_model()
    # Generate increase capacity warning for days in July
    forecast_capacity_class.generate_increase_capacity_warning(n_days=31)
