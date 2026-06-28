import unittest
from pathlib import Path
import pandas as pd

# Define the dataset path from the main project folder
DATA_PATH = Path(__file__).resolve().parents[1] / "Telco Customer Churn.csv"

class TestTelcoDataValidation(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Load the dataset once before running the tests
        cls.df = pd.read_csv(DATA_PATH)

    def prepare_data(self):
        # Create a copy of the dataset for validation
        df_prepared = self.df.copy()

        # Convert TotalCharges into numeric format
        df_prepared["TotalCharges"] = pd.to_numeric(
            df_prepared["TotalCharges"],
            errors="coerce"
        )

        # Handle missing values created after conversion
        df_prepared["TotalCharges"] = df_prepared["TotalCharges"].fillna(0)

        # Convert Churn into binary format
        df_prepared["Churn"] = df_prepared["Churn"].map({"No": 0, "Yes": 1})

        return df_prepared

    def test_dataset_file_exists(self):
        # Check whether the dataset file exists in the repository
        self.assertTrue(DATA_PATH.exists(), "Dataset file is missing.")

    def test_required_columns_exist(self):
        # Check whether important columns are available in the dataset
        required_columns = [
            "customerID", "tenure", "Contract", "MonthlyCharges",
            "TotalCharges", "Churn"
        ]

        for column in required_columns:
            self.assertIn(column, self.df.columns)

    def test_no_duplicate_customer_ids(self):
        # Check that each customer ID is unique
        duplicate_count = self.df["customerID"].duplicated().sum()
        self.assertEqual(duplicate_count, 0)

    def test_prepared_dataset_has_no_missing_values(self):
        # Check that the prepared dataset has no missing values
        df_prepared = self.prepare_data()
        missing_count = df_prepared.isnull().sum().sum()
        self.assertEqual(missing_count, 0)

    def test_totalcharges_is_numeric_after_preparation(self):
        # Check that TotalCharges is numeric after preparation
        df_prepared = self.prepare_data()
        self.assertTrue(pd.api.types.is_numeric_dtype(df_prepared["TotalCharges"]))

    def test_churn_values_are_valid_after_preparation(self):
        # Check that Churn only contains valid binary values
        df_prepared = self.prepare_data()
        valid_values = set(df_prepared["Churn"].unique())
        self.assertTrue(valid_values.issubset({0, 1}))

if __name__ == "__main__":
    unittest.main()
