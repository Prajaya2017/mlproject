# ----------------------------
# Standard library imports
# ----------------------------
import sys              # Used for detailed exception handling
import os               # Used for file path handling
from dataclasses import dataclass  # Used to create config classes

# ----------------------------
# Third-party library imports
# ----------------------------
import numpy as np       # Numerical operations
import pandas as pd      # Data manipulation

# ----------------------------
# Scikit-learn imports
# ----------------------------
from sklearn.pipeline import Pipeline                  # To create step-by-step pipelines
from sklearn.compose import ColumnTransformer           # To apply pipelines to specific columns
from sklearn.impute import SimpleImputer                # To handle missing values
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # Encoding & scaling

# ----------------------------
# Project-specific imports
# ----------------------------
from src.exception import CustomException   # Custom exception class
from src.logger import logging              # Custom logger
from src.utils import save_object           # Utility to save .pkl files
from src.utils import save_object

# ============================================================
# Configuration class (stores file paths only)
# ============================================================
@dataclass
class DataTransformationConfig:
    """
    This class stores configuration details
    related to data transformation.
    """
    preprocessor_obj_file_path = os.path.join(
        "artifacts", 
        "preprocessor.pkl"
    )


# ============================================================
# Main Data Transformation class
# ============================================================
class DataTransformation:
    def __init__(self):
        """
        Constructor initializes configuration.
        """
        self.data_transformation_config = DataTransformationConfig()

    # --------------------------------------------------------
    # Function to create preprocessing object
    # --------------------------------------------------------
    def get_data_transformer_object(self):
        """
        This function creates and returns
        a ColumnTransformer object that:
        - handles missing values
        - scales numerical features
        - encodes categorical features
        """
        try:
            # ----------------------------
            # Define column types
            # ----------------------------
            numerical_columns = ["writing_score", "reading_score"]

            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"
            ]

            # ----------------------------
            # Numerical Pipeline
            # ----------------------------
            # Step 1: Replace missing values with median
            # Step 2: Scale features using StandardScaler
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            # ----------------------------
            # Categorical Pipeline
            # ----------------------------
            # Step 1: Replace missing values with most frequent value
            # Step 2: Convert categories into one-hot encoded numeric form
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder(handle_unknown="ignore"))
                ]
            )

            logging.info("Numerical pipeline and categorical pipeline created")

            # ----------------------------
            # Combine both pipelines
            # ----------------------------
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    # --------------------------------------------------------
    # Function to apply data transformation
    # --------------------------------------------------------
    def initiate_data_transformation(self, train_path, test_path):
        """
        This function:
        1. Reads train and test data
        2. Applies preprocessing
        3. Saves preprocessing object
        4. Returns transformed arrays
        """
        try:
            # ----------------------------
            # Read datasets
            # ----------------------------
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Train and test data read successfully")

            # ----------------------------
            # Get preprocessing object
            # ----------------------------
            preprocessing_obj = self.get_data_transformer_object()

            # ----------------------------
            # Define target column
            # ----------------------------
            target_column_name = "math_score"

            # ----------------------------
            # Separate input and target features
            # ----------------------------
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing on train and test data")

            # ----------------------------
            # Apply transformations
            # ----------------------------
            input_feature_train_arr = preprocessing_obj.fit_transform(
                input_feature_train_df
            )

            input_feature_test_arr = preprocessing_obj.transform(
                input_feature_test_df
            )

            # ----------------------------
            # Combine input features and target
            # ----------------------------
            train_arr = np.c_[
                input_feature_train_arr,
                np.array(target_feature_train_df)
            ]

            test_arr = np.c_[
                input_feature_test_arr,
                np.array(target_feature_test_df)
            ]

            # ----------------------------
            # Save preprocessing object
            # ----------------------------
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            logging.info("Preprocessing object saved successfully")

            # ----------------------------
            # Return results
            # ----------------------------
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)
