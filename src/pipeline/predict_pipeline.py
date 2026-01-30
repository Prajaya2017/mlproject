import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
import logging

# Set up basic logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class PredictPipeline:
    def __init__(self):
        """
        Loads the model and preprocessor once during initialization
        to avoid repeated disk access for multiple predictions.
        """
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
            
            logging.info("Loading model and preprocessor...")
            self.model = load_object(file_path=model_path)
            self.preprocessor = load_object(file_path=preprocessor_path)
            logging.info("Model and preprocessor loaded successfully.")
            
        except Exception as e:
            raise CustomException(e, sys)

    def predict(self, features: pd.DataFrame):
        """
        Predicts the target using the pre-loaded model and preprocessor.

        Args:
            features (pd.DataFrame): Input features as a DataFrame.

        Returns:
            np.ndarray: Predictions
        """
        try:
            logging.info("Transforming input features...")
            data_scaled = self.preprocessor.transform(features)
            logging.info("Making predictions...")
            preds = self.model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(
        self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education: str,
        lunch: str,
        test_preparation_course: str,
        reading_score: int,
        writing_score: int
    ):
        """
        Stores custom input data for prediction.

        Args:
            gender (str)
            race_ethnicity (str)
            parental_level_of_education (str)
            lunch (str)
            test_preparation_course (str)
            reading_score (int)
            writing_score (int)
        """
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_data_frame(self) -> pd.DataFrame:
        """
        Converts the custom data into a pandas DataFrame
        so it can be used by the prediction pipeline.

        Returns:
            pd.DataFrame
        """
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }
            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)


# Example usage
if __name__ == "__main__":
    try:
        # Create sample input data
        custom_data = CustomData(
            gender="female",
            race_ethnicity="group B",
            parental_level_of_education="bachelor's degree",
            lunch="standard",
            test_preparation_course="none",
            reading_score=72,
            writing_score=74
        )

        input_df = custom_data.get_data_as_data_frame()

        # Create prediction pipeline and make prediction
        pipeline = PredictPipeline()
        prediction = pipeline.predict(input_df)

        logging.info(f"Prediction: {prediction}")

    except Exception as e:
        logging.error(f"Error occurred: {e}")