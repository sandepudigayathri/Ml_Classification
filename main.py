from src.train import Train_data
from logger import get_logger

if __name__ == "__main__":
    logger = get_logger(__name__)
    Train_data(
        data_path="data/campus_placement_data.csv",
        model_path="models/model.pkl"
    )
