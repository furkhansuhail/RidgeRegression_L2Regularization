import pandas as pd
import urllib.request as request
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
from dataclasses import dataclass
from pathlib import Path

Dataset_Link = "https://github.com/furkhansuhail/ProjectData/raw/refs/heads/main/RidgeRegressionDataset/HousingPriceChart.csv"
# Step 1: Configuration class for downloading data
@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    STATUS_FILE: str
    ALL_REQUIRED_FILES: list


# Step 2: Config object
config = DataIngestionConfig(
    root_dir=Path("Dataset"),
    source_URL=Dataset_Link,
    local_data_file=Path("Dataset/HousingPriceChart.csv"),
    STATUS_FILE="Dataset/status.txt",
    ALL_REQUIRED_FILES=[]
)


def download_project_file(source_URL, local_data_file):
    local_data_file.parent.mkdir(parents=True, exist_ok=True)
    if local_data_file.exists():
        print(f"✅ File already exists at: {local_data_file}")
    else:
        print(f"⬇ Downloading file from {source_URL}...")
        file_path, _ = request.urlretrieve(url=source_URL, filename=local_data_file)
        print(f"✅ File downloaded and saved to: {file_path}")




class RidgeRegression_Implementation:
    def __init__(self):
        download_project_file(config.source_URL, config.local_data_file)
        self.Dataset = pd.read_csv("Dataset/HousingPriceChart.csv")
        # print(self.Dataset.info())
        self.ModelDevelopment()

    def ModelDevelopment(self):
        X = self.Dataset.drop(columns=['Price'])
        Y = self.Dataset['Price']
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
        print(X_train.shape)
        print(X_test.shape)
        self.LinearRegression(X_train, X_test, Y_train, Y_test)

    def LinearRegression(self, X_train, X_test, Y_train, Y_test):
        # column_trans = make_column_transformer((OneHotEncoder(sparse=False), ['Location']), remainder='passthrough')
        column_trans = make_column_transformer((OneHotEncoder(sparse_output=False), ['Location']),
                                               remainder='passthrough')
        print(column_trans)

        scaler = StandardScaler()
        lr = LinearRegression()
        pipe = make_pipeline(column_trans, scaler, lr)
        pipe.fit(X_train, Y_train)
        Y_pred = pipe.predict(X_test)
        result = r2_score(Y_test, Y_pred)
        print(result)
        self.ApplyingLasso(column_trans, scaler, lr, X_train, X_test, Y_train, Y_test, Y_pred)


    def ApplyingLasso(self, column_trans, scaler, lr, X_train, X_test, Y_train, Y_test, Y_pred):
        lasso = Lasso()
        pipe = make_pipeline(column_trans, scaler, lr)
        pipe.fit(X_train, Y_train)
        Y_pred_lasso = pipe.predict(X_test)
        r2_score(Y_test, Y_pred_lasso)
        self.ApplyRidge(column_trans, scaler, lr, X_train, X_test, Y_train, Y_test, Y_pred, Y_pred_lasso)

    def ApplyRidge(self, column_trans, scaler, lr, X_train, X_test, Y_train, Y_test, Y_pred, Y_pred_lasso):
        ridge = Ridge()
        pipe = make_pipeline(column_trans, scaler, ridge)
        pipe.fit(X_train, Y_train)
        Y_pred_ridge = pipe.predict(X_test)
        r2_score(Y_test, Y_pred_ridge)
        print("No Regularization: ", r2_score(Y_test, Y_pred))
        print("Lasso: ", r2_score(Y_test, Y_pred_lasso))
        print("Ridge: ", r2_score(Y_test, Y_pred_ridge))



if __name__ == '__main__':
    RidgeRegression_Implementation()
