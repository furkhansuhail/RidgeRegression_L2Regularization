import pandas as pd
import numpy as np
# Option 1: Using pd.set_option
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

class HousePredictionSystem_Ridge_lasso:
    def __init__(self):
        self.Dataset = pd.read_csv("Dataset/Mumbai1.csv")
        # print(self.Dataset.info())
        # self.ExploratoryDataAnalysis()
        self.DataManupulationForModelDevelopment()

    def ExploratoryDataAnalysis(self):
        print(self.Dataset.head())
        print(self.Dataset.shape)
        for column in self.Dataset.columns:
            print(self.Dataset[column].value_counts())
            print("*" * 20)
        print("************************")
        print(self.Dataset.isnull().sum())

    def DataManupulationForModelDevelopment(self):
        # print(self.Dataset.info())
        print(self.Dataset.head())
        self.Dataset.drop(
            columns=['Lift Available', 'CarParking', 'Maintenance Staff', '24x7 Security', 'Clubhouse', 'Intercom',
                     'Landscaped Gardens', 'Jogging Track', 'Swimming Pool', 'Indoor Games', 'New/Resale',
                     'Gas Connection'], inplace=True)
        # print(self.Dataset.head())
        # print(self.Dataset.describe())
        self.Dataset.drop(columns=self.Dataset.columns[0], axis=1, inplace=True)
        # print(self.Dataset.head())
        self.Dataset['Price_per_sqft'] = self.Dataset['price'] * 100000 / self.Dataset['sqrt']
        # print(self.Dataset.head())
        # print(self.Dataset.describe())

        self.Dataset['location'] = self.Dataset['location'].apply(lambda x: x.strip())
        location_count = self.Dataset['location'].value_counts()
        # print(location_count)

        locations_count_less_5 = location_count[location_count <= 5]
        # print(locations_count_less_5)
        self.Dataset['location'] = self.Dataset['location'].apply(lambda x: 'other' if x in locations_count_less_5 else x)
        # print(self.Dataset['location'].value_counts())
        self.Dataset.rename(
            columns=({'No. of Bedrooms': 'bhk', 'Gymnasium': 'GYM', "Children's Play Area": 'bath',
                      '24x7 Security': 'Security'}),
            inplace=True,
        )
        # print(self.Dataset.head())

        # print((self.Dataset['sqrt'] / self.Dataset['bhk']).describe())

        data = self.Dataset[((self.Dataset['sqrt']/ self.Dataset['bhk'] >= 200))]
        print(data.info())
        # print(data.describe())
        # print(data.Price_per_sqft.describe())
        data = self.removeOutliners(data)
        print(data.info())
        print(data.describe())
        data = self.bhk_outliners(data)
        data.drop(columns=['bath', 'Price_per_sqft'], inplace=True)
        data.to_csv("cleaned_data.csv")


    def bhk_outliners(selg, df):
        exclude_indices = np.array([])
        for location, location_df in df.groupby('location'):
            bhk_stats = {}
            for bhk, bhk_df in location_df.groupby('bhk'):
                bhk_stats[bhk] = {
                    'mean': np.mean(bhk_df.Price_per_sqft),
                    'std': np.std(bhk_df.Price_per_sqft),
                    'count': bhk_df.shape[0]
                }
            for bhk, bhk_df in location_df.groupby('bhk'):
                stats = bhk_stats.get(bhk - 1)
                if stats and stats['count'] > 5:
                    exclude_indices = np.append(exclude_indices,
                                                bhk_df[bhk_df.Price_per_sqft < (stats['mean'])].index.values)
        return df.drop(exclude_indices, axis='index')

    def removeOutliners(self, df):
        df_Output = pd.DataFrame()
        for key, subdf in df.groupby('location'):
            m = np.mean(subdf.Price_per_sqft)
            st = np.std(subdf.Price_per_sqft)
            gen_df = subdf[(subdf.Price_per_sqft > (m - st)) & (subdf.Price_per_sqft <= (m + st))]
            df_Output = pd.concat([df_Output, gen_df], ignore_index=True)
        return df_Output


if __name__ =='__main__':
    HousePredictionSystem_Ridge_lasso()