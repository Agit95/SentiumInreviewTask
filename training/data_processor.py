from google.oauth2 import service_account
from google.cloud import bigquery
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np


class CDataProcessor:
    def __init__(self, credentials_path, dataset, table_name):
        self.credentials = service_account.Credentials.from_service_account_file(credentials_path,
            scopes=["https://www.googleapis.com/auth/cloud-platform"], )
        self.queryClient = bigquery.Client(credentials=self.credentials, project=self.credentials.project_id)
        self.dataset = dataset
        self.table = table_name
        self.scaler = MinMaxScaler()
        self.data = None
        self.pred_columns = list()

    def read_data(self):
        tableRef = bigquery.TableReference.from_string(
            '.'.join([self.credentials.project_id, self.dataset, self.table]))
        values = self.queryClient.list_rows(tableRef)
        self.data = values.to_dataframe()
        # return self.data.copy()

    def prepare_data(self):
        df = self.data.copy(deep=True)
        data = CDataProcessor.prepare_address_columns(df)
        data = CDataProcessor.prepare_object_str_columns(data)
        data = CDataProcessor.prepare_datetime_columns(data)
        data = data.dropna().drop_duplicates(subset=None, inplace=None)
        x_train, y_train, data = self.normalize_v1(data)
        self.pred_columns = data.columns.tolist()
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=8)
        return x_train, y_train, x_val, y_val

    def predictable_columns(self):
        return self.pred_columns.copy()

    @staticmethod
    def prepare_datetime_columns(df):
        # select all data time columns names to process
        datetime_columns = df.select_dtypes(include=['datetime64[ns, UTC]']).columns.tolist()
        print('Data time columns', datetime_columns)
        for param in datetime_columns:
            df[param + '_' + 'year'] = df[param].dt.year
            df[param + '_' + 'month'] = df[param].dt.month
            df[param + '_' + 'day'] = df[param].dt.day
        return df.drop(columns=datetime_columns, axis=1, inplace=False)

    @staticmethod
    def prepare_address_columns(df):
        return df.drop(columns=['address'], axis=1, inplace=False) # In data set already exist latitude and longitude

    @staticmethod
    def prepare_object_str_columns(df):
        # select object and string data columns names
        columns = df.columns.values[df.dtypes.values == 'object']
        print('Object columns', columns)
        # encode string data using One Hot Encoding method
        for param in columns:
            unique_values = np.sort(df[param].unique())
            for value in unique_values:
                new_name = param + '_' + value
                df[new_name] = (df[param] == value).astype(int)
        return df.drop(columns=columns, axis=1, inplace=False)

    def normalize_v1(self, df):
        # select label data
        y_train = df['price'].values
        #
        # get training data
        y_train = y_train.reshape((y_train.shape[0], 1))
        df = df.drop(['price'], axis=1)
        self.scaler.fit(df)
        scaled = self.scaler.fit_transform(df)
        df = pd.DataFrame(scaled, columns=df.columns)
        x_train = df.values

        # return all processed data
        return x_train, y_train, df
