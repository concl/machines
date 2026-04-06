

from template import *

class TrainData(Dataset):
    """
    PyTorch Dataset class for training data from a pandas DataFrame which includes a target column.
    Handles categorical encoding, label encoding, normalization, and dropping specified columns.
    """

    def __init__(
        self,
        train_df: pd.DataFrame,
        target: str,
        categorical_cols: list = None,
        encode_labels: bool = False,
        drop_columns = None,
        normalize_columns: list = None
    ): 
        """
        Args:
            train_df (pd.DataFrame): The training DataFrame.
            target (str): The name of the target column.
            categorical_cols (list, optional): List of categorical column names. Defaults to None.
            encode_labels (bool, optional): Whether to encode labels. Defaults to False.
            drop_columns (list, optional): List of columns to drop. Defaults to None.
            normalize_columns (list, optional): List of columns to normalize. Defaults to None.
        """

        self.X = train_df.drop(columns=[target])
        if drop_columns:
            self.X.drop(columns=drop_columns, inplace=True)

        if normalize_columns is not None:
            self.scaler = StandardScaler()
            self.X[normalize_columns] = self.scaler.fit_transform(self.X[normalize_columns])

        self.y = train_df[target]
        self.target = target
        self.categorical_cols = categorical_cols
        self.enc = None
        self.label_enc = None

        if categorical_cols is not None:
            self.enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            encoded_x = self.enc.fit_transform(self.X[categorical_cols])

            encoded_x_df = pd.DataFrame(
                encoded_x,
                columns=self.enc.get_feature_names_out(categorical_cols)
            )

            self.X = pd.concat(
                [
                    self.X.drop(columns=categorical_cols).reset_index(drop=True),
                    encoded_x_df.reset_index(drop=True)
                ],
                axis=1
            )
        
        if encode_labels:
            self.label_enc = LabelEncoder()
            self.y = self.label_enc.fit_transform(self.y)


    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return torch.tensor(self.X.iloc[idx].values, dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32)
    
    def transform(self, new_df: pd.DataFrame):
        if self.target in new_df.columns:
            X = new_df.drop(columns=[self.target])
        else:
            X = new_df.copy()

        if self.enc is not None:
            encoded_x = self.enc.transform(X[self.categorical_cols])
            encoded_x_df = pd.DataFrame(
                encoded_x,
                columns=self.enc.get_feature_names_out(self.categorical_cols)
            )

            X = pd.concat(
                [
                    X.drop(columns=self.categorical_cols).reset_index(drop=True),
                    encoded_x_df.reset_index(drop=True)
                ],
                axis=1
            )
        
        y = None

        if self.target in new_df.columns:
            if self.label_enc is not None:
                y = self.label_enc.transform(new_df[self.target])
            else:
                y = new_df[self.target]
        
        return X, y if y is not None else X

    def inverse_transform_labels(self, encoded_labels):
        if self.label_enc is not None:
            return self.label_enc.inverse_transform(encoded_labels)
        else:
            raise "Label Encoder is not initialized, cannot inverse transform labels"


class DFDataset(Dataset):
    """
    Torch Dataset wrapper for a pandas DataFrame
    """
    def __init__(self, X: pd.DataFrame, y: pd.Series, normalize_columns: list = None):
        """
        Args:
            X (pd.DataFrame): The feature DataFrame.
            y (pd.Series): The target Series.
            normalize_columns (list, optional): List of columns to normalize. Defaults to None.
        """
        self.X = X.copy()
        self.y = y.copy()
        self.normalize_columns = normalize_columns
        self.scaler = None

        if normalize_columns is not None:
            self.scaler = StandardScaler()
            self.X[normalize_columns] = self.scaler.fit_transform(self.X[normalize_columns])

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return torch.tensor(self.X.iloc[idx].values, dtype=torch.float32), torch.tensor(self.y.iloc[idx], dtype=torch.float32)

    def transform(self, X_test: pd.DataFrame):
        if self.scaler is not None:
            X_test = X_test.copy()
            X_test[self.normalize_columns] = self.scaler.transform(X_test[self.normalize_columns])
        return X_test