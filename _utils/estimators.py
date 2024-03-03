import pandas as pd
from _utils import *
from joblib import load, dump
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder, StandardScaler


class QuantitativeEncoder(BaseEstimator, TransformerMixin):
    """
    Renvoie un pd df restreint sur les variables quantitatives.
    Ne fait aucun traitement.
    """

    def __init__(self, columns=None):
        super().__init__()
        self.x = None  # pour les données dans la methode fit
        self.columns = columns
        # loader le standard encoder fitted sinon un encoder vide
        if os.path.exists(MyPaths.estimatorFittedStandardEncoder):
            self.stdEncoder = load(MyPaths.estimatorFittedStandardEncoder)
        else:
            self.stdEncoder = StandardScaler()

    def fit(self, x, y=None):
        if self.columns is not None:
            self.x = x[self.columns]
        else:
            print(f'Warning : Columns to fit are not specified. Encoder will try to fit on every columns of dataset.')
        try:
            self.stdEncoder.fit(self.x)
            return self
        except Exception as e:
            print(f'Exception occurs when fitting :: {e}')
        finally:
            dump(self.stdEncoder, MyPaths.estimatorFittedStandardEncoder)

    def transform(self, x):
        if self.columns is not None:
            x = x[self.columns]
        return pd.DataFrame(self.stdEncoder.transform(x), columns=self.columns)


class QualitativeEncoder(BaseEstimator, TransformerMixin):
    """
    Se base sur le labelEncoder de sklearn.
    Renvoie un pd dataframe au lieu d'un <class 'numpy.ndarray'>.
    /!\ apparemment il y a une option qui permet de renvoyer du pd.df au lieu de np.ndarray dans la màj.
    Sauvegarde l'encodeur pour la reproductibilité.
    """

    def __init__(self, columns=None):
        super().__init__()
        self.x = None
        self.columns = columns
        if os.path.exists(MyPaths.estimatorFittedLabelEncoder):
            self.lblEncoder = load(MyPaths.estimatorFittedLabelEncoder)
        else:
            self.lblEncoder = LabelEncoder()

    def fit(self, x, y=None):
        if self.columns is not None:
            self.x = x[self.columns]
        else:
            print(f'Warning : Columns to fit are not specified. Encoder will try to fit on every columns of dataset.')
        try:
            self.lblEncoder.fit(self.x)
            return self
        except Exception as e:
            print(f'Exception occurs when fitting :: {e}')
        finally:
            dump(self.lblEncoder, MyPaths.estimatorFittedLabelEncoder)

    def transform(self, x):
        if self.columns is not None:
            x = x[self.columns]
        return pd.DataFrame(self.lblEncoder.transform(x), columns=self.columns)
