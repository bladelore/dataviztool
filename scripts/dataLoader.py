import polars as pl
import io
import contextlib
from ucimlrepo import fetch_ucirepo, list_available_datasets
from typing import Optional, List

from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

class UciDataLoader:
    def __init__(self, datasetIdx: int=0, handleMissing: bool=True):
        self.datasets = self._getFormattedDatasets()
        self.datasetList = list(self.datasets.keys())
        self.dataset = fetch_ucirepo(name=self.datasetList[datasetIdx])
        self.dataTables = self.dataset.data
        self.tasks = [s.lower() for s in self.dataset.metadata.tasks]
        
        self.features = pl.DataFrame(self.dataTables.features)
        self.targets = pl.DataFrame(self.dataTables.targets)
        self.variables = pl.DataFrame(self.dataset.variables)

        if handleMissing:
            combined = pl.concat([self.features, self.targets], how="horizontal")
            self.data = combined.drop_nulls()

            self.features = self.data.select(self.features.columns)
            self.targets = self.data.select(self.targets.columns)

        else:
            self.data = pl.DataFrame(self.dataTables.original)

    @property
    def X(self):
        return self.features

    @property
    def y(self):
        return self.targets.to_numpy().ravel()

    def _getFormattedDatasets(self) -> dict:
        buf = io.StringIO()

        with contextlib.redirect_stdout(buf):
            list_available_datasets()

        output = buf.getvalue()

        datasets = {}
        # datasets = []
        lines = output.strip().splitlines()

        for line in lines[5:]:
            if not line.strip():
                continue
            parts = line.rstrip().rsplit(maxsplit=1)
            if len(parts) != 2:
                continue
            name, id_str = parts
            datasets[name.strip()] = int(id_str)
            # datasets.append(name.strip())

        return datasets
    
    def make_preprocessor(self):
    
        preprocessor = make_column_transformer(
            (StandardScaler(), self.X.select(pl.selectors.numeric()).columns),
            (OneHotEncoder(), self.X.select(pl.selectors.categorical()).columns)
        ).set_output(transform="polars")
    
        return preprocessor


# print(UciDataLoader(74).tasks)