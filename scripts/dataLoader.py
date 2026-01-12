import polars as pl
import io
import contextlib
from ucimlrepo import fetch_ucirepo, list_available_datasets
from typing import List, Dict, Optional, Any
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

class UciDataLoader:
    def __init__(self):
        """Initialize without loading a dataset"""
        self.datasets: Dict[str, int] = self._getFormattedDatasets()
        self.datasetList: List[str] = list(self.datasets.keys())
        
        self.dataset: Optional[Any] = None
        self.dataTables: Optional[Any] = None
        self.data: Optional[pl.DataFrame] = None
        self.features: Optional[pl.DataFrame] = None
        self.targets: Optional[pl.DataFrame] = None
        self.variables: Optional[pl.DataFrame] = None
        self.tasks: Optional[List[str]] = None
    
    def load(self, datasetName: str, handleMissing: bool = True) -> 'UciDataLoader':
        """Load a specific dataset by index"""
        self.dataset = fetch_ucirepo(name=datasetName)
        self.dataTables = self.dataset.data
        self.data = pl.DataFrame(self.dataTables.original) # type: ignore
        self.features = pl.DataFrame(self.dataTables.features) # type: ignore
        self.targets = pl.DataFrame(self.dataTables.targets) # type: ignore
        self.variables = pl.DataFrame(self.dataset.variables)
        
        if handleMissing:
            combined = pl.concat([self.features, self.targets], how="horizontal")
            clean_combined = combined.drop_nulls()
            self.features = clean_combined.select(self.features.columns)
            self.targets = clean_combined.select(self.targets.columns)
        
        self.tasks = [s.lower() for s in self.dataset.metadata.tasks]
        return self
    
    @property
    def X(self) -> pl.DataFrame:
        return self.features  # type: ignore
    
    @property
    def y(self):
        return self.targets.to_numpy().ravel()  # type: ignore
    
    def make_preprocessor(self):
        preprocessor = make_column_transformer(
            (StandardScaler(), self.X.select(pl.selectors.numeric()).columns),
            (OneHotEncoder(handle_unknown='ignore'), self.X.select(pl.selectors.categorical()).columns)
        ).set_output(transform="polars") # type: ignore
        return preprocessor
    
    def _getFormattedDatasets(self) -> Dict[str, int]:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            list_available_datasets()
        output = buf.getvalue()
        
        datasets = {}
        lines = output.strip().splitlines()
        for line in lines[5:]:
            if not line.strip():
                continue
            parts = line.rstrip().rsplit(maxsplit=1)
            if len(parts) != 2:
                continue
            name, id_str = parts
            datasets[name.strip()] = int(id_str)
        
        return datasets
