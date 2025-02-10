
import json
import mlflow
import hashlib

from typing import Optional, Any
from mlflow.data.dataset import Dataset
from mlflow.data.dataset_source import DatasetSource


class LocalFileDataSource(DatasetSource):

    def __init__(self, path: str):
        self._path = path
        super().__init__()
    
    def _get_source_type(self):
        return 'local'
    
    def load(self):
        return self._path
    
    def to_dict(self):
        return {
            "path": self._path
        }
    
    @classmethod
    def from_dict(cls, source_dict):
        return cls(
            path=source_dict.get('path')
        )

class SepFileDataset(Dataset):

    def __init__(
        self, 
        source: LocalFileDataSource, 
        name: Optional[str] = None, 
        digest: Optional[str] = None
    ):
         self.data = self._read_data_source(source)
         super().__init__(source, name, digest)
    
    def _read_data_source(self, source):
        path = source.load()
        data = []
        with open(path, 'r') as rf:
                for line in rf:
                    data.append(line.split('<sep>'))
        return data
		    
    def _compute_digest(self):
        all_data = '\n'.join([','.join(item) for item in self.data])
        return hashlib.sha256(all_data.encode()).hexdigest()[:6]
    
    @property
    def profile(self) -> Optional[Any]:
        """
        Optional summary statistics for the dataset, such as the number of rows in a table, the
        mean / median / std of each table column, etc.
        """
        return {
            "num_rows": len(self.data),
        }
    
    def to_dict(self) -> dict[str, str]:
        """Create config dictionary for the dataset.
        """
        config = super().to_dict()
        config.update(
            {
                "profile": json.dumps(self.profile),
            }
        )
        return config



if __name__ == '__main__':
    with mlflow.start_run() as run:
        source = LocalFileDataSource('./2_data.txt')
        dataset = SepFileDataset(source, 'demo')
        mlflow.log_input(dataset)