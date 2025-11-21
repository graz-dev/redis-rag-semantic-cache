from typing import List, Optional
from redisvl.utils.vectorize.base import BaseVectorizer

class DummyVectorizer(BaseVectorizer):
    """Dummy vectorizer that does nothing, used to satisfy SemanticCache requirements."""
    dims: int = 768

    def __init__(self, dims: int = 768, model: str = "dummy", **data):
        # BaseVectorizer requires 'model' field
        data["model"] = model
        data["dims"] = dims
        super().__init__(**data)
        
    def embed(self, text: str, preprocess: Optional[callable] = None) -> List[float]:
        return [0.0] * self.dims
        
    def embed_many(self, texts: List[str], preprocess: Optional[callable] = None) -> List[List[float]]:
        return [[0.0] * self.dims for _ in texts]
