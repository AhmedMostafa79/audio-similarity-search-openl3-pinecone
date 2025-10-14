import os
import time
from pinecone import Pinecone
try:
    # Optional imports only needed if creating a fresh index
    from pinecone import ServerlessSpec, PodSpec  # type: ignore
except Exception:  # pragma: no cover
    ServerlessSpec = None  # type: ignore
    PodSpec = None  # type: ignore

class PineConeVDB:
    def __init__(self, api_key:str, index_name:str="model-openl3", dimensions:int=512, namespace:str="ns1", use_serverless:bool=True):
        self.pc = Pinecone(api_key=api_key)
        self.index_name = index_name
        self.namespace = namespace
        self.dimensions = dimensions
        self.use_serverless = use_serverless
        self._ensure_index()
        self.index = self.pc.Index(self.index_name)

    def _ensure_index(self):
        # Create index if not exists
        existing = self.pc.list_indexes().names()
        if self.index_name not in existing:
            spec = None
            if self.use_serverless and ServerlessSpec:
                spec = ServerlessSpec(cloud="aws", region="us-east-1")
            elif PodSpec:
                spec = PodSpec(environment="production")
            self.pc.create_index(name=self.index_name, dimension=self.dimensions, metric="cosine", spec=spec)
            while not self.pc.describe_index(self.index_name).status["ready"]:
                time.sleep(1)

    def upsert_vectors(self, vectors, batch_size:int=200):
        for i in range(0, len(vectors), batch_size):
            self.index.upsert(vectors=vectors[i:i+batch_size], namespace=self.namespace)

    def describe_index_stats(self):
        return self.index.describe_index_stats()

    def search(self, vector, top_k:int=1):
        return self.index.query(vector=vector, top_k=top_k, include_metadata=True, namespace=self.namespace)
