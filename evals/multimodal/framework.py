"""Optional LlamaIndex opener using complete FastEmbed adapters, loaded lazily.

Its SimpleVectorStore instances are separate from the later manual Qdrant lab.
The framework retrieves two lists; this opener does not claim rank fusion.
"""
import os
from pathlib import Path
from lab import load_corpus
from live import require_flag


def make_adapters():
    require_flag("MM_RUN_MODELS")
    from fastembed import ImageEmbedding, TextEmbedding
    from llama_index.core.base.embeddings.base import BaseEmbedding
    from llama_index.core.bridge.pydantic import PrivateAttr
    from llama_index.core.embeddings.multi_modal_base import MultiModalEmbedding
    cache = os.getenv("MM_MODEL_CACHE", str(Path.home()/".cache"/"fastembed"))

    class LlamaTextEmbedder(BaseEmbedding):
        _model: object = PrivateAttr()

        def __init__(self, **kwargs):
            super().__init__(model_name="BAAI/bge-small-en-v1.5", **kwargs)
            self._model = TextEmbedding("BAAI/bge-small-en-v1.5", cache_dir=cache)

        def _get_text_embedding(self, text):
            return next(iter(self._model.passage_embed([text]))).tolist()

        def _get_query_embedding(self, query):
            return next(iter(self._model.query_embed([query]))).tolist()

        async def _aget_query_embedding(self, query):
            return self._get_query_embedding(query)

    class LlamaClipEmbedder(MultiModalEmbedding):
        _text_tower: object = PrivateAttr()
        _image_tower: object = PrivateAttr()

        def __init__(self, **kwargs):
            super().__init__(model_name="Qdrant/clip-ViT-B-32", **kwargs)
            self._text_tower = TextEmbedding("Qdrant/clip-ViT-B-32-text", cache_dir=cache)
            self._image_tower = ImageEmbedding("Qdrant/clip-ViT-B-32-vision", cache_dir=cache)

        def _get_image_embedding(self, img_file_path):
            return next(iter(self._image_tower.embed([str(img_file_path)]))).tolist()

        def _get_text_embedding(self, text):
            return next(iter(self._text_tower.embed([text]))).tolist()

        def _get_query_embedding(self, query):
            return self._get_text_embedding(query)

        async def _aget_query_embedding(self, query):
            return self._get_query_embedding(query)

        async def _aget_image_embedding(self, img_file_path):
            return self._get_image_embedding(img_file_path)

    return LlamaTextEmbedder(), LlamaClipEmbedder()


def build_framework_index():
    require_flag("MM_RUN_MODELS")
    from llama_index.core import StorageContext
    from llama_index.core.indices import MultiModalVectorStoreIndex
    from llama_index.core.schema import Document, ImageDocument
    from llama_index.core.vector_stores.simple import SimpleVectorStore
    text_embedder, image_embedder = make_adapters()
    docs = [Document(text=record.text, metadata={"source_id": record.source_id, "file_path": record.path}) if record.modality == "text" else ImageDocument(image_path=record.path, metadata={"source_id": record.source_id, "file_path": record.path}) for record in load_corpus()]
    context = StorageContext.from_defaults(vector_store=SimpleVectorStore(), image_store=SimpleVectorStore())
    return MultiModalVectorStoreIndex.from_documents(docs, storage_context=context, embed_model=text_embedder, image_embed_model=image_embedder, show_progress=False)
