"""Length-safe raw-text requests without provider-specific tokenizer IDs."""

import math

from langchain_core.embeddings import Embeddings


class LengthSafeEmbeddings(Embeddings):
    def __init__(self, backend):
        self.backend = backend
        self.model = backend.model
        # UTF-8 bytes conservatively bound tokens for these byte-based tokenizers.
        # Leave headroom for models with a 2,048-token limit outside OpenAI.
        self.max_bytes = 8000 if self.model.startswith("openai/") else 2000
        self.batch_size = min(32, backend.chunk_size)
        if self.batch_size < 1:
            raise ValueError("Embedding batch size must be positive.")

    def _split(self, texts):
        chunks, groups = [], []
        for text in texts:
            if not text:
                raise ValueError("Embedding input must not be empty.")
            start = len(chunks)
            current, size = [], 0
            for character in text:
                width = len(character.encode("utf-8"))
                if size + width > self.max_bytes:
                    chunks.append("".join(current))
                    current, size = [], 0
                current.append(character)
                size += width
            chunks.append("".join(current))
            groups.append((start, len(chunks)))
        return chunks, groups

    @staticmethod
    def _combine(chunks, groups, vectors):
        if len(vectors) != len(chunks):
            raise ValueError("Embedding response count does not match input chunks.")
        result = []
        for start, end in groups:
            parts = vectors[start:end]
            if not parts or not parts[0] or any(len(v) != len(parts[0]) for v in parts):
                raise ValueError("Inconsistent embedding dimensions.")
            weights = [len(chunk.encode("utf-8")) for chunk in chunks[start:end]]
            total = sum(weights)
            mean = [sum(v[i] * w for v, w in zip(parts, weights)) / total for i in range(len(parts[0]))]
            norm = math.sqrt(sum(value * value for value in mean))
            result.append([value / norm for value in mean] if norm else mean)
        return result

    def embed_documents(self, texts):
        chunks, groups = self._split(texts)
        vectors = []
        for start in range(0, len(chunks), self.batch_size):
            vectors.extend(self.backend.embed_documents(chunks[start : start + self.batch_size]))
        return self._combine(chunks, groups, vectors)

    def embed_query(self, text):
        return self.embed_documents([text])[0]

    async def aembed_documents(self, texts):
        chunks, groups = self._split(texts)
        vectors = []
        for start in range(0, len(chunks), self.batch_size):
            vectors.extend(await self.backend.aembed_documents(chunks[start : start + self.batch_size]))
        return self._combine(chunks, groups, vectors)

    async def aembed_query(self, text):
        return (await self.aembed_documents([text]))[0]
