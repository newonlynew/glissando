from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModel

from glissando.getter import Messages, Message


class EmbeddingGenerator:
    BATCH_SIZE = 8
    MODEL_URL = "ai-forever/sbert_large_nlu_ru"
    LOCAL_MODEL_DIR = Path("./assets/sbert_large_nlu_ru")

    def __init__(self) -> None:
        self._tokenizer, self._model = self._load_model()

    def generate_embeddings(self, messages: Messages) -> torch.Tensor:
        embeddings = []
        messages = messages.to_list()

        for i in range(0, len(messages), self.BATCH_SIZE):
            batch = messages[i:i + self.BATCH_SIZE]
            encoded_input = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            ).to(torch.device("cuda"))

            with torch.no_grad():
                model_output = self._model(**encoded_input)

            batch_embeddings = self._mean_pooling(model_output, encoded_input["attention_mask"])
            embeddings.append(batch_embeddings.cpu())

        return torch.cat(embeddings)

    @staticmethod
    def _mean_pooling(model_output: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        token_embeddings = model_output[0]
        input_mask_expanded = (
            attention_mask
            .unsqueeze(-1)
            .expand(token_embeddings.size())
            .float()
        )

        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask

    def _load_model(self) -> tuple:
        model_source = self.MODEL_URL
        if self.LOCAL_MODEL_DIR.is_dir():
            model_source = str(self.LOCAL_MODEL_DIR)

        tokenizer = AutoTokenizer.from_pretrained(model_source)
        model = AutoModel.from_pretrained(model_source)
        model.to(torch.device("cuda"))

        if not self.LOCAL_MODEL_DIR.exists():
            self.LOCAL_MODEL_DIR.mkdir(parents=True)
            tokenizer.save_pretrained(self.LOCAL_MODEL_DIR)
            model.save_pretrained(self.LOCAL_MODEL_DIR)

        return tokenizer, model