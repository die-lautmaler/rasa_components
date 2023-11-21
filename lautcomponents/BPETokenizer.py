import io
import logging
from typing import Dict, Text, Any, List

from rasa.engine.graph import GraphComponent, ExecutionContext
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.nlu.constants import TEXT
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.nlu.tokenizers import Tokenizer

import sentencepiece

logger = logging.getLogger(__name__)


@DefaultV1Recipe.register(
    [DefaultV1Recipe.ComponentType.MESSAGE_TOKENIZER], is_trainable=True
)
class BPETokenizer(Tokenizer):
    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> GraphComponent:
        # TODO: Implement this
        ...

    @staticmethod
    def get_default_config() -> Dict[Text, Any]:
        """Returns the component's default config."""
        return {
            # Flag to check whether to split intents
            "intent_tokenization_flag": False,
            # Symbol on which intent should be split
            "intent_split_symbol": "_",
            # path to bpe model file, or construct from training data
            "bpe_model_path": None,
            # size of vocab (only for on the fly training)
            "vocab_size": 10000,
        }

    def __init__(self, config: Dict[Text, Any]) -> None:
        """Initialize the tokenizer."""
        super().__init__(config)

        if "case_sensitive" in self._config:
            rasa.shared.utils.io.raise_warning(
                "The option 'case_sensitive' was moved from the tokenizers to the "
                "featurizers.",
                docs=DOCS_URL_COMPONENTS,
            )

        self.bpe_model = io.BytesIO()

    def train(self, training_data: TrainingData) -> Resource:
        """train a new bpe model or load"""
        logger.info("training BPE Tokenizer")
        train_texts = []
        if self.config["bpe_model_path"] is None:
            train_texts = [example[TEXT] for example in training_data.training_examples if example[TEXT] != '']


            SentencePieceTrainer.train(
                sentence_iterator=response,
                model_writer=self.bpe_model,
                vocab_size=self.config["vocab_size"],
            )

        else:
            # TODO: implement model loading
            raise NotImplementedError("BPE model loading not implemented")

    def process_training_data(self, training_data: TrainingData) -> TrainingData:
        # TODO: Implement this if your component augments the training data with
        #       tokens or message features which are used by other components
        #       during training.
        ...

        return training_data

    def process(self, messages: List[Message]) -> List[Message]:
        # TODO: This is the method which Rasa Open Source will call during inference.
        ...
        return messages
