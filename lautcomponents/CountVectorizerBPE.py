import os
import logging
import sys
from typing import Dict, List, Text, Any, Optional

from rasa.engine.graph import ExecutionContext, GraphComponent
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.nlu.constants import TOKENS_NAMES
from rasa.nlu.featurizers.sparse_featurizer.count_vectors_featurizer import (
    CountVectorsFeaturizer,
)
from rasa.shared.exceptions import FileIOException
from rasa.shared.nlu.constants import TEXT
from rasa.shared.nlu.training_data.message import Message
from sklearn.feature_extraction.text import CountVectorizer
from tokenizers import Tokenizer as HuggingfaceTokenizer

logger = logging.getLogger("rasa")
logger.addHandler(logging.StreamHandler(sys.stdout))


@DefaultV1Recipe.register(
    DefaultV1Recipe.ComponentType.MESSAGE_FEATURIZER, is_trainable=True
)
class CountVectorsFeaturizerBPE(CountVectorsFeaturizer):
    """
    Count Vectorizer using BPE Tokenizer to tokenize
    """

    tokenizer_fname = f"bpe_tokenizer.json"
    bpe_tokenizer: HuggingfaceTokenizer

    @staticmethod
    def get_default_config() -> Dict[Text, Any]:
        return {**CountVectorsFeaturizer.get_default_config(), "bpe_model": None}

    @staticmethod
    def required_packages() -> List[Text]:
        """Any extra python dependencies required for this component to run."""
        reqs = ["tokenizers"]
        reqs.extend(CountVectorsFeaturizer.required_packages())
        return reqs

    def __init__(
        self,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
        vectorizers: Optional[Dict[Text, "CountVectorizer"]] = None,
        oov_token: Optional[Text] = None,
        oov_words: Optional[List[Text]] = None,
    ) -> None:
        super().__init__(
            config,
            model_storage,
            resource,
            execution_context,
            vectorizers,
            oov_token,
            oov_words,
        )

    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> GraphComponent:
        """
            create a new instance with a loaded bpe tokenizer
        :param config:
        :param model_storage:
        :param resource:
        :param execution_context:
        :return:
        """

        bpe_model_fname = config.get("bpe_model", None)
        if bpe_model_fname:
            bpe_model_path = bpe_model_fname
        else:
            bpe_model_path = os.path.join(
                os.path.dirname(__file__), "sipgate_bpe_10k_100freq.json"
            )

        logger.info(f"CVF_BPE:: loading bpe model: {cls.tokenizer_fname}")
        try:
            tok = HuggingfaceTokenizer.from_file(bpe_model_path)
        except (ValueError, FileNotFoundError, FileIOException) as e:
            logger.error(f"CVF_BPE:: Error Could not load bpe model: {e}")
            tok = None

        ftr = cls(config, model_storage, resource, execution_context)
        ftr.bpe_tokenizer = tok

        return ftr

    def persist(self) -> None:
        """Persist this model into the passed directory."""

        super().persist()
        if self.bpe_tokenizer:
            with self._model_storage.write_to(self._resource) as model_dir:
                self.bpe_tokenizer.save(str(model_dir / "bpe_tokenizer.json"))
        else:
            logger.error("CVF_BPE:: No BPE tokenizer present. Cannot persist")

    @classmethod
    def load(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
        **kwargs: Any,
    ) -> GraphComponent:

        try:
            ftr = super().load(
                config, model_storage, resource, execution_context, **kwargs
            )
            tok: HuggingfaceTokenizer

            try:
                with model_storage.read_from(resource) as model_dir:
                    tok = HuggingfaceTokenizer.from_file(
                        str(os.path.join(model_dir, cls.tokenizer_fname))
                    )
            except (ValueError, FileNotFoundError, FileIOException):
                logger.debug(
                    f"Failed to load BPE Tokenizer model. "
                    f"Resource '{resource}' could not be found."
                )

            # Create an instance of the subclass
            instance = cls.create(config, model_storage, resource, execution_context)
            instance.__dict__.update(ftr.__dict__)
            instance.bpe_tokenizer = tok

            return instance
        except Exception as e:
            logger.error(f"CVF_BPE:: unable to restore {e}")

        return cls.create(config, model_storage, resource, execution_context)

    def process(self, messages: List[Message]) -> List[Message]:
        """Processes incoming message and compute and set features."""
        if self.vectorizers is None:
            logger.error(
                "There is no trained CountVectorizer: "
                "component is either not trained or "
                "didn't receive enough training data"
            )
            return messages

        for message in messages:
            for attribute in self._attributes:

                message_tokens = self._get_processed_message_tokens_by_attribute(
                    message, attribute
                )

                # features shape (1, seq, dim)
                sequence_features, sentence_features = self._create_features(
                    attribute, [message_tokens]
                )
                if attribute == TEXT:
                    seq_buff = sequence_features[0]
                    self.add_features_to_message(
                        sequence_features[0], sentence_features[0], attribute, message
                    )
                else:
                    self.add_features_to_message(
                        sequence_features[0], sentence_features[0], attribute, message
                    )

        return messages

    def _get_message_tokens_by_attribute(
        self, message: "Message", attribute: Text
    ) -> List[Text]:
        """Get text tokens of an attribute of a message."""
        if message.get(TOKENS_NAMES[attribute]):
            if attribute == TEXT:
                msg_tokens = []
                for t in message.get(TOKENS_NAMES[attribute]):
                    # join bpe tokens again with ws,
                    # as count vectorizer will split them again,
                    # resulting in a not so sparse count vector for the token
                    # the mismatch between token count and bpe token count
                    # is avoided
                    bpe_toks = self.bpe_tokenizer.encode(t.text).tokens
                    logger.debug(f"CVF_BPE:: sub word tokens {bpe_toks}")
                    msg_tokens.append(" ".join(bpe_toks))
                return msg_tokens
            else:
                return [
                    t.lemma if self.use_lemma else t.text
                    for t in message.get(TOKENS_NAMES[attribute])
                ]
        else:
            return []
