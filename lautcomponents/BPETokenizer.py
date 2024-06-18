import os.path
import logging

from tokenizers import Tokenizer as HuggingfaceTokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

from typing import Dict, Text, Any, List, Optional

from rasa.engine.graph import GraphComponent, ExecutionContext
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.nlu.tokenizers.tokenizer import Tokenizer
from rasa.shared.exceptions import FileIOException
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.shared.nlu.constants import TEXT

from rasa.nlu.constants import TOKENS_NAMES
from rasa.nlu.tokenizers.tokenizer import Token

logger = logging.getLogger(__name__)


@DefaultV1Recipe.register(
    DefaultV1Recipe.ComponentType.MESSAGE_TOKENIZER,
    is_trainable=True,
)
class BPETokenizer(Tokenizer, GraphComponent):
    """
    A trainable tokenizer component that uses the tiktoken library to create a bytepair encoding
    and then use it to tokenize the input text at inference time.
    """

    tokenizer_fname = f"bpe_tokenizer.json"

    def __init__(
        self,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> None:
        """Constructs a new tik tokenizer using the tiktoken library."""
        self._config = self.get_default_config()
        self._config.update(config)
        super().__init__(self._config)
        self.bpe_tokenizer: Optional[HuggingfaceTokenizer] = None

        # Store both `model_storage` and `resource` as object attributes to be able
        # to utilize them at the end of the training and persist trained BPE model.
        self._model_storage = model_storage
        self._resource = resource

        self.finetune_mode = execution_context.is_finetuning
        if self.finetune_mode:
            raise ValueError(
                "BPETokenizer: This component does not support finetuning."
            )

    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> GraphComponent:  # should be BPeTokenizer
        if config.get("model", None):
            logger.info(f"Loading BPE Tokenizer model from: {config.get('model')}.")
            try:
                pretrained = HuggingfaceTokenizer.from_file(config.get("model"))
                bpe_tok = cls(config, model_storage, resource, execution_context)
                bpe_tok.bpe_tokenizer = pretrained
                return bpe_tok
            except (FileNotFoundError, IOError, ValueError, FileIOException):
                logger.debug(
                    f"Failed to load BPE Tokenizer model. "
                    f"Resource '{resource}' could not be found."
                )

        return cls(config, model_storage, resource, execution_context)

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
            with model_storage.read_from(resource) as model_dir:
                tok = HuggingfaceTokenizer.from_file(
                    str(os.path.join(model_dir, cls.tokenizer_fname))
                )

                bt = cls(config, model_storage, resource, execution_context)
                bt.bpe_tokenizer = tok

                return bt
        except (ValueError, FileNotFoundError, FileIOException):
            logger.debug(
                f"Failed to load BPE Tokenizer model. "
                f"Resource '{resource}' could not be found."
            )
        return cls.create(config, model_storage, resource, execution_context)

    def train(self, training_data: TrainingData) -> Resource:
        """Train the BPE tokenizer using the provided training data."""
        if self._config.get("model", None) and self.bpe_tokenizer:
            logger.info("using pretrained BPE tokenizer. Skip training.")
        else:
            logger.info("Training BPE tokenizer.")
            self.bpe_tokenizer = self._train_tokenizer(training_data)

        logger.info(
            f"BPETokenizer final vocab size of: {self.bpe_tokenizer.get_vocab_size()}"
        )

        with self._model_storage.write_to(self._resource) as model_dir:
            self.bpe_tokenizer.save(str(model_dir / "bpe_tokenizer.json"))

        return self._resource

    def process_training_data(self, training_data: TrainingData) -> TrainingData:
        """Tokenize all training data."""
        for example in training_data.training_examples:
            tokens = self.tokenize(example, TEXT)
            example.set(TOKENS_NAMES[TEXT], tokens)
        return training_data

    def process(self, messages: List[Message]) -> List[Message]:
        """Tokenize the incoming messages at inferences time"""
        for message in messages:
            tokens = self.tokenize(message, TEXT)
            message.set(TOKENS_NAMES[TEXT], tokens)

        return messages

    # def tokenize(self, text: Text) -> List[Token]:
    def tokenize(self, message: Message, attribute: Text) -> List[Token]:
        """Tokenizes the text of the provided attribute of the incoming message."""
        text = message.get(attribute)
        if text is None:
            logger.warning(
                f"Message contains no {text} attribute. Skipping tokenization."
            )
            return []

        encoded = self.bpe_tokenizer.encode(text)
        tokens = []
        logger.debug(f"My Tokens: {encoded.tokens} and ids: {encoded.ids}")

        for idx, token in enumerate(encoded.tokens):
            # print("offsets", encoded.offsets[idx])
            start_idx, end_idx = encoded.offsets[idx]
            tokens.append(
                Token(
                    text=token,
                    start=start_idx,
                    end=end_idx,
                    data={"id": encoded.ids[idx]},
                    lemma=token,
                )
            )

        return tokens

    @staticmethod
    def required_packages() -> List[Text]:
        """Any extra python dependencies required for this component to run."""
        return ["tokenizers"]

    @staticmethod
    def get_default_config() -> Dict[Text, Any]:
        return {
            # maximum vocab size to train the BPE tokenizer
            "vocab_size": 5000,
            # minimum frequency of a token to be included in the vocab
            "min_frequency": 0,
            # path to a trained BPE model
            "model": None,
            # Flag to check whether to split intents
            "intent_tokenization_flag": False,
            # Symbol on which intent should be split
            "intent_split_symbol": "_",
            # Regular expression to detect tokens
            "token_pattern": None,
            # Symbol on which prefix should be split
            "prefix_separator_symbol": None,
        }

    def msg_text_generator(self, training_data: TrainingData) -> str:
        x = 0
        for example in training_data.training_examples:
            x += 1
            yield example.get(TEXT)

        logger.debug(f"Total number of examples: {x}")

    def _train_tokenizer(self, training_data: TrainingData) -> HuggingfaceTokenizer:
        """
        Train the BPE tokenizer using the provided training data.
        :param self:
        :param training_data:
        :return:
        """
        vocab_size = self._config.get("vocab_size")
        min_frequency = self._config.get("min_frequency")
        logger.info(
            f"Training BPE tokenizer with vocab size: {vocab_size} and min frequency: {min_frequency}"
        )

        tokenizer = HuggingfaceTokenizer(BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = BpeTrainer(
            special_tokens=["[UNK]", "[PAD]"],
            vocab_size=vocab_size,
            min_frequency=min_frequency,
        )
        tokenizer.train_from_iterator(self.msg_text_generator(training_data), trainer)

        return tokenizer
