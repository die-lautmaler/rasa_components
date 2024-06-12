import os.path
import logging

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace

from typing import Dict, Text, Any, List, Optional

from rasa.engine.graph import GraphComponent, ExecutionContext
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.exceptions import FileIOException
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.shared.nlu.constants import TEXT

from rasa.nlu.constants import MESSAGE_ATTRIBUTES, TOKENS_NAMES
from rasa.nlu.tokenizers.tokenizer import Token
from tokenizers.trainers import BpeTrainer

logger = logging.getLogger(__name__)

# TODO: Correctly register your component with its type
@DefaultV1Recipe.register(
    [DefaultV1Recipe.ComponentType.MESSAGE_TOKENIZER],
    is_trainable=True,
)
class BPETokenizer(GraphComponent):
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
        # super().__init__(execution_context.node_name, config)
        super().__init__()
        self.bpe_tokenizer: Optional[Tokenizer] = None

        # Store both `model_storage` and `resource` as object attributes to be able
        # to utilize them at the end of the training and persist trained BPE model.
        self._model_storage = model_storage
        self._resource = resource

        self.finetune_mode = execution_context.is_finetuning
        if self.finetune_mode:
            raise ValueError(
                "TikTokenizer: This component does not support finetuning."
            )

    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> GraphComponent:  # TikTokenizer should be TikTokenizer
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
                tok = Tokenizer.from_file(os.path.join(model_dir, cls.tokenizer_fname))

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
        logger.info("Training BPE tokenizer.")
        self.bpe_tokenizer = self._train_tokenizer(training_data)

        with self._model_storage.write_to(self._resource) as model_dir:
            self.bpe_tokenizer.save(str(model_dir / "bpe_tokenizer.json"))

        return self._resource

    def process_training_data(self, training_data: TrainingData) -> TrainingData:
        """Tokenize all training data."""
        for example in training_data.training_examples:
            text = example.get(TEXT, None)
            if text is not None:
                tokens = self.tokenize(text)
                example.set(TOKENS_NAMES[TEXT], tokens)
        return training_data

    def process(self, messages: List[Message]) -> List[Message]:
        """Tokenize the incoming messages at inferences time"""
        for message in messages:
            text = message.get(TEXT, None)
            if text is None:
                logger.warning(
                    f"Message contains no text attribute. Skipping tokenization."
                )
            else:
                tokens = self.tokenize(text)
                message.set(TOKENS_NAMES[TEXT], tokens)
        return messages

    def tokenize(self, text: Text) -> List[Token]:
        encoded = self.bpe_tokenizer.encode(text)
        tokens = []
        logger.warning(f"My Tokens: {encoded.tokens} and ids: {encoded.ids}")

        for token, tok_id in zip(encoded.tokens, encoded.ids):
            start_idx = text.find(token)
            if start_idx != -1:
                end_idx = start_idx + len(token)
                tokens.append(
                    Token(text=token, start=start_idx, end=end_idx, data={"id": tok_id})
                )
            else:
                logger.error(
                    f"Token '{token}' not found in text '{text}'. Skipping token."
                )

        return tokens

    @staticmethod
    def required_packages() -> List[Text]:
        """Any extra python dependencies required for this component to run."""
        return ["tokenizers"]

    @staticmethod
    def get_default_config() -> Dict[Text, Any]:
        return {"vocab_size": 10000}

    def msg_text_generator(self, training_data: TrainingData) -> str:
        for example in training_data.training_examples:
            yield example.get("text")

    def _train_tokenizer(self, training_data: TrainingData) -> Tokenizer:
        """
        Train the BPE tokenizer using the provided training data.
        :param self:
        :param training_data:
        :return:
        """
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = BpeTrainer(
            special_tokens=["[UNK]", "[PAD]"], vocab_size=10000, min_frequency=15
        )
        tokenizer.train_from_iterator(self.msg_text_generator(training_data), trainer)

        return tokenizer
