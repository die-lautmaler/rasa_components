from __future__ import annotations
import logging
from typing import Any, Dict, Optional, Text, List


from rasa.engine.graph import GraphComponent, ExecutionContext
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.constants import DOCS_URL_COMPONENTS
from rasa.nlu.classifiers.classifier import IntentClassifier
from rasa.shared.nlu.constants import INTENT, TEXT
import rasa.shared.utils.io
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.shared.nlu.training_data.message import Message

###################################
from rasa.shared.constants import DOCS_URL_COMPONENTS
from rasa.shared.nlu.training_data.formats.rasa_yaml import RasaYAMLReader
import rasa.shared.utils.io

logger = logging.getLogger(__name__)


@DefaultV1Recipe.register(
    DefaultV1Recipe.ComponentType.INTENT_CLASSIFIER, is_trainable=True
)
class StringMatchClassifier(GraphComponent, IntentClassifier):
    """
    Template taken from
    https://github.com/RasaHQ/rasa/blob/2.8.x/rasa/nlu/classifiers/keyword_intent_classifier.py
    with some adaptions
    Intent classifier using simple string  matching.
    The classifier takes a list of strings and associated intents as an input.
    An input message is checked if it matches any of the given strings and the intent is returned.
    classification result is not altered on no match
    """

    defaults = {"case_sensitive": False, "max_token": 2, "train_data": None}

    @staticmethod
    def get_default_config() -> Dict[Text, Any]:
        """The component's default config (see parent class for full docstring)."""
        return {"case_sensitive": False, "max_token": 2, "train_data": None}

    def __init__(
            self,
            config: Dict[Text, Any],
            model_storage: ModelStorage,
            resource: Resource,
            execution_context: ExecutionContext,
            intent_keyword_map: Optional[Dict] = None,
    ) -> None:

        # super(StringMatchClassifier, self).__init__(config)
        self.component_config = config
        self._model_storage = model_storage
        self._resource = resource
        self._execution_context = execution_context

        self.case_sensitive = self.component_config.get("case_sensitive")
        self.train_data_file = self.component_config.get("train_data")
        self.max_token = self.component_config.get("max_token")
        self.intent_keyword_map = intent_keyword_map or {}

    @classmethod
    def create(
            cls,
            config: Dict[Text, Any],
            model_storage: ModelStorage,
            resource: Resource,
            execution_context: ExecutionContext,
    ) -> StringMatchClassifier:
        """Creates a new untrained component (see parent class for full docstring)."""
        return cls(config, model_storage, resource, execution_context)

    def train(self, training_data: TrainingData) -> Resource:
        if self.train_data_file:
            self._load_train_data()
        else:
            self._compute_intent_map(training_data)

        logger.info("mapped {} utterances".format(len(self.intent_keyword_map.keys())))
        self.persist()
        return self._resource

    def _load_train_data(self):
        logger.info("reading train data file {}".format(self.train_data_file))
        train_data = RasaYAMLReader().read(self.train_data_file)

        duplicate_examples = set()
        for ex in train_data.intent_examples:
            text = ex.get(TEXT)
            # print(text)
            intent = ex.get(INTENT)
            # print(ex.get("text_spacy_doc"))

            if text in self.intent_keyword_map.keys() and intent != self.intent_keyword_map[text]:
                duplicate_examples.add(text)
                rasa.shared.utils.io.raise_warning(
                    f"text '{text}' is a trigger for intent "
                    f"'{self.intent_keyword_map[text]}' and also "
                    f"intent '{intent}', it will be removed "
                    f"from the list of keywords for both of them. "
                    f"Remove (one of) the duplicates from the training data.",
                    docs=DOCS_URL_COMPONENTS + "#string-match-classifier",
                )
            else:
                self.intent_keyword_map[text] = intent

        for utterance in duplicate_examples:
            self.intent_keyword_map.pop(utterance)
            logger.debug(
                f"Removed '{utterance}' from the list of utterances because it was "
                "an utterance for more than one intent."
            )

    def _compute_intent_map(self, train_data: TrainingData):
        logger.info('mapping utterances with max {} tokens'.format(self.max_token))

        duplicate_examples = set()
        for ex in train_data.intent_examples:
            text = ex.get(TEXT)
            intent = ex.get(INTENT)

            if len(ex.get("text_tokens")) <= self.max_token:
                if text in self.intent_keyword_map.keys() and intent != self.intent_keyword_map[text]:
                    duplicate_examples.add(text)
                    rasa.shared.utils.io.raise_warning(
                        f"text '{text}' is a trigger for intent "
                        f"'{self.intent_keyword_map[text]}' and also "
                        f"intent '{intent}', it will be removed "
                        f"from the list of keywords for both of them. "
                        f"Remove (one of) the duplicates from the training data.",
                        docs=DOCS_URL_COMPONENTS + "#string-match-classifier",
                    )
                else:
                    self.intent_keyword_map[text] = intent

        for utterance in duplicate_examples:
            self.intent_keyword_map.pop(utterance)
            logger.debug(
                f"Removed '{utterance}' from the list of utterances because it was "
                "an utterance for more than one intent."
            )

    def process(self, messages: List[Message]) -> List[Message]:
        """Set the message intent and add it to the output if it exists."""

        for message in messages:
            if len(message.get("text_tokens")) <= self.max_token:
                if message.get(TEXT).strip() in self.intent_keyword_map:
                    intent_name = self.intent_keyword_map[message.get(TEXT).strip()]
                    intent = {"name": intent_name, "confidence": 1.0}
                    message.set(INTENT, intent, add_to_output=True)

        return messages

    def persist(self) -> None:
        """Persist this model into the passed directory.
        Return the metadata necessary to load the model again.
        """
        with self._model_storage.write_to(self._resource) as model_dir:
            file_name = f"{self.__class__.__name__}.json"
            keyword_file = model_dir / file_name
            rasa.shared.utils.io.dump_obj_as_json_to_file(
                keyword_file, self.intent_keyword_map
            )

    @classmethod
    def load(
            cls,
            config: Dict[Text, Any],
            model_storage: ModelStorage,
            resource: Resource,
            execution_context: ExecutionContext,
            **kwargs: Any,
    ) -> StringMatchClassifier:
        """Loads trained component (see parent class for full docstring)."""
        try:
            with model_storage.read_from(resource) as model_dir:
                keyword_file = model_dir / f"{cls.__name__}.json"
                intent_keyword_map = rasa.shared.utils.io.read_json_file(keyword_file)
        except ValueError:
            logger.warning(
                f"Failed to load {cls.__class__.__name__} from model storage. Resource "
                f"'{resource.name}' doesn't exist."
            )
            intent_keyword_map = None

        return cls(
            config, model_storage, resource, execution_context, intent_keyword_map
        )
