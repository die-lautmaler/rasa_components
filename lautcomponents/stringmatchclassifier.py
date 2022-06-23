import os
import logging
from typing import Any, Dict, Optional, Text

from rasa.shared.constants import DOCS_URL_COMPONENTS
from rasa.shared.nlu.training_data.formats.rasa_yaml import RasaYAMLReader
from rasa.nlu import utils
from rasa.nlu.classifiers.classifier import IntentClassifier
from rasa.shared.nlu.constants import INTENT, TEXT
import rasa.shared.utils.io
from rasa.nlu.config import RasaNLUModelConfig
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.shared.nlu.training_data.message import Message
from rasa.nlu.model import Metadata

logger = logging.getLogger(__name__)


class StringMatchClassifier(IntentClassifier):
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

    def __init__(
            self,
            component_config: Optional[Dict[Text, Any]] = None,
            intent_keyword_map: Optional[Dict] = None,
    ) -> None:

        super(StringMatchClassifier, self).__init__(component_config)

        self.case_sensitive = self.component_config.get("case_sensitive")
        self.train_data_file = self.component_config.get("train_data")
        self.max_token = self.component_config.get("max_token")
        self.intent_keyword_map = intent_keyword_map or {}

    def train(self, training_data: TrainingData, config: Optional[RasaNLUModelConfig] = None, **kwargs: Any, ) -> None:
        if self.train_data_file:
            self._load_train_data()
        else:
            self._compute_intent_map(training_data)

        logger.info("mapped {} utterances".format(len(self.intent_keyword_map.keys())))

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

    def process(self, message: Message, **kwargs: Any) -> None:
        """Set the message intent and add it to the output if it exists."""

        if len(message.get("text_tokens")) <= self.max_token:
            if message.get(TEXT).strip() in self.intent_keyword_map:
                intent_name = self.intent_keyword_map[message.get(TEXT).strip()]
                intent = {"name": intent_name, "confidence": 1.0}
                message.set(INTENT, intent, add_to_output=True)

    def persist(self, file_name: Text, model_dir: Text) -> Dict[Text, Any]:
        """Persist this model into the passed directory.
        Return the metadata necessary to load the model again.
        """

        file_name = file_name + ".json"
        keyword_file = os.path.join(model_dir, file_name)
        utils.write_json_to_file(keyword_file, self.intent_keyword_map)

        return {"file": file_name}

    @classmethod
    def load(
            cls,
            meta: Dict[Text, Any],
            model_dir: Text,
            model_metadata: Metadata = None,
            cached_component: Optional["StringMatchClassifier"] = None,
            **kwargs: Any,
    ) -> "StringMatchClassifier":
        """Loads trained component (see parent class for full docstring)."""
        if meta.get("file"):
            file_name = meta.get("file")
            keyword_file = os.path.join(model_dir, file_name)
            if os.path.exists(keyword_file):
                intent_keyword_map = rasa.shared.utils.io.read_json_file(keyword_file)
            else:
                rasa.shared.utils.io.raise_warning(
                    f"Failed to load key word file for `StringMatchClassifier`, "
                    f"maybe {keyword_file} does not exist?"
                )
                intent_keyword_map = None
            return cls(meta, intent_keyword_map)
        else:
            raise Exception(
                f"Failed to load string match intent classifier model. "
                f"Path {os.path.abspath(meta.get('file'))} doesn't exist."
            )
