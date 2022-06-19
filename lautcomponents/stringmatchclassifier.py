import os
import logging
import re
from typing import Any, Dict, Optional, Text

from rasa.shared.constants import DOCS_URL_COMPONENTS
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

    defaults = {"case_sensitive": False, "train_data": "./data/nlu_kw_classify.csv"}

    def __init__(
            self,
            component_config: Optional[Dict[Text, Any]] = None,
            intent_keyword_map: Optional[Dict] = None,
    ) -> None:

        super(StringMatchClassifier, self).__init__(component_config)

        self.case_sensitive = self.component_config.get("case_sensitive")
        self.train_data_file = self.component_config.get("train_data")
        self.intent_keyword_map = intent_keyword_map or {}

    def train(
            self,
            training_data: TrainingData,
            config: Optional[RasaNLUModelConfig] = None,
            **kwargs: Any,
    ) -> None:
        # TODO: check if extra train data file is given. take normal train data if not
        td_lines = self._load_train_data()
        duplicate_examples = set()
        for exline in td_lines:
            ex = exline.split(',')
            text = ex[0].strip()
            intent = ex[1].strip()
            if (
                    text in self.intent_keyword_map.keys()
                    and intent != self.intent_keyword_map[text]
            ):
                duplicate_examples.add(text)
                rasa.shared.utils.io.raise_warning(
                    f"Keyword '{text}' is a keyword to trigger intent "
                    f"'{self.intent_keyword_map[text]}' and also "
                    f"intent '{intent}', it will be removed "
                    f"from the list of keywords for both of them. "
                    f"Remove (one of) the duplicates from the training data.",
                    docs=DOCS_URL_COMPONENTS + "#keyword-intent-classifier",
                )
            else:
                self.intent_keyword_map[text] = intent
        for keyword in duplicate_examples:
            self.intent_keyword_map.pop(keyword)
            logger.debug(
                f"Removed '{keyword}' from the list of keywords because it was "
                "a keyword for more than one intent."
            )

    # TODO: remove
    # def _validate_keyword_map(self) -> None:
    #     re_flag = 0 if self.case_sensitive else re.IGNORECASE
    #
    #     ambiguous_mappings = []
    #     for keyword1, intent1 in self.intent_keyword_map.items():
    #         for keyword2, intent2 in self.intent_keyword_map.items():
    #             if (
    #                     re.search(r"\b" + keyword1 + r"\b", keyword2, flags=re_flag)
    #                     and intent1 != intent2
    #             ):
    #                 ambiguous_mappings.append((intent1, keyword1))
    #                 rasa.shared.utils.io.raise_warning(
    #                     f"Keyword '{keyword1}' is a keyword of intent '{intent1}', "
    #                     f"but also a substring of '{keyword2}', which is a "
    #                     f"keyword of intent '{intent2}."
    #                     f" '{keyword1}' will be removed from the list of keywords.\n"
    #                     f"Remove (one of) the conflicting keywords from the"
    #                     f" training data.",
    #                     docs=DOCS_URL_COMPONENTS + "#keyword-intent-classifier",
    #                 )
    #     for intent, keyword in ambiguous_mappings:
    #         self.intent_keyword_map.pop(keyword)
    #         logger.debug(
    #             f"Removed keyword '{keyword}' from intent "
    #             f"'{intent}' because it matched a "
    #             f"keyword of another intent."
    #         )

    def _load_train_data(self):
        """load string intent mappings from a yaml file given in train_data option of the component"""
        # TODO: adapt for yaml file read in
        print('load train data from {}'.format(self.train_data_file))
        with open(self.train_data_file, 'r', encoding='utf8') as td_file:
            return td_file.readlines()

    def process(self, message: Message, **kwargs: Any) -> None:
        """Set the message intent and add it to the output if it exists."""
        # only messages with less than 3 tokens are contained
        if len(message.get(TEXT).split(' ')) > 2:
            return

        # intent_name = self._map_keyword_to_intent(message.get(TEXT))
        intent_name = self.intent_keyword_map[message.get(TEXT).strip()]

        confidence = 0.0 if intent_name is None else 1.0
        intent = {"name": intent_name, "confidence": confidence}

        # overwrite classification only if match was found
        if intent_name is not None:
            message.set(INTENT, intent, add_to_output=True)

    def _map_keyword_to_intent(self, text: Text) -> Optional[Text]:
        re_flag = 0 if self.case_sensitive else re.IGNORECASE

        for keyword, intent in self.intent_keyword_map.items():
            # if re.search(r"\b" + keyword + r"\b", text, flags=re_flag):
            if re.match(r"^" + keyword + r"$", text, flags=re_flag):
                logger.debug(
                    f"KeywordClassifier matched keyword '{keyword}' to"
                    f" intent '{intent}'."
                )
                return intent

        logger.debug("KeywordClassifier did not find any keywords in the message.")
        return None

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
