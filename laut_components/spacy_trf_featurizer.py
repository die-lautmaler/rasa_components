import numpy as np
import typing
import logging
from typing import Any, Optional, Text, Dict, List, Type

from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.components import Component
from rasa.nlu.featurizers.featurizer import DenseFeaturizer
from rasa.shared.nlu.training_data.features import Features
from rasa.nlu.utils.spacy_utils import SpacyNLP
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.shared.nlu.training_data.message import Message
from rasa.nlu.constants import (
    SPACY_DOCS,
    DENSE_FEATURIZABLE_ATTRIBUTES,
    FEATURIZER_CLASS_ALIAS,
)
from rasa.shared.nlu.constants import TEXT, FEATURE_TYPE_SENTENCE, FEATURE_TYPE_SEQUENCE
from rasa.utils.tensorflow.constants import POOLING, MEAN_POOLING

if typing.TYPE_CHECKING:
    from spacy.tokens import Doc


logger = logging.getLogger(__name__)


class SpacyTrfFeaturizer(DenseFeaturizer):

    name = 'SpacyTrfFeaturizer'
    provides = ['features']
    supported_language_list = ['de', 'en']

    @classmethod
    def required_components(cls) -> List[Type[Component]]:
        return [SpacyNLP]

    defaults = {
        # Specify what pooling operation should be used to calculate the vector of
        # the complete utterance. Available options: 'mean' and 'max'
        POOLING: MEAN_POOLING
    }

    def __init__(self, component_config: Optional[Dict[Text, Any]] = None):
        super().__init__(component_config)

        self.pooling_operation = self.component_config[POOLING]

    def _features_for_doc(self, doc: "Doc") -> np.ndarray:
        """Feature vector for a single document / sentence / tokens."""
        return np.array([t for t in doc._.trf_data.tensors[0]])

    def train(
            self,
            training_data: TrainingData,
            config: Optional[RasaNLUModelConfig] = None,
            **kwargs: Any,
    ) -> None:

        for example in training_data.training_examples:
            for attribute in DENSE_FEATURIZABLE_ATTRIBUTES:
                self._set_spacy_features(example, attribute)

    def get_doc(self, message: Message, attribute: Text) -> Any:
        return message.get(SPACY_DOCS[attribute])

    def process(self, message: Message, **kwargs: Any) -> None:
        for attribute in DENSE_FEATURIZABLE_ATTRIBUTES:
            self._set_spacy_features(message, attribute)

    def _set_spacy_features(self, message: Message, attribute: Text = TEXT) -> None:
        """Adds the spacy word vectors to the messages features."""
        doc = self.get_doc(message, attribute)

        if doc is None:
            return

        # in case an empty spaCy model was used, no vectors are present
        if len(doc._.trf_data.tensors) == 0:
            print('No features present. You are using an empty spaCy model.')
            logger.debug("No features present. You are using an empty spaCy model.")
            return

        # only add whole sentence feature vector
        #sequence_features = self._features_for_doc(doc)
        #final_sequence_features = Features(
        #    sequence_features,
        #    FEATURE_TYPE_SEQUENCE,
        #    attribute,
        #    self.component_config[FEATURIZER_CLASS_ALIAS],
        #)
        #message.add_features(final_sequence_features)

        sentence_features = doc._.trf_data.tensors[1]
        final_sentence_features = Features(
            sentence_features,
            FEATURE_TYPE_SENTENCE,
            attribute,
            self.component_config[FEATURIZER_CLASS_ALIAS],
        )
        message.add_features(final_sentence_features)