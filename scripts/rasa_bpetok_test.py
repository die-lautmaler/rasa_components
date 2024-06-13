import logging
import tempfile
from pathlib import Path
from typing import Dict, Text

from rasa.engine.graph import GraphComponent, ExecutionContext
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.shared.nlu.constants import TEXT
from rasa.nlu.constants import MESSAGE_ATTRIBUTES, TOKENS_NAMES
from lautcomponents.BPETokenizer import BPETokenizer

logger = logging.getLogger(__name__)


def create_temporary_model_storage() -> ModelStorage:
    try:
        temp_dir = tempfile.TemporaryDirectory(dir='bpeTesttmp')
        model_storage = ModelStorage.create(Path(temp_dir.name))
        return model_storage
    except Exception as e:
        logger.error(f"Failed to create model storage: {e}")
        raise


# Mock classes and data for testing
class MockModelStorage(ModelStorage):
    def __init__(self):
        self.storage = {}

    def create(self, resource: Resource) -> 'ModelStorage':
        return self

    def write_to(self, resource: Resource):
        class _MockContextManager:
            def __enter__(self_):
                # Use a temporary directory path
                self_.path = Path(tempfile.mkdtemp())
                return self_.path

            def __exit__(self_, exc_type, exc_val, exc_tb):
                # Clean up the temporary directory if needed
                pass

            def __setitem__(self_, key, value):
                self.storage[key] = value

            def __getitem__(self_, key):
                return self.storage[key]

        return _MockContextManager()

    def read_from(self, resource: Resource):
        class _MockContextManager:
            def __enter__(self_):
                # Use a temporary directory path
                self_.path = Path(tempfile.mkdtemp())
                return self_.path

            def __exit__(self_, exc_type, exc_val, exc_tb):
                # Clean up the temporary directory if needed
                pass

            def __getitem__(self_, key):
                return self.storage[key]

        return _MockContextManager()

    @staticmethod
    def from_model_archive(archive_path: Text, resource: Resource) -> 'ModelStorage':
        return MockModelStorage()

    @staticmethod
    def metadata_from_archive(archive_path: Text) -> Dict:
        return {}

    def create_model_package(self, resource: Resource, model_archive_path: Text) -> None:
        pass

class MockResource(Resource):
    def __init__(self, name: str):
        self.name = name


class MockExecutionContext(ExecutionContext):
    def __init__(self, is_finetuning: bool):
        self.is_finetuning = is_finetuning
        self.node_name = "test_node"


class MockResource(Resource):
    def __init__(self, name: str):
        self.name = name


class MockExecutionContext(ExecutionContext):
    def __init__(self, is_finetuning: bool):
        self.is_finetuning = is_finetuning
        self.node_name = "test_node"


# Create some training data
messages = [
    Message(data={TEXT: "Hello world!"}),
    Message(data={TEXT: "How are you?"}),
    Message(data={TEXT: "This is a test sentence."}),
    Message(data={TEXT: "meine mutter kann das auch"}),
    Message(data={TEXT: "ich wollte immer ich nur selber, sein"}),
]

training_data = TrainingData(training_examples=messages)

# Create the tokenizer component
# model_storage = MockModelStorage()
model_storage = create_temporary_model_storage()

resource = MockResource("bpe_tokenizer_resource")
execution_context = MockExecutionContext(is_finetuning=False)

config = {"vocab_size": 10000}

tokenizer_component = BPETokenizer.create(
    config=config,
    model_storage=model_storage,
    resource=resource,
    execution_context=execution_context,
)

# Train the tokenizer
tokenizer_component.train(training_data)

# Process the messages using the trained tokenizer
tokenized_messages = tokenizer_component.process(messages)

# Print tokenized messages
for msg in tokenized_messages:
    print(f"Original text: {msg.get(TEXT)}")
    print(f"Tokens: {[ tok.text for tok in msg.get(TOKENS_NAMES[TEXT]) ] }")
