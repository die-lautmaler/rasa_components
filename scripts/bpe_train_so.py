# Train a BPE Tokenizer from scratch and save it
# Usage: python bpe_train_so.py <path-to-utterances>
# can load rasa nlu yaml files or plain text files with one utterance per line

import logging
import sys

from rasa.shared.nlu.training_data.formats.rasa_yaml import RasaYAMLReader
from rasa.shared.nlu.constants import TEXT

from tokenizers import Tokenizer as HuggingfaceTokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

from typing import List, Generator

logging.root.setLevel(0)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


def load_utts(fpaths: List[str]) -> Generator[str, None, None]:
    utt_count = 0
    for fpath in fpaths:
        if fpath.endswith(".yml") or fpath.endswith(".yaml"):
            rasa_reader = RasaYAMLReader()
            tdata = rasa_reader.read(fpath)
            for example in tdata.training_examples:
                utt_count += 1
                yield example.get(TEXT)
        else:
            with open(fpath, "r") as f:
                for line in f.readlines():
                    utt_count += 1
                    yield line.strip()

    logger.info(f"Loaded {utt_count} utterances in total")


def train_bpe(
    utts: Generator[str, None, None], vocab_size: int = 1000, min_frequency: int = 10
) -> HuggingfaceTokenizer:
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
    tokenizer.train_from_iterator(utts, trainer)

    return tokenizer


def main():
    if len(sys.argv) > 1:

        tokenizer = train_bpe(load_utts(sys.argv), 10500, min_frequency=100)
        logger.info(
            f"BPE tokenizer trained, final vocab size: {tokenizer.get_vocab_size()}"
        )
        tokenizer.save("sipgate_bpe.json")
    else:
        print("Usage: python bpe_train_so.py <path-to-utterances>")
        sys.exit(1)


if __name__ == "__main__":
    main()
