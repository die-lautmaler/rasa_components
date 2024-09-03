# lautmaler rasa components
A package for some rasa pipeline component implementations to be shared among projects  
The main branch targets rasa 3.2.7 now, rasa 2.8.x moved to its own branch
there is a rasa 3 branch

# StringMatchIntentClassifier
A simple string matching intent classifier that can take its own set of training data.
The rasa keyword intent classifier served as a template.
https://github.com/RasaHQ/rasa/blob/2.8.x/rasa/nlu/classifiers/keyword_intent_classifier.py  

Differences:
1. It does not match substrings of the user message, only full match to one of the train examples to yield a intent classification 
2. The classification result is only overwritten on successful match, otherwise it stays untouched

The need of better handling very short utterances up to 2 tokens long has lead to the idea of using a simple string match approach for these utterances.

### Configuration

You can set the option `max_token` (default 2), it will then take the subset of utterances from the training data, that consist of up to `max_token` as its input. This value should be kept very low to avoid performance issues.

The StringMatchClassifier can also take its own training data set, given in the `train_data` option of its pipeline configuration. It expects a yaml file in standard rasa nlu format.
It will use all utterances in the file, i.e. `max_token` is ignored when `train_data` is set

```yaml
#pipeline
- name: 'lautcomponents.StringMatchClassifier'
  max_token: 2 #optional
  train_data: './pathTo/stringMatchTrain.yml' #optional
```

### Classification Behavior

The training step just builds a simple map from utterances to intents.

At classification time, the token count of the incoming utterance is computed  to avoid unecessary lookups in the map.

If the incoming utterance is a key in the map, the pertaining intent is returned. The potentially existing classification result from a previous component gets overwritten.

Classifications from the StringMatchClassifier have confidence 1.0

If no match is found, nothing is returned, leaving results from previous components intact, which also means intent classification will be None if there is no previous component.

# BPETokenizer
A simple tokenizer that uses the byte pair encoding algorithm to split words into subwords. A simple integration of the Huggingface tokenizers library.
You can train the Byte Pair Encoding model on your own your NLU training data or use a pre-trained model from the Huggingface model hub.

### Configuration
If you dont want to train the tokenizer on your training data you can load it from a model file.  
If you want to train the tokenizer you can set the `vocab_size` and `min_frequency` parameters.  

```yaml
- name: 'lautcomponents.BPETokenizer'
  model: 'pathToModel' #optional
  vocab_size: 5000 #default
  min_frequency: 0 #default
```

# CountVectorsFeaturizerBPE
A simple featurizer that uses the Byte Pair Encoding tokenizer to create count vectors from the subwords. It is dircectly derived from the CountVectorsFeaturizer of rasa.
It loads the Byte Pair Encoding model from the BPETokenizer component, and retokenizes the incoming messages to create the count vectors. This mean a traditional tokenizer can 
still be present in the pipeline, but the Byte Pair Encoding model will be used for the featurization. 
The BPETokenizer is therefore not needed in the pipeline.
Since BPE Tokens are mostly subwords, the featurizer should be used in word level, not char level mode (Just an assumption).

## Configuration
The CountVectorsFeaturizerBPE does accept the same configuration as the [CountVectorsFeaturizer](https://rasa.com/docs/rasa/components/#countvectorsfeaturizer) of rasa. The difference lier only in the shape of the Tokens that are used.

```yaml

- name: "lautcomponents.CountVectorsFeaturizerBPE"
  # Analyzer to use, either 'word', 'char', or 'char_wb'
  "analyzer": "word"
  # Set the lower and upper boundaries for the n-grams
  "min_ngram": 1
  "max_ngram": 1
  # Set the out-of-vocabulary token
  "OOV_token": "_oov_"
  # Whether to use a shared vocab
  "use_shared_vocab": False

```
