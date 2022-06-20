# lautmaler rasa components
A package for some rasa pipeline component implementations to be shared among projects  
The main branch targets rasa 2.8.x

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
