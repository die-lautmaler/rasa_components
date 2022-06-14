# lautmaler rasa components
a package for some rasa pipeline component implementations to be shared ammong projects
the main branch targets rasa 2.8.x

# StringMatchIntentClassifier
A simple string matching intent classifier that can take its own set of training data.
The rasa keyword intent classifier served as a template.
Differences:
1. It does not match substrings of the user message, only full match to one of the train examples to yield a intent classification 
2. The classification result is only overwritten on successful match, otherwise it stays untouched

# Spacy_TRF_Featurizer (obsolete?)
Rasa NLU custom component, Featurizer that get vectors from spacy transformer pipelines
