##tagger1  Without using pretrained vectors
The command line takes two arguments: the train file and the dev one:

* **POS :**
`python Ass2/part1/tagger1.py data/pos/train data/pos/dev`  
* **NER :**
`python Ass2/part1/tagger1.py data/ner/train data/ner/dev`  
##tagger3  Using pretrained
The command line takes four arguments: the train file, the dev one, the vocab text and the wordVectors:

* **POS :**
`python Ass2/part1/tagger1.py data/pos/train data/pos/dev  data/vocab.txt data/wordVectors.txt`  
* **NER :**
`python Ass2/part1/tagger1.py data/ner/train data/ner/dev  data/vocab.txt data/wordVectors.txt`
