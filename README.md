# TextRank
Implementation of TextRank with the option of using cosine similarity of word vectors from pre-trained Word2Vec embeddings as the similarity metric.

## Instructions:
The text extract from which keywords are to be extracted can be stored in sample.txt and keywords can be extracted using main.py
```
python3 main.py --data sample.txt
```

## Usage:
```
from keyword_extractor import KeywordExtractor

text = "sample text goes here"
word2vec = "path to pre-trained Word2Vec embeddings (None if pre-trained embeddings are not available"

extractor = KeywordExtractor(word2vec=word2vec)

keywords = extractor.extract(text, ratio=0.2, split=True, scores=True)
for keyword in keywords:
    print(keyword)
```

## Dependencies:
```
gensim
nltk

Use python3
```

## Reference:
-  Mihalcea, Rada, 1974- & Tarau, Paul. TextRank: Bringing Order into Texts, paper, July 2004; [Stroudsburg, Pennsylvania]. (digital.library.unt.edu/ark:/67531/metadc30962/: accessed August 7, 2018), University of North Texas Libraries, Digital Library, digital.library.unt.edu; crediting UNT College of Engineering. 
