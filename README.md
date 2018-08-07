# TextRank
Implementation of TextRank with the option of using pre-trained Word2Vec embeddings as the similarity metric

## Usage:
```
from keyword_extractor import KeywordExtractor

text = "sample text goes here"
extractor = KeywordExtractor(word2vec=args.word2vec)

keywords = extractor.extract(text, ratio=0.2, split=True, scores=True)
for keyword in keywords:
    print(keyword)
```
