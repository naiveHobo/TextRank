from keyword_extractor import KeywordExtractor
import argparse


ap = argparse.ArgumentParser()
ap.add_argument("--word2vec", default=None, help="path to word2vec pre-trained embeddings")
ap.add_argument("--data", required=True, help="path to file from which keywords are to be extracted")

args = ap.parse_args()

with open(args.data, 'r') as data_file:
    lines = data_file.readlines()

extractor = KeywordExtractor(word2vec=args.word2vec)

for text in lines:
    keywords = extractor.extract(text, ratio=0.2, split=True, scores=True)
    for keyword in keywords:
        print(keyword)
