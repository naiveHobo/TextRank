import re
import string
import unicodedata
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from syntactic_unit import SyntacticUnit


class TextProcessor:
    """
    Pre-process text data to prepare for keyword extraction
    """

    def __init__(self):
        self.STOPWORDS = TextProcessor.__load_stopwords(path="../stopwords.txt")
        self.LEMMATIZER = WordNetLemmatizer()
        self.STEMMER = SnowballStemmer("english")
        self.PUNCTUATION = re.compile('([%s])+' % re.escape(string.punctuation), re.UNICODE)
        self.NUMERIC = re.compile(r"[0-9]+", re.UNICODE)
        self.PAT_ALPHABETIC = re.compile('(((?![\d])\w)+)', re.UNICODE)

    def remove_punctuation(self, s):
        """Removes punctuation from text"""
        return self.PUNCTUATION.sub(" ", s)

    def remove_numeric(self, s):
        """Removes numeric characters from text"""
        return self.NUMERIC.sub("", s)

    def remove_stopwords(self, tokens):
        """Removes stopwords from text"""
        return [w for w in tokens if w not in self.STOPWORDS]

    def stem_tokens(self, tokens):
        """Performs stemming on text data"""
        return [self.STEMMER.stem(word) for word in tokens]

    def lemmatize_tokens(self, tokens):
        """Performs lemmatization on text data using Part-of-Speech tags"""
        if not tokens:
            return []
        if isinstance(tokens[0], str):
            pos_tags = pos_tag(tokens)
        else:
            pos_tags = tokens
        tokens = [self.LEMMATIZER.lemmatize(word[0]) if not TextProcessor.__get_wordnet_pos(word[1])
                  else self.LEMMATIZER.lemmatize(word[0], pos=TextProcessor.__get_wordnet_pos(word[1]))
                  for word in pos_tags]
        return tokens

    def part_of_speech_tag(self, tokens):
        if isinstance(tokens, str):
            tokens = self.tokenize(tokens)
        return pos_tag(tokens)

    @staticmethod
    def __load_stopwords(path="stopwords.txt"):
        """Utility function to load stopwords from text file"""
        # with open(path, "r") as stopword_file:
        #     stopwords = [line.strip() for line in stopword_file.readlines()]
        return list(set(stopwords.words('english')))

    @staticmethod
    def __get_wordnet_pos(treebank_tag):
        """Maps the treebank tags to WordNet part of speech names"""
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return None

    @staticmethod
    def deaccent(s):
        """Remove accentuation from the given string"""
        norm = unicodedata.normalize("NFD", s)
        result = "".join(ch for ch in norm if unicodedata.category(ch) != 'Mn')
        return unicodedata.normalize("NFC", result)

    def clean_text(self, text, filters=None, stem=False):
        """ Tokenizes a given text into words, applying filters and lemmatizing them.
        Returns a dict of word -> SyntacticUnit"""
        text = text.lower()
        text = self.deaccent(text)
        text = self.remove_numeric(text)
        text = self.remove_punctuation(text)
        original_words = [match.group() for match in self.PAT_ALPHABETIC.finditer(text)]
        filtered_words = self.remove_stopwords(original_words)
        pos_tags = pos_tag(filtered_words)
        if stem:
            filtered_words = self.stem_tokens(filtered_words)
        else:
            filtered_words = self.lemmatize_tokens(pos_tags)
        units = []
        if not filters:
            filters = ['N', 'J']
        for i in range(len(filtered_words)):
            if not pos_tags[i][1].startswith('N') or len(filtered_words[i]) < 3:
                continue
            token = filtered_words[i]
            text = filtered_words[i]
            tag = pos_tags[i][1]
            sentence = SyntacticUnit(text, token, tag)
            sentence.index = i
            units.append(sentence)
        return {unit.text: unit for unit in units}

    def tokenize(self, text):
        """Performs basic preprocessing and tokenizes text data"""
        text = text.lower()
        text = self.deaccent(text)
        return [match.group() for match in self.PAT_ALPHABETIC.finditer(text)]

    def clean_sentence(self, text):
        """Cleans sentence for word2vec training"""
        text = text.lower()
        text = self.deaccent(text)
        text = self.remove_numeric(text)
        text = self.remove_punctuation(text)
        return text
