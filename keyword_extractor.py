from itertools import combinations
from queue import Queue
from graph import Graph
from preprocessing import TextProcessor
from gensim.models import KeyedVectors


class KeywordExtractor:
    """
    Extracts keywords from text using TextRank algorithm
    """

    def __init__(self, word2vec=None):
        self.preprocess = TextProcessor()
        self.graph = Graph()
        if word2vec:
            print("Loading word2vec embedding...")
            self.word2vec = KeyedVectors.load_word2vec_format(word2vec, binary=True)
            print("Succesfully loaded word2vec embeddings!")
        else:
            self.word2vec = None

    def init_graph(self):
        self.preprocess = TextProcessor()
        self.graph = Graph()

    def extract(self, text, ratio=0.4, split=False, scores=False):
        """
        :param: text: text data from which keywords are to be extracted
        :return: list of keywords extracted from text
        """
        self.init_graph()
        words = self.preprocess.tokenize(text)
        tokens = self.preprocess.clean_text(text)
        for word, item in tokens.items():
            if not self.graph.has_node(item.token):
                self.graph.add_node(item.token)
        self.__set_graph_edges(self.graph, tokens, words)
        del words
        KeywordExtractor.__remove_unreachable_nodes(self.graph)
        if len(self.graph.nodes()) == 0:
            return [] if split else ""
        pagerank_scores = self.__textrank()
        extracted_lemmas = KeywordExtractor.__extract_tokens(self.graph.nodes(), pagerank_scores, ratio)
        lemmas_to_word = KeywordExtractor.__lemmas_to_words(tokens)
        keywords = KeywordExtractor.__get_keywords_with_score(extracted_lemmas, lemmas_to_word)
        combined_keywords = self.__get_combined_keywords(keywords, text.split())
        return KeywordExtractor.__format_results(keywords, combined_keywords, split, scores)

    def __textrank(self, initial_value=None, damping=0.85, convergence_threshold=0.0001):
        """Implementation of TextRank on a undirected graph"""
        if not initial_value:
            initial_value = 1.0 / len(self.graph.nodes())
        scores = dict.fromkeys(self.graph.nodes(), initial_value)

        iteration_quantity = 0
        for iteration_number in range(100):
            iteration_quantity += 1
            convergence_achieved = 0
            for i in self.graph.nodes():
                rank = 1 - damping
                for j in self.graph.neighbors(i):
                    neighbors_sum = sum(self.graph.edge_weight((j, k)) for k in self.graph.neighbors(j))
                    rank += damping * scores[j] * self.graph.edge_weight((j, i)) / neighbors_sum
                if abs(scores[i] - rank) <= convergence_threshold:
                    convergence_achieved += 1
                scores[i] = rank
            if convergence_achieved == len(self.graph.nodes()):
                break
        return scores

    @staticmethod
    def __format_results(_keywords, combined_keywords, split, scores):
        """
        :param _keywords:dict of keywords:scores
        :param combined_keywords:list of word/s
        """
        combined_keywords.sort(key=lambda w: KeywordExtractor.__get_average_score(w, _keywords), reverse=True)
        if scores:
            return [(word, KeywordExtractor.__get_average_score(word, _keywords)) for word in combined_keywords]
        if split:
            return combined_keywords
        return "\n".join(combined_keywords)

    @staticmethod
    def __get_average_score(concept, _keywords):
        """Calculates average score"""
        word_list = concept.split()
        word_counter = 0
        total = 0
        for word in word_list:
            total += _keywords[word]
            word_counter += 1
        return total / word_counter

    def __strip_word(self, word):
        """Preprocesses given word"""
        stripped_word_list = list(self.preprocess.tokenize(word))
        return stripped_word_list[0] if stripped_word_list else ""

    def __get_combined_keywords(self, _keywords, split_text):
        """
        :param _keywords:dict of keywords:scores
        :param split_text: list of strings
        :return: combined_keywords:list
        """
        result = []
        _keywords = _keywords.copy()
        len_text = len(split_text)
        for i in range(len_text):
            word = self.__strip_word(split_text[i])
            if word in _keywords:
                combined_word = [word]
                if i + 1 == len_text:
                    result.append(word)  # appends last word if keyword and doesn't iterate
                for j in range(i + 1, len_text):
                    other_word = self.__strip_word(split_text[j])
                    if other_word in _keywords and other_word == split_text[j] \
                            and other_word not in combined_word:
                        combined_word.append(other_word)
                    else:
                        for keyword in combined_word:
                            _keywords.pop(keyword)
                        result.append(" ".join(combined_word))
                        break
        return result

    @staticmethod
    def __get_keywords_with_score(extracted_lemmas, lemma_to_word):
        """
        :param extracted_lemmas:list of tuples
        :param lemma_to_word: dict of {lemma:list of words}
        :return: dict of {keyword:score}
        """
        keywords = {}
        for score, lemma in extracted_lemmas:
            keyword_list = lemma_to_word[lemma]
            for keyword in keyword_list:
                keywords[keyword] = score
        return keywords

    @staticmethod
    def __lemmas_to_words(tokens):
        """Returns the corresponding words for the given lemmas"""
        lemma_to_word = {}
        for word, unit in tokens.items():
            lemma = unit.token
            if lemma in lemma_to_word:
                lemma_to_word[lemma].append(word)
            else:
                lemma_to_word[lemma] = [word]
        return lemma_to_word

    @staticmethod
    def __extract_tokens(lemmas, scores, ratio):
        lemmas.sort(key=lambda s: scores[s], reverse=True)
        length = len(lemmas) * ratio
        return [(scores[lemmas[i]], lemmas[i],) for i in range(int(length))]

    @staticmethod
    def __remove_unreachable_nodes(graph):
        for node in graph.nodes():
            if sum(graph.edge_weight((node, other)) for other in graph.neighbors(node)) == 0:
                graph.del_node(node)

    def __set_graph_edges(self, graph, tokens, words):
        self.__process_first_window(graph, tokens, words)
        self.__process_text(graph, tokens, words)

    def __process_first_window(self, graph, tokens, split_text):
        first_window = KeywordExtractor.__get_first_window(split_text)
        for word_a, word_b in combinations(first_window, 2):
            self.__set_graph_edge(graph, tokens, word_a, word_b)

    def __process_text(self, graph, tokens, split_text):
        queue = KeywordExtractor.__init_queue(split_text)
        for i in range(2, len(split_text)):
            word = split_text[i]
            self.__process_word(graph, tokens, queue, word)
            KeywordExtractor.__update_queue(queue, word)

    def __set_graph_edge(self, graph, tokens, word_a, word_b):
        if word_a in tokens and word_b in tokens:
            lemma_a = tokens[word_a].token
            lemma_b = tokens[word_b].token
            edge = (lemma_a, lemma_b)

            if graph.has_node(lemma_a) and graph.has_node(lemma_b) and not graph.has_edge(edge):
                if not self.word2vec:
                    graph.add_edge(edge)
                else:
                    try:
                        similarity = self.word2vec.similarity(lemma_a, lemma_b)
                        if similarity < 0:
                            similarity = 0.0
                    except:
                        similarity = 0.2
                    graph.add_edge(edge, wt=similarity)

    def __process_word(self, graph, tokens, queue, word):
        for word_to_compare in KeywordExtractor.__queue_iterator(queue):
            self.__set_graph_edge(graph, tokens, word, word_to_compare)

    @staticmethod
    def __get_first_window(split_text):
        return split_text[:2]

    @staticmethod
    def __init_queue(split_text):
        queue = Queue()
        first_window = KeywordExtractor.__get_first_window(split_text)
        for word in first_window[1:]:
            queue.put(word)
        return queue

    @staticmethod
    def __update_queue(queue, word):
        queue.get()
        queue.put(word)
        assert queue.qsize() == 1

    @staticmethod
    def __queue_iterator(queue):
        iterations = queue.qsize()
        for i in range(iterations):
            var = queue.get()
            yield var
            queue.put(var)
