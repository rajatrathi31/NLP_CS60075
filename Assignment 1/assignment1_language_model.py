'''The purpose of this code is to generate sentences using n-gram language model'''
from itertools import product
import math
import random
from sys import set_coroutine_origin_tracking_depth
import nltk
from assignment1_preprocess import preprocess

class NgramLanguageModel(object):
    def __init__(self, train_data, n, laplace = 1):
        self.n = n
        self.laplace = laplace
        self.tokens = preprocess(train_data, n)
        self.vocab = nltk.FreqDist(self.tokens)
        self.model = self.model_creation()
        self.masks = list(reversed(list(product((0,1), repeat=n))))

    def model_creation(self):
        '''This function is used to apply Laplace Smoothing'''

        if self.n > 1:
            return self._smooth()
        else:
            num_tokens = len(self.tokens)
            return { (unigram,): count / num_tokens for unigram, count in self.vocab.items() }

    def _smooth(self):
        '''This function is used to apply Laplace Smoothing'''

        vocab_size = len(self.vocab)
        n_grams = nltk.ngrams(self.tokens, self.n)
        n_vocab = nltk.FreqDist(n_grams)

        m_grams = nltk.ngrams(self.tokens, self.n-1)
        m_vocab = nltk.FreqDist(m_grams)

        _dict = {n_gram: (count + self.laplace) / (m_vocab[n_gram[:-1]] + (self.laplace * vocab_size)) for n_gram, count in n_vocab.items()}
        return _dict

    def best_candidate(self, prev, blocked_tokens=[]):
        '''This function picks up the best token for the next word'''

        # As the name suggests this blocked_tokens list contains the words that should not be included
        blocked_tokens = blocked_tokens + ["<UNK>"]
        
        candidates_words = []
        candidates_prob = []
        for ngram, prob in self.model.items():
            if ngram[:-1] == prev and ngram[-1] not in blocked_tokens:
                candidates_words.append(ngram[-1])
                candidates_prob.append(prob)
        
        # zipping both the words and probability together
        candidates = list(zip(candidates_words, candidates_prob))
        candidates = sorted(candidates, key=lambda candidate: candidate[1], reverse=True)
        
        if len(candidates) != 0:
            return candidates[0 if prev != () and prev[-1] != "<s>" else random.randint(0, len(candidates)-1)]
        else:
            return ("</s>", 1)
            

    def generate_sentences(self, num, min_len=10, max_len=20):
        '''This function is used to generate sentences by predicting next word over and over again'''
        
        for i in range(num):
            prob = 1
            sent = ["<s>"] * max(1, self.n-1)
            
            # This loop runs till the last token is </s>
            while sent[-1] != "</s>":
                if self.n == 1:
                    prev = ()
                else:
                    prev = tuple(sent[-(self.n-1):])

                if len(sent) < min_len:
                    blocked_tokens = sent + ["</s>"]
                
                next_token, next_prob = self.best_candidate(prev, blocked_tokens)
                sent.append(next_token)
                prob *= next_prob

                if len(sent) >= max_len:
                    sent.append("</s>")

            yield ' '.join(sent)

    def convert_oov(self, ngram):
        mask = lambda ngram, bitmask: tuple((token if flag == 1 else "<UNK>" for token,flag in zip(ngram, bitmask)))

        ngram = (ngram,) if type(ngram) is str else ngram
        for possible_known in [mask(ngram, bitmask) for bitmask in self.masks]:
            if possible_known in self.model:
                return possible_known

    def perplexity(self, test_data):
        '''This function calculates the model perplexity on the testing data
        Less the perplexity better the model'''
        
        # preprocesses test data and generate n-grams
        test_tokens = preprocess(test_data, self.n)
        test_ngrams = nltk.ngrams(test_tokens, self.n)
        N = len(test_tokens)

        known_ngrams = (self.convert_oov(ngram) for ngram in test_ngrams)
        probabilities = [self.model[ngram] for ngram in known_ngrams]

        cross_entropy_loss = 0
        for prob in probabilities:
            cross_entropy_loss -= math.log(prob)
        cross_entropy_loss = cross_entropy_loss / N 
        
        ppl = math.exp(cross_entropy_loss)
        return ppl

def load_corpus():
    '''Load training and testing data'''

    # Change the path relative to this main file
    with open("./corpus/train.txt", 'r') as f:
        train = [l.strip() for l in f.readlines()]
    with open("./corpus/test.txt", 'r') as f:
        test = [l.strip() for l in f.readlines()]

    return train, test


if __name__ == '__main__':
    '''This n-gram-model takes the value of "n" as input as well as 
    the number of sentences user wants to generate and then generate 
    the number of sentences.
    
    It also calculates the perplexity and shows it as ouput.'''

    print("Attention: Please put the training and testing files in the corpus folder and then load the program")
    
    n = int(input("Enter the value of n for a n-gram model: "))
    
    print("Laplace value is set to 1 by default")
    
    num = int(input("Enter the number of sentences you want to generate: "))
    
    train, test = load_corpus()
    
    print("Loading {}-gram model...".format(n))
    lm = NgramLanguageModel(train, n, laplace=1)

    print("Vocabulary size: {}".format(len(lm.vocab)))

    for sentence in lm.generate_sentences(num):
        if n <= 2:
            print("{}".format(sentence))
        else:
            sentence_tokens = sentence.split()
            sentence_tokens = sentence_tokens[n-2:]
            new_sentence = ' '.join(sentence_tokens)
            print("{}".format(new_sentence))


    ppl = lm.perplexity(test)
    print("The model perplexity came out to be: {:.3f}".format(ppl))