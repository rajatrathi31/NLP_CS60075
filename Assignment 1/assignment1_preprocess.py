import nltk
SOS = "<s> "
EOS = "</s>"

def replace_unknown(tokens):
    '''Replaces unknown words with UNK token'''
    
    vocab = nltk.FreqDist(tokens)
    
    result = []

    for token in tokens:
        if vocab[token] > 1:
            result.append(token)
        else:
            result.append("<UNK>")
    
    return result

def add_sentence_tokens(sentences, n):
    '''This function adds Start and End tokens'''
    
    if n > 1:
        sos = SOS * (n-1)
    else:
        sos = SOS
    
    return ['{}{} {}'.format(sos, s, EOS) for s in sentences]

def preprocess(sentences, n):
    '''This function adds Start and End tokens also replaces unknown words with UNK token'''
    
    sentences = add_sentence_tokens(sentences, n)
    
    # Splitting sentence through white space into tokens
    tokens = ' '.join(sentences).split()
    
    # Calling replace_unknown to replace unknown tokens
    tokens = replace_unknown(tokens)
    
    return tokens

if __name__ == '__main__':
    sentences = ["Hello How are you", "Hmm Yeah", "what are these"]
    sentences = preprocess(sentences,3)
    print(sentences)