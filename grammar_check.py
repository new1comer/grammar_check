from pathlib import Path
import pathlib
import nltk
from nltk import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import numpy as np

def get_data_address(data_folder):
    data_paths = Path(data_folder).glob("*.txt")
    return data_paths

def get_textdata(data_paths):
    corpus = []
    for path in data_paths:
        if path.name != "Dataset_Xiang-Kuperberg_2015.txt":
            with Path(path).open("r",encoding="utf-8") as file:
                doc = file.read().strip()
                corpus.append(doc)
        else:
            with Path(path).open("r",encoding="utf-8") as file:
                docs = file.readlines()
                for text in docs[1:]:
                    tokenized_text = text.split("\t")
                    doc = tokenized_text[2]
                    corpus.append(doc)
    return corpus

def data_preprocess(corpus):
    for doc in corpus:
        doc = doc.lower()
        
    #sentence tokenzie
    sentences = []
    for doc in corpus:
        doc_sentences = sent_tokenize(doc)
        sentences.extend(doc_sentences)
    
    #word tokenize
   
        
    #add start mark <s> and end mark </s> into sentences
    for i in range(len(sentences)):
        sentences[i] = word_tokenize(sentences[i])
        sentences[i] = ["<s>"] + sentences[i] + ["</s>"]
    
    return sentences

def word_fre_cal(sentences):
        
    #token_frequency count dict
    tk_fre = {}
    
    for sent in sentences:
        for word in sent:
            if word in tk_fre:
                tk_fre[word] += 1
            else:
                tk_fre[word] = 1
    
    return tk_fre

def word_prob_cal(tk_fre):
    #token_probabilities
    tk_prob = {}
    total_count = sum(tk_fre.values())
    for tk in tk_fre:
        tk_prob[tk] = tk_fre[tk] / total_count
    
    
    return tk_prob

def bigram_match(sentences):
    bigrams = {}
    for sent in sentences:
        length = len(sent)
        for i in range(len(sent)-1):
            current_bigram = (sent[i], sent[i+1])
            if current_bigram in bigrams:
                bigrams[current_bigram] += 1
            else:
                bigrams[current_bigram] = 1
    return bigrams


def bigram_condition_prob(bigrams, tk_fre):
    bigram_condition_probs = {}
    for key, value in bigrams.items():
        initial_word = key[0]
        prob = value / tk_fre[initial_word]
        bigram_condition_probs[key] = prob
    return bigram_condition_probs

def bigram_prob_check(bigram, bigram_condition_probs):
    if bigram_condition_probs.get(bigram, 0):
        return bigram_condition_probs[bigram]
    else:
        return False
    
def bigram_construct(sent):
    tokenized_sentence = ["<s>"] + word_tokenize(sent) + ["</s>"]
    length = len(tokenized_sentence)
    sent_bigrams = []
    for i in range(length - 1):
        sent_bigrams.append((tokenized_sentence[i],tokenized_sentence[i+1]))
    return sent_bigrams

def bigram_prob_calculate(sent_bigrams, bigram_condition_probs):
    bigram_prob = 1
    for bigram in sent_bigrams:
        if bigram_prob_check(bigram, bigram_condition_probs):
            bigram_prob *= bigram_prob_check(bigram, bigram_condition_probs)
        else:
            bigram_prob *= 1
    return bigram_prob

        


def check_grammar(sent1, sent2, bigram_condition_probs):
    sent1_prob = bigram_prob_calculate(bigram_construct(sent1),bigram_condition_probs)
    sent2_prob = bigram_prob_calculate(bigram_construct(sent2),bigram_condition_probs)
    if sent1_prob > sent2_prob:
        return sent1, sent1_prob
    else:
        return sent2, sent2_prob
    
   
    
def generate_bigram_probs():
    datapaths = get_data_address(Path.cwd().joinpath("data"))
    corpus = get_textdata(datapaths)
    sentences = data_preprocess(corpus)
    bigram_condition_probs = bigram_condition_prob(bigram_match(sentences),word_fre_cal(sentences))
    
    return bigram_condition_probs
    
    


if __name__ == "__main__":
    
    sent1 = "The many important contributions of Omodeo to oligochaetological research are briefly mentioned."
    sent2 = "Information is provided for 38 genus-group names in Mutillidae, of which 13 have been proposed as new since 2008."
    print(check_grammar(sent1, sent2, generate_bigram_probs()))
    
    