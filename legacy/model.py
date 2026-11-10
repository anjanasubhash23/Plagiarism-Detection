from numpy.linalg import norm
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.util import ngrams
from nltk import word_tokenize
from nltk.corpus import indian
from nltk.tag import tnt
from nltk.util import unique_list
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from stopwords import hi_stop, hi_stop1


def Calculate_Percentage(text_1, text_2):
    final_list = []
    Original_text = text_1
    Plagarized_text = text_2
    # Original file

    word_tokens = word_tokenize(Original_text)
    stop = [w for w in hi_stop if w in word_tokens]
    filtered_sentence = [w for w in word_tokens if not w in hi_stop]
    filtered_sentence = []
    for w in word_tokens:
        if w not in hi_stop:
            filtered_sentence.append(w)
    after_stopword = (' '.join(filtered_sentence))

    # Plagarized File

    word_tokens1 = word_tokenize(Plagarized_text)
    stop1 = [w for w in hi_stop1 if w in word_tokens1]

    filtered_sentence1 = [w for w in word_tokens1 if not w in hi_stop1]
    filtered_sentence1 = []

    for w in word_tokens1:
        if w not in hi_stop1:
            filtered_sentence1.append(w)

    after_stopword1 = (' '.join(filtered_sentence1))

    # Original File

    with open('lemma (2).txt', encoding="utf-8") as f:
        dict = {}
        for line in f:
            x, y = line.split(":")
            dict[x.strip()] = y.strip()
        # print(dict)

    lemma = []
    b = list(dict.keys())

    for word in filtered_sentence:
        if word in b:
            lemma.append(dict[word])
        else:
            lemma.append(word)

    after_lemma = (' '.join(lemma))

    # Plagarized File

    with open('lemma (2).txt', encoding="utf-8") as f:
        dict1 = {}
        for line in f:
            x, y = line.split(":")
            dict1[x.strip()] = y.strip()
        # print(dict)

    lemma1 = []
    b = list(dict1.keys())

    for word in filtered_sentence1:
        if word in b:
            lemma1.append(dict1[word])
        else:
            lemma1.append(word)

    after_lemma1 = (' '.join(lemma1))

    # Original File

    def hindi_model():
        train_data = indian.tagged_sents('hindi.txt')
        tnt_pos_tagger = tnt.TnT()
        tnt_pos_tagger.train(train_data)
        return tnt_pos_tagger

    model = hindi_model()
    new_tagged = (model.tag(nltk.word_tokenize(after_lemma)))

    # Plagarized File

    def hindi_model():
        train_data = indian.tagged_sents('hindipos.txt')
        tnt_pos_tagger = tnt.TnT()
        tnt_pos_tagger.train(train_data)
        return tnt_pos_tagger
    model = hindi_model()
    new_tagged1 = (model.tag(nltk.word_tokenize(after_lemma1)))

    # Original File

    temp = []
    for item in new_tagged:
        if item[1] != "NNP":
            temp.append(item)
    myone = []
    for item1, tag1 in temp:
        myone.append(item1)
    new_data = ' '.join(myone)

    # Plagarized File
    temp1 = []
    for item in new_tagged1:
        if item[1] != "NN":
            temp1.append(item)
    myone1 = []
    for item1, tag1 in temp1:
        myone1.append(item1)
    ner_data1 = ' '.join(myone1)
    with open('synonym (2).txt', encoding="utf-8") as f:
        dict = {}
        for line in f:
            x, y = line.split(":")
            dict[x.strip()] = y.strip()
    # print(dict)

    synonym = []
    b = list(dict.keys())

    unique_synonym = []
    for word in myone1:
        if word in b:
            synonym.append(dict[word])
        else:
            synonym.append(word)

    synonym_data = ' '.join(synonym)

    # Original File

    # Tokenize the text
    tokens = word_tokenize(new_data)

    # Create trigrams from the tokens
    trigrams = list(ngrams(tokens, 3))

    # Tokenize the text
    tokens1 = word_tokenize(ner_data1)

    # Create trigrams from the tokens
    trigrams1 = list(ngrams(tokens1, 3))

    # Original File

    # import required module
    # create object
    vectorizer = TfidfVectorizer(analyzer=lambda x: x)
    result = vectorizer.fit_transform(trigrams)

    # Plagarized File

    # import required module
    # create object
    tfidf = TfidfVectorizer()
    # get tf-df values
    result1 = vectorizer.fit_transform(trigrams1)

    # Original File

    c = result  # word in which resul
    d = {}
    d = c.todok()
    x = list(d.values())

    # Plagarized File

    c1 = result1  # word in which resul
    d1 = {}
    d1 = c1.todok()
    x1 = list(d1.values())

    # import required libraries

    # define two lists or array
    A = list(np.array(x))
    B = list(np.array(x1))
    a_len = len(A)
    b_len = len(B)
    c = a_len if a_len > b_len else b_len
    modfi = B if a_len > b_len else A

    if a_len > b_len:
        min = b_len
    else:
        min = a_len
    # print(modfi)
    for i in range(c+1):
        if i > min:
            modfi.append(0.0)
    # print(b_len)
    # print(a_len)
    cosine = np.dot(A, B)/(norm(A)*norm(B))
    return (cosine*100)
