import os
import string
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
import spacy
from gensim import corpora
from gensim.models import LdaModel
from tqdm import tqdm


try:
    nltk.data.find('tokenizers/stopwords')
except LookupError:
    nltk.download('stopwords')
STOP_WORDS = set(stopwords.words('english'))
STOP_WORDS.add('n\'t')
STOP_WORDS.add('\'s')
PUNCTUATION = set(string.punctuation)
PUNCTUATION.add('br')
PUNCTUATION.add('<br>')
PUNCTUATION.add('<\\br>')
PUNCTUATION.add('``')
PUNCTUATION.add('\'\'')


def file_search(extension, dir_string=os.getcwd()):
    """
    searches a directory, with the current working directory as default, for a given filetype.
    :param dir_string: string of directory to search
    :param extension: string of filetype, input as a string in the format: '.type'
    :return: list of file names with extensions that match search term
    """
    print('Searching for .txt files ...')
    file_list = []
    # Run through list and add files with .ast extension to ast_list
    for list_item in os.listdir(dir_string):
        if list_item.__contains__(extension):
            file_list.append(list_item)
    return sorted(file_list, key=str.casefold)


def read_file_list(file_path, name_list, encoding):
    """
    reads stl file and stores to array
    :param file_path: file path of ast or stl file
    :param name_list: list of file names to be appended to file_path
    :param encoding: type of encoding to use when opening file
    """
    print('Reading all txt files ...')
    text_data = []
    for name in name_list:
        read_file = []
        opened_file = open(file_path + name, 'r', encoding=encoding)
        for line in opened_file:
            read_file.append(line)
        opened_file.close()
        text_data.append(read_file)
    return text_data


def split_sentence(text_data):
    """
    Splits each comment from list of comments into sentences
    :param text_data: list of comments
    :return: list of lists of comments split into sentences
    """
    # Make sure 'punkt' is installed
    print('Splitting all comments by sentences ...')
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    text_split_sentence = text_data
    for index in range(len(text_data)):
        text_split_sentence[index] = nltk.sent_tokenize(text_data[index][0])
    return text_split_sentence


def split_words(text_data, exclusion=False):
    """
    Splits each comment from list of comments into words
    :param text_data: list of comments
    :param exclusion: Boolean to exclude STOP_WORDS and PUNCTUATION
    :return: list of lists of comments split into words
    """
    print('Splitting all comments by words ...')
    if exclusion:
        print('Excluding stop words and punctuation ...')
    text_split_word = []
    for index in tqdm(range(len(text_data))):
        comment_words = []
        for sentence in text_data[index]:
            temp = nltk.word_tokenize(sentence)
            if exclusion:
                temp = [token for token in temp if token.lower() not in STOP_WORDS and token not in PUNCTUATION]
            for word in temp:
                comment_words.append(word)
        text_split_word.append(comment_words)
    return text_split_word


def lemmatization(text_split_sentence, exclusion=False):
    """
    lemmatizes each comment from list of comments into lemmatized words
    :param text_split_sentence: list of lists of comments split into sentences
    :param exclusion: Boolean to exclude STOP_WORDS and PUNCTUATION
    :return: list of lists of comments split into lemmatized words
    """
    print('Lemmatizing all comments ... ')
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')
    lemmatizer = WordNetLemmatizer()
    all_comments_lemmatized = []
    for comment in tqdm(text_split_sentence):
        comment_lda = []
        for sentence in comment:
            words = word_tokenize(sentence)
            for word in words:
                word = word.lower()
                if exclusion and (word not in STOP_WORDS and word not in PUNCTUATION):
                    comment_lda.append(lemmatizer.lemmatize(word))
        all_comments_lemmatized.append(comment_lda)
    return all_comments_lemmatized


def collect_all_words(text_split_word, exclusion=False):
    """
    collects all words from text_split_word or all_comments_lemmatized into one array of words
    :param text_split_word:
    :param exclusion: Boolean to exclude STOP_WORDS and PUNCTUATION
    :return: list of all words
    """
    if exclusion is not True:
        insert = ''
    else:
        insert = ', excluding stop words and punctuation,'
    print('Collecting all words{} in all comments into one list ...'.format(insert))
    all_words = []
    for comment in text_split_word:
        for word in comment:
            all_words.append(word)
    return all_words


def lda(all_comments_lemmatized, n):
    """

    :param all_comments_lemmatized:
    :param n:
    :return:
    """
    print('Performing Latent Drichlet Allocation ...')

    lda_dict = corpora.Dictionary(all_comments_lemmatized)
    lda_corpus = []
    for comment in all_comments_lemmatized:
        lda_corpus.append(lda_dict.doc2bow(comment))
    lda_model = LdaModel(lda_corpus, n, id2word=lda_dict)
    for topic_id, topic, in lda_model.print_topics():
        print(f"Topic {topic_id + 1}: {topic}")
