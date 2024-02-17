import numpy as np
import matplotlib.pyplot as plt


def collect_all_words(text_split_word, exclusion=False):
    """
    collects all words from text_split_word or all_comments_lemmatized into one array of words
    :param text_split_word:
    :param exclusion: Boolean to exclude STOP_WORDS and PUNCTUATION
    :return: list of all words
    """
    insert = ''
    if exclusion is True:
        insert = ', excluding stop words and punctuation,\n'
    print('\nCollecting all words{} in {} comments into one list ...'.format(insert, len(text_split_word)))
    all_words = []
    for comment in text_split_word:
        for word in comment:
            all_words.append(word)
    return all_words


def unique(all_words):
    """
    creatse list of unique words in list
    :param all_words: list of all words
    :return: list of unique words
    """
    print('Total unique words = {}'.format(len(all_words)))
    unique_words = np.ndarray.tolist(np.unique(all_words))
    return unique_words


def average_sent_per_comment(text_split_sentence):
    print('Average number of setences per comment = ', end="")
    average_sent_num_list = []
    for comment in text_split_sentence:
        average_sent_num_list.append(len(comment))
    average_sent_stat = sum(average_sent_num_list)/len(text_split_sentence)
    print(average_sent_stat)
    return average_sent_stat


def average_words_per_comment(comment_word_list, exclusion=False):
    if exclusion is not True:
        print('   Average number of words per comment = ', end="")
    else:
        print('   Average number of words per comment,\n'
              '   excluding stop words and punctuation= ', end="")
    average_tokens_num_list = []
    for comment in comment_word_list:
        average_tokens_num_list.append(len(np.unique(comment)))
    average_tokens_stat = sum(average_tokens_num_list) / len(comment_word_list)
    print(average_tokens_stat)
    return average_tokens_stat


def plot_lda_distribution(lda_model):
    output_string = lda_model.print_topics()
    topic_num = len(output_string)
    print(topic_num)
    for output in output_string:
        print(output)
    return
