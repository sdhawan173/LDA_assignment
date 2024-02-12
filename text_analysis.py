import numpy as np


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


def average_words_per_comment(text_split_word, exclusion=False):
    if exclusion is not True:
        print('   Average number of words per comment = ', end="")
    else:
        print('   Average number of words per comment,\n'
              '  excluding stop words and punctuation = ', end="")
    average_tokens_num_list = []
    for comment in text_split_word:
        average_tokens_num_list.append(len(np.unique(comment)))
    average_tokens_stat = sum(average_tokens_num_list) / len(text_split_word)
    print(average_tokens_stat)
    return average_tokens_stat
