import os
import text_file_operations as tfo
import text_functions as tf
import text_analysis as ta

PWD = os.getcwd()


def lda_main(file_name, topic_size, exclusion_boolean=True, multiword_boolean=True):
    """
    A function to load, preprocess, and run LDA for specific text files in the comments1k folder
    :param file_name: name of file to analyze
    :param topic_size: number of topics for lda to produce
    :param exclusion_boolean: Boolean to exclude (True) or include (False) stop words and punctuation
    :param multiword_boolean: Boolean to run multiword replacement function
    :return: lda model object
    """
    print('\nFile Name = \'{}\'--------------------'.format(file_name))
    print('\n-----LOADING DATA...')
    directory = PWD + '/' + 'comments1k/'
    file_names = tfo.file_search(file_name, dir_string=directory, match_term=True)
    data = tfo.read_file_list(directory, file_names, encoding='cp1252')
    print('\n-----PREPROCESSING DATA ...')
    data_split_sentence = tf.split_sentence(data)
    data_processed = tf.lemmatization(data_split_sentence, exclusion=exclusion_boolean, multiword=multiword_boolean)
    # data_split_words = tf.split_words(data, exclusion=True)
    # data_processed = tf.stemming(data_split_words, exclusion=exclusion_boolean, multiword=multiword_boolean)
    print('\n-----RUNNING LDA CODE ...')
    tf.lda(data_processed, n=topic_size)


print('\n-----LOADING-----')
text_dir = PWD + '/' + 'comments1k/'
text_file_names = tfo.file_search(search_term='.txt', dir_string=text_dir)
comment_data = tfo.read_file_list(text_dir, text_file_names, encoding='cp1252')

print('\n-----PREPROCESSING-----')
text_split_word = tf.split_words(comment_data)
text_split_word_exclusion = tf.split_words(comment_data, exclusion=True)
text_split_sentence = tf.split_sentence(comment_data)

all_words = ta.collect_all_words(text_split_word)
unique_words = ta.unique(all_words)

all_words_exclusion = ta.collect_all_words(text_split_word_exclusion, exclusion=True)
unique_words_exclusion = ta.unique(all_words_exclusion)

all_comments_stemmed = tf.stemming(text_split_word, exclusion=False)
all_comments_lemmatized = tf.lemmatization(text_split_sentence, exclusion=False)

all_comments_stemmed_exclusion = tf.stemming(text_split_word_exclusion, exclusion=True)
all_comments_lemmatized_exclusion = tf.lemmatization(text_split_sentence, exclusion=True)


print('\n-----QUESTION 1-----')
print('-----PART 1')
average_sent_stat = ta.average_sent_per_comment(text_split_sentence)
print('-----PART 2')
average_tokens_stat = ta.average_words_per_comment(text_split_word)
print('-----PART 3')
average_tokens_stat_exclusion = ta.average_words_per_comment(text_split_word_exclusion, exclusion=True)
print('-----PART 4')
print('---Lemmatization')
ats_lemmatized = ta.average_words_per_comment(all_comments_lemmatized)
print('---Lemmatization, Exclusion')
ats_lemmatized_exclusion = ta.average_words_per_comment(all_comments_lemmatized_exclusion, exclusion=True)
print('-----Stemming')
ats_stemmed = ta.average_words_per_comment(all_comments_stemmed)
print('-----Stemming, Exclusion')
ats_stemmed_exclusion = ta.average_words_per_comment(all_comments_stemmed_exclusion, exclusion=True)

print('\n-----QUESTION 2-----')
print('-----PART 1')
print('---3 Topics, Exclusion Lemmatization')
tf.lda(all_comments_lemmatized_exclusion, n=3)
print('---3 Topics, Exclusion Stemming')
tf.lda(all_comments_stemmed_exclusion, n=3)

print('-----PART 2')
all_comments_stemmed_excl_multi = tf.stemming(text_split_word_exclusion, exclusion=True, multiword=True)
all_comments_lemmatized_excl_multi = tf.lemmatization(text_split_sentence, exclusion=True, multiword=True)
tf.lda(all_comments_stemmed_excl_multi, n=1)
tf.lda(all_comments_lemmatized_excl_multi, n=1)

print('-----PART 3')
lda_main('0_9.txt', 1)
lda_main('1_7.txt', 1)
lda_main('2_9.txt', 1)
