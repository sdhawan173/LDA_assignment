import os
import text_functions as tf
import text_analysis as ta

PWD = os.getcwd()


def lda_main(file_name, topic_size, exclusion_boolean=True):
    """
    A function to load, preprocess, and run LDA for specific text files in the comments1k folder
    :param file_name: name of file to analyze
    :param topic_size: number of topics for lda to produce
    :param exclusion_boolean: Boolean to exclude (True) or include (False) stop words and punctuation
    :return: lda model object
    """
    print('\nFile Name = \'{}\'--------------------'.format(file_name))
    print('\n-----LOADING DATA...')
    directory = PWD + '/' + 'comments1k/'
    file_names = tf.file_search(file_name, dir_string=directory, match_term=True)
    data = tf.read_file_list(directory, file_names, encoding='cp1252')
    print('\n-----PREPROCESSING DATA ...')
    data_split_sentence = tf.split_sentence(data)
    data_lemmatized = tf.lemmatization(data_split_sentence, exclusion=exclusion_boolean)
    print('\n-----RUNNING LDA CODE ...')
    for topic_size in range(1, 6):
        tf.lda(data_lemmatized, n=topic_size)


print('\n-----LOADING-----')
text_dir = PWD + '/' + 'comments1k/'
text_file_names = tf.file_search(search_term='.txt', dir_string=text_dir)
comment_data = tf.read_file_list(text_dir, text_file_names, encoding='cp1252')

print('\n-----PREPROCESSING-----')
text_split_word = tf.split_words(comment_data)
text_split_word_exclusion = tf.split_words(comment_data, exclusion=True)
all_comments_stemmed = tf.stemming(text_split_word, exclusion=False)
all_comments_stemmed_exclusion = tf.stemming(text_split_word_exclusion, exclusion=True)
text_split_sentence = tf.split_sentence(comment_data)
all_comments_lemmatized = tf.lemmatization(text_split_sentence, exclusion=False)
all_comments_lemmatized_exclusion = tf.lemmatization(text_split_sentence, exclusion=True)
all_words = tf.collect_all_words(text_split_word)
unique_words = ta.unique(all_words)

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
ats_stemmed_exclusion = ta.average_words_per_comment(all_comments_stemmed_exclusion)

print('\n-----QUESTION 2-----')
print('-----PART 1')
print('---1 Topics, Regular Lemmatization')
tf.lda(all_comments_lemmatized, n=1)
print('---2 Topics, Regular Lemmatization')
tf.lda(all_comments_lemmatized, n=2)
print('---3 Topics, Regular Lemmatization')
tf.lda(all_comments_lemmatized, n=3)
print('-----PART 2')
print('---1 Topics, Exclusion Lemmatization')
tf.lda(all_comments_lemmatized_exclusion, n=1)
print('---2 Topics, Exclusion Lemmatization')
tf.lda(all_comments_lemmatized_exclusion, n=2)
print('---3 Topics, Exclusion Lemmatization')
tf.lda(all_comments_lemmatized_exclusion, n=3)
print('-----PART 3')
lda_main('0_9.txt', 2)
lda_main('1_7.txt', 2)
lda_main('2_9.txt', 2)
