import os
import text_functions as tf
import text_analysis as ta


pwd = os.getcwd()
text_dir = pwd + '/' + 'comments1k/'
text_file_names = tf.file_search(extension='.txt', dir_string=text_dir)
text_data = tf.read_file_list(text_dir, text_file_names, encoding='cp1252')

text_split_sentence = tf.split_sentence(text_data)
print('text_split_sentence:', text_split_sentence[0])
all_comments_lemmatized_exclusion = tf.lemmatization(text_split_sentence, exclusion=True)
print('all_comments_lemmatized_exclusion:', all_comments_lemmatized_exclusion[0])
tf.lda(all_comments_lemmatized_exclusion, n=2)
# print('all_comments_lemmatized:', all_comments_lemmatized[0])

text_split_word = tf.split_words(text_data)
# print('text_split_word:', text_split_word[0])
text_split_word_exclusion = tf.split_words(text_data, exclusion=True)
# print('text_split_word_exclusion:', text_split_word_exclusion[0])
all_words = tf.collect_all_words(text_split_word)
# print('all_words:', all_words[0])
unique_words = tf.unique(all_words)
# print('unique_words:', unique_words[0])

average_sent_stat = ta.average_sent_per_comment(text_split_sentence)
print('average_sent_stat:', average_sent_stat)
average_tokens_stat = ta.average_words_per_comment(text_split_word)
# print('average_tokens_stat:', average_tokens_stat)
average_tokens_stat_exclusion = ta.average_words_per_comment(text_split_word_exclusion, exclusion=True)
# print('average_tokens_stat_exclusion:', average_tokens_stat_exclusion)
