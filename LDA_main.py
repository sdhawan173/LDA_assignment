import text_preprocessing as tp


pwd = os.getcwd()
text_dir = pwd + '/' + 'comments1k/'
text_file_names = tp.file_search(extension='.txt', dir_string=text_dir)
text_data = tp.read_file_list(text_dir, text_file_names)
