import os


def file_search(extension, dir_string=os.getcwd()):
    """
    searches a directory, with the current working directory as default, for a given filetype.
    :param dir_string: string of directory to search
    :param extension: string of filetype, input as a string in the format: '.type'
    :return: list of file names with extensions that match search term
    """
    file_list = []

    # Run through list and add files with .ast extension to ast_list
    for list_item in os.listdir(dir_string):
        if list_item.__contains__(extension):
            file_list.append(list_item)
    return sorted(file_list, key=str.casefold)


def read_file_list(file_path, name_list):
    """
    reads stl file and stores to array
    :param file_path: file path of ast or stl file
    :param name_list: list of file names to be appended to file_path
    """
    file_list = []
    for name in name_list:
        read_file = []
        opened_file = open(file_path + name, 'r')
        for line in opened_file:
            read_file.append(line)
        opened_file.close()
        file_list.append(read_file)
    return file_list