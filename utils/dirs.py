import os


def createMissingDirectories(list_of_directories):
    """
    If some directories are missing from the project structure, this function creates them. This is essentially an
    error correction method.
    :param list_of_directories: A list of all the necessary directories
    :return: returns a code 0:success and -1:failure
    """
    try:
        for directory in list_of_directories:
            if not os.path.exists(directory):
                os.makedirs(directory)
        return 0
    except Exception as err:
        print("Error in creating the directory: {0}".format(err))
        exit(-1)
