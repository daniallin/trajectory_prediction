import pickle


FILE_NAME = ''

with open(FILE_NAME, 'rb') as f:
    content = pickle.load(f)
    print(content)


