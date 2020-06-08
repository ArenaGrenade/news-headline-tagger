from gensim.models import KeyedVectors
from tensorflow.keras.preprocessing.sequence import pad_sequences


def main():
    wv = KeyedVectors.load("C:\\Users\\rohan\\Documents\\Coding\\Python\\news-headline-tagger\\models\\embedder\\model_trial_1")
    MAX_NB_WORDS = len(wv.vocab)
    MAX_SEQUENCE_LENGTH = 200





if __name__ == '__main__':
    main()