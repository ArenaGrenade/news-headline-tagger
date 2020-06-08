import csv
import re

import SentenceToVector


def trainEmbedder(tokenized_dataset, feature_size, filename):
    print("training the model")
    s2v = SentenceToVector.WordToVector(train_new_model=True,
                                        tokenized_dataset=tokenized_dataset
                                        , vector_size=feature_size)
    model_file = "C:\\Users\\rohan\\Documents\\Coding\\Python\\news-headline-tagger\\models\\embedder\\" + filename
    print("saving model now")
    s2v.to_train_model.wv.save(model_file)


def getProcessedData():
    tokenizer = SentenceToVector.bag_of_words_converter.BOWConvert()
    train_data = []
    test_data = []
    train_labels = []
    test_labels = []
    with open("C:/Users/rohan/Documents/coding/python/news-headline-tagger/datasets/training _data/training_data.csv",
              newline='', encoding='utf-8') as train_file:
        csvreader = csv.reader(train_file)
        for row in csvreader:
            train_data.append((re.sub('[^a-z]', ' ', (row[0] + ' ' + row[1]).lower())).split())
            train_labels.append(int(row[2]))

    with open("C:/Users/rohan/Documents/coding/python/news-headline-tagger/datasets/test_data/test_data.csv",
              newline='', encoding='utf-8') as test_file:
        csvreader = csv.reader(test_file)
        for row in csvreader:
            # test_data.append(tokenizer.convert(row[0] + ' ' + row[1]))
            test_data.append((re.sub('[^a-z]', ' ', (row[0] + ' ' + row[1]).lower())).split())
            test_labels.append(int(row[2]))

    return (train_data, SentenceToVector.np.array(train_labels)), (test_data, SentenceToVector.np.array(test_labels))


if __name__ == '__main__':
    getProcessedData()
