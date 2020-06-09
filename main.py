from data_loader.news_dataset_loader import NewsDatasetLoader
from models.CNN_tagger_model import *
from trainers.CNN_tagger_trainer import CNNModelTrainer
from utils.config import processConfig
from utils.dirs import createMissingDirectories
from utils.args import get_args
from tensorflow.train import latest_checkpoint

MAX_SEQ_LEN = 300
WV_DIM = 300
NB_WORDS = 0


def main():
    try:
        args = get_args()
        config = processConfig(args.config)
    except:
        print("missing or invalid arguments")
        exit(0)

    createMissingDirectories([config.callbacks.tensorboard_log_dir, config.callbacks.checkpoint_dir])

    print("Create the data generator.")
    data_loader = NewsDatasetLoader(config)
    train_data, train_labels = data_loader.getTrainingData()
    test_data, test_labels = data_loader.getTestData()

    print("Creating model.")
    glove_model, NB_WORDS = load_glove_model(config.exp.GLOVE_PATH)
    train_seq, test_seq, wv_mat = embeddingLayerBuild(glove_model, train_data, test_data, MAX_SEQ_LEN, WV_DIM, NB_WORDS)
    model_class = ConvTaggerModel(config, MAX_SEQ_LEN, NB_WORDS, WV_DIM, wv_mat)
    model = model_class.buildModel()

    if config.saveorload.load:
        latest = latest_checkpoint(config.callbacks.checkpoint_dir)
        if latest is not None:
            model.load_weights(latest)

    print("Create the trainer.")
    trainer = CNNModelTrainer(model, train_seq, train_labels, config)

    print("Start training the model.")
    trainer.train()


if __name__ == '__main__':
    main()
