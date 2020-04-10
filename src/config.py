import os
import tokenizers

MAX_LEN = 128
BASE_PATH = 'f:/PycharmProjects/tweet_sentiment_BERT'
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 16
EPOCHS = 10
BERT_PATH =  os.path.abspath(BASE_PATH + "/input/bert-base-uncased/")
MODEL_PATH = os.path.abspath(BASE_PATH + "/input/model.bin")
TRAINING_FILE = os.path.abspath(BASE_PATH + "/input/train.csv")
TOKENIZER = tokenizers.BertWordPieceTokenizer(
    os.path.join(BERT_PATH, "vocab.txt"),
    lowercase=True
)