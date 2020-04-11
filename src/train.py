import config
import torch
import dataset
import engine
import pandas as pd
import numpy as np
from model import BERTBaseUncased
from sklearn import model_selection
from torch.utils.data import DataLoader
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn import metrics


def run():
    dfx = pd.read_csv(config.TRAINING_FILE).dropna().reset_index(drop=True)
    
    df_train, df_valid = model_selection.train_test_split(
        dfx,
        test_size=0.1,
        random_state=42,
        stratify=dfx.sentiment.values
    )
    df_train = df_train.reset_index(drop=True)
    df_valid = df_valid.reset_index(drop=True)

    train_dataset = dataset.TweetDataset(
        tweet=df_train.text.values,
        sentiment=df_train.sentiment.values,
        selected_text=df_train.selected_text.values
    )

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=config.TRAIN_BATCH_SIZE,
                                  num_workers=4)

    valid_dataset = dataset.TweetDataset(
        tweet=df_train.text.values,
        sentiment=df_train.sentiment.values,
        selected_text=df_train.selected_text.values
    )

    valid_dataloader = DataLoader(valid_dataset,
                                  batch_size=config.VALID_BATCH_SIZE,
                                  num_workers=4)

    device = torch.device('cpu')

    model = BERTBaseUncased()
    model.to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': .001},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': .001}
    ]

    num_train_steps = int(len(df_train) / config.TRAIN_BATCH_SIZE * config.EPOCHS)
    optimizer = AdamW(optimizer_parameters, lr=3e-5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=num_train_steps
    )

    best_accuracy = 0.0
    for epoch in range(config.EPOCHS):
        engine.train_fn(train_dataloader, model, optimizer, device, scheduler)
        # outputs, targets = engine.eval_fn(valid_dataloader, model, device)
        # outputs = np.array(outputs) >= 0.5
        # accuracy = metrics.accuracy_score(targets, outputs)
        # print(f'accuracy score: {accuracy}')
        # if accuracy > best_accuracy:
        #     torch.save(model.state_dict(), config.MODEL_PATH)
        #     best_accuracy = accuracy


if __name__ == '__main__':
    run()