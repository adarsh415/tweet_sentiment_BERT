import utils
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import gc

def loss_fn(o1, o2, t1, t2):
    l1 = nn.BCEWithLogitsLoss()(o1, t1)
    l2 = nn.BCEWithLogitsLoss()(o2, t2)

    return l1+l2

def train_fn(data_loader, model, optimizer, device, scheduler):
    model.train()
    for batch, dataset in tqdm(enumerate(data_loader), total=len(data_loader)):
        ids = dataset['ids']
        token_type_ids = dataset['token_type_ids']
        mask = dataset['mask']
        targets_start = dataset['targets_start']
        targets_end = dataset['targets_end']

        ids = ids.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        targets_start = targets_start.to(device, dtype=torch.float)
        targets_end = targets_end.to(device, dtype=torch.float)

        optimizer.zero_grad()
        o1, o2 = model(
            ids=ids,
            mask=mask,
            token_type_ids=token_type_ids
        )
        loss = loss_fn(o1, o2, targets_start, targets_start)
        loss.backward()

        optimizer.step()
        scheduler.step()
        gc.collect()


def eval_fn(data_loader, model, device):
    model.eval()
    fin_output_start = []
    fin_output_end = []
    fin_padding_lens = []
    fin_tweet_tokens = []
    fin_orig_sentiment = []
    fin_orig_selected = []
    fin_orig_tweet = []

    with torch.no_grad():
        for batch, dataset in tqdm(enumerate(data_loader), total=len(data_loader)):
            ids = dataset['ids']
            token_type_ids = dataset['token_type_ids']
            mask = dataset['mask']
            tweet_tokens = dataset['tweet_tokens']
            padding_len = dataset['padding_len']
            orig_sentiment = dataset['orig_sentiment']
            orig_selected = dataset['orig_selected']
            orig_tweet = dataset['orig_tweet']

            ids = ids.to(device, dtype=torch.long)
            token_type_ids = token_type_ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)
            
            o1, o2 = model(
                ids=ids,
                mask=mask,
                token_type_ids=token_type_ids
            )

            
            fin_output_start.extend(torch.sigmoid(o1).cpu().detach().numpy().tolist())
            fin_output_end.extend(torch.sigmoid(o2).cpu().detach().numpy().tolist())
            fin_padding_lens.extend(padding_len.cpu().detach().numpy().tolist())

            fin_tweet_tokens.extend(tweet_tokens)
            fin_orig_sentiment.extend(orig_sentiment)
            fin_orig_selected.extend(orig_selected)
            fin_orig_tweet.extend(orig_tweet)

            gc.collect()
    fin_output_start = np.vstack(fin_output_start)
    fin_output_end = np.vstack(fin_output_end)

    threshold = 0.2
    jaccard = []

    for i in range(fin_tweet_tokens):
        target_string = fin_orig_selected[j]
        tweet_tokens = fin_tweet_tokens[j]
        padding_len = fin_padding_lens[j]
        original_tweet = fin_orig_tweet[j]
        sentiment = fin_orig_sentiment[j]

        if padding_len > 0:
            mask_start = fin_output_start[j,:][:-padding_len] >= threshold
            mask_end = fin_output_end[j, :][:-padding_len]>= threshold
        else:
            mask_start = fin_output_start[j,:] >= threshold
            mask_end = fin_output_end[j, :]>= threshold

        mask = [0]*mask_start
        idx_start = np.nonzero(mask_start)[0]
        idx_end = np.nonzero(mask_end)[0]

        if len(idx_start) > 0:
            idx_start = idx_start[0]
            if len(idx_end)>0:
                idx_end = idx_end[0]
            else:
                idx_end = idx_start
        else:
            idx_start =0
            idx_end = 0
        # Need more work

    return fin_outputs, fin_targets