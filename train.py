"""
Based on https://github.com/batra-mlp-lab/visdial-challenge-starter-pytorch
"""

import argparse
import os
import yaml
from copy import deepcopy
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn

from torch import optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm

from dataset.dataset import CXRVisDialDataset
from models.models import LateFusionModel, RecursiveAttentionModel, StackedAttentionModel
from utils import report_metric, load_embeddings, match_embeddings


if __name__ == "__main__":
    # ------------------------------------------------------------------------
    # ARGUMENTS
    # ------------------------------------------------------------------------л

    parser = argparse.ArgumentParser()

    parser.add_argument("--train_json",
                        required=True,
                        help="Location of the train json file")

    parser.add_argument("--val_json",
                        required=True,
                        help="Location of the val json file")

    parser.add_argument("--train_img_feats",
                        required=True,
                        help="Path to train image features h5 file")

    parser.add_argument("--val_img_feats",
                        required=True,
                        help="Path to val image features h5 file")

    parser.add_argument("--config_yml",
                        default="config.yaml",
                        help="Location of the config yaml file")

    parser.add_argument("--word_counts",
                        default=None,
                        help="Location of the word counts file used for constructing the vocabulary."
                             "If None, BERT model and BERT vocabulary will be used.")
    
    parser.add_argument("--output_dir",
                        required=True,
                        help="Output location for saving the weights")

    parser.add_argument("--embeddings",
                        default=None,
                        help="Whether pretrained embeddings should be used. "
                             "If yes, the argument is a path to a pickled file.")

    parser.add_argument("--bert_path",
                        default=None,
                        help="If using BERT embeddings, the location for the pre-trained bin model")

    parser.add_argument("--model",
                        default="lf",
                        help="Model to use for training. Valid options are: lf (LateFusion), "
                             "rva (RecursiveVisualAttention), and san (Stacked Attention Network)")

    parser.add_argument("--log_dir",
                        default='./logs',
                        help="Path to the directory where TensorBoard logs will be stored for tracking training." )

    parser.add_argument("--label",
                        default="",
                        help="Optional label for an experiment. Will be used for naming the file with the model weights.")

    args = parser.parse_args()

    with open(args.config_yml) as f:
        config = yaml.safe_load(f)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # ------------------------------------------------------------------------
    # DATASET & DATALOADER
    # ------------------------------------------------------------------------
    if args.model == "rva":
        # RvA treats history as sequences of turns
        mode = 'seq'
    elif args.model == "lf" or args.model == "san":
        # Other models treat history as concatenation of all turns
        mode = 'concat'
    else:
        raise ValueError("Unknown model")

    train_dataset = CXRVisDialDataset(args.train_img_feats, args.train_json, args.word_counts, mode)
    val_dataset = CXRVisDialDataset(args.val_img_feats, args.val_json, args.word_counts, mode)

    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=True)

    # ------------------------------------------------------------------------
    # SETUP
    # ------------------------------------------------------------------------
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_epochs = config['num_epochs']

    # If word_counts are passed, use them to construct the vocabulary. If word embeddings are also passed,
    # use them for initializing the embedding layer. Otherwise, use BERT
    if args.word_counts is None:
        use_bert = True
        embeddings = None
        bert_path = args.bert_path
    else:
        use_bert = False
        bert_path = None
        if args.embeddings is not None:
            embeddings_dict = load_embeddings(args.embeddings)
            embeddings = match_embeddings(train_dataset.vocabulary, embeddings_dict)
            embeddings = torch.Tensor(embeddings).float().to(device)
        else:
            embeddings = None

    if args.model == "lf":
        model = LateFusionModel(config, train_dataset.vocabulary, embeddings)
    elif args.model == "rva":
        model = RecursiveAttentionModel(config, train_dataset.vocabulary, embeddings)
    elif args.model == "san":
        model = StackedAttentionModel(config, train_dataset.vocabulary, embeddings, use_bert, bert_path)
    
    model = model.to(device)

    # Weights of the loss tend to affect the performance (especially SAN). Adjust if needed
    weight = torch.Tensor([5, 5, 10, 1]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

    # ------------------------------------------------------------------------
    # LOGGING
    # ------------------------------------------------------------------------
    experiment_name = f"{args.model}_{args.label}_{datetime.today().strftime('%Y-%m-%d-%H:%M:%S')}"

    # TensorBoard outputs
    log_dir = args.log_dir
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    experiment_logs = os.path.join(log_dir, experiment_name)
    writer = SummaryWriter(log_dir=experiment_logs)

    # Saving weights
    model_path = os.path.join(args.output_dir, 'weights_{}.pt'.format(experiment_name))

    # ------------------------------------------------------------------------
    # TRAINING/VALIDATION LOOP
    # ------------------------------------------------------------------------
    best_macro_f1 = None
    for epoch in range(1, num_epochs + 1):
        batch_losses = []

        model.train()
        for batch in tqdm(train_dataloader, desc="Train"):
            image, history, question, options, caption, turn = batch['image'], batch['history_ids'], batch['question_ids'],\
                                                         batch['options'], batch['caption_ids'], batch['turn']
            image, history, question, options, caption, turn = image.to(device), history.to(device), question.to(device), \
                                                         options.to(device), caption.to(device), turn.to(device)

            optimizer.zero_grad()

            output = model(image, history, question, options, caption, turn)
            target = batch["answer_ind"].to(device)

            batch_loss = criterion(output.view(-1, output.size(-1)), target.view(-1))

            batch_losses.append(batch_loss.item())
            batch_loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

            optimizer.step()

        epoch_loss = np.mean(np.array(batch_losses))
        print(f'Train loss after epoch {epoch}: {epoch_loss}')
        writer.add_scalar("Train loss", epoch_loss, epoch)

        model.eval()
        targets = []
        outputs = []
        batch_losses = []
        turns = []

        for batch in tqdm(val_dataloader, desc="Validation"):
            image, history, question, options, caption, turn = batch['image'], batch['history_ids'], \
                                                               batch['question_ids'], batch['options'], \
                                                               batch['caption_ids'], batch['turn']
            image, history, question, options, caption, turn = image.to(device), history.to(device), question.to(device),\
                                                               options.to(device), caption.to(device), turn.to(device)

            output = model(image, history, question, options, caption, turn)
            target = batch["answer_ind"].to(device)

            batch_loss = criterion(output.view(-1, output.size(-1)), target.view(-1))
            batch_losses.append(batch_loss.item())
            
            targets.append(target.detach().cpu().numpy())
            outputs.append(output.detach().cpu().numpy())
            turns.append(turn.detach().cpu().numpy())

        epoch_loss = np.mean(np.array(batch_losses))

        f1_scores, conf_matrix, macro_f1, accuracies = report_metric(targets, outputs, turns)
        scores_dict = {'Yes': f1_scores[0], 'No': f1_scores[1], 'Maybe': f1_scores[2], 
                       'N/A': f1_scores[3], 'macro_f1': macro_f1}
        scores_by_round = {str(r+1):acc for r, acc in enumerate(accuracies)}

        if best_macro_f1 is None or macro_f1 > best_macro_f1:
            best_loss = epoch_loss
            torch.save(model.state_dict(), model_path)
            best_macro_f1 = macro_f1
            best_scores = deepcopy(scores_dict)

        writer.add_scalars("F1 scores", scores_dict, epoch)
        writer.add_scalars("Round accuracies", scores_by_round, epoch)

    writer.close()
    print(experiment_name)
    print(f"The best validation macro-f1 score: {best_macro_f1}")
    print(best_scores)