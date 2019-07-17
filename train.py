"""
Based on https://github.com/batra-mlp-lab/visdial-challenge-starter-pytorch
"""

import argparse
import os
import yaml
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm

from dataset.dataset import CXRVisDialDataset
from models.models import LateFusionModel, RecursiveAttentionModel, StackedAttentionModel
from models.utils.utils import report_metric, load_embeddings, match_embeddings


if __name__ == "__main__":
    # ------------------------------------------------------------------------
    # ARGUMENTS
    # ------------------------------------------------------------------------

    parser = argparse.ArgumentParser()

    parser.add_argument("--train_json",
                        required=True,
                        help="Location of the train json file")

    parser.add_argument("--val_json",
                        required=True,
                        help="Location of the val json file")

    parser.add_argument("--img_feats_train",
                        required=True,
                        help="Location of train images features")

    parser.add_argument("--img_feats_val",
                        required=True,
                        help="Location of train images features")

    parser.add_argument("--config_yml",
                        default="config.yaml",
                        help="Location of the config yaml file")

    parser.add_argument("--word_counts",
                        required=True,
                        help="Location of the word counts file")
    
    parser.add_argument("--output_dir",
                        required=True,
                        help="Output location for saving the weights")

    args = parser.parse_args()

    with open(args.config_yml) as f:
        config = yaml.safe_load(f)
        
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)


    # ------------------------------------------------------------------------
    # DATASET & DATALOADER
    # ------------------------------------------------------------------------

    train_dataset = CXRVisDialDataset(args.img_feats_train, args.train_json, args.word_counts)
    val_dataset = CXRVisDialDataset(args.img_feats_val, args.val_json, args.word_counts)

    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'])
    val_dataloader = DataLoader(val_dataset, batch_size=config['batch_size'])  

    # ------------------------------------------------------------------------
    # SETUP
    # ------------------------------------------------------------------------
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_epochs = config['num_epochs']

    embeddings_dict = load_embeddings('./embeddings/glove_bio_asq_mimic.no_clean.300d.pickled')
    embeddings = match_embeddings(train_dataset.vocabulary, embeddings_dict)
    embeddings = torch.Tensor(embeddings).float().to(device)

    model = LateFusionModel(config, train_dataset.vocabulary, embeddings)
    # model = RecursiveAttentionModel(config, train_dataset.vocabulary, embeddings)
    # model = StackedAttentionModel(config, train_dataset.vocabulary, embeddings=None)
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

    # ------------------------------------------------------------------------
    # LOGGING
    # ------------------------------------------------------------------------
    experiment_name = "lf_mednli_embeddings_unfreeze"
    writer = SummaryWriter(log_dir="./logs/" + experiment_name)
    model_path = os.path.join(args.output_dir, 'weights_{}.pt'.format(experiment_name))

    # ------------------------------------------------------------------------
    # TRAINING LOOP
    # ------------------------------------------------------------------------
    best_micro_f1 = None
    for epoch in range(1, num_epochs + 1):
        batch_losses = []

        model.train()
        for batch in tqdm(train_dataloader, desc="Train"):
            image, history, question, options = batch['image'], batch['history'], batch['question'], batch['options']
            image, history, question, options = image.to(device), history.to(device), \
                                                question.to(device), options.to(device)

            optimizer.zero_grad()

            output = model(image, history, question, options)
            target = batch["answer_ind"].to(device)

            batch_loss = criterion(output.view(-1, output.size(-1)), target.view(-1))

            batch_losses.append(batch_loss.item())
            batch_loss.backward()
            optimizer.step()

        epoch_loss = np.mean(np.array(batch_losses))
        print(f'Train loss after epoch {epoch}: {epoch_loss}')
        writer.add_scalar("Train loss", epoch_loss, epoch)

        model.eval()
        targets = []
        outputs = []
        batch_losses = []

        for batch in tqdm(val_dataloader, desc="Validation"):
            image, history, question, options = batch['image'], batch['history'], batch['question'], batch['options']
            image, history, question, options = image.to(device), history.to(device), \
                                                question.to(device), options.to(device)

            output = model(image, history, question, options)
            target = batch["answer_ind"].to(device)

            batch_loss = criterion(output.view(-1, output.size(-1)), target.view(-1))
            batch_losses.append(batch_loss.item())
            
            targets.append(target.detach().cpu().numpy())
            outputs.append(output.detach().cpu().numpy())

        epoch_loss = np.mean(np.array(batch_losses))
        

        f1_scores, conf_matrix, micro_f1 = report_metric(targets, outputs)
        scores_dict = {'Yes': f1_scores[0], 'No': f1_scores[1], 'Maybe': f1_scores[2], 'N/A': f1_scores[3]}

        if best_micro_f1 is None or micro_f1 > best_micro_f1:
            best_loss = epoch_loss
            torch.save(model.state_dict(), model_path)
            best_micro_f1 = micro_f1

        writer.add_scalars("Metrics", scores_dict, epoch)

    writer.close()
    print(best_micro_f1)
