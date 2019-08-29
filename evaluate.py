import argparse
import yaml

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.dataset import CXRVisDialDataset
from dataset.vocabulary import Vocabulary
from models.models import LateFusionModel, RecursiveAttentionModel, StackedAttentionModel
from utils import report_metric



if __name__ == "__main__":
    # ------------------------------------------------------------------------
    # ARGUMENTS
    # ------------------------------------------------------------------------
    parser = argparse.ArgumentParser()

    parser.add_argument("--test_json",
                        required=False,
                        help="Location of the test json file")

    parser.add_argument("--test_img_feats",
                        required=False,
                        help="Location of test images features")

    parser.add_argument("--word_counts",
                        required=True,
                        help="Location of the word counts file")

    parser.add_argument("--config_yml",
                        default="config.yaml",
                        help="Location of the config yaml file")

    parser.add_argument("--model_path",
                        required=True,
                        help="Output location where the weights are stored")

    parser.add_argument("--model",
                        default="lf",
                        help="Model to use for training. Valid options are: `lf` (LateFusion), "
                             "`rva` (RecursiveVisualAttention), and `san` (Stacked Attention Network)")


    args = parser.parse_args()

    with open(args.config_yml) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # ------------------------------------------------------------------------
    # DATASET & DATALOADER
    # ------------------------------------------------------------------------
    if args.model == "rva":
        mode = 'seq'
    elif args.model == "lf" or args.model == "san":
        mode = 'concat'
    else:
        raise ValueError("Unknown model")


    # If word_counts are passed, use them to construct the vocabulary. If word embeddings are also passed,
    # use them for initializing the embedding layer. Otherwise, use BERT
    train_vocabulary = Vocabulary(args.word_counts)

    test_dataset = CXRVisDialDataset(args.test_img_feats, args.test_json, args.word_counts, mode)
    test_dataloader = DataLoader(test_dataset, batch_size=config['batch_size'])

    # ------------------------------------------------------------------------
    # SETUP
    # ------------------------------------------------------------------------
    if args.model == "lf":
        model = LateFusionModel(config, train_vocabulary, args.embeddings)
    elif args.model == "rva":
        model = RecursiveAttentionModel(config, train_vocabulary)
    elif args.model == "san":
        model = StackedAttentionModel(config, train_vocabulary)

    model = model.to(device)
    model.load_state_dict(torch.load(args.model_path))
    print("Model weights restored")
    model.eval()

    # ------------------------------------------------------------------------
    # EVALUATION
    # ------------------------------------------------------------------------
    targets = []
    outputs = []
    turns = []
    for batch in tqdm(test_dataloader, desc="Evaluation"):
        image, history, question, options, caption, turn = batch['image'], batch['history_ids'], \
                                                           batch['question_ids'], batch['options'], \
                                                           batch['caption_ids'], batch['turn']
        image, history, question, options, caption, turn = image.to(device), history.to(device), question.to(device), \
                                                           options.to(device), caption.to(device), turn.to(device)

        output = model(image, history, question, options, caption, turn)
        target = batch["answer_ind"].to(device)

        targets.append(target.detach().cpu().numpy())
        outputs.append(output.detach().cpu().numpy())
        turns.append(turn.detach().cpu().numpy())

    f1_scores, conf_matrix, macro_f1, accuracies = report_metric(targets, outputs, turns)
    scores_dict = {'Yes': f1_scores[0], 'No': f1_scores[1], 'Maybe': f1_scores[2],
                   'N/A': f1_scores[3], 'macro_f1': macro_f1}
    print(conf_matrix)
    print(scores_dict)
