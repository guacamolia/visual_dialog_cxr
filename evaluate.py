import argparse
import yaml

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.dataset import CXRVisDialDataset
from models.models import LateFusionModel, RecursiveAttentionModel, StackedAttentionModel
from models.utils.utils import report_metric



if __name__ == "__main__":
    # ------------------------------------------------------------------------
    # ARGUMENTS
    # ------------------------------------------------------------------------
    parser = argparse.ArgumentParser()

    parser.add_argument("--test_json",
                        required=False,
                        help="Location of the test json file")

    parser.add_argument("--img_feats_test",
                        required=False,
                        help="Location of test images features")

    parser.add_argument("--word_counts",
                        required=True,
                        help="Location of the word counts file")

    parser.add_argument("--config_yml",
                        default="config.yaml",
                        help="Location of the config yaml file")

    parser.add_argument("--output_dir",
                        required=True,
                        help="Output location where the weights are stored")

    args = parser.parse_args()

    with open(args.config_yml) as f:
        config = yaml.safe_load(f)

    # ------------------------------------------------------------------------
    # DATASET & DATALOADER
    # ------------------------------------------------------------------------

    test_dataset = CXRVisDialDataset(args.img_feats_test, args.test_json, args.word_counts)
    test_dataloader = DataLoader(test_dataset, batch_size=config['batch_size'])

    # ------------------------------------------------------------------------
    # SETUP
    # ------------------------------------------------------------------------
    model = LateFusionModel(config, test_dataset.vocabulary)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.load_state_dict(args.output_dir)
    model.eval()

    # ------------------------------------------------------------------------
    # EVALUATION
    # ------------------------------------------------------------------------
    targets = []
    outputs = []
    for batch in tqdm(test_dataloader, desc="Evaluation"):
        image, history, question, options = batch['image'], batch['history'], batch['question'], batch['options']
        image, history, question, options = image.to(device), history.to(device), \
                                            question.to(device), options.to(device)

        output = model(image, history, question, options)
        target = batch["answer_ind"].to(device)

        targets.append(target.detach().cpu().numpy())
        outputs.append(output.detach().cpu().numpy())

    f1_scores, conf_matrix = report_metric(targets, outputs)
    scores_dict = {'Yes': f1_scores[0], 'No': f1_scores[1], 'Maybe': f1_scores[2], 'N/A': f1_scores[3]}
    print(conf_matrix)
    print(scores_dict)
