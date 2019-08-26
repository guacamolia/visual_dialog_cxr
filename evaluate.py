import argparse
import yaml

import torch
from pytorch_transformers import BertTokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.dataset import CXRVisDialDataset
from dataset.vocabulary import Vocabulary
from models.models import LateFusionModel, RecursiveAttentionModel, StackedAttentionModel
from utils import report_metric, load_embeddings, match_embeddings



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

    parser.add_argument("--embeddings",
                        default=None,
                        help="Whether pretrained embeddings should be used. "
                             "If yes, the argument is a path to a pickled file.")

    parser.add_argument("--bert_path",
                        default=None,
                        help="If using BERT embeddings, the location for the pre-trained bin model")

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
    if args.word_counts is None:
        use_bert = True
        embeddings = None
        bert_path = args.bert_path
        train_vocabulary = BertTokenizer.from_pretrained('bert-base-cased').vocab
    else:
        train_vocabulary = Vocabulary(args.word_counts)
        use_bert = False
        bert_path = None
        if args.embeddings is not None:
            embeddings_dict = load_embeddings(args.embeddings)
            embeddings = match_embeddings(train_vocabulary, embeddings_dict)
            embeddings = torch.Tensor(embeddings).float().to(device)
        else:
            embeddings = None

    test_dataset = CXRVisDialDataset(args.img_feats_test, args.test_json, args.word_counts, mode, views=config['views'])
    test_dataloader = DataLoader(test_dataset, batch_size=config['batch_size'])

    # ------------------------------------------------------------------------
    # SETUP
    # ------------------------------------------------------------------------
    if args.model == "lf":
        model = LateFusionModel(config, train_vocabulary, args.embeddings)
    elif args.model == "rva":
        model = RecursiveAttentionModel(config, train_vocabulary, embeddings)
    elif args.model == "san":
        model = StackedAttentionModel(config, train_vocabulary, embeddings, use_bert, bert_path)

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
