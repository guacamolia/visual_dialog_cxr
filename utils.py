import pickle

import numpy as np

from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
from tqdm import tqdm

    
def report_metric(targets, outputs, turns):
    """
    Report f1-score per label class and accuracy per dialog round.
    Args:
        targets (list): List of numpy arrays with ground truth labels
        outputs (list): List of numpy arrays with scores per label
        turns (list): List of turn indices

    Returns:
        scores (tuple): f1_score per class, confusion matrix, macro-averaged f1, accuracies per round

    """
    targets = np.concatenate(targets)
    outputs = np.concatenate(outputs)
    turns = np.concatenate(turns)
    predictions = np.argmax(outputs, axis=-1)

    accuracies = []
    for round in np.unique(turns):
        indices = np.where(turns == round)
        round_predictions = predictions[indices]
        round_targets = targets[indices]
        round_accuracy = accuracy_score(round_targets, round_predictions)
        accuracies.append(round_accuracy)

    f1_scores = f1_score(targets.flatten(), predictions.flatten(), average=None)
    macro_f1 = f1_score(targets.flatten(), predictions.flatten(), average='macro')
    conf_matrix = confusion_matrix(targets.flatten(), predictions.flatten())

    return f1_scores, conf_matrix, macro_f1, accuracies


def match_embeddings(vocabulary, embeddings):
    """

    Args:
        vocabulary (Vocabulary): vocabulary class with word-to-idx mapping
        embeddings (dict): Word-to-vector dictionary of word embeddings

    Returns:
        (np.array): An array of embeddings ordered accordingly with the vocabulary indices

    """
    dim = list(embeddings.values())[0].shape[0]

    print("Matching word embeddings...")
    embs = []
    for i in tqdm(range(len(vocabulary))):
        token = vocabulary.index2word[i]
        token_emb = embeddings.get(token, np.random.uniform(-1, 1, dim))
        embs.append(token_emb)
    embs = np.stack(embs)
    return embs


def load_embeddings(path):
    """
    Load pre-trained embeddings from a pickled file.

    Args:
        path (str): Path to pickled embeddings

    Returns:
        (dict): Word-to-vector dictionary of word embeddings

    """
    with open(path, 'rb') as f:
        embeddings_dict = pickle.load(f)
    return embeddings_dict

