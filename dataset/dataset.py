import json
from itertools import chain

import h5py
import torch
from nltk import word_tokenize
from torch.utils.data import Dataset
from tqdm import tqdm

from dataset.vocabulary import Vocabulary

PAD_TOKEN = '<PAD>'
# TODO: remove ^ from here 


class CXRVisDialDataset(Dataset):
    """
    TODO: add an option for merging labels
    """
    def __init__(self, image_features_path, dialog_path, vocab_path):
        super().__init__()
        self.image_reader = ImageFeaturesReader(image_features_path)
        self.dialog_reader = DialogReader(dialog_path)

        self.image_ids = list(self.image_reader.features.keys())
        self.questions = self.dialog_reader.visdial_data['data']['questions']
        self.answers = self.dialog_reader.visdial_data['data']['answers']

        self.vocabulary = Vocabulary(vocab_path)

    def __len__(self):
        return len(self.dialog_reader.dialogs)

    def __getitem__(self, idx):
        dialog = self.dialog_reader.dialogs[idx]
        image_id = dialog['image_id']
        image_vector = self.image_reader.features[image_id]

        history_ids = [self.vocabulary.to_indices(history) for history in dialog['padded_history']]
        question_ids = [self.vocabulary.to_indices(question) for question in dialog['padded_question']]
        option_ids = [[self.vocabulary.to_indices(item) for item in option] for option in dialog['padded_options']]

        history_ids = torch.Tensor(history_ids).long()
        question_ids = torch.Tensor(question_ids).long()
        option_ids = torch.Tensor(option_ids).long()
        answer_inds = torch.Tensor(dialog['answer_ind']).long()

        example = {'history': history_ids,
                   'question': question_ids,
                   'answer': dialog['answer'],
                   'image': image_vector,
                   'options': option_ids,
                   'answer_ind': answer_inds}

        return example
    

class ImageFeaturesReader(object):
    """
    TODO: read multiple images per report
    """
    def __init__(self, image_features_path):
        self.features = self._load(image_features_path)

    def __len__(self):
        return len(self.image_id_list)
    
    def _load(self, image_features_path):
        print('Loading image vectors...')
        with h5py.File(image_features_path, "r") as features_hdf:
            self.image_id_list = list(features_hdf["image_id"])
            features = {}
            for i, image_id in enumerate(tqdm(self.image_id_list)):
                image_id_features = features_hdf["features"][i]
                features[image_id] = image_id_features
        print(f'Loaded {len(self.image_id_list)} image vectors.')
        return features
        

class DialogReader(object):
    """
    TODO: keep full dialogs? Add an option for permutation
    """
    def __init__(self, dialog_path):
        self.visdial_data = self._load(dialog_path)
        self.questions = [word_tokenize(question.lower().replace('_', ' ')) \
                          for question in self.visdial_data['data']['questions']]
        self.answers = [word_tokenize(answer.lower().replace('_', ' ')) \
                        for answer in self.visdial_data['data']['answers']]
        self.dialogs = self._get_possible_dialogs()
        self._pad_dialogs()

    @staticmethod
    def _load(dialog_path):
        with open(dialog_path, "r") as visdial_file:
            visdial_data = json.load(visdial_file)

        return visdial_data

    def _get_possible_dialogs(self):
        # TODO tokenize? add question id answer id? add a dialog class?
        dialogs = []
        print("Loading dialogs...")
        for i, dialog in enumerate(tqdm(self.visdial_data['data']['dialogs'])):
            num_rounds = len(dialog['dialog'])
            image_questions = [self.questions[turn['question']] for turn in dialog['dialog']]
            dialog_answer_inds = [turn['answer'] for turn in dialog['dialog']]
            dialog_answers = [self.answers[turn['answer']] for turn in dialog['dialog']]
            image_caption = word_tokenize(dialog['caption'])
            dialog_options = [[self.answers[option] for option in turn['answer_options']] for turn in dialog['dialog']]

            image_pairs = [question + answer for question, answer in zip(image_questions, dialog_answers)]
            histories = [image_caption + list(chain.from_iterable(image_pairs[:i])) for i in range(num_rounds)]
            dialogs.append({'history': histories,
                             'question': image_questions,
                             'answer': dialog_answers,
                             'image_id': dialog['image_id'],
                            'options': dialog_options,
                            'answer_ind': dialog_answer_inds})
        print(f"Loaded {len(dialogs)} dialogs.")
        return dialogs
            
    def _pad_dialogs(self, max_q_len=5, max_a_len=5, max_o_len=5, max_h_len=100):
        print(f'Padding dialogs...')
        for dialog in self.dialogs:
            dialog['padded_history'] = []
            for history in dialog['history']:
                if len(history) < max_h_len:
                    padded_history = history + [PAD_TOKEN]*(max_h_len - len(history))
                else:
                    padded_history = history[:max_h_len]
                dialog['padded_history'].append(padded_history)

            dialog['padded_question'] = []
            for question in dialog['question']:
                if len(question) < max_q_len:
                    padded_question = question + [PAD_TOKEN]*(max_q_len - len(question))
                else:
                    padded_question = question[:max_q_len]
                dialog['padded_question'].append(padded_question)

            dialog['padded_options'] = []
            for options in dialog['options']:
                padded_options = []
                for option in options:
                    if len(option) < max_o_len:
                        padded_option = option + [PAD_TOKEN] * (max_o_len - len(option))
                    else:
                        padded_option = option[:max_o_len]
                    padded_options.append(padded_option)
                dialog['padded_options'].append(padded_options)
        print(f'Finished padding.')




