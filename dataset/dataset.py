import json
import os
import random
from itertools import chain

import h5py
import numpy as np
import torch
from nltk import RegexpTokenizer
from pytorch_transformers import BertModel, BertTokenizer
from torch.utils.data import Dataset
from tqdm import tqdm

from dataset.vocabulary import Vocabulary


class CXRVisDialDataset(Dataset):
    def __init__(self, image_features_dir, dialog_path, vocab_path=None, mode='concat', permute=False, views=['PA']):
        """
        A dataset class.

        Args:
            image_features_dir (str): path to image features file/folder?
            dialog_path (str): path to .json file with dialog data
            vocab_path (str, optional): path to word counts. If None, BERT vocabulary is used instead
            permute (bool, optional): Whether to permute dialog turns in random order. Defaults to False
            views (list, optional): List of views for which image vectors are extracted and concatenated

        """
        super().__init__()
        if vocab_path is not None:
            self.vocabulary = Vocabulary(vocab_path)
            self.tokenizer = RegexpTokenizer('\w+')
            self.bert = False
        else:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
            self.vocabulary = self.tokenizer.vocab
            self.bert = True

        # Read dialogs and corresponding image ids
        self.dialog_reader = DialogReader(dialog_path, self.vocabulary, self.tokenizer, mode, permute)
        self.image_ids = self.dialog_reader.image_ids

        # Read image vectors
        self.image_reader = ImageFeaturesReader(image_features_dir, self.image_ids, views)

        # Get all possible questions and answers
        self.questions = self.dialog_reader.visdial_data['data']['questions']
        self.answers = self.dialog_reader.visdial_data['data']['answers']

        self.all_dialogs = self.get_all_dialogs(mode)

    def get_all_dialogs(self, mode):
        """
        Extract all dialog examples.
        Args:
            mode (str): If 'seq', dialog history consists of separate sequences of turns.
                If 'concat', dialog turns are concatenated.

        Returns:
            a list of dictionaries with dialog examples. Tokens are replaced with their ids based on vocabulary.
                Returned dialogs are padded.

        """
        all_dialogs = []
        print('Extracting all possible data examples...')
        for dialog in tqdm(self.dialog_reader.dialogs):
            image_id_frontal = dialog['image_id']
            # image_id_frontal = dialog['image_id_frontal']
            # image_id_lateral = dialog['image_id_lateral']
            try:
                image_vector_frontal = self.image_reader.image_vectors[image_id_frontal]
                # image_vector_lateral = self.image_reader.image_vectors[image_id_lateral]
                # image_vector = np.concatenate((image_vector_frontal, image_vector_lateral), axis=0)
                image_vector = image_vector_frontal
            except Exception:
                continue
            num_turns = len(dialog['history'])
            for turn in range(num_turns):
                if mode == 'concat':
                    if self.bert:
                        history_ids = self.tokenizer.convert_tokens_to_ids(dialog['padded_history'][turn])
                        caption_ids = self.tokenizer.convert_tokens_to_ids(dialog['padded_history'][0])
                        question_ids = self.tokenizer.convert_tokens_to_ids(dialog['padded_question'][turn])
                    else:
                        history_ids = self.vocabulary.to_indices(dialog['padded_history'][turn])
                        caption_ids = self.vocabulary.to_indices(dialog['padded_history'][0])
                        question_ids = self.vocabulary.to_indices(dialog['padded_question'][turn])
                elif mode == 'seq':
                    if self.bert:
                        history_ids = [self.tokenizer.convert_tokens_to_ids(sequence)
                                       for sequence in dialog['padded_history'][turn]]
                        caption_ids = history_ids[0]
                        question_ids = [self.tokenizer.convert_tokens_to_ids(sequence)
                                        for sequence in dialog['padded_all_questions'][turn]]
                    else:
                        history_ids = [self.vocabulary.to_indices(sequence)
                                       for sequence in dialog['padded_history'][turn]]
                        caption_ids = history_ids[0]
                        question_ids = [self.vocabulary.to_indices(sequence)
                                        for sequence in dialog['padded_all_questions'][turn]]
                if self.bert:
                    option_ids = [self.tokenizer.convert_tokens_to_ids(option)
                                  for option in dialog['padded_options'][turn]]
                else:
                    option_ids = [self.vocabulary.to_indices(option)
                                  for option in dialog['padded_options'][turn]]


                history_ids = torch.Tensor(history_ids).long()
                question_ids = torch.Tensor(question_ids).long()
                option_ids = torch.Tensor(option_ids).long()
                caption_ids = torch.Tensor(caption_ids).long()
                answer_ind = dialog['answer_ind'][turn]

                all_dialogs.append({'history_ids': history_ids,
                                    'history': dialog['history'][turn],
                                    'question': dialog['question'][turn],
                                    'question_ids': question_ids,
                                    'answer': dialog['answer'][turn],
                                    'image': image_vector,
                                    'options': option_ids,
                                    'answer_ind': answer_ind,
                                    'caption_ids': caption_ids,
                                    'turn': turn})

        return all_dialogs

    def __len__(self):
        return len(self.all_dialogs)

    def __getitem__(self, idx):
        return self.all_dialogs[idx]
    

class ImageFeaturesReader(object):
    def __init__(self, image_features_dir, image_ids, views=["PA"]):
        """
        A class for reading image data.

        Args:
            image_features_dir (str):
            image_ids (list): Image ids from dialogs to load corresponding image vectors
            views (list, optional): List of views for which image vectors are extracted and concatenated
        """
        self.image_ids = image_ids
        self.image_vectors = self._load(image_features_dir, views=views)

    def __getitem__(self, image_id):
        return self.image_vectors[image_id]

    def __len__(self):
        return len(self.image_ids)
    
    def _load(self, image_features_dir, views):
        print('Loading image vectors...')
        features = {}
        for view in views:
            path = os.path.join(image_features_dir, view, 'images.h5')
            with h5py.File(path, "r") as features_hdf:
                for image_id in tqdm(self.image_ids, desc=view):
                    try:
                        image_id_features = features_hdf.get(image_id).value

                        # Image vectors are either concatenated regions or densenet/resnet penultimate layers
                        if image_id_features.shape[0] == 11 or len(image_id_features.shape) == 3:
                            features[image_id] = image_id_features
                    except Exception:
                        print(f"Vector not found for image {image_id}")
            print(f'Loaded {len(self.image_ids)} image vectors.')
        return features

class DialogReader(object):
    def __init__(self, dialog_path, vocabulary, tokenizer, mode='concat', permute=False):
        """

        Args:
            dialog_path (str): path to .json file with dialog data
            vocabulary: either Vocabulary class constructed from word counts or BERT vocabulary
            tokenizer: either a RegExp nltk tokenizer or BERT tokenizer
            mode (str): If 'seq', dialog history consists of separate sequences of turns.
                If 'concat', dialog turns are concatenated.
            permute (bool, optional): Whether to permute dialog turns in random order. Defaults to False
        """
        self.tokenizer = tokenizer
        try:
            self.pad_token = vocabulary.PAD_TOKEN
            self.eos_token = vocabulary.EOS_TOKEN
        except Exception:
            self.pad_token = tokenizer.pad_token
            self.eos_token = tokenizer.eos_token

        self.visdial_data = self._load(dialog_path)
        self.questions = [self.tokenizer.tokenize(question.lower().replace('_', ' ')) \
                          for question in self.visdial_data['data']['questions']]
        self.answers = [self.tokenizer.tokenize(answer.lower().replace('_', ' ')) \
                        for answer in self.visdial_data['data']['answers']]
        self.dialogs = self._get_possible_dialogs(mode, permute)

        ### TODO edit below ####
        self.image_ids = [dialog['image_id'] for dialog in self.dialogs]
        # self.image_ids = list(set([dialog['image_id_frontal'] for dialog in self.dialogs] +
        #                           [dialog['image_id_lateral'] for dialog in self.dialogs]))
        self._pad_dialogs(mode)

    @staticmethod
    def _load(dialog_path):
        with open(dialog_path, "r") as visdial_file:
            visdial_data = json.load(visdial_file)

        return visdial_data

    def _get_possible_dialogs(self, mode, permute):
        """
        Extract <num_turns> sub-dialogs from a given dialog.

        Args:
            mode (str): If 'seq', dialog history consists of separate sequences of turns.
                If 'concat', dialog turns are concatenated.
            permute (bool, optional): Whether to permute dialog turns in random order. Defaults to False

        Returns:

        """
        dialogs = []
        print("Loading dialogs...")
        for i, dialog in enumerate(tqdm(self.visdial_data['data']['dialogs'])):
            num_rounds = len(dialog['dialog'])
            if num_rounds == 0:
                print(f"No questions found for {dialog['image_id']}.")
                continue
            image_questions = [self.questions[turn['question']] for turn in dialog['dialog']]
            dialog_answer_inds = [turn['answer'] for turn in dialog['dialog']]
            dialog_answers = [self.answers[turn['answer']] for turn in dialog['dialog']]
            image_caption = self.tokenizer.tokenize(dialog['caption'])
            dialog_options = [[self.answers[option] for option in turn['answer_options']] for turn in dialog['dialog']]

            if permute:
                to_shuffle = list(zip(image_questions, dialog_answer_inds, dialog_answers, dialog_options))
                random.shuffle(to_shuffle)
                image_questions, dialog_answer_inds, dialog_answers, dialog_options = (list(item) for item in zip(*to_shuffle))

            image_pairs = [question + answer for question, answer in zip(image_questions, dialog_answers)]
            if mode == 'concat':
                histories = [image_caption + list(chain.from_iterable(image_pairs[:i])) for i in range(num_rounds)]
            elif mode == 'seq':
                histories = [[image_caption] + image_pairs[:i] for i in range(num_rounds)]
            dialogs.append({'history': histories,
                            'question': image_questions,
                            'answer': dialog_answers,
                            # 'image_id_frontal': dialog['image_id_frontal'],
                            # 'image_id_lateral': dialog['image_id_lateral'],
                            'image_id': dialog['image_id'],
                            'options': dialog_options,
                            'answer_ind': dialog_answer_inds})
        print(f"Loaded {len(dialogs)} dialogs.")
        return dialogs
            
    def _pad_dialogs(self, mode, max_ques_len=5, max_opt_len=5, max_hist_len=100, max_turn_len=20,  max_turns=5):
        """

        Args:
            mode (str): If 'seq', dialog history consists of separate sequences of turns.
            max_ques_len (int, optional): Maximum length to pad questions
            max_opt_len (int, optional: Maximum length to pad answer options
            max_hist_len (int, optional): Maximum length to pad history if turns are concatenated
            max_turn_len (int, optional): Maximum length to pad individual turn if turns are not concatenated
            max_turns (int, optional): Maximum number of turns per dialog to pad if turns are not concatenated

        """
        print(f'Padding dialogs...')
        for dialog in self.dialogs:
            dialog['padded_history'] = []
            for history in dialog['history']:
                if mode == 'concat':
                    if len(history) < max_hist_len:
                        padded_history = history + [self.pad_token]*(max_hist_len - len(history))
                    else:
                        padded_history = history[:max_hist_len]
                    dialog['padded_history'].append(padded_history)
                elif mode == 'seq':
                    new_history = []
                    for turn in history:
                        if len(turn) < max_turn_len:
                            new_turn = turn + [self.pad_token] * (max_turn_len - len(turn))
                        else:
                            new_turn = turn[:max_turn_len]
                        new_history.append(new_turn)
                    if len(new_history) < max_turns:
                        padded_history = new_history + [[self.eos_token] + [self.pad_token]*(max_turn_len - 1)]*(max_turns - len(new_history))
                    else:
                        padded_history = new_history[:max_turns]
                    dialog['padded_history'].append(padded_history)

            dialog['padded_question'] = []
            for i, question in enumerate(dialog['question']):
                if len(question) < max_ques_len:
                    padded_question = question + [self.pad_token]*(max_ques_len - len(question))
                else:
                    padded_question = question[:max_ques_len]

                dialog['padded_question'].append(padded_question)

            if mode == "seq":
                dialog['padded_all_questions'] = []
                for i, question in enumerate(dialog['padded_question']):
                    dialog['padded_all_questions'].append(dialog['padded_question'][:i+1] + \
                                                         [[self.eos_token] + [self.pad_token]*(max_ques_len - 1)]*(max_turns - len(dialog['question'][:i+1])))

            dialog['padded_options'] = []
            for options in dialog['options']:
                padded_options = []
                for option in options:
                    if len(option) < max_opt_len:
                        padded_option = option + [self.pad_token] * (max_opt_len - len(option))
                    else:
                        padded_option = option[:max_opt_len]
                    padded_options.append(padded_option)
                dialog['padded_options'].append(padded_options)
        print(f'Finished padding.')


