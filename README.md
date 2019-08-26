# Visual Dialog for Radiology
Introducing a Visual Dialog task in radiology. The general-domain task description can be found [here](https://visualdialog.org/).

## Introduction
We provide the baseline models and results for the Visual Dialog task that uses MIMIC<sup>[1](#mimic)</sup> chest X-ray images and associated reports. Our silver-standard dataset is constructed using [CheXpert annotating tool](https://stanfordmlgroup.github.io/competitions/chexpert/).

Our baseline models include:
- LateFusion<sup>[2](#lf)</sup> model (provided with the general-domain challenge [starter code](https://github.com/batra-mlp-lab/visdial-challenge-starter-pytorch)).
- Recursive Visual Attention<sup>[3](#rva)</sup> model, the 2019 winner of the general-domain challenge ([repository](https://github.com/yuleiniu/rva)).
- Stacked Attention Network<sup>[4](#san)</sup>. We make modifications to the architecture of the model to take into account the history of the dialog turns.

## Prerequisites
Our models are implemented in PyTorch. Install dependencies as
```
pip install -r requirements.txt
```

## Usage
To train one of the three models (LateFusion model by default) run the train script as:

```
python train.py \ 
    --train_json <path_to_train_json>  \
    --val_json <path_to_val_json> \
    --train_img_feats_dir <path_to_train_img_features> \
    --val_img_feats_dir <path_to_val_img_features> \
    --word_counts <path_to_word_count_json> \
    --output_dir <path_to_output_dir>
```
You can select a different model passing a `--model` argument with valid options being `lf`, `rva` and `san`.
If you want to use pre-trained word embeddings, pass an extra argument as `--embeddings <path_to_pickled_embeddings_dict>`. MedNLI domain-specific embeddings used in our experiments can be found [here](https://github.com/jgc128/mednli).



## References
<a name="mimic">1</a>: [MIMIC-CXR: A LARGE PUBLICLY AVAILABLE DATABASE OF LABELED CHEST RADIOGRAPHS](https://arxiv.org/pdf/1901.07042.pdf)

<a name="lf">2</a>: [Visual Dialog](https://arxiv.org/pdf/1611.08669.pdf)

<a name="rva">3</a>: [Recursive Visual Attention in Visual Dialog](https://arxiv.org/pdf/1812.02664.pdf)

<a name="san">4</a>: [Stacked Attention Networks for Image Question Answering](https://arxiv.org/pdf/1511.02274.pdf)

