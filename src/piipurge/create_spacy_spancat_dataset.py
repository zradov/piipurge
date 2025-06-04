import spacy
import logging
import argparse
from . import consts
from pathlib import Path
from random import shuffle
from spacy.language import Vocab
from typing import Dict, List, Tuple
from spacy.tokens import Doc, DocBin


logging.basicConfig(format=consts.LOG_FORMAT, level=logging.INFO)


def _get_file_samples(file_path: str, vocab: Vocab) -> Tuple[List, List]:
    doc_bin = DocBin().from_disk(file_path)
    docs = doc_bin.get_docs(vocab)
    positive_samples = []
    negative_samples = []
    for doc in docs:
        if doc.spans["sc"]:
            positive_samples.append(doc)
        else:
            negative_samples.append(doc)
    logging.info(f"= File '{file_path}' =")
    logging.info(f"==> positive samples: {len(positive_samples)}")
    logging.info(f"==> negative samples: {len(negative_samples)}")

    return positive_samples, negative_samples


def _get_samples(spacy_dir: str, language_model_name: str) -> Dict[str, List] | None:
    spacy_files = list(Path(spacy_dir).rglob("**/*.spacy"))
    positive_samples = []
    negative_samples = []
        
    if spacy_files:
        nlp = spacy.load(language_model_name)

        for file_path in spacy_files:
            temp_positive_samples, temp_negative_samples = _get_file_samples(file_path, nlp.vocab)
            positive_samples.extend(temp_positive_samples)
            negative_samples.extend(temp_negative_samples)

    return positive_samples, negative_samples


def _split_samples(samples, train_pct: float, val_pct: float) -> Tuple[List, List, List]:
    train_samples = samples[0:int(len(samples)*train_pct)]
    val_samples = samples[len(train_samples):int(len(samples)*(train_pct+val_pct))]
    test_samples = samples[len(train_samples)+len(val_samples):]
    
    return train_samples, val_samples, test_samples


def _interleave_proportional(list1, list2):
    merged = []
    shorter_list = list1 if len(list1) < len(list2) else list2
    longer_list = list1 if len(list1) > len(list2) else list2
    len_short = len(shorter_list)
    len_long = len(longer_list)

    insert_indices = [round(i * (len_long + len_short) / len_short) for i in range(len_short)]
    
    j = k = 0 
    for i in range(len_long + len_short):
        if j < len_short and i == insert_indices[j]:
            merged.append(shorter_list[j])
            j += 1
        elif k < len_long:
            merged.append(longer_list[k])
            k += 1

    return merged


def create_dataset(spacy_dir_path: str,
                   output_dir_path: str,
                   language_model_name: str, 
                   train_pct: float, 
                   val_pct: float)-> None:
    
    positive_samples, negative_samples = _get_samples(spacy_dir_path, language_model_name)
    logging.info(f"Total positive samples: {len(positive_samples)}")
    logging.info(f"Total negative samples: {len(negative_samples)}")
    train_pos_samples, val_pos_samples, test_pos_samples = _split_samples(positive_samples, train_pct, val_pct)
    train_neg_samples, val_neg_samples, test_neg_samples = _split_samples(negative_samples, train_pct, val_pct)
    train_samples = _interleave_proportional(train_pos_samples, train_neg_samples)
    shuffle(train_samples)
    val_samples = val_pos_samples + val_neg_samples
    test_samples = test_pos_samples + test_neg_samples

    logging.info(f"Total training samples: {len(train_samples)}")
    logging.info(f"Total validation samples: {len(val_samples)}")
    logging.info(f"Total test samples: {len(test_samples)}")

    _save_to_disk(train_samples, output_dir_path, "train")
    _save_to_disk(val_samples, output_dir_path, "validation")
    _save_to_disk(test_samples, output_dir_path, "test")
    

def _save_to_disk(samples: List[Doc], output_dir: str, file_name: str, ext=".spacy") -> None:
    doc_bin = DocBin(docs=samples)
    output_file_path = Path(output_dir) / f"{file_name}{ext}"
    doc_bin.to_disk(output_file_path)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-s", "--spacy-dir", help="Path to the folder containing .spacy files.")
    parser.add_argument("-o", "--output-dir", help="Path to the folder where training, validation and the test .spacy files will be created.")
    parser.add_argument("-l", "--language-model", default="en_core_web_trf", help="Name of the language model used as a storage class for vocabulary.")
    parser.add_argument("-t", "--train-pct", default=0.7, type=float, help="Percentage of the total number of samples that will be used for the model training.")
    parser.add_argument("-v", "--val-pct", default=0.2, type=float, help="Percentage of the total number of samples that will be used for the model validation.")
    # The reason why the percentage argument for the test dataset size is left out is that, by default,
    # the percentage of the samples that will be used for the model evaluation will be equal to:
    # total_samples_count - int(total_samples_count * (train_pct + val_pct))

    args = parser.parse_args()

    create_dataset(args.spacy_dir, args.output_dir, args.language_model, args.train_pct, args.val_pct)
