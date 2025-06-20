import os
import sys
import json
import spacy
import logging
import argparse
from .. import consts
from typing import Tuple, List
from spacy.tokens import DocBin
from spacy.training.example import Example
from .nlp import balance_examples, generate_examples


logging.basicConfig(format=consts.LOG_FORMAT, level=logging.INFO)


def get_text(file_path):
    with open(file_path, encoding="utf8") as fh:
        txt = fh.read()
        return txt


def get_spans(annotations):
    spans = [
        (i["begin"], i["end"], i["value"])
        for i in annotations["%FEATURE_STRUCTURES"]
        if i["%TYPE"] == "de.tudarmstadt.ukp.dkpro.core.api.ner.type.NamedEntity"
        and i.get("value", None)
        in ["ORG", "POST_CODE", "ADDR", "LOC", "DATE", "PERSON"]
    ]
    # i.get("value", None) in ["ORG"]]
    spans = sorted(spans, key=lambda i: i[0])

    return spans


def _get_sentences_boundaries(annotations):
    sentences = [
        (i["begin"], i["end"])
        for i in annotations["%FEATURE_STRUCTURES"]
        if i["%TYPE"] == "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence"
    ]
    sentences = sorted(sentences, key=lambda i: i[0])

    return sentences


def _get_sent_annots(source_txt, spans, sent_bounds):
    span_index = 0
    sent_index = 0
    spans_len = len(spans)
    curr_sent_start, curr_sent_end = sent_bounds[sent_index]
    curr_sent = source_txt[curr_sent_start:curr_sent_end]
    docs = []
    doc_spans = []

    while span_index < spans_len:
        span = spans[span_index]
        # If the current custom span begin index is greater then
        # the current sentence end index move to the next sentence.
        if span[0] > curr_sent_end:
            while curr_sent_end <= span[0]:
                docs.append((curr_sent, doc_spans))
                doc_spans = []
                sent_index += 1
                curr_sent_start, curr_sent_end = sent_bounds[sent_index]
                curr_sent = source_txt[curr_sent_start:curr_sent_end]
        # if the current span boundaries extend over multiple sentences, change
        # the current sentence boundaries so that all the sentences are included.
        while (span[0] >= curr_sent_start) and (span[1] > curr_sent_end):
            sent_index += 1
            curr_sent_end = sent_bounds[sent_index][1]
            curr_sent = source_txt[curr_sent_start:curr_sent_end]
        doc_spans.append(
            (span[0] - curr_sent_start, span[1] - curr_sent_start, span[2])
        )
        span_index += 1

    # Add the last doc's spans.
    docs.append((curr_sent, doc_spans))

    return docs


def check_unknown_tokens(doc):
    unk_indices = [i for i, token in enumerate(doc) if token == "[UNK]"]
    if unk_indices:
        print(f"Found [UNK] tokens in text '{doc}' at indices {unk_indices}.")
        print(f"Tokens list: {doc}")


def create_spacy_doc(doc_annots: Tuple[str, List], language_model="en_core_web_trf"):
    doc_bin = DocBin()
    nlp = spacy.load(language_model)

    if "spancat" not in nlp.pipe_names:
        nlp.add_pipe("spancat", config={"spans_key": "sc"}, last=True)

    if len(doc_annots) > 0:
        for text, annots in doc_annots:
            doc = nlp.make_doc(text)
            check_unknown_tokens(doc)
            # because of the "expand" value for the alignment_mode, in some cases, a text, that isn't related to
            # the entity type, will be included in the Span object e.g. for token www.iescrow.com/2TheMart
            # if just the "2TheMart" is labeled as a separate entity, the entire URL string will be included.
            # This can be resolved by using a custom tokenizer rule, modifing the input text so that
            # the "2TheMart" is its own token or simply ignoring "2TheMart" string if it occurs inside an URL
            # and
            spans = [
                doc.char_span(s[0], s[1], label=s[2], alignment_mode="expand")
                for s in annots
            ]
            valid_spans = [s for s in spans if s is not None]
            if len(valid_spans) != len(annots):
                logging.info(
                    f"The count of valid spans and the count of annotations mismatch: {len(valid_spans)} != {len(annots)}."
                )
                logging.info(f"Document text: {doc.text}")
            doc.spans["sc"] = valid_spans
            # because of the "expand" alignment mode, which expands the span to include all tokens that partially
            # overlap with the provided character offsets, the starting and the ending position of characters in a span
            # can be changed compared to the original annotations, after calling the char_span method.
            aligned_annots = [(s.start_char, s.end_char, s.label_) for s in valid_spans]
            example = Example.from_dict(doc, {"spans": {"sc": aligned_annots}})
            doc_bin.add(example.reference)

        return doc_bin

    return None


def are_group_args_provided(args_group, args):
    for group_action in args_group._group_actions:
        long_option = (
            group_action.option_strings[-1].replace("--", "").replace("-", "_")
        )
        if getattr(args, long_option) is not None:
            return True

    return False


def add_single_file_arg_group(parser, group_name="Single file"):
    group = parser.add_argument_group(group_name)

    group.add_argument(
        "-a", "--annots-file", help="The path to the .json file with annotations."
    )
    group.add_argument("-s", "--txt-file", help="The path to the original .txt file.")
    group.add_argument(
        "-o",
        "--out-dir",
        help="The path to a folder in which the .spacy file will be created.",
    )

    return group


def add_multiple_files_arg_group(parser, group_name="Multiple files"):
    group = parser.add_argument_group(group_name)

    group.add_argument(
        "-l",
        "--annots-dir",
        help="The path to a folder containing .json files with annotations.",
    )
    group.add_argument(
        "-t",
        "--txt-dir",
        help="The path to a folder containing the original .txt files.",
    )
    group.add_argument(
        "-d",
        "--dest-dir",
        help="The path to a folder in which the .spacy files will be created.",
    )

    return group


def process_program_args():
    parser = argparse.ArgumentParser()

    single_file_arg_group = add_single_file_arg_group(parser)
    multiple_files_arg_group = add_multiple_files_arg_group(parser)
    parser.add_argument(
        "-g",
        "--synth-gen",
        action=argparse.BooleanOptionalAction,
        help="Whether or not to add synthetically generated data to the original dataset.",
    )

    args = parser.parse_args()

    single_file_group_used = are_group_args_provided(single_file_arg_group, args)
    multiple_file_group_used = are_group_args_provided(multiple_files_arg_group, args)

    if single_file_group_used and multiple_file_group_used:
        print(
            f"\nERROR: Arguments from '{single_file_arg_group.title}' and '{multiple_files_arg_group.title}'",
            "parameters group, are mutually exclusive.\n",
        )
        parser.print_help()
        sys.exit(2)

    return args


def _process_annotations(annots_file, txt_file):
    annotations_txt = get_text(annots_file)
    annotations = json.loads(annotations_txt)
    source_txt = get_text(txt_file)
    spans = get_spans(annotations)
    sentences_boundaries = _get_sentences_boundaries(annotations)
    sent_annots = _get_sent_annots(source_txt, spans, sentences_boundaries)

    return sent_annots


def _get_unique_annots(doc_annots: List[Tuple[str, List]]) -> List[Tuple[str, List]]:
    unique_keys = set()
    unique_annots = []

    for text, annots in doc_annots:
        if text not in unique_keys:
            unique_keys.add(text)
            unique_annots.append((text, annots))
        else:
            logging.info(f"Text {text} is already in unique annotations list.")

    return unique_annots


def _augment_annots(
    docs_annots: List[Tuple[str, List]], max_augmented_examples=5
) -> List[Tuple[str, List]]:
    augmented_docs_annots = []

    for annots in docs_annots:
        augmented_examples = generate_examples(annots, max_augmented_examples)
        augmented_docs_annots.extend(augmented_examples)

    return augmented_docs_annots


def _process_single_file(
    annots_file_path: str, txt_file_path: str, out_dir: str, synth_gen: bool
) -> None:
    """
    Processes a single file containing annotations.

    Args:
        annots_file_path: a path to the file containing the annotations.
        txt_file_path: a path to the file containing the original text.
        out_dir: the path to the folder where the .spacy file will be stored.
        synth_gen: whether or not to augment the original text samples with synthetically generated data.

    Returns:
        None
    """
    logging.info(f"Processing annotations file: {annots_file_path}")
    doc_annots = _process_annotations(annots_file_path, txt_file_path)
    unique_annots = _get_unique_annots(doc_annots)
    """
    if synth_gen:
        augmented_annots = _augment_annots(unique_annots)
        unique_annots += augmented_annots
    """
    unique_annots = balance_examples(unique_annots)
    logging.info(
        f"Annotations count: {len(doc_annots)}, unique annotations count: {len(unique_annots)}"
    )
    doc_bin = create_spacy_doc(unique_annots)
    out_file = os.path.splitext(annots_file_path.split(os.path.sep)[-1])[0] + ".spacy"
    doc_bin.to_disk(os.path.join(out_dir, out_file))


def _process_multiple_files(annots_dir, txt_dir, out_dir, synth_gen):
    for annot_file in os.listdir(annots_dir):
        txt_file_name = os.path.splitext(annot_file)[0] + ".txt"
        txt_file_path = os.path.join(txt_dir, txt_file_name)
        if not os.path.exists(txt_file_path):
            logging.error(f"TXT file '{txt_file_path}' not found, skipping it.")
            continue
        annot_file_path = os.path.join(annots_dir, annot_file)
        _process_single_file(annot_file_path, txt_file_path, out_dir, synth_gen)


def main():
    args = process_program_args()

    if getattr(args, "annots_file", None):
        _process_single_file(
            args.annots_file, args.txt_file, args.out_dir, args.synth_gen
        )
    else:
        _process_multiple_files(
            args.annots_dir, args.txt_dir, args.dest_dir, args.synth_gen
        )
