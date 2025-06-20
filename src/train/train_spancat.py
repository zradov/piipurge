import spacy
import random
from pathlib import Path
from spacy.tokens import DocBin
from spacy.util import minibatch
from spacy.training import Example
from spacy.language import Language, Vocab


def unfreeze_transformer(nlp: Language):
    if "transformer" in nlp.pipe_names:
        nlp.get_pipe("transformer").frozen = False


"""
@spacy.Language.component("unfreeze_scheduler")
def unfreeze_scheduler(nlp: Language):
    Callback to unfreeze the transformer after a certain step/epoch.
    Adjust the logic as needed.
    print("Inside unfreeze_scheduler")
    print(type(pipe))
    if epoch == 1:  # Example: Unfreeze after epoch 1, step 500
        print("Unfreezing transformer ...")
        unfreeze_transformer(pipe)
"""


def load_data(data_path: str, nlp: Language):
    """Load training and dev data (adjust for your data format)."""
    doc_bin = DocBin().from_disk(data_path)
    docs = list(doc_bin.get_docs(nlp.vocab))
    """
    examples = [Example.from_dict(
        nlp.make_doc(doc.text), {"spans": {"sc": doc.spans["sc"]}})
        for doc in docs]
    """
    examples = []
    for doc in docs:
        gold_standard_annots = [
            (a.start_char, a.end_char, a.label_) for a in doc.spans["sc"]
        ]
        ex = Example.from_dict(
            nlp.make_doc(doc.text), {"spans": {"sc": gold_standard_annots}}
        )
        examples.append(ex)

    return examples


def train_spancat(config_path: str, train_path: str, dev_path: str, output_dir: str):
    # Load the config and create the nlp object
    nlp = spacy.util.load_model_from_config(
        spacy.util.load_config(config_path), auto_fill=True
    )

    # Add the unfreeze callback to the pipeline (runs during training)
    # nlp.add_pipe("unfreeze_scheduler", first=True)

    # Load data
    train_data = load_data(train_path, nlp)
    dev_data = load_data(dev_path, nlp)

    # Initialize the pipeline
    optimizer = nlp.initialize(lambda: train_data)

    # Training loop
    best_score = 0.0
    for epoch in range(10):  # Adjust epochs
        random.shuffle(train_data)
        losses = {}
        batches = minibatch(train_data, size=16)  # Adjust batch size

        for step, batch in enumerate(batches):
            nlp.update(batch, drop=0.1, losses=losses, sgd=optimizer)

            print(f"Epoch {epoch}, Step {step}, Losses: {losses}")

        # Evaluate on dev set
        dev_scores = nlp.evaluate(dev_data)
        print(f"Epoch {epoch} Dev Scores: {dev_scores}")

        # Save best model
        if dev_scores["spans_sc_f"] > best_score:
            best_score = dev_scores["spans_sc_f"]
            nlp.to_disk(Path(output_dir) / "best_model")

    # Save final model
    nlp.to_disk(Path(output_dir) / "final_model")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="Path to config.cfg", required=True)
    parser.add_argument("-t", "--train", help="Path to training data", required=True)
    parser.add_argument("-d", "--dev", help="Path to dev data", required=True)
    parser.add_argument("-o", "--output", help="Output directory", required=True)
    args = parser.parse_args()

    train_spancat(args.config, args.train, args.dev, args.output)
