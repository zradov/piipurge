import random
from typing import List
from itertools import filterfalse
from collections import defaultdict
from typing import Tuple, List, Dict, Callable
from .synth_text_generator import generate_text


def generate_examples(
        doc_annots: Tuple[str, List[Tuple[int, int, str]]], 
        max_examples: int) -> List[Tuple[str, List]]:
    return generate_text(doc_annots, max_examples)


class EntityBalancer:

    def __init__(self, ngram_delim=" "):
        self.ngram_delim = ngram_delim
    

def is_single_entity_annotation(annot: Tuple[str, list]) -> bool:
    entity_types = set([a[2] for a in annot[1]])
    return len(entity_types) == 1
        

def _group_entity_examples(docs_annots: List[Tuple[str, List]]) \
    -> Dict[str, List[Tuple[str, List]]]:
    """Count occurrences of each entity type."""
    annots_counts = defaultdict(list)

    for text, annotations in docs_annots:
        entity_types = set([a[2] for a in annotations])
        for ent_type in entity_types:
            annots_counts[ent_type].append((text, annotations))

    return dict(annots_counts)
    

def _augment_annots(docs_annots: List[Tuple[str, List]], 
                    max_augmented_examples: int) -> List[Tuple[str, List]]:
    
    augmented_docs_annots = []

    for annots in docs_annots:
        augmented_examples = generate_text(annots, max_augmented_examples)
        augmented_docs_annots.extend(augmented_examples)

    return augmented_docs_annots


def _oversample(annots_groupped_by_entity: Dict[str, List[Tuple[str, List]]],
                max_majority_minority_ratio: float) \
    -> Dict[str, List[Tuple[str, List]]]:

    sorted_annots = dict(sorted(annots_groupped_by_entity.items(), key=lambda i: len(i[1][1])))
    max_count = max([len(annots) for _, annots in sorted_annots.items()])
    entity_types = list(sorted_annots.keys())
    new_single_entities_annots = {
        entity_types[0]: sorted_annots[entity_types[0]]
    }

    for entity_type in entity_types[1:]:
        entity_type_examples_len = len(sorted_annots[entity_type])
        examples_ratio = max_count / entity_type_examples_len
        if examples_ratio > max_majority_minority_ratio:
            total_examples_to_generate = int(round(entity_type_examples_len * examples_ratio))
            num_augmentations_per_example = (total_examples_to_generate // entity_type_examples_len) + 1
            augmented_examples = generate_text(sorted_annots[entity_type], num_augmentations_per_example)
            new_single_entities_annots[entity_type] = augmented_examples
        else:
            new_single_entities_annots[entity_type] = sorted_annots[entity_type]

    return new_single_entities_annots


def _undersample(annots_groupped_by_entity: Dict[str, List[Tuple[str, List]]],
                 max_majority_minority_ratio: float) \
                -> Dict[str, List[Tuple[str, List]]]:
    
    sorted_annots = dict(sorted(annots_groupped_by_entity.items(),
                                         key=lambda i: len(i[1][1]), reverse=True))
    min_count = min([len(annots) for _, annots in sorted_annots.items()])
    entity_types = list(sorted_annots.keys())
    new_single_entities_annots = {
        entity_types[0]: sorted_annots[entity_types[0]]
    }
    # Maximum allowed number of samples per entity type.
    max_allowed_samples = int(round(max_majority_minority_ratio) * min_count)

    for entity_type in entity_types[1:]:
        entity_type_examples_len = len(sorted_annots[entity_type])
        examples_ratio = entity_type_examples_len / min_count
        if examples_ratio > max_majority_minority_ratio:
            new_single_entities_annots[entity_type] = random.sample(
                sorted_annots[entity_type], max_allowed_samples)
        else:
            new_single_entities_annots[entity_type] = sorted_annots[entity_type]

    return new_single_entities_annots


def balance_examples(docs_annots: List[Tuple[str, List]], 
                     max_majority_minority_ratio: float=1.5) -> List[Tuple[str, List]]:
    """Performs examples balancing"""
    single_entity_annots = list(filter(is_single_entity_annotation, docs_annots))
    grouped_single_entity_annots = _group_entity_examples(single_entity_annots)
    multiple_entities_annots = list(filterfalse(is_single_entity_annotation, docs_annots))
    grouped_single_entity_annots = dict(sorted(grouped_single_entity_annots.items(), key=lambda i: len(i[1][1])))
    augmented_single_entities_annots = _oversample(grouped_single_entity_annots, max_majority_minority_ratio)
    single_entities_annots = _undersample(augmented_single_entities_annots, max_majority_minority_ratio)
    balanced_examples = single_entities_annots.values() + multiple_entities_annots.values()

    return balanced_examples

