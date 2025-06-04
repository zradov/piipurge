import re
from .. import consts
import logging
from faker import Faker
from .paraphraser import Paraphraser
from typing import List, Tuple, Dict
from .nlp_metrics import get_text_similarity


logging.basicConfig(format=consts.LOG_FORMAT, level=logging.INFO)


ORIGINAL_TEXT_SIMILARITY_THRESH_RANGE = (0.85, 0.9950)
NEW_TEXT_SIMILARITY_THRESH_BELOW = 0.9
LABEL_TYPE_DESC_MAPPINGS = {
    "ORG": "Organization",
    "LOC": "Location",
    "POST_CODE": "Post office box",
    "ADDR": "Address",
    "PERSON": "Person",
    "DATE": "Date"
}
DATE_FORMATS_REGEX = [
    (re.compile("\\w+\\s\\d+", flags=re.IGNORECASE), "%B %d"),
    (re.compile("\\w+\\s\\d+", flags=re.IGNORECASE), "%B %d"),
    (re.compile("\\w+\\s\\d{1,2},\\s*\\d{4}", flags=re.IGNORECASE), "%B %d, %Y"),
    (re.compile("\\d{4}\\s\\w+\\s\\d{1,2}", flags=re.IGNORECASE), "%Y %B %d"),
    (re.compile("\\d{1,2}\\s\\w+\\s\\d{4}", flags=re.IGNORECASE), "%d %B %Y"),
    (re.compile("\\d+\\s\\w+", flags=re.IGNORECASE), "%B %d"),
    (re.compile("\\d{1,2}/\\d{1,2}/\\d{4}", flags=re.IGNORECASE), "%d/%m/%Y"),
    (re.compile("\\d{4}-\\d{1,2}-\\d{1,2}", flags=re.IGNORECASE), "%Y-%m-%d") 
]


def _get_date_format(date_value: str, default_format_index: int=6) -> str:
    """
    Returns the exact date format that corresponds to the given date value.

    Args:
        date_value: a string value containing a date.
        default_format_index: an index of the default date format that will be used if, for the 
                              given date value, there is no corresponding data format.

    Returns:
        a string containing the date format.
    """
    
    for pattern, date_format in DATE_FORMATS_REGEX:
        if pattern.match(date_value):
            return date_format
        
    logging.info(f"There is no date format found for the date value of '{date_value}.")
    logging.info(f"Falling back to the default format: {DATE_FORMATS_REGEX[default_format_index]}")

    return DATE_FORMATS_REGEX[default_format_index][1]


def _get_label_desc(label_type: str, label_value: str) -> Tuple[str, str]:
    """
    Returns a short description for the given Spacy's label type and the format of the value.
    The value format used primarily for date values.

    Args:
        label_type: a label type, e.g. ORG, LOC etc.
        label_value: a label value, e.g. 'March 21'
        
    Returns:
        a Tuple containing short label type description and the label value format.
    """

    if label_type.lower() == "date":
        return LABEL_TYPE_DESC_MAPPINGS[label_type], _get_date_format(label_value)
    
    return LABEL_TYPE_DESC_MAPPINGS[label_type], None


def _remove_overlapping_loc_annots(annotations: List[Tuple[int, int, str]]) -> List[Tuple[int, int, str]]:
    """
    Removes LOC annotations that are part of the ADDR annotation because the entire ADDR
    label value will be replaced by Faker's address value instead of replacing individual
    LOC label values.

    Args:
        annotations: List of annotations as (start_idx, end_idx, label_type) tuples

    Returns:
        annotations without 
    """

    loc_annots = [a for a in annotations if a[2] in ["LOC", "POST_CODE"]]
    addr_annots = [a for a in annotations if a[2] == "ADDR"]
    other_annots = [a for a in annotations if a[2] not in ["LOC", "ADDR", "POST_CODE"]]
    
    filtered_annots = []
    for loc_annot in loc_annots:
        overlap = False
        for addr_annot in addr_annots:
            if loc_annot[0] >= addr_annot[0] and loc_annot[1] <= addr_annot[1]:
                overlap = True
                break
        if not overlap:
            filtered_annots.append(loc_annot)
        
    filtered_annots = filtered_annots + addr_annots + other_annots
    filtered_annots = sorted(filtered_annots, key=lambda a: a[0])

    return filtered_annots


def _insert_labels_placeholders(text: str, annotations: List[Tuple[int, int, str]]) -> Tuple[str, Dict[str, List[str]]]:
    """
    Replace labeled spans in text with placeholders and track label values.
    
    Args:
        text: Original text containing labeled spans
        annotations: List of annotations as (start_idx, end_idx, label_type) tuples
        
    Returns:
        Tuple of (text with placeholders, dictionary mapping label types to values)
    """

    new_prompt_parts = []
    labels: Dict[str, List[Tuple[str, str]]] = {}
    start_pos = 0

    filtered_annots = _remove_overlapping_loc_annots(annotations)

    for start_char_idx, end_char_idx, label_type in filtered_annots:
        label_value = text[start_char_idx:end_char_idx]
        label_desc, label_desc_format = _get_label_desc(label_type, label_value)
        new_prompt_parts.append(text[start_pos:start_char_idx])
        
        # Update label tracking and add placeholder
        if label_type not in labels:
            labels[label_type] = []
            
        if label_value in labels[label_type]:
            label_index = labels[label_type].index((label_value, label_desc_format)) + 1
        else:
            labels[label_type].append((label_value, label_desc_format))
            label_index = len(labels[label_type])
            
        new_prompt_parts.append(f"[{label_desc}{label_index}]")
        start_pos = end_char_idx
    
    new_prompt_parts.append(text[start_pos:])
    
    return "".join(new_prompt_parts), labels


def _filter_text(original_text: str, new_sentences: List[str]) -> List[str]:
    """
    Filter generated sentences based on similarity thresholds.
    
    Args:
        original_text: The source text that was paraphrased
        new_sentences: List of generated sentences to filter
        
    Returns:
        List of filtered sentences meeting similarity criteria
    """

    if not new_sentences:
        return []
        
    # First filter: similarity to original text
    low_thresh, high_thresh = ORIGINAL_TEXT_SIMILARITY_THRESH_RANGE
    filtered = [
        sent for sent in new_sentences
        if low_thresh <= get_text_similarity(original_text, sent).item() <= high_thresh
    ]
    
    if len(filtered) <= 1:
        return filtered
    
    similarity_matrix = get_text_similarity(filtered, filtered)
    unique_sentences = []
    
    for i in range(len(filtered)):
        if filtered[i] not in unique_sentences:
            unique_sentences.append(filtered[i])
        
        for j in range(i + 1, len(filtered)):
            if similarity_matrix[i][j] < NEW_TEXT_SIMILARITY_THRESH_BELOW:
                if filtered[j] not in unique_sentences:
                    unique_sentences.append(filtered[j])
    
    return unique_sentences


def _get_fake_value(faker: Faker, label_type: str, label_value_format: str) -> str:
    """
    Generates a fake value for different types of entity descriptions.

    Args:
        faker: an instance of the Faker class.
        label_type: an entity type e.g. ORG, LOC, POST_CODE etc.
        label_value_format: a format of the label value, used primarily for dates.

    Returns:
        a fake value for the given entity description.
    """
    
    if label_type == "LOC":
        return faker.country()
    if label_type == "ORG":
        return faker.company()
    if label_type == "PERSON":
        return faker.name()
    if label_type == "POST_CODE":
        return faker.postalcode()
    if label_type == "ADDR":
        return faker.address().replace("\n", ", ")
    if label_type == "DATE":
        return faker.date(label_value_format)


def _find_entity_positions(sentence: str, labels: Dict[str, List[Tuple[str, str]]]) -> List[Tuple[int, int, str]]:
    """
    Identify all entity placeholder positions in a sentence.
    
    Args:
        sentence: Input sentence containing placeholders
        labels: Entity label mappings
    
    Returns:
        List of (start, end, entity_type) tuples sorted by start position
    """

    positions = []
    
    for entity_type in labels:
        pattern = re.compile(f"\\[{LABEL_TYPE_DESC_MAPPINGS[entity_type]}\\d+(?:[:])?([^\\]]+)?\\]", flags=re.IGNORECASE)
        for idx, match in enumerate(pattern.finditer(sentence)):
            label_value_format = "" if entity_type != "DATE" else labels[entity_type][idx][1]
            positions.append((match.start(), match.end(), entity_type, label_value_format))
    
    return sorted(positions, key=lambda x: x[0])


def _process_address_components(address: str, start_pos: int) -> List[Tuple[int, int, str]]:
    """
    Process address components into separate location entities.
    
    Args:
        address: Full address string
        start_pos: Starting position of the address in the sentence
    
    Returns:
        List of component positions (LOC and POST_CODE)
    """
    components = []
    loc_parts = address.split(",")
    current_pos = start_pos
    
    for idx, part in enumerate(loc_parts):
        # Clean whitespace and get actual content positions
        stripped = part.strip()
        if not stripped:
            current_pos += len(part) + 1  # +1 for comma
            continue
            
        content_start = part.find(stripped[0])
        
        # Add LOC component
        loc_start = current_pos + content_start
        loc_end = loc_start + len(stripped)
        components.append((loc_start, loc_end, "LOC"))
        
        # Add POST_CODE if this is the last part
        if idx == len(loc_parts) - 1:
            post_code = stripped.split()[-1]
            post_start = loc_end - len(post_code)
            components.append((post_start, loc_end, "POST_CODE"))
        
        current_pos += len(part) + 1  # +1 for comma
    
    return components


def _process_sentence(
    sentence: str,
    entity_positions: List[Tuple[int, int, str]],
    faker: Faker
) -> Tuple[str, List[Tuple[int, int, str]]]:
    """
    Process a sentence by replacing placeholders with fake values and tracking new positions.
    
    Args:
        sentence: Original sentence with placeholders
        entity_positions: Sorted list of entity positions
        faker: Faker instance for generating fake values
    
    Returns:
        Tuple of (processed_sentence, new_entity_positions)
    """
    new_sentence = []
    new_entities = []
    offset = 0
    pos = 0
    
    for start, end, label_type, label_value_format in entity_positions:
        # Add text before the current entity
        new_sentence.append(sentence[pos:start])
        
        # Generate fake value and update position tracking
        fake_value = _get_fake_value(faker, label_type, label_value_format)
        new_start = start + offset
        new_end = new_start + len(fake_value)
        
        # Add the fake value
        new_sentence.append(fake_value)
        
        # Track the new entity position
        new_entities.append((new_start, new_end, label_type))
        
        # Handle special cases like addresses
        if label_type == "ADDR":
            new_entities.extend(_process_address_components(fake_value, new_start))
        
        # Update offsets and position
        offset += len(fake_value) - (end - start)
        pos = end
    
    # Add remaining text after last entity
    new_sentence.append(sentence[pos:])
    
    return "".join(new_sentence), new_entities


def _insert_fake_entities(
    sentences: List[str], 
    labels: Dict[str, List[str]]
) -> List[Tuple[str, List[Tuple[int, int, str]]]]:
    """
    Replace placeholders containing entity descriptions with generated fake values.
    Example: [Organization1] → [Some Company Name Inc.], [Person1] → [John Doe], etc.

    Args:
        sentences: List of text sentences containing entity description placeholders to be replaced.
        labels: Dictionary mapping entity descriptions to original values in source text.
    
    Returns:
        Dictionary mapping processed sentences to lists of entity positions (start, end, type).
    """

    faker = Faker(["en_US"])
    processed_sentences = []

    for sentence in sentences:
        # Find all placeholder positions and types
        entity_positions = _find_entity_positions(sentence, labels)
        
        # Process the sentence and generate fake values
        processed_sentence, new_entities = _process_sentence(
            sentence, 
            entity_positions, 
            faker
        )
        
        processed_sentences.append((processed_sentence, new_entities))

    return processed_sentences


def generate_text(
    doc_annots: Tuple[str, List[Tuple[int, int, str]]], 
    max_examples: int
) -> List[Tuple[str, List]]:
    """
    Generate paraphrased versions of text with optional annotation handling.
    
    Args:
        doc_annots: Tuple of (text, annotations) where annotations are
                   (start_idx, end_idx, label_type) tuples
        max_examples: Maximum number of examples to generate
        
    Returns:
        List of tuples containing (paraphrased_text, annotations)
    """

    text, annotations = doc_annots
    paraphraser = Paraphraser(max_sentences=max_examples)
    
    # In case when there are no annotations just run synthetic text generation 
    # without creating entitites placeholders.
    if not annotations:
        new_sentences = paraphraser.rephrase(text)
        return [(sentence, []) for sentence in new_sentences]
    
    labeled_prompt, labels = _insert_labels_placeholders(text, annotations)
    new_sentences = paraphraser.rephrase(labeled_prompt)
    filtered_sentences = _filter_text(labeled_prompt, new_sentences)
    filtered_sentences = _insert_fake_entities(filtered_sentences, labels)

    return filtered_sentences
  