import re
import fitz
import spacy
import string
import consts
import numpy as np
from numpy.linalg import norm
from spacy.util import filter_spans
from spacy.language import Language
from spacy.pipeline import EntityRuler, Sentencizer
from sentence_transformers import SentenceTransformer


text = """
Common tasks for an IP address include both the identification of a host or a network, or identifying the location of a device. 
An IP address is not random. The creation of an IP address has the basis of math. 
The Internet Assigned Numbers Authority (IANA) allocates the IP address and its creation. 
The full range of IP addresses can go from 0.0.0.0 to 255.255.255.255. There are also IPv6 addresses supported.
Their range goes from ::2:3:4:5:6:7:8 to ::2:3:4:5:9:9:9. 
You can also access our services using Multi-Protocol Label Switching or MPLS.
If you have any questions please contact our tech support either 
at techsupport@gmail.com or at tel. numbers +1 718 222 2222 or at +33 1 09 75 83 51. For the lastest information
please visit our web site at https://company.com. The web presentation is developed using HTML's code exclusively.
The FakeCompany Ltd. (FCMP) is developing different products. There are many products produces by the company FCMP.
FCMP also offers various services in their domain of business.
"""

PATTERNS = {
    "email": re.compile(
        r"([^\x00-\x20\x22\x28\x29\x2c\x2e\x3a-\x3c\x3e\x40\x5b-\x5d\x7f-\xff]+|"
        + r"\x22([^\x0d\x22\x5c\x80-\xff]|\x5c[\x00-\x7f])*\x22)(\x2e([^\x00-\x20\x22\x28\x29\x2c\x2e\x3a-"
        + r"\x3c\x3e\x40\x5b-\x5d\x7f-\xff]+|\x22([^\x0d\x22\x5c\x80-\xff]|\x5c[\x00-\x7f])*\x22))*\x40("
        + r"[^\x00-\x20\x22\x28\x29\x2c\x2e\x3a-\x3c\x3e\x40\x5b-\x5d\x7f-\xff]+|\x5b([^\x0d\x5b-\x5d\x80"
        + r"-\xff]|\x5c[\x00-\x7f])*\x5d)(\x2e([^\x00-\x20\x22\x28\x29\x2c\x2e\x3a-\x3c\x3e\x40\x5b-\x5d"
        + r"\x7f-\xff]+|\x5b([^\x0d\x5b-\x5d\x80-\xff]|\x5c[\x00-\x7f])*\x5d))*"
    ),
    "ipv4": re.compile(
        r"(\b25[0-5]|\b2[0-4][0-9]|\b[01]?[0-9][0-9]?)(\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)){3}"
    ),
    "ipv6": re.compile(
        r"(([0-9a-fA-F]{1,4}:){7,7}[0-9a-fA-F]{1,4}|([0-9a-fA-F]{1,4}:){1,7}:|"
        + r"([0-9a-fA-F]{1,4}:){1,6}:[0-9a-fA-F]{1,4}|([0-9a-fA-F]{1,4}:){1,5}(:[0-9a-fA-F]{1,4}){1,2}|"
        + r"([0-9a-fA-F]{1,4}:){1,4}(:[0-9a-fA-F]{1,4}){1,3}|([0-9a-fA-F]{1,4}:){1,3}(:[0-9a-fA-F]{1,4}){1,4}|"
        + r"([0-9a-fA-F]{1,4}:){1,2}(:[0-9a-fA-F]{1,4}){1,5}|[0-9a-fA-F]{1,4}:((:[0-9a-fA-F]{1,4}){1,6})|"
        + r":((:[0-9a-fA-F]{1,4}){1,7}|:)|fe80:(:[0-9a-fA-F]{0,4}){0,4}%[0-9a-zA-Z]{1,}|"
        + r"::(ffff(:0{1,4}){0,1}:){0,1}((25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,3}(25[0-5]|"
        + r"(2[0-4]|1{0,1}[0-9]){0,1}[0-9])|([0-9a-fA-F]{1,4}:){1,4}:((25[0-5]|(2[0-4]|"
        + r"1{0,1}[0-9]){0,1}[0-9])\.){3,3}(25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9]))"
    ),
    "phone": re.compile(r"([\+]\d+\s)?[(]?\d+[)]?[-\s\.]?\d+[-\s\.]?[0-9]{4,6}"),
    "ssn": re.compile(
        r"^(?!0{3})(?!6{3})[0-8]\d{2}-(?!0{2})\d{2}-(?!0{4})\d{4}$",
    ),
    "medicare": re.compile(
        r"([a-z]{0,3})[-\s]?(\d{3})[-\s]?(\d{2})[-\s]?(\d{4})[-\s]?([0-9a-z]{1,3})"
    ),
    "vin": re.compile(r"\b[(A-H|J-N|P|R-Z|0-9)]{17}\b"),
    "url": re.compile(
        r"(((?:http[s]?:\/\/.)(?:www\.)?)|((?:ftp[s]?:\/\/.)))[-a-zA-Z0-9@%._\+~#=]{2,256}\.[a-z]{2,6}\b(?:[-a-zA-Z0-9@:%_\+.~#?&\/\/=]*)"
    ),
}

COMMON_ACRONYMS = [
    "IANA",
    "Internet Assigned Numbers Authority",
    "HTML",
    "Hyper Text Markup Language",
    "MPLS",
    "Multi - Protocol Label Switching",
    "PSTN",
    "Public Switched Telecommunications Network",
    "Simple Mail Transfer Protocol",
    "SMTP",
]

ENTITY_DESC = {
    "ORG": ("Company", string),
    "LOC": ("Location", string),
    "PERSON": ("Person", string),
    "IPv4": ("IPv4 Address", int),
    "IPv6": ("IPv6 Address", int),
    "PHONE": ("Phone Number", int),
    "EMAIL": ("Email Address", int),
    "SSN": ("Social Security Number", int),
    "MEDICARE": ("Medicare Number", int),
    "VIN": ("Vehicle Identification Number", int),
    "URL": ("URL Address", int),
}


def find_matching_entities(doc, regex, label):
    matches = regex.finditer(doc.text)
    spans = [
        doc.char_span(match.start(), match.end(), label=label) for match in matches
    ]
    spans = [s for s in spans if s is not None]
    spans = filter_spans(list(doc.ents) + spans)
    doc.ents = spans
    return doc


@Language.component("ipv4")
def ipv4_component(doc):
    doc = find_matching_entities(doc, PATTERNS["ipv4"], "IPv4")
    return doc


@Language.component("ipv6")
def ipv6_component(doc):
    doc = find_matching_entities(doc, PATTERNS["ipv6"], "IPv6")
    return doc


@Language.component("phone")
def phone_component(doc):
    doc = find_matching_entities(doc, PATTERNS["phone"], "PHONE")
    return doc


@Language.component("email")
def email_component(doc):
    doc = find_matching_entities(doc, PATTERNS["email"], "EMAIL")
    return doc


@Language.component("ssn")
def ssn_component(doc):
    doc = find_matching_entities(doc, PATTERNS["ssn"], "SSN")
    return doc


@Language.component("medicare")
def medicare_component(doc):
    doc = find_matching_entities(doc, PATTERNS["medicare"], "MEDICARE")
    return doc


@Language.component("vin")
def vin_component(doc):
    doc = find_matching_entities(doc, PATTERNS["vin"], "VIN")
    return doc


@Language.component("url")
def url_component(doc):
    doc = find_matching_entities(doc, PATTERNS["url"], "URL")
    return doc


@Language.component("custom_sentencizer")
def get_sentencizer(doc):
    sentencizer = Sentencizer(punct_chars=[r"\n"])
    return sentencizer(doc)


def get_spans_similarity(span1, span2, text_encoder):
    span1_embedding = text_encoder.encode(span1)
    span2_embedding = text_encoder.encode(span2)
    similarity = text_encoder.similarity(span1_embedding, span2_embedding)

    return similarity


def check_common_acronyms(
    span_text, common_acronyms, text_encoder, similarity_score_threshold=0.95
):
    max_similarity = 0
    optim_acronym = None
    for acronym in common_acronyms:
        span_similarity_score = get_spans_similarity(acronym, span_text, text_encoder)
        if (span_similarity_score > max_similarity) and (
            span_similarity_score > similarity_score_threshold
        ):
            optim_acronym = acronym
            max_similarity = span_similarity_score

    return optim_acronym


def get_most_similar_text(
    texts_to_compare, texts, text_encoder, similarity_score_threshold=0.95
):
    most_similar_text = None

    if (texts_to_compare is not None) and (texts is not None):
        curr_max_similarity = 0
        for text1 in texts_to_compare:
            for text2 in texts:
                if isinstance(text2, tuple):
                    text_similarity_score = max(
                        get_spans_similarity(text2[0], text1, text_encoder),
                        get_spans_similarity(text2[1], text1, text_encoder),
                    )
                else:
                    text_similarity_score = get_spans_similarity(
                        text2, text1, text_encoder
                    )
                if (text_similarity_score > curr_max_similarity) and (
                    text_similarity_score > similarity_score_threshold
                ):
                    most_similar_text = text2
                    curr_max_similarity = text_similarity_score

    return most_similar_text


def get_ent_replacement(ent, suffix_type, entities_count):
    if suffix_type is string:
        return (
            f'"{ENTITY_DESC[ent.label_][0]} {string.ascii_uppercase[entities_count]}"',
            ent.sent,
            ent.start_char,
            ent.end_char,
        )
    elif suffix_type is int:
        return (
            f'"{ENTITY_DESC[ent.label_][0]} {entities_count+1}"',
            ent.sent,
            ent.start_char,
            ent.end_char,
        )
    else:
        raise Exception(f"Unsupported suffix type '{suffix_type}")


def get_ent_replacement2(ent, suffix_type, entities_count):
    if suffix_type is string:
        suffix = (
            int(round(entities_count / len(string.ascii_uppercase))) * "A"
            + string.ascii_uppercase[entities_count % len(string.ascii_uppercase)]
        )
        return (
            f'"{ENTITY_DESC[ent.label_][0]} {suffix}"',
            get_matching_line(ent.sent.text, ent.text),
        )
    elif suffix_type is int:
        return (
            f'"{ENTITY_DESC[ent.label_][0]} {entities_count+1}"',
            get_matching_line(ent.sent.text, ent.text),
        )
    else:
        raise Exception(f"Unsupported suffix type '{suffix_type}")


def get_closest_ent_name(texts, ent_cat_subs, relations, text_encoder):
    relation = get_most_similar_text(texts, relations, text_encoder)
    closest_key_name = None
    if relation:
        closest_key_name = (
            relation[0]
            if relation[0] in ent_cat_subs
            else relation[1] if relation[1] in ent_cat_subs else None
        )
        if not closest_key_name:
            closest_key_name = get_most_similar_text(
                texts, list(ent_cat_subs.keys()), text_encoder
            )
    return closest_key_name


def get_matching_line(text, pattern, linesep="\n"):
    lines = text.split(linesep)

    for l in lines:
        if pattern in l:
            # Strip the whitespace characters because these
            # characters are stripped from the pdf lines as well.
            return l.strip()

    return None


def get_entities_subs2(ents, relations, common_acronyms, text_encoder):
    subs = {}

    for ent in ents:
        if ent.label_ not in ENTITY_DESC:
            continue
        if ent.label_ == "ORG":
            acronym = check_common_acronyms(ent.text, common_acronyms, text_encoder)
            if acronym:
                continue
        ent_cat_subs = subs.get(ent.label_, {})
        ent_sub = ent_cat_subs.get(ent.text)
        if ent_sub is None:
            if ent.label_ == "ORG":
                relation = get_most_similar_text([ent.text], relations, text_encoder)
                if relation:
                    closest_key_name = get_closest_ent_name(
                        [ent.text, *relation], ent_cat_subs, relations, text_encoder
                    )
                    if closest_key_name:
                        ent_cat_subs[ent.text] = [
                            (
                                ent_cat_subs[closest_key_name][0][0],
                                get_matching_line(ent.sent.text, ent.text),
                            )
                        ]
                        continue
            suffix_type = ENTITY_DESC[ent.label_][1]
            ent_cat_subs[ent.text] = [
                get_ent_replacement2(ent, suffix_type, len(ent_cat_subs))
            ]
        else:
            ent_sub.append((ent_sub[-1][0], get_matching_line(ent.sent.text, ent.text)))
        subs[ent.label_] = ent_cat_subs

    return subs


def get_lines_subs(subs):
    lines_subs = {}

    for ent_label in subs:
        for old_val, occurences_list in subs[ent_label].items():
            for new_val, sent_text in occurences_list:
                lines = lines_subs.get(sent_text, [])
                new_item = (ent_label, old_val, new_val)
                if new_item not in lines:
                    lines.append(new_item)
                    lines_subs[sent_text] = lines

    return lines_subs


def get_entities_subs(ents, relations, common_acronyms, text_encoder):
    subs = {}

    for ent in ents:
        if ent.label_ not in ENTITY_DESC:
            continue
        if ent.label_ == "ORG":
            acronym = check_common_acronyms(ent.text, common_acronyms, text_encoder)
            if acronym:
                continue
        ent_cat_subs = subs.get(ent.label_, {})
        ent_sub = ent_cat_subs.get(ent.text)
        if ent_sub is None:
            if ent.label_ == "ORG":
                relation = get_most_similar_text([ent.text], relations, text_encoder)
                if relation:
                    closest_key_name = get_closest_ent_name(
                        [ent.text, *relation], ent_cat_subs, relations, text_encoder
                    )
                    if closest_key_name:
                        ent_cat_subs[ent.text] = [
                            (
                                ent_cat_subs[closest_key_name][0][0],
                                ent.sent,
                                ent.start_char,
                                ent.end_char,
                            )
                        ]
                        continue
            suffix_type = ENTITY_DESC[ent.label_][1]
            ent_cat_subs[ent.text] = [
                get_ent_replacement(ent, suffix_type, len(ent_cat_subs))
            ]
        else:
            ent_sub.append((ent_sub[-1][0], ent.sent, ent.start_char, ent.end_char))
        subs[ent.label_] = ent_cat_subs

    return subs


def is_acronym(short_form, long_form_entities):
    long_acronym = "".join(ent.text[0].upper() for ent in long_form_entities)
    long_acronym = re.sub(r"[^\w]", "", long_acronym)
    return short_form.upper() == long_acronym


def find_short_long_relations(acronyms_ents):
    if len(acronyms_ents) == 0:
        return
    long_form = []
    long_form_range = (0, 0)
    relations = []

    for ent in acronyms_ents:
        tag, _, start, end = ent.label_, ent.text, ent.start, ent.end
        if (
            tag.startswith("B-long") or tag.startswith("I-long")
        ) and start - long_form_range[1] > 2:
            long_form = [ent]
            long_form_range = (start, end)
        elif (tag.startswith("I-long")) or (
            (tag.startswith("B-long") and (start - long_form_range[1]) <= 2)
        ):
            long_form.append(ent)
            long_form_range = (long_form_range[0], end)
        elif tag.startswith("B-short"):
            if long_form and (start - long_form_range[1]) <= 2:
                if is_acronym(ent.text, long_form) or (
                    (start - long_form_range[1] <= 2)
                    and (long_form[-1].sent == ent.sent)
                ):
                    relations.append((" ".join([e.text for e in long_form]), ent.text))
                    long_form_range = (0, 0)

    return relations


def substitute_entities(text, subs, min_distance=3):
    orgs_to_replace = []
    new_text = text
    diff = 0

    for ent_type in subs:
        for old_val, occurences_list in subs[ent_type].items():
            for new_val, sent, start_char, end_char in occurences_list:
                if ent_type == "ORG":
                    if orgs_to_replace:
                        (
                            last_org_name,
                            last_org_sent,
                            last_org_start_char,
                            last_org_end_char,
                        ) = orgs_to_replace[-1]
                        if (
                            (start_char - last_org_end_char <= min_distance)
                            and (last_org_name == new_val)
                            and (last_org_sent == sent)
                        ):
                            if text[end_char] == ")":
                                end_char += 1
                            orgs_to_replace[-1] = (
                                last_org_name,
                                last_org_sent,
                                last_org_start_char,
                                end_char,
                            )
                            continue
                    orgs_to_replace.append((new_val, sent, start_char, end_char))
                else:
                    new_text = (
                        new_text[: start_char + diff]
                        + new_val
                        + new_text[end_char + diff :]
                    )
                    diff += len(new_val) - (end_char - start_char)

    for org in orgs_to_replace:
        new_org_name, _, org_start_char, org_end_char = org
        new_text = (
            new_text[: org_start_char + diff]
            + new_org_name
            + new_text[org_end_char + diff :]
        )
        diff += len(new_org_name) - (org_end_char - org_start_char)

    return new_text


def replace_text(pdf, pdf_lines, old_val, new_val):
    for pdf_line in pdf_lines:
        # Temporarly lowercase the pdf line text.
        new_sent = pdf_line["text"].lower().replace(old_val, new_val)
        x, y = pdf_line["origin"]
        xb0, yb0, xb1, yb1 = pdf_line["bbox"]
        rect = fitz.Rect(xb0, yb0, xb1, yb1)
        page = pdf[pdf_line["page_number"]]
        page.draw_rect(rect, color=(1, 1, 1), fill=(1, 1, 1))
        page.insert_text(
            (x, y),
            new_sent,
            fontsize=pdf_line["size"],
            fontname=pdf_line["font"],
            color=pdf_line["color"],
            fill_opacity=pdf_line["alpha"],
        )


def substitute_page_entities(pdf, pdf_spans, lines_subs, min_distance=3):
    orgs_to_replace = []

    for line in lines_subs:
        for ent_label, old_val, new_val in lines_subs[line]:
            if ent_label == "ORG":
                if orgs_to_replace:
                    last_org_old_name, last_org_new_name, last_org_sent = (
                        orgs_to_replace[-1]
                    )
                    last_org_old_name_end = last_org_sent.index(
                        last_org_old_name
                    ) + len(last_org_old_name)
                    last_org_new_name_start = last_org_sent.index(last_org_new_name)
                    if (
                        last_org_new_name_start - last_org_old_name_end <= min_distance
                    ) and (last_org_old_name == last_org_new_name):
                        last_org_new_name_end = last_org_new_name_start + len(
                            new_org_name
                        )
                        if last_org_sent[last_org_old_name_end] == ")":
                            last_org_new_name_end += 1
                        orgs_to_replace[-1] = (
                            last_org_old_name,
                            last_org_sent[
                                last_org_new_name_start:last_org_old_name_end
                            ],
                            last_org_sent,
                        )
                        continue
                orgs_to_replace.append((old_val, new_val, line))
            else:
                line_text_orig = re.sub(r"\s.\s?$", "", line)
                old_val_orig = re.sub(r"\s.\s?$", "", old_val)
                replace_text(pdf, pdf_spans[line_text_orig], old_val, new_val)

    for org in orgs_to_replace:
        old_org_name, new_org_name, sent_text = org
        sent_text_orig = re.sub(r"\s.\s?$", "", sent_text)
        old_org_name_orig = re.sub(r"\s.\s?$", "", old_org_name)
        replace_text(pdf, pdf_spans[sent_text_orig], old_org_name_orig, new_org_name)


def run_nlp(text):
    nlp = spacy.load("en_core_web_trf")
    nlp_acronyms = spacy.load(consts.ACRONYMS_MODEL_DIR)

    components = ["ipv4", "ipv6", "phone", "email", "ssn", "medicare", "vin", "url"]

    ruler = nlp.add_pipe("entity_ruler", before="ner")
    patterns = [
        {"label": "DATE", "pattern": [{"TEXT": {"REGEX": r"\d{1,2}/\d{1,2}/\d{4}"}}]}
    ]
    ruler.add_patterns(patterns)

    # nlp.add_pipe("custom_sentencizer", before="parser")
    nlp.add_pipe("sentencizer")
    nlp_acronyms.add_pipe("sentencizer")
    for c in components:
        nlp.add_pipe(c, last=True)

    doc = nlp(text.lower())
    acronyms_doc = nlp_acronyms(text)

    relations = find_short_long_relations(acronyms_doc.ents)

    for ent in doc.ents:
        print(ent.text, ent.label_)

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    subs = get_entities_subs2(doc.ents, relations, COMMON_ACRONYMS, model)

    return subs


def get_pdf_text(pdf):
    raw_text = ""

    for page in pdf:
        raw_text += page.get_text("text") + "\n"

    return raw_text


def get_pdf_spans(pdf):
    pdf_spans = {}

    for page in pdf:
        spans = get_pdf_page_spans(page)
        for span, occurences in spans.items():
            if span in pdf_spans:
                pdf_spans[span].extend(occurences)
            else:
                pdf_spans[span] = occurences

    return pdf_spans


def extract_pdf_text(pdf_spans, line_gap_threshold=5):
    all_paragraphs = []
    sorted_lines = []

    pdf_spans = get_pdf_spans(pdf)
    for line, infos in pdf_spans.items():
        for line_info in infos:
            temp_line_info = line_info.copy()
            temp_line_info["normalized_text"] = line
            sorted_lines.append(temp_line_info)

    sorted_lines = list(
        sorted(sorted_lines, key=lambda l: (l["page_number"], l["spans_bbox"][-1]))
    )

    paragraphs = []
    current_paragraph = []
    last_y = None

    for line in sorted_lines:
        x0, y0, x1, y1 = line["spans_bbox"]

        if last_y is not None and (y0 - last_y) > line_gap_threshold:
            if current_paragraph:
                paragraphs.append(current_paragraph)
            current_paragraph = []

        current_paragraph.append(
            {
                "text": line["text"],
                "bbox": line["bbox"],
                "spans_bbox": (x0, y0, x1, y1),
                "page_number": line["page_number"],
            }
        )
        last_y = y1

    if current_paragraph:
        paragraphs.append(current_paragraph)

    for paragraph in paragraphs:
        paragraph_text = " ".join([line["text"] for line in paragraph])
        all_paragraphs.append(
            {
                "text": paragraph_text,
                "lines": paragraph,
                "page_number": paragraph[0]["page_number"],
            }
        )

    return all_paragraphs


def get_pdf_span_boundaries(span):
    a = span["ascender"]
    d = span["descender"]
    r = fitz.Rect(span["bbox"])
    o = fitz.Point(span["origin"])
    r.y1 = o.y - span["size"] * d / (a - d)
    r.y0 = r.y1 - span["size"]

    return r


def get_pdf_page_spans(page):
    pdf_spans = {}
    blocks = page.get_text("dict")["blocks"]

    for block_idx, block in enumerate(blocks):
        is_text_block = block["type"] == 0
        if is_text_block:
            for line_idx, line in enumerate(block["lines"]):
                line_text = ""
                spans = []
                for span in line["spans"]:
                    span_text = span["text"]
                    if span_text.strip() != "":
                        spans.append(
                            {
                                "page_number": page.number,
                                "block_index": block_idx,
                                "line_index": line_idx,
                                "size": span["size"],
                                "font": "helv",  # Temporarly use the 'helv' font instead of the span["font"],
                                "color": span["color"],
                                "alpha": span["alpha"],
                                "origin": span["origin"],
                                "bbox": span["bbox"],
                                "span_bbox": get_pdf_span_boundaries(span),
                                # Use original text instead of the whitespaces the stripped text
                                # will be used to match tet of sentences returned by the spaCy model.
                                "text": span_text,
                            }
                        )
                        line_text += span_text
                if line_text != "":
                    line_span = spans[0].copy()
                    line_span["bbox"] = (
                        spans[0]["bbox"][0],
                        spans[0]["bbox"][1],
                        spans[-1]["bbox"][-2],
                        spans[-1]["bbox"][-1],
                    )
                    spans_max_y = max([s["span_bbox"][-1] for s in spans])
                    line_span["spans_bbox"] = (
                        spans[0]["span_bbox"][0],
                        spans[0]["span_bbox"][1],
                        spans[-1]["span_bbox"][-2],
                        spans_max_y,
                    )
                    line_span["text"] = line_text
                    key_name = line_text.strip().lower()
                    occurences = pdf_spans.get(key_name, [])
                    occurences.extend([line_span])
                    pdf_spans[key_name] = occurences

    return pdf_spans


if __name__ == "__main__":
    """
    nlp = spacy.load("en_core_web_trf")
    doc = nlp((r"Name: DOE, JOHN  \nAccession ID: PM-XX-12345 \nDOB: 12/31/1999 \nMRN: 0123456789 \n"
        "Specimen: Blood, Peripheral \nSex: Male \nReferring facility: Double Helix Hospital   \n"
        "Lab Control Number: ABC123 \nRace/Ethnicity: White                           Referring physician: Dr. DNA "))
    ents = doc.ents
    print(doc.ents)
    """
    # main(text)
    pdf = fitz.open(consts.PDFS_FOLDER / "health_report.pdf")
    paragraphs = extract_pdf_text(pdf)
    print(len(paragraphs))
    """
    pdf = fitz.open(consts.PDFS_FOLDER / "bank_form.pdf")
    pdf_spans = get_pdf_spans(pdf)
    #raw_text = get_pdf_text(pdf)
    raw_text = r" . ".join(list(pdf_spans.keys()))
    #raw_text = r" \n ".join(list(pdf_spans.keys()))
    
    print(raw_text)
    subs = run_nlp(raw_text)
    lines_subs = get_lines_subs(subs)
    substitute_page_entities(pdf, pdf_spans, lines_subs)
    
    pdf.save("./test.pdf")
    pdf.close()
    """
