import re
import os
import torch
from pathlib import Path

ROOT_FOLDER = Path(os.path.abspath(__file__)).parent.parent.parent
DATA_FOLDER = ROOT_FOLDER.joinpath("data")
MODELS_FOLDER = ROOT_FOLDER.joinpath("models")
OUTPUT_FOLDER = ROOT_FOLDER.joinpath("output")
ACRONYMS_TRAINING_FOLDER = Path(DATA_FOLDER).joinpath("acronyms_training_data")
ACRONYMS_MODEL_DIR = Path(MODELS_FOLDER).joinpath("acronyms_ner", "model-best")
SPANCAT_NER_TRAINING_FOLDER = Path(OUTPUT_FOLDER).joinpath("spancat_ner_training")
PDFS_FOLDER = DATA_FOLDER.joinpath("pdfs")
HANDWRITTEN_SIGNATURES_MODEL_PATH = Path(MODELS_FOLDER).joinpath(
    "handwrittings_classification_model.joblib"
)
REPLACEMENT_FONTS_PATH = DATA_FOLDER.joinpath("font_replacements.csv")

_ = [
    d.mkdir(parents=True, exist_ok=True)
    for d in [
        DATA_FOLDER,
        OUTPUT_FOLDER,
        ACRONYMS_TRAINING_FOLDER,
        SPANCAT_NER_TRAINING_FOLDER,
    ]
]

HANDWRITTEN_SIGNATURES_MEAN = torch.tensor([219.8001, 219.8001, 219.8001])
HANDWRITTEN_SIGNATURES_STD = torch.tensor([73.0633, 73.0633, 73.0633])


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
        r"(((?:http[s]?:\/\/.)?(?:www\.)?)|((?:ftp[s]?:\/\/.)))[-a-zA-Z0-9@%._\+~#=]{2,256}\.[a-z]{2,6}\b(?:[-a-zA-Z0-9@:%_\+.~#?&\/\/=]*)"
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
    "ORG": ("Company", str),
    "LOC": ("Location", str),
    "PERSON": ("Person", str),
    "IPv4": ("IPv4 Address", int),
    "IPv6": ("IPv6 Address", int),
    "PHONE": ("Phone Number", int),
    "EMAIL": ("Email Address", int),
    "SSN": ("Social Security Number", int),
    "MEDICARE": ("Medicare Number", int),
    "VIN": ("Vehicle Identification Number", int),
    "URL": ("URL Address", int),
}

LOG_FORMAT = "%(levelname)s-%(asctime)s: %(message)s"
