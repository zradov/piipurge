import os
import fitz
import spacy
from piipurge import consts
from piipurge.documents_processor import get_paragraphs, add_utf8_normalization


def get_files(dir_path):
    files = []

    for entry in os.scandir(dir_path):
        if entry.is_dir(follow_symlinks=False):
            files.extend(get_files(os.path.join(dir_path, entry.name)))
        if os.path.splitext(entry.name.lower())[-1] == ".pdf":
            files.append(entry.path)

    return files


if __name__ == "__main__":
    source_folder = "C:\\Users\\zoki\\Downloads\\CUAD_v1\\full_contract_pdf"
    exclude_files = [
        "2ThemartComInc_19990826_10-12G_EX-10.10_6700288_EX-10.10_Co-Branding Agreement_ Agency Agreement.pdf".lower(),
        "ABILITYINC_06_15_2020-EX-4.25-SERVICES AGREEMENT.pdf".lower(),
        "ACCELERATEDTECHNOLOGIESHOLDINGCORP_04_24_2003-EX-10.13-JOINT VENTURE AGREEMENT.pdf".lower(),
        "ACCURAYINC_09_01_2010-EX-10.31-DISTRIBUTOR AGREEMENT.pdf".lower(),
        "ADAMSGOLFINC_03_21_2005-EX-10.17-ENDORSEMENT AGREEMENT.pdf".lower(),
        "ADAPTIMMUNETHERAPEUTICSPLC_04_06_2017-EX-10.11-STRATEGIC ALLIANCE AGREEMENT.pdf.lower()",
        "ADUROBIOTECH,INC_06_02_2020-EX-10.7-CONSULTING AGREEMENT.pdf".lower(),
        "AIRSPANNETWORKSINC_04_11_2000-EX-10.5-Distributor Agreement.pdf".lower(),
    ]
    keywords = [" St.", "Street ", " located at ", "Suite "]
    output_file_path = consts.OUTPUT_FOLDER / "extracted_entities2.txt"
    nlp = spacy.load("en_core_web_sm")
    _ = [nlp.remove_pipe(c) for c in nlp.component_names]
    nlp.add_pipe("sentencizer")
    add_utf8_normalization(nlp)
    files = get_files(source_folder)
    if files:
        with open(output_file_path, mode="w", encoding="utf8") as fh:
            for file_path in files:
                print(f"Processing file {file_path} ...")
                if os.path.basename(file_path).lower() not in exclude_files:
                    pdf = fitz.open(file_path)
                    for paragraph in get_paragraphs(pdf):
                        doc = nlp(paragraph["text"])
                        for sent in doc.sents:
                            for keyword in keywords:
                                if keyword in doc.text:
                                    fh.write(sent.text)
                                    fh.write(os.linesep)
                                    break
