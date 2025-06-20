![Project Status](https://img.shields.io/badge/status-active-brightgreen)

## 1. The project description
The main goal of the project is to demonstrate how Spacy, Tesseract OCR, TrOCR and PyMuPDF could be used to identify
and redact PII information in PDF documents.

## 2. The project structure

* [config](./config) - Spacy configuration files
  * [acronym_identification_gpu.cfg](./config/acronym_identification_gpu.cfg) - Spacy's configuration for training the acronym identification model on GPU instance.
  * [spancat_training_gpu.cfg](./config/spancat_training_gpu.cfg) - Spacy's configuration for training the custom model for 
  spans categorization
* [data](./data) - PDF documents and Spacy's model training data
  * [acronyms_training](./data/acronyms_training) - .spacy files used for training the acronyms identification model
    * [dev.spacy](./data/acronyms_training/dev.spacy) - serialized Doc objects used for the model performance validation
    * [train.spacy](./data/acronyms_training/train.spacy) - serialized Doc objects used for the model training
    * [test.spacy](./data/acronyms_training/test.spacy) - serialized Doc objects used for the model final evaluation
    > The original data comes from the **amirveyseh/acronym_identification** dataset that is available in the Hugging Face platform.
  * [pdfs](./data/pdfs) - sample PDF documents used for testing the solution
* [models](./models) - serialized ML models
  * [acronyms_ner/model-best](./models/acronyms_ner/model-best) - serialized NLP object representing the acronyms identification model 
  * [handwrittings_classification_model.joblib](./models/handwrittings_classification_model.joblib) - serialized model for handwritten/printed signatures classification
* [notebooks](./notebooks) - Jupyter notebooks
  * [acronym_identification.ipynb](./notebooks/acronym_identification.ipynb) - notebook for training the acronyms identification model

    > When evaluating the acronyms identification model on the test subset, the precision, recall and the F-Score metrics will most
  probably all be equal to zero. The reason for this is that all the tokens in the test subset are labeled with the "4" value which
  corresponds to the "O" tag used in the BIO tagging format. The "O" tag refers to the token that is outside of tokens sequence related to an acronym definition. 
  * [handwritten_signature_classification.ipynb](./notebooks/handwritten_signature_classification.ipynb)
* [output](./output) - the output folder where results, of running the PII redaction tool, are stored
* [src](./src) - the app code and utility scripts
  * [piipurge](./src/piipurge) - main package folder
    * [utils](./src/piipurge/utils) - utility scripts
      * [cv_metrics.py](./src/piipurge/utils/cv_metrics.py) - evaluation metrics for computer vision
      * [drawing.py](./src/piipurge/utils/drawing_utils.py) - graphics utility functions
      * [fonts.py](./src/piipurge/utils/fonts.py) - fonts utility functions
      * [nlp_metrics.py](./src/piipurge/utils/nlp_metrics.py) - evaluation metrics for NLP
      * [paraphraser.py](./src/piipurge/utils/paraphraser.py) - used for rephrasing sentences
      * [synth_text_generator.py](./src/piipurge/utils/synth_text_generator.py) - synthetic text generator
      * [uima_cas_to_spacy.py](./src/piipurge/utils/uima_cas_to_spacy.py) - converts UIMA CAS, JSON-based serialization format, annotations, exported using INCEpTION semantic annotation platform, into a binary .spacy format.
    * [analyze_images.py](./src/piipurge/analyze_images.py) - scans images for handwritten and printed signatures and analyzes them using 
  the TrOCR CV model and the Tesseract OCR engine.
    * [consts.py](./src/piipurge/consts.py) - shared constant values used by various scripts
    * [document_processor](./src/piipurge/document_processor.py) - utility functions for images preprocessing before signature recognition analysis.
    * [schemas.py](./src/piipurge/schemas.py) - contains custom data structures definitions.
  * [train](./src/train) - folder containing NLP and other ML models training scripts
    * [acronym_ner_training.py](./src/train/acronym_ner_training.py) - converts acronym identification dataset "amirveyseh/acronym_identification" into serialized collection of Doc objects and splits it into train, validation and the test subsets.
    * [create_spacy_spancat_dataset.py](./src/piipurge/create_spacy_spancat_dataset.py) - splits a .spacy binary file into a train, validation and a test subsets and stores them as separate files.
    * [train_spancat](./src/piipurge/train_spancat.py) - a custom script for training a span categorizer model.
  * [run_document_processing.py](./src/run_document_processing.py) - main script used for running the document processing.
* [tests](./src/tests) - unit tests
* [app.Dockerfile](./app.Dockerfile) - instructions file for building a Docker image for a Streamlit app.

## 3. Architecture

🚧 _This section will be defined in a future update._

## 4. Prerequisities

Tesseract-OCR 5.5.0 is required to be installed on the system running the code. In order for Tesseract-OCR commands
to be available inside terminal Tesseract-OCR installation folder needs to be added to the **PATH** environment variable.

## 4. Data annotations

🚧 _This section will be defined in a future update._

## 5. Synthetic data generation

For the synthetic text data generation, the script **src/piipurge/utils/synth_text_generator.py** is primarily used.
It replaces a specific placeholder for entities description with the fakes values generated using the Faker package.

For an example, in the following input text:
"ESCROW SERVICES" means services for auction sellers and auction bidders whereby an agent holds a buyer's money in trust until the buyer approves the applicable item that was physically delivered, at which time the agent releases the buyer's money to seller, after subtracting the escrow fees.
, the "ESCROW SERVICES" and the "escrow" strings will be replaced with the appropriate entity description, in this case with
the **"[Organization1]"** and the **"[Organization2]"** strings, where the square brackets "[]" mark a placeholder string.

In the background, the **synth_text_generator.py** uses the **src/piipurge/utils/paraphraser.py** script to reformulate the 
input text before inserting fake values into the entity descriptions placeholders.
The script **paraphraser.py**, by default, uses the **Vamsi/T5_Paraphrase_Paws** model to rephrase the input text. It performs 
better when the entity names placeholders contain more descriptive name for the entity instead of the shorter version used 
for some of the Spacy's entity types such as the ORG or the LOC entity types.

## 6. Fine-tunning the spancat model

🚧 _This section will be defined in a future update._

## Issues

The function **_find_entity_boundaries** finds matching string using test similarity because of it and because the search is performed on pre-normalized text, the search results will be suboptimal. In some cases some of the characters will be left out from the search results.
One of the alternate solutions would be to search for the string by comparing n-grams.

## Running the code

Before running the code the required prerequisites need to be installed. 
This can be done by running the following command:
```bash
pip install -r requirements.txt
```

The main script for running the program is the **run_document_processing.py**(./src/piipurge/run_document_processing.py).
The script can be run directly from the command line by run the command:
```bash
python src/run_document_processing.py -i PATH_TO_THE_INPUT_FILE -o PATH_TO_THE_OUTPUT_FOLDER [-r]
```
> Note: The **-r** program argument is optional and it indicates whether or not the text reconstruction will be performed
at the end of the document processing. The text reconstruction is, by default, excluded.

The program can also be run using Docker container. The Docker image **app.Dockerfile**(./src/app.Dockerfile) contains
instructions for build the Docker image containing the program code and a simple frontend app.
To build the Docker image run the following command from within the root project folder:
```bash
docker build -t piipurge:latest -f app.Dockerfile .
```
After the Docker image is built, the container can be created via command:
```bash
docker run -d -p 8501:8501 -v ~/.cache/huggingface:/tmp --name test piipurge:latest
```
> Note: In the example above the folder path **~/.cache/huggingface**, on a host machine, is mapped to the **/tmp** folder
in the container in order to avoid unnecessary download of ML models from the Huggingface Hub inside the container instance.
This is recommended if the ML models are already downloaded on the host machine otherwise the folder mapping instruction can be
removed from the command. 
