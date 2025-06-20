import os
import sys
from pathlib import Path
from argparse import ArgumentParser

sys.path.append(str(Path(__file__).absolute().parent.parent))

import torch
import tempfile
import streamlit as st
from piipurge.documents_processor import process_document

torch.classes.__path__ = []
results_placeholder = None

st.title("PDF Processor with Piipurge")

parser = ArgumentParser()
parser.add_argument("--output-dir", "-o", default="./output", type=str)
args = parser.parse_args()

# File upload widget
uploaded_file = st.file_uploader(
    "Upload PDF documents", type=["pdf"], accept_multiple_files=False
)

reconstruct_text = st.checkbox(
    label="Reconstruct",
    help="Whether or not to reconstruct the missing text after processing PDF document.",
)

if uploaded_file:
    if st.button("Process Document"):
        if not results_placeholder:
            results_placeholder = st.empty()
        else:
            results_placeholder.empty()

        with st.spinner("Processing..."):
            # Get file bytes from uploaded files
            file_bytes = uploaded_file.getvalue()

            with tempfile.NamedTemporaryFile(
                mode="wb+", suffix=".pdf", delete=False
            ) as temp_file:
                temp_file.write(file_bytes)
                temp_file.flush()
                os.fsync(temp_file.fileno())

                try:
                    # Call your package's functionality
                    process_document(
                        temp_file.name, args.output_dir, reconstruct=reconstruct_text
                    )
                    print("Processing complete!")
                    st.success("Successfully processed the uploaded file.")
                    output_file_path = os.path.join(
                        args.output_dir, os.path.basename(temp_file.name)
                    )
                    with open(output_file_path, "rb") as fh:
                        processed_data = fh.read()
                    results_placeholder.download_button(
                        "Download processed filed",
                        data=processed_data,
                        file_name=os.path.basename(temp_file.name),
                        mime="application/pdf",
                    )
                except Exception as ex:
                    st.error(f"Failed to process the uploaded file<br/>{ex}.")
