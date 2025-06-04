import tempfile
import argparse
from piipurge import process_document


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--input-file", type=str, required=True)
    parser.add_argument("-o", "--output-dir", type=str, default=tempfile.gettempdir())
    parser.add_argument("-r", "--reconstruct", action=argparse.BooleanOptionalAction)

    args = parser.parse_args()

    process_document(args.input_file, args.output_dir, args.reconstruct)



















