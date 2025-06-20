import torch
import joblib
import torchvision
import pytesseract
from . import consts
from PIL import Image
from typing import List, Dict, Tuple
from torchvision.transforms import v2
from torch.nn import Sequential, Linear
from .utils.nlp_metrics import get_text_similarity
from .schemas import ImageTextInfo, SavedImageInfo
from .images_preprocessing import clear_background
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


MODEL_NAME = "microsoft/trocr-large-handwritten"


def analyze_print(saved_images_info: List[SavedImageInfo]) -> Dict[str, ImageTextInfo]:
    """
    Analyzes images for printed text and returns metadata about the
    printed text if such text is found in the images.

    Args:
        img_paths: a list of absolute paths to images on a local file system.

    Returns:
        a dictionary where the key refers to an absolute path to a saved image on a
        local file system, while values refer to the text metadata found in the images.
    """

    processor = _get_print_processor(MODEL_NAME)
    model = _get_vision_model(MODEL_NAME, processor)
    images_text = {}

    for img_info in saved_images_info:
        image, ocr_data = _run_ocr(img_info.img_path)
        text_bboxes = _get_text_bboxes(image, ocr_data, model, processor)
        text_lines = _get_text_lines(text_bboxes)
        images_text[img_info.img_path] = ImageTextInfo(
            image_size=image.size, text_lines=text_lines
        )

    return images_text


def analyze_handwriting(img_paths: List[str]) -> List[Tuple[str, bool]]:
    """
    Runs a prediction, using handwriting recognition model, on a given list
    of image paths and returns prediction results.

    Args:
        img_paths: a list of absolute paths to images on a local file system.

    Returns:
        a list of tuples where each tuple contains a pair of an absolute path
        to an image, on a local file system, and a boolean value representing
        whether the image contains handwrittings in it or not.
    """

    transform = _get_image_transformations(
        consts.HANDWRITTEN_SIGNATURES_MEAN, consts.HANDWRITTEN_SIGNATURES_STD
    )
    results = []
    model = joblib.load(consts.HANDWRITTEN_SIGNATURES_MODEL_PATH)

    for img_path in img_paths:
        clear_background(img_path)
        img = Image.open(img_path)
        img_tensor = transform(img)
        batch = img_tensor.unsqueeze(0)
        features = _extract_features(batch).cpu().numpy()
        y_pred = model.predict(features)
        results.append((img_path, bool(y_pred)))

    return results


def _extract_features(img):
    vgg16 = torchvision.models.vgg16()
    feature_extractor = Sequential(*list(vgg16.children())[:-1])
    projection_layer = Linear(512 * 7 * 7, 512)

    with torch.no_grad():
        features = feature_extractor(img)
        features = torch.flatten(features, start_dim=1)
        features = projection_layer(features)

        return features


def _get_image_transformations(mean, std):
    """
    Returns the transformation pipeline used for transforming images prior to
    running a prediction using a handwriting recognition model.
    """
    transform = v2.Compose(
        [
            v2.ToImage(),
            v2.Resize((224, 224)),
            v2.ToDtype(torch.float32, scale=False),
            v2.Normalize(mean, std),
        ]
    )

    return transform


def _get_print_processor(model_name: str) -> object:
    """
    Loads and returns a wrapper for vision image processor and a TrOCR tokenizer.

    Args:
        model_name: a name of the processor model.

    Returns:
        a single processor object wrapping vision image processor and a TrOCR tokenizer.
    """

    processor = TrOCRProcessor.from_pretrained(model_name)
    return processor


def _get_vision_model(model_name: str, processor, num_return_sequences: int = 3):
    """
    Loads and returns an instance of the pretrained TrOCR decoder model.

    Args:
        model_name: a name of the pretrained TrOCR decoder model.
        processor: a model used for preprocessing the input image and decoding
                  the generated target tokens to the target string.
        num_return_sequences: how many sequences of text to return from the model
                              prediction along with their confidence score.

    Returns:
        an instance of the vision encoder decoder model that accepts images as input and
        makes use of generate() to autoregressively generate text given the input image.
    """

    model = VisionEncoderDecoderModel.from_pretrained(model_name)

    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.config.max_length = 50
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3
    model.config.length_penalty = 2.0
    model.config.num_beams = 4
    model.config.num_return_sequences = num_return_sequences

    return model


def _analyze_print_from_image(image, model, processor):
    # input = image_processor(image, input_data_format="channels_last")
    pixel_values = processor(
        image, return_tensors="pt", input_data_format="channels_last"
    ).pixel_values
    generate_result = model.generate(
        pixel_values,
        output_scores=True,
        # num_return_sequences=num_return_sequences,
        return_dict_in_generate=True,
    )
    ids, scores = generate_result["sequences"], generate_result["sequences_scores"]
    generated_text = processor.batch_decode(ids, skip_special_tokens=True)

    return generated_text, scores


def _are_textboxes_adjacent(text_box: List[int], text_boxes: List[List[int]]) -> bool:
    """
    Check whether the specified text box is adjacent to any of the text boxes
    in the provided list of text boxes.

    Args:
        text_box: a text box for which we need to check whether it is adjacent to any of the other text boxes or not.

    Returns:
        True if the text box is adjacent to any of the provided text boxes otherwise False.
    """

    for tb in text_boxes:
        if (abs(tb[2] - text_box[0]) < 10 and abs(tb[1] - text_box[1]) < 5) or (
            abs(tb[0] - text_box[0]) < 5 and abs(tb[3] - text_box[1]) < 10
        ):
            return True

    return False


def _run_ocr(img_path):
    image = Image.open(img_path).convert("RGB")
    ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)

    return image, ocr_data


def _get_text_bboxes(
    image,
    ocr_data,
    vision_model,
    image_processor,
    include_empty_text=True,
    score_threshold=-0.05,
    text_similarity_threshold=0.95,
):
    text_bboxes = []

    for idx, ocr_text in enumerate(ocr_data["text"]):
        if ocr_text.strip() != "" or include_empty_text:
            text_area = [
                ocr_data["left"][idx],
                ocr_data["top"][idx],
                ocr_data["left"][idx] + ocr_data["width"][idx],
                ocr_data["top"][idx] + ocr_data["height"][idx],
            ]
            cropped_img = image.crop(text_area)
            generated_texts, scores = _analyze_print_from_image(
                cropped_img, vision_model, image_processor
            )
            max_score_index = torch.argmax(scores).item()
            if scores[max_score_index].item() > score_threshold:
                best_text = None
                best_similarity_score = 0.0
                for gen_text in generated_texts:
                    text_similarity_score = get_text_similarity(
                        gen_text, ocr_text
                    ).item()
                    if (
                        text_similarity_score > text_similarity_threshold
                        and text_similarity_score > best_similarity_score
                    ):
                        best_similarity_score = text_similarity_score
                        best_text = gen_text
                if best_text:
                    text_bboxes.append({"text": best_text, "bbox": text_area})

    return text_bboxes


def _get_enclosing_bbox(bboxes: List[Dict]) -> List[int]:
    """
    Returns a bounding box that surrounds all bounding boxes in the given list.

    Args:
        bboxes: a list of bounding boxes.

    Returns:
        a bounding box that surrounds all the bounding boxes in the list.
    """

    bbox = [
        min(b["bbox"][0] for b in bboxes),
        min(b["bbox"][1] for b in bboxes),
        max(b["bbox"][2] for b in bboxes),
        max(b["bbox"][3] for b in bboxes),
    ]

    return bbox


def _get_text_lines(text_bboxes: Dict) -> List[Dict]:
    """
    For the given dictionary, consisting of text lines and lines' bounding boxes,
    it returns a list of unique dictionary items, regarding the text of the items.

    Args:
        text_bboxes: a dictionary containing text lines and lines' bounding boxes.

    Returns:
        a list of unique items regarding the text they contain.
    """

    if not text_bboxes:
        return []

    lines = []

    for i in range(len(text_bboxes) - 1):
        related_text_boxes = [text_bboxes[i]]
        for j in range(i + 1, len(text_bboxes)):
            if _are_textboxes_adjacent(
                text_bboxes[j]["bbox"], [t["bbox"] for t in related_text_boxes]
            ):
                related_text_boxes.append(text_bboxes[j])
        bbox = _get_enclosing_bbox(related_text_boxes)
        lines.append(
            {"text": " ".join([t["text"] for t in related_text_boxes]), "bbox": bbox}
        )

    return _get_unique_lines(lines)


def _get_unique_lines(lines: List[Dict]) -> List[Dict]:
    """
    Returns a list of unique lines of text for the given list of lines.

    Args:
        lines: a list of lines dictionaries.

    Returns:
        a list of unique lines.
    """

    sorted_lines = list(sorted(lines, key=lambda i: len(i["text"]), reverse=True))
    unique_lines = []

    for l1 in sorted_lines:
        is_unique = True
        for l2 in unique_lines:
            if l1["text"] in l2["text"]:
                is_unique = False
                break
        if is_unique:
            unique_lines.append(l1)

    return unique_lines
