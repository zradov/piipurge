import torch
import pytest
import numpy as np
from PIL import Image
from torchvision.transforms import v2
from unittest.mock import patch, MagicMock
from src.piipurge.schemas import ImageTextInfo, SavedImageInfo
from src.piipurge.analyze_images import (
    _run_ocr, analyze_print, _get_text_bboxes, _extract_features, 
    _get_vision_model, analyze_handwriting, _get_print_processor, 
    _are_textboxes_adjacent, _analyze_print_from_image, _get_image_transformations, 
    _get_enclosing_bbox, _get_text_lines, _get_unique_lines)


@pytest.fixture
def sample_saved_images_info():
    return [
        SavedImageInfo(img_path="img1.png", xref="123", page_number=1),
        SavedImageInfo(img_path="img2.png", xref="456", page_number=2),
    ]


@pytest.fixture
def fake_image():
    img = MagicMock()
    img.size = (100, 200)
    return img


@pytest.fixture
def fake_ocr_data():
    return {
        "text": ["Hello", "World"],
        "left": [0, 50],
        "top": [0, 100],
        "width": [40, 40],
        "height": [20, 20],
    }


@pytest.fixture
def fake_text_bboxes():
    return [
        {"text": "Hello", "bbox": [0, 0, 40, 20]},
        {"text": "World", "bbox": [50, 100, 90, 120]},
    ]


@pytest.fixture
def fake_text_lines():
    return [
        {"text": "Hello", "bbox": [0, 0, 40, 20]},
        {"text": "World", "bbox": [50, 100, 90, 120]},
    ]


@patch("src.piipurge.analyze_images._get_print_processor")
@patch("src.piipurge.analyze_images._get_vision_model")
@patch("src.piipurge.analyze_images._run_ocr")
@patch("src.piipurge.analyze_images._get_text_bboxes")
@patch("src.piipurge.analyze_images._get_text_lines")
def test_analyze_print_basic(
    mock_get_text_lines,
    mock_get_text_bboxes,
    mock_run_ocr,
    mock_get_vision_model,
    mock_get_print_processor,
    sample_saved_images_info,
    fake_image,
    fake_ocr_data,
    fake_text_bboxes,
    fake_text_lines,
):
    mock_get_print_processor.return_value = MagicMock()
    mock_get_vision_model.return_value = MagicMock()
    mock_run_ocr.return_value = (fake_image, fake_ocr_data)
    mock_get_text_bboxes.return_value = fake_text_bboxes
    mock_get_text_lines.return_value = fake_text_lines

    result = analyze_print(sample_saved_images_info)

    assert isinstance(result, dict)
    assert set(result.keys()) == {"img1.png", "img2.png"}
    for info in result.values():
        assert isinstance(info, ImageTextInfo)
        assert info.image_size == (100, 200)
        assert info.text_lines == fake_text_lines


@patch("src.piipurge.analyze_images._get_print_processor")
@patch("src.piipurge.analyze_images._get_vision_model")
@patch("src.piipurge.analyze_images._run_ocr")
@patch("src.piipurge.analyze_images._get_text_bboxes")
@patch("src.piipurge.analyze_images._get_text_lines")
def test_analyze_print_empty_text_lines(
    mock_get_text_lines,
    mock_get_text_bboxes,
    mock_run_ocr,
    mock_get_vision_model,
    mock_get_print_processor,
    sample_saved_images_info,
    fake_image,
    fake_ocr_data,
    fake_text_bboxes,
):
    mock_get_print_processor.return_value = MagicMock()
    mock_get_vision_model.return_value = MagicMock()
    mock_run_ocr.return_value = (fake_image, fake_ocr_data)
    mock_get_text_bboxes.return_value = fake_text_bboxes
    mock_get_text_lines.return_value = []

    result = analyze_print(sample_saved_images_info)

    assert isinstance(result, dict)
    for info in result.values():
        assert isinstance(info, ImageTextInfo)
        assert info.text_lines == []


@patch("src.piipurge.analyze_images._get_print_processor")
@patch("src.piipurge.analyze_images._get_vision_model")
@patch("src.piipurge.analyze_images._run_ocr")
@patch("src.piipurge.analyze_images._get_text_bboxes")
@patch("src.piipurge.analyze_images._get_text_lines")
def test_analyze_print_no_images(
    mock_get_text_lines,
    mock_get_text_bboxes,
    mock_run_ocr,
    mock_get_vision_model,
    mock_get_print_processor,
):
    result = analyze_print([])

    assert result == {}


@patch("src.piipurge.analyze_images._get_image_transformations")
@patch("src.piipurge.analyze_images.joblib.load")
@patch("src.piipurge.analyze_images.clear_background")
@patch("src.piipurge.analyze_images.Image.open")
@patch("src.piipurge.analyze_images._extract_features")
def test_analyze_handwriting_basic(
    mock_extract_features,
    mock_image_open,
    mock_clear_background,
    mock_joblib_load,
    mock_get_image_transformations,
):
    img_paths = ["img1.png", "img2.png"]
    fake_transform = MagicMock()
    fake_img = MagicMock()
    fake_img_tensor = MagicMock()
    fake_img_tensor.unsqueeze.return_value = MagicMock()
    fake_features = MagicMock()
    fake_features.cpu.return_value.numpy.return_value = "features"
    fake_model = MagicMock()
    fake_model.predict.side_effect = [1, 0]

    mock_get_image_transformations.return_value = fake_transform
    fake_transform.return_value = fake_img_tensor
    mock_image_open.return_value = fake_img
    mock_extract_features.return_value = fake_features
    mock_joblib_load.return_value = fake_model

    result = analyze_handwriting(img_paths)

    assert result == [("img1.png", True), ("img2.png", False)]
    assert mock_clear_background.call_count == 2
    assert mock_image_open.call_count == 2
    assert fake_model.predict.call_count == 2


@patch("src.piipurge.analyze_images._get_image_transformations")
@patch("src.piipurge.analyze_images.joblib.load")
@patch("src.piipurge.analyze_images.clear_background")
@patch("src.piipurge.analyze_images.Image.open")
@patch("src.piipurge.analyze_images._extract_features")
def test_analyze_handwriting_empty(
    mock_extract_features,
    mock_image_open,
    mock_clear_background,
    mock_joblib_load,
    mock_get_image_transformations,
):
    result = analyze_handwriting([])

    assert result == []


@patch("src.piipurge.analyze_images._get_image_transformations")
@patch("src.piipurge.analyze_images.joblib.load")
@patch("src.piipurge.analyze_images.clear_background")
@patch("src.piipurge.analyze_images.Image.open")
@patch("src.piipurge.analyze_images._extract_features")
def test_analyze_handwriting_all_false(
    mock_extract_features,
    mock_image_open,
    mock_clear_background,
    mock_joblib_load,
    mock_get_image_transformations,
):
    img_paths = ["img1.png", "img2.png"]
    fake_transform = MagicMock()
    fake_img = MagicMock()
    fake_img_tensor = MagicMock()
    fake_img_tensor.unsqueeze.return_value = MagicMock()
    fake_features = MagicMock()
    fake_features.cpu.return_value.numpy.return_value = "features"
    fake_model = MagicMock()
    fake_model.predict.side_effect = [0, 0]

    mock_get_image_transformations.return_value = fake_transform
    fake_transform.return_value = fake_img_tensor
    mock_image_open.return_value = fake_img
    mock_extract_features.return_value = fake_features
    mock_joblib_load.return_value = fake_model

    result = analyze_handwriting(img_paths)

    assert result == [("img1.png", False), ("img2.png", False)]


@patch("src.piipurge.analyze_images.vgg16")
@patch("src.piipurge.analyze_images.Linear")
def test_extract_features_basic(mock_linear, mock_vgg16):
    # Arrange

    # Fake input tensor (batch of 2 images, 3 channels, 224x224)
    images = torch.randn(2, 3, 224, 224)

    # Mock VGG16 model and its methods
    mock_vgg = MagicMock()
    mock_features = torch.randn(2, 512, 7, 7)
    mock_avgpool = torch.randn(2, 512, 1, 1)
    mock_flatten = torch.randn(2, 512)
    mock_vgg.features.return_value = mock_features
    mock_vgg.avgpool.return_value = mock_avgpool
    mock_vgg16.return_value = mock_vgg

    # Patch torch.flatten to return mock_flatten
    with patch("torch.flatten", return_value=mock_flatten):
        # Mock Linear layer
        mock_proj = MagicMock()
        mock_proj.return_value = torch.randn(2, 512)
        mock_linear.return_value = mock_proj

        # Act
        out = _extract_features(images)

    # Assert
    mock_vgg.eval.assert_called_once()
    mock_vgg.features.assert_called_once_with(images)
    mock_vgg.avgpool.assert_called_once_with(mock_features)
    mock_linear.assert_called_once_with(mock_flatten.shape[1], 512)
    mock_proj.assert_called_once_with(mock_flatten)

    assert out.shape == (2, 512)


@patch("src.piipurge.analyze_images.vgg16")
@patch("src.piipurge.analyze_images.Linear")
def test_extract_features_single_image(mock_linear, mock_vgg16):

    images = torch.randn(1, 3, 224, 224)
    mock_vgg = MagicMock()
    mock_features = torch.randn(1, 512, 7, 7)
    mock_avgpool = torch.randn(1, 512, 1, 1)
    mock_flatten = torch.randn(1, 512)
    mock_vgg.features.return_value = mock_features
    mock_vgg.avgpool.return_value = mock_avgpool
    mock_vgg16.return_value = mock_vgg

    with patch("torch.flatten", return_value=mock_flatten):
        mock_proj = MagicMock()
        mock_proj.return_value = torch.randn(1, 512)
        mock_linear.return_value = mock_proj

        out = _extract_features(images)

    assert out.shape == (1, 512)


@patch("src.piipurge.analyze_images.vgg16")
@patch("src.piipurge.analyze_images.Linear")
def test_extract_features_projection_layer_called_with_correct_shape(mock_linear, mock_vgg16):

    images = torch.randn(4, 3, 224, 224)
    mock_vgg = MagicMock()
    mock_features = torch.randn(4, 256, 7, 7)
    mock_avgpool = torch.randn(4, 256, 1, 1)
    mock_flatten = torch.randn(4, 256)
    mock_vgg.features.return_value = mock_features
    mock_vgg.avgpool.return_value = mock_avgpool
    mock_vgg16.return_value = mock_vgg

    with patch("torch.flatten", return_value=mock_flatten):
        mock_proj = MagicMock()
        mock_proj.return_value = torch.randn(4, 512)
        mock_linear.return_value = mock_proj

        _extract_features(images)

    mock_linear.assert_called_once_with(256, 512)


def test_get_image_transformations_returns_compose():

    mean = torch.tensor([0.5, 0.5, 0.5])
    std = torch.tensor([0.2, 0.2, 0.2])
    transform = _get_image_transformations(mean, std)

    assert isinstance(transform, v2.Compose)
    assert len(transform.transforms) == 4
    assert any(isinstance(t, v2.ToImage) for t in transform.transforms)
    assert any(isinstance(t, v2.Resize) for t in transform.transforms)
    assert any(isinstance(t, v2.ToDtype) for t in transform.transforms)
    assert any(isinstance(t, v2.Normalize) for t in transform.transforms)


def test_get_image_transformations_pipeline_applies_all_transforms():

    mean = torch.tensor([0.5, 0.5, 0.5])
    std = torch.tensor([0.2, 0.2, 0.2])
    transform = _get_image_transformations(mean, std)

    arr = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
    img = Image.fromarray(arr)

    out = transform(img)

    assert isinstance(out, torch.Tensor)
    assert out.shape == (3, 224, 224)
    assert out.dtype == torch.float32


def test_get_image_transformations_normalization_effect():

    mean = torch.tensor([127.0, 127.0, 127.0])
    std = torch.tensor([127.0, 127.0, 127.0])
    transform = _get_image_transformations(mean, std)

    arr = np.random.randint(low=0, high=255, size=(224, 224, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    out = transform(img)
    
    assert torch.where(out > 1.0)[0].numel() == 0
    assert torch.where(out < -1.0)[0].numel() == 0
    assert out.shape == (3, 224, 224)
    
    
@patch("src.piipurge.analyze_images.TrOCRProcessor")
def test_get_print_processor_calls_from_pretrained(mock_processor):

    mock_instance = MagicMock()
    mock_processor.from_pretrained.return_value = mock_instance

    model_name = "some-model"
    result = _get_print_processor(model_name)

    mock_processor.from_pretrained.assert_called_once_with(model_name)
    assert result == mock_instance


@patch("src.piipurge.analyze_images.TrOCRProcessor")
def test_get_print_processor_returns_processor_instance(mock_processor):

    mock_instance = MagicMock()
    mock_processor.from_pretrained.return_value = mock_instance

    result = _get_print_processor("test-model")
    assert result is mock_instance
    
    
@patch("src.piipurge.analyze_images.VisionEncoderDecoderModel")
def test_get_vision_model_sets_config_correctly(mock_vedm):

    mock_model = MagicMock()
    mock_config = MagicMock()
    mock_decoder = MagicMock()
    mock_config.decoder = mock_decoder
    mock_decoder.vocab_size = 1234
    mock_model.config = mock_config
    mock_vedm.from_pretrained.return_value = mock_model

    mock_tokenizer = MagicMock()
    mock_tokenizer.cls_token_id = 101
    mock_tokenizer.pad_token_id = 0
    mock_tokenizer.sep_token_id = 102
    mock_processor = MagicMock()
    mock_processor.tokenizer = mock_tokenizer

    model_name = "test-model"
    num_return_sequences = 5

    result = _get_vision_model(model_name, mock_processor, num_return_sequences)

    mock_vedm.from_pretrained.assert_called_once_with(model_name)

    assert result is mock_model
    assert mock_model.config.decoder_start_token_id == 101
    assert mock_model.config.pad_token_id == 0
    assert mock_model.config.vocab_size == 1234
    assert mock_model.config.eos_token_id == 102
    assert mock_model.config.max_length == 50
    assert mock_model.config.early_stopping is True
    assert mock_model.config.no_repeat_ngram_size == 3
    assert mock_model.config.length_penalty == 2.0
    assert mock_model.config.num_beams == 4
    assert mock_model.config.num_return_sequences == num_return_sequences


@patch("src.piipurge.analyze_images.VisionEncoderDecoderModel")
def test_get_vision_model_default_num_return_sequences(mock_vedm):

    mock_model = MagicMock()
    mock_config = MagicMock()
    mock_decoder = MagicMock()
    mock_config.decoder = mock_decoder
    mock_decoder.vocab_size = 999
    mock_model.config = mock_config
    mock_vedm.from_pretrained.return_value = mock_model
    mock_tokenizer = MagicMock()
    mock_tokenizer.cls_token_id = 1
    mock_tokenizer.pad_token_id = 2
    mock_tokenizer.sep_token_id = 3
    mock_processor = MagicMock()
    mock_processor.tokenizer = mock_tokenizer

    model_name = "default-model"

    result = _get_vision_model(model_name, mock_processor)

    assert result.config.num_return_sequences == 3
    assert result.config.vocab_size == 999
    assert result.config.decoder_start_token_id == 1
    assert result.config.pad_token_id == 2
    assert result.config.eos_token_id == 3


@patch("src.piipurge.analyze_images.PreTrainedModel")
def test_analyze_print_from_image_basic(mock_model_cls):

    mock_image = MagicMock()
    mock_model = MagicMock()
    mock_processor = MagicMock()
    mock_pixel_values = MagicMock()
    mock_processor.return_value.pixel_values = mock_pixel_values
    mock_sequences = [[1, 2, 3], [4, 5, 6]]
    mock_scores = torch.tensor([0.9, 0.8])
    mock_generate_result = {
        "sequences": mock_sequences,
        "sequences_scores": mock_scores,
    }
    mock_model.generate.return_value = mock_generate_result

    mock_processor.batch_decode.return_value = ["hello", "world"]

    result_texts, result_scores = _analyze_print_from_image(mock_image, mock_model, mock_processor)

    mock_processor.assert_called_once_with(
        mock_image, return_tensors="pt", input_data_format="channels_last"
    )
    mock_model.generate.assert_called_once_with(
        mock_pixel_values,
        output_scores=True,
        return_dict_in_generate=True,
    )
    mock_processor.batch_decode.assert_called_once_with(mock_sequences, skip_special_tokens=True)
    assert result_texts == ["hello", "world"]
    assert torch.equal(result_scores, mock_scores)


@patch("src.piipurge.analyze_images.PreTrainedModel")
def test_analyze_print_from_image_empty_sequences(mock_model_cls):

    mock_image = MagicMock()
    mock_model = MagicMock()
    mock_processor = MagicMock()
    mock_pixel_values = MagicMock()
    mock_processor.return_value.pixel_values = mock_pixel_values
    mock_generate_result = {
        "sequences": [],
        "sequences_scores": torch.tensor([]),
    }
    mock_model.generate.return_value = mock_generate_result
    mock_processor.batch_decode.return_value = []

    result_texts, result_scores = _analyze_print_from_image(mock_image, mock_model, mock_processor)

    assert result_texts == []
    assert torch.equal(result_scores, torch.tensor([]))


def test_analyze_print_from_image_calls_batch_decode_with_skip_special_tokens():

    mock_image = MagicMock()
    mock_model = MagicMock()
    mock_processor = MagicMock()
    mock_pixel_values = MagicMock()
    mock_processor.return_value.pixel_values = mock_pixel_values

    mock_sequences = [[7, 8, 9]]
    mock_scores = torch.tensor([0.5])
    mock_generate_result = {
        "sequences": mock_sequences,
        "sequences_scores": mock_scores,
    }
    mock_model.generate.return_value = mock_generate_result
    mock_processor.batch_decode.return_value = ["foo"]

    _analyze_print_from_image(mock_image, mock_model, mock_processor)
    mock_processor.batch_decode.assert_called_once_with(mock_sequences, skip_special_tokens=True)
    def test_are_textboxes_adjacent_true_horizontal():
        # tb[2] - text_box[0] < 10 and tb[1] - text_box[1] < 5
        text_box = [50, 10, 100, 30]
        text_boxes = [
            [40, 10, 49, 30],  # tb[2]=49, text_box[0]=50, diff=1 < 10, tb[1]=10, text_box[1]=10, diff=0 < 5
        ]
        assert _are_textboxes_adjacent(text_box, text_boxes) is True


def test_are_textboxes_adjacent_true_vertical():

    text_box = [50, 10, 100, 30]
    text_boxes = [[50, 5, 100, 19]]

    assert _are_textboxes_adjacent(text_box, text_boxes) is True


def test_are_textboxes_adjacent_false_far_apart():

    text_box = [100, 100, 150, 150]
    text_boxes = [
        [0, 0, 50, 50],
        [200, 200, 250, 250]
    ]

    assert _are_textboxes_adjacent(text_box, text_boxes) is False


def test_are_textboxes_adjacent_multiple_boxes_one_adjacent():
    text_box = [50, 10, 100, 30]
    text_boxes = [
        [0, 0, 10, 10],
        [40, 10, 49, 30], 
        [200, 200, 250, 250],
    ]

    assert _are_textboxes_adjacent(text_box, text_boxes) is True


def test_are_textboxes_adjacent_empty_list():

    text_box = [10, 10, 20, 20]
    text_boxes = []

    assert _are_textboxes_adjacent(text_box, text_boxes) is False


def test_are_textboxes_adjacent_edge_case_exact_threshold():

    text_box = [50, 10, 100, 30]
    text_boxes = [
        [40, 15, 60, 20],
        [50, 0, 100, 0]
    ]

    assert _are_textboxes_adjacent(text_box, text_boxes) is False


@patch("src.piipurge.analyze_images.pytesseract")
@patch("src.piipurge.analyze_images.Image")
def test_run_ocr_returns_image_and_ocr_data(mock_image_module, mock_pytesseract):

    mock_img_instance = MagicMock()
    mock_converted_img = MagicMock()
    mock_image_module.open.return_value = mock_img_instance
    mock_img_instance.convert.return_value = mock_converted_img

    fake_ocr_data = {"text": ["abc"], "left": [0], "top": [0], "width": [10], "height": [10]}
    mock_pytesseract.image_to_data.return_value = fake_ocr_data
    mock_pytesseract.Output.DICT = "DICT"

    img_path = "some_path.png"
    image, ocr_data = _run_ocr(img_path)

    mock_image_module.open.assert_called_once_with(img_path)
    mock_img_instance.convert.assert_called_once_with("RGB")
    mock_pytesseract.image_to_data.assert_called_once_with(mock_converted_img, output_type="DICT")
    
    assert image == mock_converted_img
    assert ocr_data == fake_ocr_data


@patch("src.piipurge.analyze_images.pytesseract")
@patch("src.piipurge.analyze_images.Image")
def test_run_ocr_handles_empty_ocr_data(mock_image_module, mock_pytesseract):

    mock_img_instance = MagicMock()
    mock_converted_img = MagicMock()
    mock_image_module.open.return_value = mock_img_instance
    mock_img_instance.convert.return_value = mock_converted_img

    mock_pytesseract.image_to_data.return_value = {}
    mock_pytesseract.Output.DICT = "DICT"

    img_path = "empty.png"
    image, ocr_data = _run_ocr(img_path)

    assert image == mock_converted_img
    assert ocr_data == {}


@patch("src.piipurge.analyze_images.pytesseract")
@patch("src.piipurge.analyze_images.Image")
def test_run_ocr_image_open_and_convert_called(mock_image_module, mock_pytesseract):

    mock_img_instance = MagicMock()
    mock_converted_img = MagicMock()
    mock_image_module.open.return_value = mock_img_instance
    mock_img_instance.convert.return_value = mock_converted_img

    mock_pytesseract.image_to_data.return_value = {"dummy": "data"}
    mock_pytesseract.Output.DICT = "DICT"

    img_path = "file.png"
    _run_ocr(img_path)

    mock_image_module.open.assert_called_once_with(img_path)
    mock_img_instance.convert.assert_called_once_with("RGB")
    mock_pytesseract.image_to_data.assert_called_once_with(mock_converted_img, output_type="DICT")
    
    
@patch("src.piipurge.analyze_images._analyze_print_from_image")
@patch("src.piipurge.analyze_images.get_text_similarity")
def test_get_text_bboxes_basic(mock_get_text_similarity, mock_analyze_print_from_image):

    ocr_data = {
        "text": ["foo", "bar"],
        "left": [0, 10],
        "top": [0, 10],
        "width": [5, 5],
        "height": [5, 5],
    }
    fake_image = MagicMock()
    fake_cropped = MagicMock()
    fake_image.crop.return_value = fake_cropped

    mock_analyze_print_from_image.return_value = (["foo", "baz"], torch.tensor([0.9, 0.8]))
    
    def similarity_side_effect(gen_text, ocr_text):
        print(f"Comparing '{gen_text}' with '{ocr_text}'")
        if gen_text == ocr_text:
            return torch.tensor(0.99)
        return torch.tensor(0.5)
    mock_get_text_similarity.side_effect = similarity_side_effect

    result = _get_text_bboxes(
        fake_image, ocr_data, MagicMock(), MagicMock(),
        include_empty_text=True, score_threshold=0.0, text_similarity_threshold=0.95
    )

    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0]["text"] == "foo"
    assert "bbox" in result[0]
    assert isinstance(result[0]["bbox"], list)


@patch("src.piipurge.analyze_images._analyze_print_from_image")
@patch("src.piipurge.analyze_images.get_text_similarity")
def test_get_text_bboxes_score_below_threshold(mock_get_text_similarity, mock_analyze_print_from_image):

    ocr_data = {
        "text": ["foo"],
        "left": [0],
        "top": [0],
        "width": [5],
        "height": [5],
    }
    fake_image = MagicMock()
    fake_image.crop.return_value = MagicMock()
    mock_analyze_print_from_image.return_value = (["foo"], torch.tensor([-1.0]))
    mock_get_text_similarity.return_value = torch.tensor(1.0)

    result = _get_text_bboxes(
        fake_image, ocr_data, MagicMock(), MagicMock(),
        include_empty_text=True, score_threshold=0.0, text_similarity_threshold=0.95
    )
    assert result == []


@patch("src.piipurge.analyze_images._analyze_print_from_image")
@patch("src.piipurge.analyze_images.get_text_similarity")
def test_get_text_bboxes_similarity_below_threshold(mock_get_text_similarity, mock_analyze_print_from_image):

    ocr_data = {
        "text": ["foo"],
        "left": [0],
        "top": [0],
        "width": [5],
        "height": [5],
    }
    fake_image = MagicMock()
    fake_image.crop.return_value = MagicMock()
    mock_analyze_print_from_image.return_value = (["foo"], torch.tensor([1.0]))
    mock_get_text_similarity.return_value = torch.tensor(0.5)

    result = _get_text_bboxes(
        fake_image, ocr_data, MagicMock(), MagicMock(),
        include_empty_text=True, score_threshold=0.0, text_similarity_threshold=0.95
    )
    assert result == []


@patch("src.piipurge.analyze_images._analyze_print_from_image")
@patch("src.piipurge.analyze_images.get_text_similarity")
def test_get_text_bboxes_include_empty_text_false(mock_get_text_similarity, mock_analyze_print_from_image):

    ocr_data = {
        "text": ["", "foo"],
        "left": [0, 10],
        "top": [0, 10],
        "width": [5, 5],
        "height": [5, 5],
    }
    fake_image = MagicMock()
    fake_image.crop.return_value = MagicMock()
    mock_analyze_print_from_image.return_value = (["foo"], torch.tensor([1.0]))
    mock_get_text_similarity.return_value = torch.tensor(1.0)

    result = _get_text_bboxes(
        fake_image, ocr_data, MagicMock(), MagicMock(),
        include_empty_text=False, score_threshold=0.0, text_similarity_threshold=0.95
    )
    
    assert len(result) == 1
    assert result[0]["text"] == "foo"


@patch("src.piipurge.analyze_images._analyze_print_from_image")
@patch("src.piipurge.analyze_images.get_text_similarity")
def test_get_text_bboxes_best_similarity_selected(mock_get_text_similarity, mock_analyze_print_from_image):

    ocr_data = {
        "text": ["foo"],
        "left": [0],
        "top": [0],
        "width": [5],
        "height": [5],
    }
    fake_image = MagicMock()
    fake_image.crop.return_value = MagicMock()
    mock_analyze_print_from_image.return_value = (["foo", "bar"], torch.tensor([1.0, 1.0]))

    def similarity_side_effect(gen_text, ocr_text):
        if gen_text == "bar":
            return torch.tensor(0.99)
        return torch.tensor(0.96)
    mock_get_text_similarity.side_effect = similarity_side_effect

    result = _get_text_bboxes(
        fake_image, ocr_data, MagicMock(), MagicMock(),
        include_empty_text=True, score_threshold=0.0, text_similarity_threshold=0.95
    )
    
    assert len(result) == 1
    assert result[0]["text"] == "bar"


def test_get_enclosing_bbox_basic():
    bboxes = [
        {"bbox": [10, 20, 30, 40]},
        {"bbox": [15, 10, 35, 50]},
        {"bbox": [5, 25, 25, 45]},
    ]
    result = _get_enclosing_bbox(bboxes)
    assert result == [5, 10, 35, 50]


def test_get_enclosing_bbox_single_box():
    bboxes = [
        {"bbox": [1, 2, 3, 4]}
    ]
    result = _get_enclosing_bbox(bboxes)
    assert result == [1, 2, 3, 4]


def test_get_enclosing_bbox_negative_coordinates():
    bboxes = [
        {"bbox": [-10, -20, 0, 0]},
        {"bbox": [-5, -25, 5, 5]},
    ]
    result = _get_enclosing_bbox(bboxes)
    assert result == [-10, -25, 5, 5]


def test_get_enclosing_bbox_overlapping_boxes():
    bboxes = [
        {"bbox": [0, 0, 10, 10]},
        {"bbox": [5, 5, 15, 15]},
        {"bbox": [8, 8, 12, 12]},
    ]
    result = _get_enclosing_bbox(bboxes)
    assert result == [0, 0, 15, 15]


def test_get_enclosing_bbox_all_same():
    bboxes = [
        {"bbox": [2, 2, 4, 4]},
        {"bbox": [2, 2, 4, 4]},
        {"bbox": [2, 2, 4, 4]},
    ]
    result = _get_enclosing_bbox(bboxes)
    assert result == [2, 2, 4, 4]


def test_get_text_lines_empty_returns_empty():
    assert _get_text_lines([]) == []


def test_get_text_lines_single_box():
    text_bboxes = [{"text": "Hello", "bbox": [0, 0, 10, 10]}]
    result = _get_text_lines(text_bboxes)
    assert isinstance(result, list)
    assert len(result) == 0 


def test_get_text_lines_two_non_adjacent(monkeypatch):
    text_bboxes = [
        {"text": "Hello", "bbox": [0, 0, 10, 10]},
        {"text": "World", "bbox": [100, 100, 110, 110]},
    ]
    monkeypatch.setattr("src.piipurge.analyze_images._are_textboxes_adjacent", lambda tb, tbs: False)
    monkeypatch.setattr("src.piipurge.analyze_images._get_enclosing_bbox", lambda bbs: bbs[0]["bbox"])
    monkeypatch.setattr("src.piipurge.analyze_images._get_unique_lines", lambda lines: lines)
    result = _get_text_lines(text_bboxes)

    assert len(result) == 1
    assert result[0]["text"] == "Hello"
    assert result[0]["bbox"] == [0, 0, 10, 10]


def test_get_text_lines_two_adjacent(monkeypatch):
    text_bboxes = [
        {"text": "Hello", "bbox": [0, 0, 10, 10]},
        {"text": "World", "bbox": [11, 0, 21, 10]},
    ]
    monkeypatch.setattr("src.piipurge.analyze_images._are_textboxes_adjacent", lambda tb, tbs: True)
    monkeypatch.setattr("src.piipurge.analyze_images._get_enclosing_bbox", lambda bbs: [0, 0, 21, 10])
    monkeypatch.setattr("src.piipurge.analyze_images._get_unique_lines", lambda lines: lines)
    result = _get_text_lines(text_bboxes)

    assert len(result) == 1
    assert result[0]["text"] == "Hello World"
    assert result[0]["bbox"] == [0, 0, 21, 10]


def test_get_text_lines_multiple_boxes_some_adjacent(monkeypatch):
    text_bboxes = [
        {"text": "A", "bbox": [0, 0, 10, 10]},
        {"text": "B", "bbox": [11, 0, 21, 10]},
        {"text": "C", "bbox": [100, 100, 110, 110]},
    ]
    # Only the first two are adjacent
    def fake_adjacent(tb, tbs):
        if tb == [11, 0, 21, 10]:
            return True
        return False
    monkeypatch.setattr("src.piipurge.analyze_images._are_textboxes_adjacent", fake_adjacent)
    monkeypatch.setattr("src.piipurge.analyze_images._get_enclosing_bbox", lambda bbs: [0, 0, 21, 10])
    monkeypatch.setattr("src.piipurge.analyze_images._get_unique_lines", lambda lines: lines)
    result = _get_text_lines(text_bboxes)

    assert len(result) == 2
    assert result[0]["text"] == "A B"
    assert result[1]["text"] == "B"


def test_get_text_lines_calls_get_unique_lines(monkeypatch):
    text_bboxes = [
        {"text": "A", "bbox": [0, 0, 10, 10]},
        {"text": "B", "bbox": [11, 0, 21, 10]},
    ]
    monkeypatch.setattr("src.piipurge.analyze_images._are_textboxes_adjacent", lambda tb, tbs: True)
    monkeypatch.setattr("src.piipurge.analyze_images._get_enclosing_bbox", lambda bbs: [0, 0, 21, 10])
    called = {}
    def fake_get_unique_lines(lines):
        called["lines"] = lines
        return lines
    monkeypatch.setattr("src.piipurge.analyze_images._get_unique_lines", fake_get_unique_lines)
    _get_text_lines(text_bboxes)

    assert "lines" in called
    assert isinstance(called["lines"], list)


def test_get_unique_lines_empty():
    assert _get_unique_lines([]) == []


def test_get_unique_lines_all_unique():
    lines = [
        {"text": "test1", "bbox": [0, 0, 1, 1]},
        {"text": "test2", "bbox": [1, 1, 2, 2]},
        {"text": "test3", "bbox": [2, 2, 3, 3]},
    ]
    result = _get_unique_lines(lines)

    assert len(result) == 3
    assert all(line in result for line in lines)


def test_get_unique_lines_duplicates_removed():
    lines = [
        {"text": "test1", "bbox": [0, 0, 1, 1]},
        {"text": "test2", "bbox": [1, 1, 2, 2]},
        {"text": "test1", "bbox": [2, 2, 3, 3]},
    ]
    result = _get_unique_lines(lines)
    texts = [l["text"] for l in result]

    assert "test1" in texts
    assert "test2" in texts
    assert len(result) == 2


def test_get_unique_lines_longer_text_first():
    lines = [
        {"text": "abc", "bbox": [0, 0, 1, 1]},
        {"text": "a", "bbox": [1, 1, 2, 2]},
        {"text": "ab", "bbox": [2, 2, 3, 3]},
    ]
    result = _get_unique_lines(lines)
    texts = [l["text"] for l in result]

    assert "abc" in texts
    assert "ab" not in texts
    assert "a" not in texts
    assert len(result) == 1


def test_get_unique_lines_remove_overlaps():
    lines = [
        {"text": "test1", "bbox": [0, 0, 1, 1]},
        {"text": "test1test2", "bbox": [1, 1, 2, 2]},
        {"text": "test2", "bbox": [2, 2, 3, 3]}
    ]
    
    result = _get_unique_lines(lines)
    texts = [l["text"] for l in result]

    assert "test1test2" in texts
    assert len(result) == 1



