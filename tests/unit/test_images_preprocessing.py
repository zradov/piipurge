import cv2
import numpy as np
from src.piipurge.images_preprocessing import clear_background


def test_clear_background_creates_white_background(tmp_path):
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    img[2:8, 2:8] = [150, 150, 150] 
    img_path = tmp_path / "test_img.png"
    cv2.imwrite(str(img_path), img)

    clear_background(str(img_path), threshold_color=200)

    result = cv2.imread(str(img_path))
    unique_colors = np.unique(result.reshape(-1, 3), axis=0)
    
    assert all((np.array_equal(c, [0, 0, 0]) or np.array_equal(c, [255, 255, 255])) for c in unique_colors)


def test_clear_background_threshold(tmp_path):
    img = np.ones((2, 2, 3), dtype=np.uint8) * 255
    img[0, 0] = [100, 100, 100]
    img[1, 1] = [210, 210, 210]
    img_path = tmp_path / "test_img2.png"
    cv2.imwrite(str(img_path), img)

    clear_background(str(img_path), threshold_color=200)
    result = cv2.imread(str(img_path))

    assert np.array_equal(result[0, 0], [0, 0, 0])
    assert np.array_equal(result[1, 1], [255, 255, 255])


def test_clear_background_calls_repair_lines(monkeypatch, tmp_path):
    called = {}
    def fake_repair_lines(img):
        called['called'] = True
        return img
    import src.piipurge.images_preprocessing as ip
    monkeypatch.setattr(ip, "repair_lines", fake_repair_lines)

    img = np.ones((5, 5, 3), dtype=np.uint8) * 255
    img_path = tmp_path / "test_img3.png"
    cv2.imwrite(str(img_path), img)

    clear_background(str(img_path))

    assert called.get('called', False)
