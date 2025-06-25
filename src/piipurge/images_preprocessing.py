import os
import cv2
import numpy as np
from typing import List


def get_files(dir_path: str, ext: str=".png") -> List[str]:
    """
    Returns a list of files in a given directory that have the specifice file extension.

    Args:
        dir_path: a folder path.
        ext: a matching file extension.

    Returns:
        a list of files in the specified directory.
    """

    files = []

    for file_name in os.listdir(dir_path):
        file_path = os.path.join(dir_path, file_name)
        if os.path.isdir(file_path):
            files.extend(get_files(file_path, ext))
        else:
            if os.path.splitext(file_path)[1].lower() == ext:
                files.append(file_path)

    return files


def repair_lines(img: np.ndarray) -> np.ndarray:
    """
    Repairs lines by smoothing out noise and sharp edges.

    Args:
        img: an array representing image pixels
    
    Returns:
        blurred image pixels array.
    """
    
    gauss_win_size = 1
    gauss_sigma = 3

    img_blur = cv2.GaussianBlur(img, (gauss_win_size, gauss_win_size), gauss_sigma)

    return img_blur


def clear_background(img_path: str, threshold_color: int=200) -> None:
    """
    Clears the background artifacts and repairs the lines in the process.
    Any pixel color lower than the specified threshold color will be reset
    to the white color.
    
    Args:
        img_path: a path to the image file.
        threshold_color: a threshold above which all pixel colors will be preserved,
                         otherwise the pixel color will be reset to white.
    
    Returns:
        None
    """

    img = cv2.imread(img_path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    black_mask = gray < threshold_color
    output = np.ones_like(img) * 255
    output[black_mask] = [0, 0, 0]
    img = repair_lines(output)

    cv2.imwrite(img_path, img)
