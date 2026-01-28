import argparse
from collections import defaultdict

import cv2
import pytesseract
from pytesseract import Output


def parse_args():
    parser = argparse.ArgumentParser(
        description="Print bounding boxes and text for OCR-detected text blocks"
    )
    parser.add_argument(
        "-i",
        "--input",
        dest="input_path",
        required=True,
        help="Path to the input image",
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output_path",
        help="Path to save an output image with detected text blocks outlined",
    )
    return parser.parse_args()


def collect_blocks(ocr_data):
    blocks = defaultdict(list)
    num_items = len(ocr_data["text"])
    for i in range(num_items):
        text = ocr_data["text"][i].strip()
        if not text:
            continue
        try:
            conf = float(ocr_data["conf"][i])
        except ValueError:
            conf = -1.0
        if conf < 0:
            continue
        block_id = ocr_data["block_num"][i]
        left = int(ocr_data["left"][i])
        top = int(ocr_data["top"][i])
        width = int(ocr_data["width"][i])
        height = int(ocr_data["height"][i])
        blocks[block_id].append((i, text, left, top, width, height))
    return blocks


def block_to_output(block_items):
    block_items.sort(key=lambda item: item[0])
    texts = [item[1] for item in block_items]
    lefts = [item[2] for item in block_items]
    tops = [item[3] for item in block_items]
    rights = [item[2] + item[4] for item in block_items]
    bottoms = [item[3] + item[5] for item in block_items]
    left = min(lefts)
    top = min(tops)
    width = max(rights) - left
    height = max(bottoms) - top
    return left, top, width, height, " ".join(texts)


def main():
    args = parse_args()
    image = cv2.imread(args.input_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image at {args.input_path}")

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    config = "--psm 11 -c load_system_dawg=0 -c load_freq_dawg=0"
    ocr_data = pytesseract.image_to_data(
        rgb,
        output_type=Output.DICT,
        lang="eng",
        config=config,
    )
    blocks = collect_blocks(ocr_data)

    if not blocks:
        print("No text blocks found.")
        return

    annotated = image.copy() if args.output_path else None

    for block_id in sorted(blocks.keys()):
        left, top, width, height, text = block_to_output(blocks[block_id])
        print(
            f"block {block_id}: x={left}, y={top}, w={width}, h={height}, text={text}"
        )
        if annotated is not None:
            cv2.rectangle(
                annotated,
                (left, top),
                (left + width, top + height),
                (0, 255, 0),
                2,
            )

    if annotated is not None:
        saved = cv2.imwrite(args.output_path, annotated)
        if not saved:
            raise IOError(f"Could not write output image to {args.output_path}")


if __name__ == "__main__":
    main()
