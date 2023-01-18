"""Detects objects and draws bounding boxes.

Single image usage:
  python bboxer.py \
    --input /some/image.jpg \
    --output /some/image_w_bboxes.jpg

Directory mode usage:
  python bboxer.py \
    --input /some/directory/with/images \
    --output /some/directory/to/save
"""
from pathlib import Path

from absl import app
from absl import flags
from PIL import Image
from torchvision.utils import draw_bounding_boxes
from torchvision.utils import save_image
import torchvision

import transforms


FLAGS = flags.FLAGS


flags.DEFINE_string('input', None, 'The image to detect objects for.')
flags.DEFINE_string('output', None, 'Where to save the bboxes for.')


class Detector:
    def __init__(self):
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.COCO_V1)
        self.model.eval()

    def detect_and_draw(self, image_in, image_out):
        im = Image.open(image_in)
        im_tensor, _ = transforms.PILToTensor()(im)
        im_tensor_01 = im_tensor / 255
        pred = self.model([im_tensor_01])
        bbox_im = draw_bounding_boxes(im_tensor, pred[0]['boxes'], width=5)
        save_image(bbox_im / 255, image_out)


def main(_):
    if FLAGS.input is None or FLAGS.output is None:
        print("Need to specify input and output.")
        exit(1)

    input_path = Path(FLAGS.input)
    if input_path.is_dir():
        if Path(FLAGS.output).is_file():
            print(f"Error: Input {FLAGS.input} is a directory.")
            print(f"Output {FLAGS.output} is a file but should be a directory.")
            exit(1)
        for f in input_path.iterdir():
            detector = Detector()
            image_in = f"{FLAGS.input}/{f.name}"
            image_out = f"{FLAGS.output}/{f.name}"
            detector.detect_and_draw(image_in, image_out)
    else:
        Detector().detect_and_draw(FLAGS.input, FLAGS.output)


if __name__ == '__main__':
    app.run(main)
