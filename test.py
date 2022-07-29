from datasets.coco import build

class args():
    coco_path = "data"

dataset = build("train", args())