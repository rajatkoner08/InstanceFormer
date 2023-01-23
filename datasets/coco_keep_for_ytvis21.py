# ------------------------------------------------------------------------
# InstanceFormer data pre-processing
# ------------------------------------------------------------------------

import json

def coto_keep_for_yvis21(coco_path):
    ids_json_file = f'{coco_path}/annotations/instances_train2017.json'
    with open(ids_json_file, 'r') as fh:
        samples = json.load(fh)
    cat_ids_to_keep = [1, 2, 3, 4, 5, 7, 8, 9, 16, 17, 18, 19, 21, 22, 23, 24, 25, 36, 41, 42, 43, 74]
    samples['annotations'] = [ann for ann in samples['annotations'] if ann['category_id'] in cat_ids_to_keep]
    image_ids_to_keep = list(set([ann['image_id'] for ann in samples['annotations']]))
    samples['images'] = [img for img in samples['images'] if img['id'] in image_ids_to_keep]

    with open('coco_keepfor_ytvis21.json', 'w', encoding='utf-8') as f:
        json.dump(samples, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    coco_path = "set_coco_path"
    coto_keep_for_ovis(coco_path)