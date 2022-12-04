# ------------------------------------------------------------------------
# InstanceFormer data pre-processing
# ------------------------------------------------------------------------

import json
import os
import yaml

def coto_keep_for_ovis(coco_path):
    ids_json_file =f'{coco_path}/annotations/instances_train2017.json'
    with open(ids_json_file, 'r') as fh:
        samples = json.load(fh)
    with open(os.path.join('./meta/coco.yaml'), 'r') as fh:
        category_details = yaml.load(fh, Loader=yaml.SafeLoader)
        category_details = {cat['id']: cat for cat in category_details}
    cat_ids_to_keep = [cat_id for cat_id, attribs in category_details.items() if attribs['keep_ovis']]
    samples['annotations'] = [ann for ann in samples['annotations'] if ann['category_id'] in cat_ids_to_keep]
    image_ids_to_keep = list(set([ann['image_id'] for ann in samples['annotations']]))
    samples['images'] = [img for img in samples['images'] if img['id'] in image_ids_to_keep]

    with open('coco_keepfor_ovis.json', 'w', encoding='utf-8') as f:
        json.dump(samples, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    coco_path = "set_coco_path"
    coto_keep_for_ovis(coco_path)