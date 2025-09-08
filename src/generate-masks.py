import os
import json
import base64
import imgviz
# import PIL.Image
from labelme import utils

def create_segmentation_mask(json_file, output_dir, class_labels):
    data = json.load(open(json_file))
    imageData = data.get("imageData")

    if not imageData:
        imagePath = os.path.join(os.path.dirname(json_file), data["imagePath"])
        with open(imagePath, "rb") as f:
            imageData = f.read()
            imageData = base64.b64encode(imageData).decode("utf-8")
    img = utils.img_b64_to_arr(imageData)

    label_name_to_value = {"_background_": 0}
    for label, value in class_labels.items():
        label_name_to_value[label] = value

    lbl, _ = utils.shapes_to_label(img.shape, data["shapes"], label_name_to_value)

    label_names = [None] * (max(label_name_to_value.values()) + 1)
    for name, value in label_name_to_value.items():
        label_names[value] = name

    lbl_viz = imgviz.label2rgb(lbl, imgviz.asgray(img), label_names=label_names, loc="rb")

    base_name = os.path.splitext(os.path.basename(json_file))[0]
    # PIL.Image.fromarray(img).save(os.path.join(output_dir, f"{base_name}_img.png"))
    utils.lblsave(os.path.join(output_dir, f"{base_name}_label.png"), lbl) # Só o que interessa para o processamento da rede neural
    # PIL.Image.fromarray(lbl_viz).save(os.path.join(output_dir, f"{base_name}_label_viz.png"))

    with open(os.path.join(output_dir, f"{base_name}_label_names.txt"), "w") as f:
        for lbl_name in label_names:
            f.write(lbl_name + "\n")


def main():
    input_dir = './dataset/processed/labels-json' # "/path to json files"
    output_dir = "./dataset/processed/labels-png"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    class_labels = {
        "AreaIndustrial":1,
        "AreaVegetada":2,
        "Edificacao":3,
        "Agua":4,
        "Agropecuaria":5,    
    } 

    json_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.json')]
    
    for json_file in json_files:
        create_segmentation_mask(json_file, output_dir, class_labels)

if __name__ == "__main__":
    main()