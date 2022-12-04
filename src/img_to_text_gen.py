import torch
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer

itt_model = None
itt_feature_extractor = None
itt_tokenizer = None
itt_device = None
itt_inited = False

def init_itt():
    global itt_model
    global itt_feature_extractor
    global itt_tokenizer
    global itt_device
    global itt_inited

    if itt_inited:
        return itt_inited

    try:
        itt_model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        itt_feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        itt_tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

        itt_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        itt_model.to(itt_device)
    except:
        itt_inited = False
        return False

    itt_inited = True
    return itt_inited

def gen_itt(image):
    if not itt_inited:
        return None
    if image is None:
        return None

    max_length = 16
    num_beams = 4
    gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

    if image.mode != "RGB":
        image = image.convert(mode="RGB")

    images = [image]

    pixel_values = itt_feature_extractor(images=images, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(itt_device)

    output_ids = itt_model.generate(pixel_values, **gen_kwargs)

    preds = itt_tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]

    return preds[0]