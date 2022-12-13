import torch
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer


class IttGenerator(object):
    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(IttGenerator, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        self.model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        self.feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        self.tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def gen(self, image, max_length, min_length, num_beams):
        image = None
        if image is None:
            raise ValueError("An image is None")

        if max_length < min_length:
            return "Max length must be greater than min length"

        if image.mode != "RGB":
            image = image.convert(mode="RGB")

        images = [image]

        pixel_values = self.feature_extractor(images=images, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.device)

        output_ids = self.model.generate(
            pixel_values, max_length=max_length, min_length=min_length, num_beams=num_beams
        )

        preds = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        preds = [pred.strip() for pred in preds]

        return preds[0]
