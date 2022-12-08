import torch
from transformers import VisionEncoderDecoderModel, \
    ViTFeatureExtractor, AutoTokenizer


class IttGenerator(object):
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(IttGenerator, cls).__new__(cls)
        return cls.instance

    model = VisionEncoderDecoderModel\
        .from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    feature_extractor = ViTFeatureExtractor\
        .from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    tokenizer = AutoTokenizer\
        .from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    device = torch\
        .device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    max_length = 64
    num_beams = 1
    min_length = 16

    def gen(self, image):
        if image is None:
            return None

        if image.mode != "RGB":
            image = image.convert(mode="RGB")

        images = [image]

        pixel_values = self.feature_extractor(
            images=images,
            return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.device)

        output_ids = self.model.generate(pixel_values,
                                         max_length=self.max_length,
                                         min_length=self.min_length,
                                         num_beams=self.num_beams)

        preds = self.tokenizer.batch_decode(
            output_ids,
            skip_special_tokens=True)
        preds = [pred.strip() for pred in preds]

        return preds[0]
