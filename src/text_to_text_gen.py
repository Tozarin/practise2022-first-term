from transformers import pipeline, set_seed

class TttGenerator(object):
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(TttGenerator, cls).__new__(cls)
        return cls.instance

    ttt_generator = pipeline('text-generation', model='gpt2')
    set_seed(42)

    def gen(self, base_text):
        output = self.ttt_generator(base_text, max_length=512, num_return_sequences=1)
        return output[0]['generated_text']