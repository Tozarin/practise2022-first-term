from transformers import pipeline, set_seed

ttt_generator = None
ttt_inited = False

def init_ttt():
    global ttt_generator
    global ttt_inited

    if ttt_inited:
        return ttt_inited

    try:
        ttt_generator = pipeline('text-generation', model='gpt2')
        set_seed(42)
    except:
        ttt_inited = False
        return ttt_inited

    ttt_inited = True
    return ttt_inited

def gen_ttt(base_text):
    print(base_text)
    if not ttt_inited:
        print('1')
        return None

    output = ttt_generator(base_text, max_length=256, num_return_sequences=1)
    return output[0]['generated_text']