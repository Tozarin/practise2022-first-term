import sys

from img_to_text_gen import init_itt, gen_itt
from text_to_text_gen import  init_ttt, gen_ttt

def gen_text(path):
  text = gen_itt(image_path=path)

  if text == None:
    return 'Failed wile generating text from image'

  text = gen_ttt(base_text=text)

  if text == None:
    return 'Faield wile generating text from text'


  return text

if __name__ == '__main__':
  if len(sys.argv) > 1:
    if init_itt() and init_ttt():
      text = gen_text(path=sys.argv[1])
      print(text)