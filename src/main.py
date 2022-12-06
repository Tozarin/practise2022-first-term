import sys
import io
import streamlit as st

from PIL import Image
from img_to_text_gen import IttGenerator
from text_to_text_gen import TttGenerator


def gen_text(image):

    itt_generator = IttGenerator()
    ttt_generator = TttGenerator()

    text = itt_generator.gen(image=image)

    if text is None:
        return 'Failed wile generating text from image'

    text = ttt_generator.gen(base_text=text)

    if text is None:
        return 'Faield wile generating text from text'

    return text

if __name__ == '__main__':

    st.set_page_config(
      page_title='Funny descriptions generator',
      page_icon=':skull:',
      layout='wide')

    with st.container():
        left_column, right_column = st.columns(2)

        with left_column:
            uploaded_file = st.file_uploader(label="Choose a file")

            if uploaded_file is not None:
                data = uploaded_file.getvalue()
                image = Image.open(io.BytesIO(data))
            else:
                image = Image.open('../images/cat.jpg')

            st.image(image)

        with right_column:
            discription = gen_text(image=image)

            st.title('Discription')
            st.write(discription)
