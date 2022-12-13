import io
import streamlit as st

from PIL import Image
from img_to_text_gen import IttGenerator


def gen_text(image, max_length, min_length, num_beams):
    itt_generator = IttGenerator()

    try:
        text = itt_generator.gen(
            image=image,
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams)
    except ValueError:
        return "Failed wille opening an image"

    return text


def main():
    st.set_page_config(
        page_title='Funny descriptions generator',
        page_icon=':skull:',
        layout='wide')

    num_beams = st.sidebar.slider("Num beams", 1, 16, 4)
    max_length = st.sidebar.slider("Max length", 1, 256, 128)
    min_length = st.sidebar.slider("Min length", 1, 256, 16)

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
            description = gen_text(
                image=image,
                max_length=max_length,
                min_length=min_length,
                num_beams=num_beams)

            st.title('Description')
            st.write(description)


if __name__ == '__main__':
    main()
