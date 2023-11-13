import math
from io import BytesIO

import streamlit as st
import numpy as np
from skimage import io, transform
from scipy.ndimage import rotate
from scipy.optimize import fsolve
from skimage.util import img_as_ubyte


def main():
    st.title("Collage Generator")
    uploaded_files = st.file_uploader("Choose images to create collage ("
                                      "solo pngs)",
                                      type=["png"],
                                      accept_multiple_files=True)
    if uploaded_files is not None:
        if len(uploaded_files) > 5:
            st.warning(f"Selecciona como máximo {5} archivos.")
            uploaded_files = uploaded_files[:5]
    if uploaded_files:
        st.header("Selected Images")
        images = [io.imread(file) for file in uploaded_files]
        st.image(images,
                 caption=[f"Image {i + 1}" for i in range(len(images))],
                 width=100)
        # Color picker para el fondo
        fondo_color = st.color_picker("Selecciona el color de fondo")
        collage_button = st.button("Generate Collage")
        if collage_button:
            collage = generate_collage(images, fondo_color)
            st.session_state['collage'] = collage
        if 'collage' in st.session_state:
            st.header("Generated Collage")
            st.image(st.session_state['collage'],
                     caption="Generated Collage", )
            # Convertir la imagen a BytesIO
            buffer = BytesIO()
            data = img_as_ubyte(st.session_state['collage'])
            io.imsave(buffer, data,
                      type='png', format='png')
            buffer.seek(0)
            st.download_button("Descargar", data=buffer,
                               file_name='collage.png', )


def generate_collage(images, fondo_color):
    # Rotate each image randomly
    rotated_images = [rotate(image, angle=np.random.randint(-30, 30)) for
                      image
                      in images]
    # Resize all images to the same dimensions
    area = \
        sorted([(image.shape[0] * image.shape[1],
                 image.shape) for image in rotated_images])[0][0]
    resized_images = []
    for image in rotated_images:
        proportion = image.shape[0] / image.shape[1]
        result = fsolve(lambda x: equations(x, proportion, area),
                        np.array([image.shape[0], image.shape[1]]))
        resized = transform.resize(np.array(image), result)
        resized_images.append(resized)
    np.random.shuffle(resized_images)
    collage = resized_images[0]
    for resized_image in resized_images[1:]:
        side = np.random.choice([0, 1])
        if side == 0:
            diff = collage.shape[0] - resized_image.shape[0]
            if diff > 0:
                relleno1 = np.zeros((math.ceil(diff / 2), resized_image.shape[
                    1], 4))
                relleno2 = np.zeros((math.floor(diff / 2), resized_image.shape[
                    1], 4))
                resized_image = np.vstack((relleno1, resized_image, relleno2))
            elif diff < 0:
                diff = -diff
                relleno1 = np.zeros((math.ceil(diff / 2), collage.shape[
                    1], 4))
                relleno2 = np.zeros((math.floor(diff / 2), collage.shape[
                    1], 4))
                collage = np.vstack((relleno1, collage, relleno2))
            collage = np.hstack((collage, resized_image))
        else:
            diff = collage.shape[1] - resized_image.shape[1]
            if diff > 0:
                relleno1 = np.zeros((resized_image.shape[
                                         0], math.ceil(diff / 2), 4))
                relleno2 = np.zeros((resized_image.shape[
                                         0], math.floor(diff / 2), 4))
                resized_image = np.hstack((relleno1, resized_image, relleno2))
            elif diff < 0:
                diff = -diff
                relleno1 = np.zeros((collage.shape[
                                         0], math.ceil(diff / 2), 4))
                relleno2 = np.zeros((collage.shape[
                                         0], math.floor(diff / 2), 4))
                collage = np.hstack((relleno1, collage, relleno2))
            collage = np.vstack((collage, resized_image))
    # Color de fondo (en este caso, blanco)
    # Crear un fondo del mismo tamaño que la imagen original
    hex_color = fondo_color.lstrip('#')

    # Convertir el valor hexadecimal a RGB
    rgb_color = tuple(int(hex_color[i:i + 2], 16) / 255 for i in (0, 2, 4))

    fondo = np.ones((collage.shape[0], collage.shape[1], 4)) * [*rgb_color, 1]
    collage = np.where(collage[:, :, :3] == 0,
                       fondo[:, :, :3], collage[:, :, :3])
    return collage


def equations(vars, proportion, area):
    x, y = vars
    eq1 = x / y - proportion
    eq2 = x * y - area
    return [eq1, eq2]


if __name__ == "__main__":
    main()
