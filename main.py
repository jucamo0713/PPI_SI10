# Librerías nativas de Python
import math
from io import BytesIO

# Librerías de terceros
import numpy as np
import streamlit as st
from skimage import io, transform
from scipy.ndimage import rotate
from scipy.optimize import fsolve
from skimage.util import img_as_ubyte


def main():
    """
    Función principal que muestra la interfaz de usuario para crear collages.
    """
    st.title("Collage Generator")

    # Seleccionar archivos
    uploaded_files = st.file_uploader(
        "Choose images to create collage",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True)

    # Validar número máximo de archivos
    if uploaded_files is not None:
        if len(uploaded_files) > 5:
            st.warning(f"Selecciona como máximo {5} archivos.")
            uploaded_files = uploaded_files[:5]

    if uploaded_files:
        st.header("Selected Images")

        # Leer y mostrar imágenes seleccionadas
        images = [io.imread(file) for file in uploaded_files]
        st.image(images,
                 caption=[f"Image {i + 1}" for i in range(len(images))],
                 width=100)

        # Color picker para el fondo
        fondo_color = st.color_picker("Selecciona el color de fondo")
        collage_button = st.button("Generate Collage")

        # Generar collage al hacer clic en el botón
        if collage_button:
            collage = generate_collage(images, fondo_color)
            st.session_state['collage'] = collage

        # Mostrar collage generado y botón de descarga
        if 'collage' in st.session_state:
            st.header("Generated Collage")
            st.image(st.session_state['collage'],
                     caption="Generated Collage", )

            # Convertir la imagen a BytesIO y agregar botón de descarga
            buffer = BytesIO()
            data = img_as_ubyte(st.session_state['collage'])
            io.imsave(buffer, data, type='png', format='png')
            buffer.seek(0)
            st.download_button("Descargar", data=buffer,
                               file_name='collage.png', )


def generate_collage(images, fondo_color):
    """
    Genera un collage a partir de una lista de imágenes y un color de fondo.

    Args:
        images (list): Lista de imágenes.
        fondo_color (str): Color de fondo en formato hexadecimal.

    Returns:
        np.ndarray: Imagen de collage generada.
    """

    num_channels = max(image.shape[2] for image in images)
    # Rotar cada imagen de forma aleatoria
    rotated_images = []
    for image in images:
        canalized_image = np.ones(
            (image.shape[0], image.shape[1], num_channels))
        canalized_image[:image.shape[0], :image.shape[1], :image.shape[2]] = \
            image / 255
        rotated_images.append(
            rotate(canalized_image, angle=np.random.randint(-30, 30)))

    area = sorted([(image.shape[0] * image.shape[1], image.shape) for image in
                   rotated_images])[0][0]
    resized_images = []
    # Calcular el tamaño optimizado para cada imagen
    for image in rotated_images:
        proportion = image.shape[0] / image.shape[1]
        result = fsolve(lambda x: equations(x, proportion, area),
                        np.array([image.shape[0], image.shape[1]]))
        resized = transform.resize(np.array(image), result)
        resized_images.append(resized)

    np.random.shuffle(resized_images)
    collage = resized_images[0]

    # Combinar imágenes para formar el collage
    for resized_image in resized_images[1:]:
        side = np.random.choice([0, 1])
        if side == 0:
            # Añadir imagen en el lado derecho
            collage = combine_images_horizontal(collage, resized_image)
        else:
            # Añadir imagen en la parte inferior
            collage = combine_images_vertical(collage, resized_image)

    # Color de fondo
    hex_color = fondo_color.lstrip('#')

    # Convertir el valor hexadecimal a RGB
    rgb_color = tuple(int(hex_color[i:i + 2], 16) / 255 for i in (0, 2, 4))

    # Crear un fondo del mismo tamaño que la imagen original
    fondo = np.ones((collage.shape[0], collage.shape[1], 4)) * [*rgb_color, 1]

    # Reemplazar píxeles negros con el fondo
    collage = np.where(collage[:, :, :3] == 0, fondo[:, :, :3],
                       collage[:, :, :3])

    return np.clip(collage, 0., 1.)


def combine_images_horizontal(image1, image2):
    """
    Combina dos imágenes horizontalmente.

    Args:
        image1 (np.ndarray): Primera imagen.
        image2 (np.ndarray): Segunda imagen.

    Returns:
        np.ndarray: Imagen combinada.
    """
    diff = image1.shape[0] - image2.shape[0]

    # Ajustar el tamaño de la imagen más pequeña
    if diff > 0:
        image2 = add_padding(image2, diff, 0)
    elif diff < 0:
        diff = -diff
        image1 = add_padding(image1, diff, 0)

    # Concatenar imágenes horizontalmente
    collage = np.hstack((image1, image2))

    return collage


def combine_images_vertical(image1, image2):
    """
    Combina dos imágenes verticalmente.

    Args:
        image1 (np.ndarray): Primera imagen.
        image2 (np.ndarray): Segunda imagen.

    Returns:
        np.ndarray: Imagen combinada.
    """
    diff = image1.shape[1] - image2.shape[1]

    # Ajustar el tamaño de la imagen más pequeña
    if diff > 0:
        image2 = add_padding(image2, diff, 1)
    elif diff < 0:
        diff = -diff
        image1 = add_padding(image1, diff, 1)

    # Concatenar imágenes verticalmente
    collage = np.vstack((image1, image2))

    return collage


def equations(vars, proportion, area):
    """
    Ecuaciones utilizadas para calcular el tamaño de las imágenes.

    Args:
        vars (list): Lista de variables [x, y].
        proportion (float): Proporción de la imagen.
        area (int): Área de la imagen.

    Returns:
        list: Lista de ecuaciones [eq1, eq2].
    """
    x, y = vars
    eq1 = x / y - proportion
    eq2 = x * y - area
    return [eq1, eq2]


def add_padding(image, diff, axis):
    """
    Añade relleno a la imagen según la dirección y el lado especificados.

    Args:
        image (np.ndarray): Imagen a la que se le agregará el relleno.
        diff (int): Diferencia en tamaño con la otra imagen.
        axis (int): Eje a lo largo del cual se agrega el relleno.

    Returns:
        np.ndarray: Imagen con relleno agregado.
    """
    padding_start = np.zeros(
        (
            *[image.shape[i] if i != axis else math.ceil(diff / 2) for i in
              range(len(image.shape) - 1)], 4
        )
    )

    padding_end = np.zeros(
        (
            *[image.shape[i] if i != axis else math.floor(diff / 2) for i in
              range(len(image.shape) - 1)], 4
        )
    )
    return np.concatenate((padding_start, image, padding_end), axis=axis)


if __name__ == "__main__":
    main()
