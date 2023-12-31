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


# Definir la función principal
def main():
    """
    Función principal que muestra la interfaz de usuario para crear collages.
    """
    # Título de la aplicación
    st.title("Collage Generator")

    # Seleccionar archivos de imagen desde el usuario
    uploaded_files = st.file_uploader(
        "Choose images to create collage",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True)

    # Validar el número máximo de archivos cargados
    if uploaded_files is not None:
        if len(uploaded_files) > 5:
            st.warning(f"Select a maximum of {5} files.")
            uploaded_files = uploaded_files[:5]

    if uploaded_files:
        # Sección de imágenes seleccionadas
        st.header("Selected Images")

        # Leer y mostrar las imágenes seleccionadas
        images = [io.imread(file) for file in uploaded_files]
        st.image(images,
                 caption=[f"Image {i + 1}" for i in range(len(images))],
                 width=100)

        # Color picker para el fondo del collage
        fondo_color = st.color_picker("Select background color")
        collage_button = st.button("Generate Collage")

        # Generar el collage al hacer clic en el botón correspondiente
        if collage_button:
            collage = generate_collage(images, fondo_color)
            st.session_state['collage'] = collage

        # Mostrar el collage generado y el botón de descarga
        if 'collage' in st.session_state:
            st.header("Generated Collage")
            st.image(st.session_state['collage'],
                     caption="Generated Collage", )

            # Convertir la imagen a BytesIO y agregar botón de descarga
            buffer = BytesIO()
            data = img_as_ubyte(st.session_state['collage'])
            io.imsave(buffer, data, type='png', format='png')
            buffer.seek(0)
            st.download_button("Download", data=buffer,
                               file_name='collage.png', )


# Función para generar el collage
def generate_collage(images, fondo_color):
    """
    Genera un collage a partir de una lista de imágenes y un color de fondo.

    Args:
        images (list): Lista de imágenes.
        fondo_color (str): Color de fondo en formato hexadecimal.

    Returns:
        np.ndarray: Imagen de collage generada.
    """

    # Determinar el número máximo de canales de color entre las imágenes
    num_channels = max(image.shape[2] for image in images)

    # Rotar cada imagen de forma aleatoria
    rotated_images = []
    for image in images:
        # Normalizar y agregar canales para imágenes con diferente número de
        # canales
        canalized_image = np.ones(
            (image.shape[0], image.shape[1], num_channels))
        canalized_image[:image.shape[0], :image.shape[1], :image.shape[2]] = \
            image / 255
        # Rotar la imagen y agregarla a la lista de imágenes rotadas
        rotated_images.append(
            rotate(canalized_image, angle=np.random.randint(-30, 30)))

    # Calcular el área de cada imagen y seleccionar la más pequeña
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

    # Barajar aleatoriamente las imágenes redimensionadas
    np.random.shuffle(resized_images)
    # Tomar la primera imagen como base del collage
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
    rgb_color = list(int(hex_color[i:i + 2], 16) / 255 for i in (0, 2, 4))

    # Crear un fondo del mismo tamaño que la imagen original
    fondo = (np.ones((collage.shape[0], collage.shape[1], collage.shape[
        2]))) * ([*rgb_color, 1] if collage.shape[2] == 4 else rgb_color)

    # Reemplazar píxeles negros con el fondo
    collage = np.where(collage[:, :, :3] == 0, fondo[:, :, :3],
                       collage[:, :, :3])

    return np.clip(collage, 0., 1.)


# Funciones para combinar imágenes horizontal y verticalmente
def combine_images_horizontal(image1, image2):
    """
    Combina dos imágenes horizontalmente.

    Args:
        image1 (np.ndarray): Primera imagen.
        image2 (np.ndarray): Segunda imagen.

    Returns:
        np.ndarray: Imagen combinada.
    """
    # Calcular la diferencia en altura entre las dos imágenes
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
    # Calcular la diferencia en anchura entre las dos imágenes
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


# Función para las ecuaciones utilizadas en el redimensionamiento
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


# Función para agregar relleno a una imagen
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
    # Crear matrices de relleno en el inicio y final de la imagen
    padding_start = np.zeros(
        tuple(
            [image.shape[i] if i != axis else math.ceil(diff / 2) for i in
             range(len(image.shape))]
        )
    )

    padding_end = np.zeros(
        tuple(
            [image.shape[i] if i != axis else math.floor(diff / 2) for i in
             range(len(image.shape))]
        )
    )
    # Concatenar matrices de relleno y la imagen original a lo largo del eje
    # especificado
    return np.concatenate((padding_start, image, padding_end), axis=axis)


# Verificar si el script se está ejecutando como un programa independiente
if __name__ == "__main__":
    main()
