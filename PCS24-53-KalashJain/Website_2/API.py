import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image

def load_and_preprocess_image(image_path, target_size=(1000, 1000), num_channels=3):
    if not isinstance(image_path, str):
        raise TypeError("Image path must be a string.")

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at: {image_path}")

    print(f"Original image shape: {image.shape}")

    # Ensure the image has 3 channels
    if image.shape[2] != num_channels:
        print(f"Converting image to {num_channels} channels.")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if image.shape[0] > target_size[0] or image.shape[1] > target_size[1]:
        print(f"Resizing image to target size: {target_size}")
        image = cv2.resize(image, target_size)

    image = image.astype(np.float32)[np.newaxis, ...] / 255.0
    return image


def transfer_style(content_image, style_image, model_path, alpha=0.7):
    # Resize the images to (1000,1000) if greater than (2000 x 2000)
    size_threshold = 2000
    resizing_shape = (1000, 1000)
    content_shape = content_image.shape
    style_shape = style_image.shape

    resize_content = True if content_shape[0] > size_threshold or content_shape[1] > size_threshold else False
    resize_style = True if style_shape[0] > size_threshold or style_shape[1] > size_threshold else False

    if resize_content:
        print("Content Image bigger than (2000x2000), resizing to (1000x1000)")
        content_image = cv2.resize(
            content_image, (resizing_shape[0], resizing_shape[1]))
        content_image = np.array(content_image)

    if resize_style:
        print("Style Image bigger than (2000x2000), resizing to (1000x1000)")
        style_image = cv2.resize(
            style_image, (resizing_shape[0], resizing_shape[1]))
        style_image = np.array(style_image)

    # Resizing and Normalizing images
    content_image = content_image.astype(np.float32)[np.newaxis, ...] / 255.
    style_image = style_image.astype(np.float32)[np.newaxis, ...] / 255.

    # Optionally resize the images
    style_image = tf.image.resize(style_image, (256, 256))

    # Loading pre-trained model
    hub_module = hub.load(model_path)

    print("Generating stylized image... Please wait.")
    # Stylize image.
    outputs = hub_module(tf.constant(content_image), tf.constant(style_image))
    stylized_image = outputs[0].numpy()

    # Reshape the stylized image
    stylized_image = stylized_image.reshape(
        stylized_image.shape[1], stylized_image.shape[2], stylized_image.shape[3])

    # Blend the stylized image with the content image
    blended_image = (alpha * stylized_image +
                     (1 - alpha) * content_image) * 255
    blended_image = np.clip(blended_image, 0, 255).astype(np.uint8)
    blended_image = blended_image.squeeze()  # Remove singleton dimensions
    blended_image = cv2.cvtColor(
        blended_image, cv2.COLOR_BGR2RGB)  # Convert to RGB

    # Create and display PIL Image
    pil_image = Image.fromarray(blended_image)
    pil_image.show()

    print("Stylizing completed...")
    return blended_image
