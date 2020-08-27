from imgaug import augmenters as iaa
import glob
import os
from PIL import Image


seq = iaa.Sequential([
    iaa.Crop(px=(0, 16)), # crop images from each side by 0 to 16px (randomly chosen)
    iaa.Fliplr(0.5), # horizontally flip 50% of the images
    iaa.GaussianBlur(sigma=(0, 3.0)) # blur images with a sigma of 0 to 3.0
])

def main():
    aug()

def aug():
    os.chdir('./six_dof_1cm5deg_text')
    for picture in range(sorted(glob.glob('*.png'))):
    # 'images' should be either a 4D numpy array of shape (N, height, width, channels)
    # or a list of 3D numpy arrays, each having shape (height, width, channels).
    # Grayscale images must have shape (height, width, 1) each.
    # All images must have numpy's dtype uint8. Values are expected to be in
    # range 0-255.
        images = Image.open(picture)
        images_aug = seq.augment_images(images)  # done by the library
        j = Image.fromarray(images_aug, mode='RGB')
        j.save(picture)