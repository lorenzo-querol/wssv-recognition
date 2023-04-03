import albumentations as A
from alive_progress import alive_bar
import numpy as np

transform = A.Compose([
    A.HorizontalFlip(),
    A.VerticalFlip(),
    A.Rotate(),
    A.GaussianBlur()
])


def augment_images(images, labels, num_aug=5):
    augmented_images = []
    augmented_labels = []

    with alive_bar(len(images), title="Augmenting Images", bar='smooth', spinner=None) as bar:
        for image, label in images:
            for i in range(num_aug):
                augmented = transform(image=image)["image"]
                augmented_images.append(augmented)
                augmented_labels.append(label)
            bar()

    zipped = zip(augmented_images, augmented_labels)

    new_dataset = np.vstack((zip(images), zipped))
    new_labels = np.hstack((labels, augmented_labels))

    return new_dataset, new_labels
