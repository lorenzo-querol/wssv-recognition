import albumentations as A
from alive_progress import alive_bar

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
            augmented_images.append(image)
            augmented_labels.append(label)
            
            for i in range(num_aug):
                augmented = transform(image=image)["image"]
                augmented_images.append(augmented)
                augmented_labels.append(label)
            bar()

    new_dataset = list(zip(augmented_images, augmented_labels))
    
    return new_dataset
