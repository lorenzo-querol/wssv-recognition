# -*- coding: utf-8 -*-

import albumentations as A
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
    
    for image, label in images:
        for i in range(num_aug):
            augmented = transform(image=image)["image"]
            augmented_images.append(augmented)
            augmented_labels.append(label)
            
            # filename = img_fn.replace('.jpg','') # Get image base filename
            # new_filename = base_fn + ('_aug%d' % (i+1)) + IMG_EXTENSION # Append "aug#" to filename
            # img_aug_bgr1 = cv2.cvtColor(img_aug1, cv2.COLOR_RGB2BGR) # Re-color to BGR from RGB\
    
    zipped = list(zip(augmented_images, augmented_labels))            
    new_dataset = np.vstack((images, zipped))
    
    new_labels = np.hstack((labels, augmented_labels))
    
    return new_dataset, new_labels