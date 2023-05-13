import argparse
import cv2
import os
import albumentations as A
import glob
import shutil


def augment_images(
    data_dir: str, dest_dir: str, num_aug: str, img_ext: str, classes: list[str]
):
    """
    Augment images in a directory.

    Args:
    - data_dir (str): Path to the directory containing the input images.
    - dest_dir (str): Path to the directory where the augmented images will be saved.
    - num_aug (int): Number of augmented images to generate per input image. Default is 5.
    - img_ext (str): Extension of the input images. (e.g. .jpg)
    - classes (list): List of classes to augment. (Separate classes with a space, e.g. --classes healthy wssv)
    """
    transforms = A.Compose(
        [
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.Rotate(interpolation=cv2.INTER_CUBIC),
            A.GaussianBlur(),
            A.RandomBrightnessContrast(),
            A.ImageCompression(quality_lower=85, quality_upper=100),
        ]
    )

    if not os.path.isdir(data_dir):
        raise ValueError(f"{data_dir} is not a directory.")

    # Reset destination directory
    if os.path.exists(dest_dir):
        shutil.rmtree(dest_dir)

    for c in classes:
        os.makedirs(os.path.join(dest_dir, c))

    image_paths = glob.glob(f"{data_dir}/*/*{img_ext}", recursive=True)

    if not image_paths:
        raise ValueError(f"No images found in {data_dir} with extension {img_ext}.")

    for idx, image_path in enumerate(image_paths):
        image = cv2.imread(image_path)
        label = image_path.split(os.path.sep)[-2]

        # Save original image
        filename = os.path.join(dest_dir, label, str(idx)) + img_ext
        cv2.imwrite(filename, image)

        # Save augmented images
        for i in range(int(num_aug)):
            augmented_image = transforms(image=image)["image"]

            filename = os.path.join(dest_dir, label, str(idx))
            augmented_filename = filename + ("_augmented_%d" % (i + 1)) + img_ext

            cv2.imwrite(augmented_filename, augmented_image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Augment images in a directory.")
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to the directory containing the input images.",
    )
    parser.add_argument(
        "--dest-dir",
        type=str,
        required=True,
        default="augmented",
        help="Path to the directory where the augmented images will be saved. Default is 'augmented'.",
    )
    parser.add_argument(
        "--num-aug",
        type=int,
        default=5,
        help="Number of augmented images to generate per input image. Default is 5.",
    )
    parser.add_argument(
        "--img-ext",
        type=str,
        default=".jpg",
        help="Extension of the input images. Default is '.jpg'.",
    )
    parser.add_argument(
        "--classes",
        type=str,
        nargs="+",
        default=["healthy", "wssv"],
        help="List of classes to augment. Default is ['healthy', 'wssv']",
    )
    args = parser.parse_args()

    try:
        augment_images(
            args.data_dir, args.dest_dir, args.num_aug, args.img_ext, args.classes
        )
    except Exception as e:
        print(f"Error: {e}")
