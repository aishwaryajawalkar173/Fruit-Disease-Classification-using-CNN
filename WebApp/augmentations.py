import albumentations as A
import os
import cv2

path = input('Input the folder:')
images = os.listdir(path)
print(images)
for img in images:
    image = cv2.imread(img)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    transform = A.Compose([
        A.RandomCrop(width=512, height=512),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
    ])

    transformed = transform(image=image)
    transformed_image = transformed["image"]

    cv2.imwrite('augmented_'+img, transformed_image)


