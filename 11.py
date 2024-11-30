print('Lesson 11: AI model. Threat image')

# cmd command for install the next libraries:
#   pip install opencv-python
#   pip install pillow

import cv2
from PIL import Image

image_cat_path = 'cat11.jpeg'
image_sunglasses_path = 'glasses10.png'
image_cat = cv2.imread(image_cat_path)

cat_face_handler = cv2.CascadeClassifier('haarcascade_frontalcatface_extended.xml')

# get cooridinates of pattern
cat_face_coordinates = cat_face_handler.detectMultiScale(image_cat)
# print('cat face coordinates', cat_face_coordinates)

for (x, y, w, h) in cat_face_coordinates:
    cv2.rectangle(image_cat, (x, y), (x + w, y + h), (255, 0, 0), 3)

cv2.imshow('Bob cat', image_cat)

# prepare image for union
cat = Image.open(image_cat_path)
sun_glasses = Image.open(image_sunglasses_path)

cat = cat.convert('RGB')
sun_glasses = sun_glasses.convert('RGB')

(x, y, w, h) = cat_face_coordinates[0]
sun_glasses = sun_glasses.resize((w, int(h/3)))

# union of two images
cat.paste(sun_glasses, (x, int(y + h/4)))
cat.save('cat_with_sunglasses.jpeg')

# save and read the new image after threatment by AI image model
cat_with_sunglasses = cv2.imread('cat_with_sunglasses.jpeg')
cv2.imshow('Bob cat with sunglasses', cat_with_sunglasses)

cv2.waitKey()

