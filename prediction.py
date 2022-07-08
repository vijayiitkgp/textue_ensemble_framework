import numpy as np
from IPython.display import display
from PIL import Image


def predict(tf_rep):
  print('Image 1:')
  img = Image.open('banded_0002.jpg').resize((1,3,224,224)).convert('L')
  display(img)
  output = tf_rep.run(np.asarray(img, dtype=np.float32))
  print('The texture is classified as ', np.argmax(output))
  # print('Image 2:')
  # img = Image.open('three.png').resize((28, 28)).convert('L')
  # display(img)
  # output = tf_rep.run(np.asarray(img, dtype=np.float32)[np.newaxis, np.newaxis, :, :])
  # print('The digit is classified as ', np.argmax(output)) 