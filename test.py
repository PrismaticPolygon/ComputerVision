import cv2
import numpy as np
import re
import sys

image = (512, 1392)

with open("dataset/poses/00.txt") as pose_00:

    data = pose_00.read().split(" ")

    print(len(data))    # 49952

    # 112 × 446 is closest.

    # 1392 x 512 = 712704   # Hm. No direct correlation. But it'd be a scale factor.

    print(data)

    # 49952

# And we can easily convert...
# Is this any image?

def load_pfm(file):
  color = None
  width = None
  height = None
  scale = None
  endian = None

  header = file.readline().decode('utf-8').rstrip()
  if header == 'PF':
    color = True
  elif header == 'Pf':
    color = False
  else:
    raise Exception('Not a PFM file.')

  dimensions = [int(x) for x in str(file.readline()).split(" ")]

  print(dimensions)

  # dim_match = re.match(r'^(\d+)\s(\d+)\s$', str(file.readline()))

  #
  #
  # if dim_match:
  #   width, height = map(int, dim_match.groups())
  # else:
  #   raise Exception('Malformed PFM header.')

  scale = float(file.readline().rstrip())
  if scale < 0: # little-endian
    endian = '<'
    scale = -scale
  else:
    endian = '>' # big-endian

  data = np.fromfile(file, endian + 'f')
  shape = (height, width, 3) if color else (height, width)
  return np.flipud(np.reshape(data, shape)), scale

'''
Save a Numpy array to a PFM file.
'''
def save_pfm(file, image, scale = 1):
  color = None

  if image.dtype.name != 'float32':
    raise Exception('Image dtype must be float32.')

  if len(image.shape) == 3 and image.shape[2] == 3: # color image
    color = True
  elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1: # greyscale
    color = False
  else:
    raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

  file.write('PF\n' if color else 'Pf\n')
  file.write('%d %d\n' % (image.shape[1], image.shape[0]))

  endian = image.dtype.byteorder

  if endian == '<' or endian == '=' and sys.byteorder == 'little':
    scale = -scale

  file.write('%f\n' % scale)

  image.tofile(file)

def get_grayscale(filename, max_val_pct=0.1):
    """
    Handle any infinite values, set those equal to largest value + largest value * max_val_pct
    DocTest .pfm from http://vision.middlebury.edu/stereo/data/scenes2014/datasets/Adirondack-perfect/
    :param filename: str
    :param max_val_pct: float
    :return: numpy.array
    >>> grayscale = get_grayscale('images/Adirondack-perfect/disp1.pfm')
    >>> grayscale = cv2.resize(grayscale, dsize=(0, 0), fx=0.2, fy=0.2)
    >>> cv2.imshow('image', cv2.applyColorMap(grayscale, cv2.COLORMAP_JET))
    >>> cv2.waitKey(0)
    """
    data, _ = load_pfm(open(filename, 'rb'))
    data = np.where(data == np.inf, -1, data)
    max_val = np.max(data)
    max_val += max_val * max_val_pct
    data = np.where(data == -1, max_val, data)
    return cv2.normalize(data, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)



asd = get_grayscale("disp1.pfm")

print(asd)