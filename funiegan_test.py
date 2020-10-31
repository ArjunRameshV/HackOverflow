import os
import time
import ntpath
import numpy as np
from PIL import Image
from os.path import join, exists
from keras.models import model_from_json
## local libs
from utils.data_utils import getPaths, read_and_resize, preprocess, deprocess

## for testing arbitrary local data
img_path = r"F:\Docs\Hackathon\Hackerflow\Dataset\test\test_1.jpeg"
# from utils.data_utils import get_local_test_data
# test_paths = getPaths(data_dir)
# print ("{0} test images are loaded".format(len(test_paths)))

## create dir for log and (sampled) validation data
output_dir = r"F:\Docs\Hackathon\Hackerflow\\"
if not exists("samples_dir"): os.makedirs("samples_dir")

checkpoint_dir  = 'models/gen_p/'
model_name_by_epoch = "model_15320_" 


model_h5 = checkpoint_dir + model_name_by_epoch + ".h5"  
model_json = checkpoint_dir + model_name_by_epoch + ".json"

assert (exists(model_h5) and exists(model_json))

# load model
with open(model_json, "r") as json_file:
    loaded_model_json = json_file.read()

funie_gan_generator = model_from_json(loaded_model_json)
funie_gan_generator.load_weights(model_h5)
print("\nLoaded data and model")

times = []; s = time.time()

inp_img = read_and_resize(img_path, (256, 256))
im = preprocess(inp_img)
im = np.expand_dims(im, axis=0) # (1,256,256,3)

s = time.time()
gen = funie_gan_generator.predict(im)
gen_img = deprocess(gen)[0]
tot = time.time()-s
times.append(tot)

img_name = ntpath.basename(img_path)
output_file_path = join(output_dir,img_name)

out_img = gen_img.astype('uint8')
Image.fromarray(out_img).resize((1280,720)).save(output_file_path)

del funie_gan_generator

#Call YOLO on image 