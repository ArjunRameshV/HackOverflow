import os
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
import json

import numpy as np
import glob
import random
import os
import time
import ntpath
import numpy as np
from PIL import Image
from os.path import join, exists
from keras.models import model_from_json
## local libs
from utils.data_utils import getPaths, read_and_resize, preprocess, deprocess

UPLOAD_FOLDER = 'F:\Docs\Hackathon\Hackerflow\\'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}
IMAGE_EXTENSIONS = ALLOWED_EXTENSIONS
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

import cv2



@app.route('/')
def upload_file():
    return render_template("upload.html")


count = 23

with open("count_value.txt", "r") as f:
    a = f.readlines()
    print("The file contents are : ")
    print(a)
    count = int(a[0]) + 1

print(count)
@app.route('/upload', methods=['GET', 'POST'])
def load_faces():
    global count
    Files = request.files.getlist("files")
    # ID = request.form.get("personID")
    print(Files)
    imagelist = list()
    num = list()
    F = Files[0]
    print(F.filename)
    if(F.filename == ""):
        return render_template("upload.html")

    if F.filename.rsplit('.', 1)[1].lower() in IMAGE_EXTENSIONS:
        for f in Files:
            if f.filename.rsplit('.', 1)[1].lower() in IMAGE_EXTENSIONS:
                f.save("static/images/"+f.filename)
                path = "static/images/"+f.filename
                # imagelist.append(path)
                # img = cv2.imread(path, 0)
                # cv2.imwrite("static/images/modified_image.jpeg", img)
                
                """ADD the model here"""


                img_path = path
                # from utils.data_utils import get_local_test_data
                # test_paths = getPaths(data_dir)
                # print ("{0} test images are loaded".format(len(test_paths)))

                ## create dir for log and (sampled) validation data
                output_dir = r"F:\Docs\Hackathon\Hackerflow\static\images"
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
                output_file_path = join(output_dir,"processed_"+img_name)

                out_img = gen_img.astype('uint8')
                Image.fromarray(out_img).resize((1280,720)).save(output_file_path)

                del funie_gan_generator





                net = cv2.dnn.readNet("yolov3_training_last.weights", "yolov3_testing.cfg")

                # Name custom object
                classes = ["gate", "obstacle", "bucket"]

                # Images path
                # images_path = glob.glob(r'F:\Docs\Computer Vision\Projects\Object Detection using YOLO\Classification using YOLO\Dataset\Datasets\car\2182.jpg')
                # images_path.extend(glob.glob(r"F:\Docs\Computer Vision\Projects\Object Detection using YOLO\DataSet\bmw10_release\bmw10_ims\Test_dataset\car.jpeg"))
                # images_path.append(r'F:\Docs\Computer Vision\Projects\Object Detection using YOLO\DataSet\bmw10_release\bmw10_ims\Test_dataset\8057028683_a8eb0a2605_z.jpeg')
                images_path = [output_file_path]
                print(images_path)

                layer_names = net.getLayerNames()
                output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
                colors = np.random.uniform(0, 255, size=(len(classes), 3))

                # Insert here the path of your images
                random.shuffle(images_path)
                # loop through all the images
                input_image = cv2.imread(path)
                for img_path in images_path:
                    # Loading image
                    img = cv2.imread(img_path)
                    shape = img.shape
                    shape = shape[0 : 2]
                    shape = list(shape)
                    shape[0], shape[1] = shape[1], shape[0]
                    shape = tuple(shape)
                    print(shape)
                    img = cv2.resize(img, None, fx=0.4, fy=0.4)
                    input_image = cv2.resize(input_image, None, fx=0.4, fy=0.4)
                    height, width, channels = img.shape

                    # Detecting objects
                    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

                    net.setInput(blob)
                    outs = net.forward(output_layers)

                    # Showing informations on the screen
                    class_ids = []
                    confidences = []
                    boxes = []
                    for out in outs:
                        for detection in out:
                            scores = detection[5:]
                            class_id = np.argmax(scores)
                            confidence = scores[class_id]
                            if confidence > 0.5:
                                # Object detected
                                print(class_id)
                                center_x = int(detection[0] * width)
                                center_y = int(detection[1] * height)
                                w = int(detection[2] * width)
                                h = int(detection[3] * height)
                                # Rectangle coordinates
                                x = int(center_x - w / 2)
                                y = int(center_y - h / 2)

                                boxes.append([x, y, w, h])
                                confidences.append(float(confidence))
                                class_ids.append(class_id)

                    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
                    print(indexes)
                    font = cv2.FONT_HERSHEY_PLAIN
                    for i in range(len(boxes)):
                        if i in indexes:
                            x, y, w, h = boxes[i]
                            label = str(classes[class_ids[i]])
                            print(label)
                            if(label == "gate"):
                                color = colors[class_ids[i]]
                                cv2.rectangle(input_image, (x, y), (x + w, y + h), color, 2)
                                cv2.circle(input_image, center = (x + w//2, y + h//2), radius = 5, color = color, thickness = -1)
                                cv2.putText(input_image, "class : " + label + " confidence : " + "{:.2f}".format(confidences[i]), (x, y + 30), font, 1, color, 2)


                    # cv2.imshow("Image", input_image)
                    img = cv2.resize(input_image, shape)
                    print("static/images/modified_image_" + str(count) + ".jpeg")
                    cv2.imwrite("static/images/modified_image_" + str(count) + ".jpeg", input_image)
                    # key = cv2.waitKey(0)

                # cv2.destroyAllWindows()
                
                """Model ends here"""
                
                imagelist.append("static/images/modified_image_" + str(count) + ".jpeg") # this shd contain the path to the final modified image
                h = "GATE FOUND"
                response = json.dumps(
                    h)+str(app.response_class(response=json.dumps(h), status=200, mimetype='application/json'))
                print(response)
        count += 1
        print(count)
        with open("count_value.txt", "w") as f:
            f.write(str(count))
        # Return = {"imagelist": imagelist, "array": num}
        # encodedNumpyData = json.dumps(Return, cls=NumpyArrayEncoder)
        # return Response(encodedNumpyData)
        return render_template("extract.html", results=imagelist)


if(__name__ == "__main__"):
    app.run(debug = True)