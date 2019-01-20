# Specific-object-recognition
When given different views of an object as input, it can tell us if that specific object is present in a larger picture or not.

## Run
- Please download the weights file for YOLO if don't have it already:

  `wget https://pjreddie.com/media/files/yolov3.weights`
- Here `image1.jpg`, `image2.jpg`, `image3.jpg` are the inputs. These are the front-view, side-view and the other side-view of the query object (laptop in this case) respectively. please feel free to test this with any other object as the query object.

- There shall be another input, that is the test image in which the algorithm has to check for the query object. You can use either `occ_lap.jpg`, `two_laps_demo.jpg` and `white.jpg` as the test image. You can also choose an entirely different image to test. While providing images of an query object of your choice, please make sure that images are focused on the object as much as possible and there are no other objects which might be a disturbance.
- run `python3 yolo_opencv.py` and pass all the three images (`image1.jpg`, `image2.jpg`, `image3.jpg`) and you will get `image_yolo1.jpg`, `image_yolo2.jpg` and `image_yolo3.jpg` as the outputs. These are the crop fitted images of the query object. These images will be used for checking with the test image.
- run `python3 specific_recog.py` and you will get the output as whether the exact query object is inside the test image or not.

## outputs

Here are the outputs when different test images are given as input for the query object laptop, which is present in `image_yolo1.jpg`, `image_yolo2.jpg` and `image_yolo3.jpg`:

for `two_laps_demo.jpg`:

![output_two_laps_demo](https://github.com/nvinayvarma189/Specific-object-recognition/blob/master/output_imgs/output_two_laps_demo.png)

for `white.jpg`:

![output_white](https://github.com/nvinayvarma189/Specific-object-recognition/blob/master/output_imgs/output_white.png)

**The good part is that it is also occlusion resistant and is producing accurate results in most cases.**

for `occ_lap.jpg`:

![output_occ_lap](https://github.com/nvinayvarma189/Specific-object-recognition/blob/master/output_imgs/output_occ_lap.png)
