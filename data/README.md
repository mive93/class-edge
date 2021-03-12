# Configuration YAML file explanation:

You can check the yaml files in this folder for reference.

## General parameters:
```
* net: yolo4_berkeley_fp16.rt
* type: y
* width: 960
* height: 540
* classes: 10
* tif: ../data/masa_map.tif
* password: ""
* filter: 1
* [stream: 1]
* [record: 0]
* cameras
```

* net: String value. Neural network used for detection. An .rt file in the tkDNN format should be specified (for example: yolo4_berkeley_fp16.rt, yolo3_berkeley_fp16.rt, yolo4_berkeley_fp32.rt, ex.). There is no need to generate it in advance, if it is missing and the name is correct, class-edge will automatically create it. 
* type: Char value. Default: y. Eligible values: ```y, c, m```. It represents the network family. Accepted type: ```y, c, m``` for the corresponding families: yolo, centernet, modilenet. 
* width : Integer value. Default: 960. Width of the image class-edge will work onto (resize, undistortion and homography). Be careful: it does not necessarily correspond with the size used for calibration!
* height: Integer value. Default: 540. Height of the image class-edge will work onto (resize, undistortion and homography). Be careful: it does not necessarily correspond with the size used for calibration!
* classes: Integer value. Default: 10. Eligible values: 10, 80. Number of classes of the used network. It will change also the dataset considered. Right now, two options are allowed: 10 (Berkeley dataset) and 80 (COCO dataset). 
* tif: String value. Default: ```../data/masa_map.tif```. Path to the tif file used for homography. If you have none, but you specify the default one class-edge will automatically download the needed map. 
* password: String value. Default: "". Password needed to decrypt the encrypted input streams for the camera
* filter: Integer value. Default: 1. Eligible values: 0, 1. Kind of Kalman filter to use in class-edge. Two values are currently allowed: 0: EKF, 1:UKF.
* [stream]: optional parameter. Integer value. Default: 0. Eligible values: 0, 1. You must set it to 1 if you use a camera stream.
* [record]: optional parameter. Integer value. Default: 0. Eligible values: 0, 1. If you want to record the input video (useful for the camera stream) and the acquisition timestamp of the video capture, you can set this filed to 1. Two types of files are generated: the mp4 file and the txt file with the timestamps (in the build folder). If you run a demo with more input streaming, it will be generated a couple of these files for each stream. 
* cameras: array value. It specifies all the camera inputs handled by class-edge. 

## Camera parameters:
```
* encrypted: 1
* pmatrix: ../data/pmat_new/pmat_07-03-20937_20p.txt
* id: 20937
* input: "U2FsdGVkX1+9iqrN6hqxdsAZrETpIFKr69Qgv46TFI4mlMeNPA/BEEn5OcApDFZP\nNh/AcxPz3SC2rsBEjaEDVAKT6sK66+cwki+MWupx9CY=\n"
* [cameraCalib: ../data/calib_cameras/20937.params]
* [resolution: 800x450]
* [maskFileOrient: ../data/masks_orient/1920-1080_mask_null.jpg]
* [maskfile: ../data/masks/20937_mask.jpg]

```

* encrypted: Integer value. Default: 0. Eligible values: 0, 1. Flag that tells if the input filed is encrypted (1) or not (0).
* pmatrix: String value. Path to the homography matrix for a given camera
* id:  Integer value. Id of the camera that will be used inside class-edge.
* input: String value. Path to the stream of the camera. It can be a path to a file or an RTSP address. It can be encrypted or not. 
* [cameraCalib]: optional parameter. String value. Path to the calibration file for the given camera
* [resolution]: optional parameter. String value. Default: empty string. It is used to set the camera resolution in the RTSP URL. It is an RTSP parameter. 
* [maskFileOrient]: optional parameter. String value. Path to the orientation mask file for the given camera. Currently not used by class-edge.
* [maskfile]: optional parameter. String value. Path to the orientation mask file for the given camera. Currently not used by class-edge.

