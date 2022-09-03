#!/usr/bin/env python3
from onnxruntime.quantization import quantize_dynamic, QuantType
model_fp32 = "models/crestereo_combined_iter5_240x320.onnx"
model_quant = "models/crestereo_combined_iter5_240x320_quant.onnx"
augmented_model_path = "models/crestereo_combined_iter5_240x320_augmented_model.onnx"
# quantized_model = quantize_dynamic(model_fp32, model_quant, weight_type=QuantType.QUInt8)

from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType, create_calibrator, write_calibration_table
import os
import numpy as np
import cv2 as cv

def hwc2chw(img_input):
    img_input = img_input.transpose(2, 0, 1)
    # img_input = img_input[np.newaxis,:,:,:]    
    return img_input

def prepare_input(left_img, right_img, input_width, input_height):
    left_img = cv.resize(left_img, (input_width, input_height))
    right_img = cv.resize(right_img, (input_width,input_height))
    init_left_img = cv.resize(left_img, (input_width//2, input_height//2))
    init_right_img = cv.resize(right_img, (input_width//2, input_height//2))
    left_img = hwc2chw(left_img)
    right_img = hwc2chw(right_img)
    init_left_img = hwc2chw(init_left_img)
    init_right_img = hwc2chw(init_right_img)
    # left_img = np.expand_dims(left_img,2)
    # right_img = np.expand_dims(right_img,2)
    # init_left_img = np.expand_dims(init_left_img,2)
    # init_right_img = np.expand_dims(init_right_img,2)

    return  np.expand_dims(init_left_img, 0).astype(np.float32), np.expand_dims(init_right_img, 0).astype(np.float32), \
         np.expand_dims(left_img, 0).astype(np.float32), np.expand_dims(right_img, 0).astype(np.float32)

def preprocess_image(image_path, image_path_r, height, width, channels=3):
    img_l = cv.imread(image_path, cv.IMREAD_COLOR)
    img_r = cv.imread(image_path_r, cv.IMREAD_COLOR)
    data = prepare_input(img_l, img_r, width, height)
    return data
    
def preprocess_func(images_folder, height, width, size_limit=0):
    image_names = os.listdir(images_folder)
    if size_limit > 0 and len(image_names) >= size_limit:
        batch_filenames = [image_names[i] for i in range(size_limit)]
    else:
        batch_filenames = image_names
    unconcatenated_batch_data = []

    for image_name in batch_filenames:
        if "left" in image_name:
            image_filepath_left = images_folder + '/' + image_name
            image_filepath_right = images_folder + '/' + image_name.replace("left", "right")
            image_data = preprocess_image(image_filepath_left, image_filepath_right, height, width)
            unconcatenated_batch_data.append(image_data)
    # batch_data = np.concatenate(np.expand_dims(unconcatenated_batch_data, axis=0), axis=0)
    return unconcatenated_batch_data


class StereoDataReader(CalibrationDataReader):
    def __init__(self, calibration_image_folder):
        self.image_folder = calibration_image_folder
        self.preprocess_flag = True
        self.enum_data_dicts = []
        self.datasize = 0
        self.image_height = 240
        self.image_width = 320

    def get_next(self):
        if self.preprocess_flag:
            self.preprocess_flag = False
            nhwc_data_list = preprocess_func(self.image_folder, self.image_height, self.image_width, size_limit=0)
            self.datasize = len(nhwc_data_list)
            self.enum_data_dicts = iter([{'init_left': nhwc_data[0], 'init_right': nhwc_data[1], 'next_left': nhwc_data[2], 'next_right': nhwc_data[3] } for nhwc_data in nhwc_data_list])
        return next(self.enum_data_dicts, None)

calibration_data_folder = "/home/dji/output/stereo_calib"
dr = StereoDataReader(calibration_data_folder)
# print("Static quantization...")
# quantize_static(model_fp32, model_quant, dr, weight_type=QuantType.QInt8, activation_type=QuantType.QInt8)
# print('ONNX full precision model size (MB):', os.path.getsize(model_fp32)/(1024*1024))
# print('ONNX quantized model size (MB):', os.path.getsize(model_quant)/(1024*1024))

#Quant for TensorRT
print("Quantization for TensorRT...")
import os
os.environ["ORT_TENSORRT_FP16_ENABLE"] = "1"  # Enable FP16 precision
os.environ["ORT_TENSORRT_INT8_ENABLE"] = "1"  # Enable INT8 precision
os.environ["ORT_TENSORRT_INT8_CALIBRATION_TABLE_NAME"] = "calibration.flatbuffers"  # Calibration table name
os.environ["ORT_TENSORRT_ENGINE_CACHE_ENABLE"] = "1"  # Enable engine caching
calibrator = create_calibrator(model_fp32, [], augmented_model_path=augmented_model_path)
calibrator.set_execution_providers(["CUDAExecutionProvider"])      
calibrator.collect_data(dr)
write_calibration_table(calibrator.compute_range())