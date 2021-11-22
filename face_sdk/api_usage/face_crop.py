"""
@author: JiXuan Xu, Jun Wang
@date: 20201015
@contact: jun21wangustc@gmail.com 
"""
import os
import sys
sys.path.append('.')
import argparse

import cv2

from core.image_cropper.arcface_cropper.FaceRecImageCropper import FaceRecImageCropper

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--image_root', type=str, required=True)
    parser.add_argument('--lmk_file', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()

    face_cropper = FaceRecImageCropper()

    with open(args.lmk_file, 'r') as f:
        for line in f.read().strip().split('\n'):

            line_sp = line.strip().split(" ")
            image_path = os.path.join(args.image_root, line_sp[0])
            landmarks_str = line_sp[1:]
            landmarks = [float(num) for num in landmarks_str]

            image = cv2.imread(image_path)
            cropped_image = face_cropper.crop_image_by_mat(image, landmarks)

            output_image_path = os.path.join(args.output_dir, line_sp[0])
            output_dir = os.path.dirname(output_image_path)
            os.makedirs(output_dir, exist_ok=True)

            cv2.imwrite(output_image_path, cropped_image)
