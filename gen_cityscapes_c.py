import os
import random
from imagecorruptions import corrupt
import cv2
corruptions = [
    'gaussian_noise','shot_noise', 'impulse_noise', 'defocus_blur',
    'glass_blur', 'motion_blur', 'zoom_blur',
    'snow', 'frost', 'fog',
    'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression'
]

def process_images(base_path, output_base_path, corruption_name):
    for subdir, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith(".png"):
                file_path = os.path.join(subdir, file)
                img = cv2.imread(file_path)
                print(file_path)
                if img is not None:
                    for a in range(1,6):
                        corrupted_img = corrupt(img,corruption_name=corruption_name, severity=a)
                        
                        relative_subdir = os.path.relpath(subdir, base_path)
                        output_dir = os.path.join(output_base_path,corruption_name,str(a),relative_subdir)
                        os.makedirs(output_dir, exist_ok=True)
                        
                        output_file_path = os.path.join(output_dir, file)
                        cv2.imwrite(output_file_path, corrupted_img)

base_path = '/home/xuxiaoran/datasets/cityscapes/leftImg8bit/val'
output_base_path = '/home/xuxiaoran/datasets/cityscapes-c'

for corruption_name in corruptions:
    process_images(base_path, output_base_path, corruption_name)