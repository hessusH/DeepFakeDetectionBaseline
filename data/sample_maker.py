import os
import cv2
import json
import random
import argparse


def cut_video(filepath, out_dir, label=None):
    counter = 0
    new_annotation = dict()
    cap = cv2.VideoCapture(filepath)
    assert cap.isOpened() is True, f'Error opening video stream or file: {filepath}'
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            if counter % 30 == 0:
                filename = filepath.split('/')[-1][:-4] + str(counter) + '.jpg'
                new_annotation[filename] = int(label == 'FAKE')
                cv2.imwrite(os.path.join(out_dir, filename), frame)
        else:
            cap.release()
        counter += 1
    return new_annotation


def main(input_dir, output_dir, annotation_file):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    assert os.path.exists(annotation_file), 'No annotation file found'
    
    train_annotation = dict()
    if not os.path.exists(os.path.join(output_dir, 'train/')):
        os.mkdir(os.path.join(output_dir, 'train/'))
    
    val_annotation = dict()
    if not os.path.exists(os.path.join(output_dir, 'validation/')):
        os.mkdir(os.path.join(output_dir, 'validation/'))
    
    with open(annotation_file, 'r') as f:
        annotation = json.load(f)
     
    images_list = list(annotation.keys())
    random.shuffle(images_list)
    val_keys = images_list[:30]
    train_keys = images_list[30:]
    
    for key in val_keys:
        new_annotation = cut_video(os.path.join(input_dir, key), os.path.join(output_dir, 'validation/'), annotation[key]['label'])
        val_annotation.update(new_annotation)
        
    with open(os.path.join(os.path.join(output_dir, 'validation/'), 'annotation.json'), 'w') as f:
        json.dump(val_annotation, f)
        
    for key in train_keys:
        new_annotation = cut_video(os.path.join(input_dir, key), os.path.join(output_dir, 'train/'), annotation[key]['label'])
        train_annotation.update(new_annotation)           
    
    with open(os.path.join(os.path.join(output_dir, 'train/'), 'annotation.json'), 'w') as f:
        json.dump(train_annotation, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True, type=str,
                        help='Choose directory containing your dataset')
    parser.add_argument('--output_dir', required=False, type=str,
                        default='./training_data/',
                        help='Choose output directory')
    parser.add_argument('--annotation_file', required=True, type=str, 
                        help='Choose dataset annotation file')
    args = parser.parse_args()
    main(**vars(args))
