import os
import cv2
import json
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


def main(input_dir, output_dir, dataset_type, annotation_file):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    assert (annotation_file is not None and dataset_type == 'train') or \
           (annotation_file is None and dataset_type == 'test'), 'Arguments mismatch'
    full_annotation = dict()
    if annotation_file is not None:
        with open(os.path.join(input_dir, annotation_file), 'r') as f:
            annotation = json.load(f)
        for key in annotation.keys():
            new_annotation = cut_video(os.path.join(input_dir, key), output_dir, annotation[key]['label'])
            full_annotation.update(new_annotation)
            print(full_annotation)
    else:
        files = os.listdir(input_dir)
        for file in files:
            new_annotation = cut_video(os.path.join(input_dir, file), output_dir)
            full_annotation.update(new_annotation)
           
    with open(os.path.join(output_dir, 'annotation.json'), 'w') as f:
        json.dump(full_annotation, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=False, type=str,
                        default='./test_videos/',
                        help='Choose directory containing your dataset')
    parser.add_argument('--output_dir', required=False, type=str,
                        default='./test/',
                        help='Choose output directory')
    parser.add_argument('--dataset_type', required=False, type=str, default='test',
                        help='Choose dataset type')
    parser.add_argument('--annotation_file', required=False, type=str, default=None,
                        help='Choose dataset annotation file')
    args = parser.parse_args()
    main(**vars(args))
