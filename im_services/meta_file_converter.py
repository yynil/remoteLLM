import os
import json
import argparse
import shutil

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, help='output file')
    parser.add_argument('--image_dir', type=str, help='image dir')
    args = parser.parse_args()
    image_dir = args.image_dir
    output_dir = args.output_dir
    #put all file end with .jpg in image_dir into image_files
    with open(os.path.join(output_dir,"metadata.jsonl"), 'w',encoding="UTF-8") as f:
        for root, dirs, files in os.walk(image_dir):
            for file in files:
                if file.endswith('.jpg') or file.endswith('.png') or file.endswith('.gif'):
                    full_file_path = os.path.join(root, file)
                    target_file_path = os.path.join(output_dir, file)
                    pure_file_name = os.path.splitext(file)[0]
                    text_file_name = pure_file_name+'.txt'
                    text_file_path = os.path.join(image_dir, text_file_name)
                    if os.path.exists(text_file_path):
                        #read text from text_file_path
                        with open(text_file_path, 'r',encoding="UTF-8") as text_file:
                            tags = text_file.read().split(',')
                            tags = ','.join([tag.strip().replace('_',' ') for tag in tags])
                            jsonStr = json.dumps({"file_name": file, "text": tags}, ensure_ascii=False)
                            f.write(jsonStr + '\n')
                        #copy image file to output_dir
                        shutil.copy(full_file_path, target_file_path)
                    else:
                        print("no text file for image file: ", text_file_path)
                


