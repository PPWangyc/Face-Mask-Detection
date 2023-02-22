import os
data_path="/home/yanchen/Data/breathe"
# get all mp4 files under the data_path
file_list = []
for root, dirs, files in os.walk(data_path):
    for file in files:
        if file.endswith(".mp4"):
            file_list.append(os.path.join(root, file))

# read processed.txt
processed_file = open("processed.txt", "r")
processed_list = processed_file.readlines()
processed_file.close()

for file in file_list:
    file=file + "\n"
    if file not in processed_list:
        file=file.strip()
        file=file.replace(" ", "\ ")
        print(file)
        # run the detect_mask_video.py
        os.system("python3 detect_mask_video.py --video " + file)
        # write the file name to processed.txt
        processed_file = open("processed.txt", "a")
        file = file.replace("\ ", " ")
        processed_file.write(file + '\n')
        # delete the file
        # os.remove(file)
