import os
from shutil import copy

# # Copy file from YOLOv5 -> YOLOv7
for d1 in ['hairnet', 'handgun_and_rifle', 'license_plate', 'mask', 'person', 'sack', 'safety_hardhat_vest']:
    for d2 in ['train', 'test', 'validation']:
        idx = 0
        for i, f in enumerate(os.listdir(f"./dataset/YOLOv5/{d1}/images/{d2}")):
            if os.path.exists(f"./dataset/YOLOv5/{d1}/images/{d2}/{f}") and os.path.exists(f"./dataset/YOLOv5/{d1}/labels/{d2}/{f[:-4]}.txt"):
                copy(f"./dataset/YOLOv5/{d1}/images/{d2}/{f}", f"./dataset/YOLOv7/{d2}/images/{d1}_{d2}_{idx}.jpg")
                copy(f"./dataset/YOLOv5/{d1}/labels/{d2}/{f[:-4]}.txt", f"./dataset/YOLOv7/{d2}/labels/{d1}_{d2}_{idx}.txt")
                idx += 1
            else:
                print(f"./dataset/YOLOv5/{d1}/images/{d2}/{f}")
                print(os.path.exists(f"./dataset/YOLOv5/{d1}/images/{d2}/{f}"))
                print(f"./dataset/YOLOv5/{d1}/labels/{d2}/{f[:-4]}.txt")
                print(os.path.exists(f"./dataset/YOLOv5/{d1}/labels/{d2}/{f[:-4]}.txt"))


# Merge classes YOLOv7
# 0. hairnet
# 1. handgun
# 2. hardhat
# 3. license_plate
# 4. mask_weared_incorrect
# 5. person
# 6. rifle
# 7. sack
# 8. vest
# 9. with_mask
# 10. without_mask

map = {'hairnet': {0: 0}, 'handgun': {0: 1, 1: 6}, 'safety': {0: 2, 1: 8}, 'license': {0: 3}, 'mask': {0: 9, 1: 10, 2: 4}, 'person': {0: 5}, 'sack': {0: 7}}
for d1 in ['train', 'test', 'validation']:
    for i, f in enumerate(os.listdir(f"./dataset/YOLOv7/{d1}/labels")):
        key = f.split('_')[0]
        out = []
        with open(f"./dataset/YOLOv7/{d1}/labels/{f}", 'r') as f1:
            lines = f1.readlines()
        for line in lines:
            _temp = line.split(' ')
            try:
                _temp[0] = str(map[key][int(_temp[0])])
            except Exception:
                if int(_temp[0]) not in list(map[key].values()):
                    print(f)
            out.append(' '.join(_temp))
        with open(f"./dataset/YOLOv7/{d1}/labels/{f}", 'w') as f1:
            f1.writelines(out)