# utilityMeterReader

TRAIN MODEL:
1 - Place your dpictures and labels on the datasets folder (You can use Label Studio to create the labels)
2 - Run trainModelZo.py

EXTRACT ZOI
1 - Place the desired image in the /input/inputZoi folder
2 - Run detectZoi.py
3 - The detected ZOI will be saved as a cropped image in the /input/inputNumber folder

EXTRACT DATA
1 - Run detectNumber.py
2 - It will extract the data from the cropped images in /input/inputNumber folder [WIP]
