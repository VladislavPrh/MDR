import scipy.misc
import numpy as np
import csv
import os


def read_csv(path):
    """Reads csv file and puts data into python dictionary"""
    dict_csv = {}
    with open(path) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row["FileName"] in dict_csv:
                if int(row["DigitLabel"]) == 10:
                    dict_csv[row["FileName"]]["DigitLabel"].append(0)
                else:
                    dict_csv[row["FileName"]]["DigitLabel"].append(int(row["DigitLabel"]))
                dict_csv[row["FileName"]]["Left"].append(int(row["Left"]))
                dict_csv[row["FileName"]]["Top"].append(int(row["Top"]))
                dict_csv[row["FileName"]]["Width"].append(int(row["Width"]))
                dict_csv[row["FileName"]]["Height"].append(int(row["Height"]))
            else:
                dict_csv[row["FileName"]] = {"DigitLabel": [], "Left": [], "Top": [], "Width": [], "Height": []}
                if int(row["DigitLabel"]) == 10:
                    dict_csv[row["FileName"]]["DigitLabel"].append(0)
                else:
                    dict_csv[row["FileName"]]["DigitLabel"].append(int(row["DigitLabel"]))
                dict_csv[row["FileName"]]["Left"].append(int(row["Left"]))
                dict_csv[row["FileName"]]["Top"].append(int(row["Top"]))
                dict_csv[row["FileName"]]["Width"].append(int(row["Width"]))
                dict_csv[row["FileName"]]["Height"].append(int(row["Height"]))
    return dict_csv        


def get_list_of_image(path):
    """Makes sorted list of image names"""
    list_of_image_names = []
    for f in os.listdir(path):
        if ".png" in f:
            list_of_image_names.append(f)
    return sorted(list_of_image_names, key=lambda x: int(x.split(".")[0]))


def get_overall_bounding_box(dict_csv, image):
    """Gets location of the bounding boxes"""
    left = min(dict_csv[image]["Left"])
    if left < 0:
        left = 0
    top = min(dict_csv[image]["Top"])
    
    for e in range(len(dict_csv[image]["DigitLabel"])):
        if e == 0:
            l = []
        l.append(dict_csv[image]["Top"][e] + dict_csv[image]["Height"][e])    
        bottom = max(l)
        
    for e in range(len(dict_csv[image]["DigitLabel"])):
        if e == 0:
            li = []
        li.append(dict_csv[image]["Left"][e] + dict_csv[image]["Width"][e])
        right = max(li)

    return bottom, right, left, top


def crop_and_resize(bottom, right, left, top, image_array):
    """Crops images based on location their bounding boxes
    and rezises the crop to 32x32 pixels """
    crop_image = image_array[top:bottom, left:right]
    new_image = scipy.misc.imresize(crop_image, (32, 32, 3))
    return new_image


def get_x(path):
    """Makes array of x data
    x.shape ==> (number_of_images, 32, 32, 3)
    """
    if "test" in path:
        dict_csv = read_csv("data/digitStruct_test.csv")
    if "train" in path:
        dict_csv = read_csv("data/digitStruct_train.csv")
    count = 0
    list_of_image_names = get_list_of_image(path)
    number_of_images = len(list_of_image_names)
    x = np.empty((number_of_images, 32, 32, 3))
    for image_name in list_of_image_names:
        image_array = scipy.misc.imread(path + "/" + image_name, flatten=False)
        bottom, right, left, top = get_overall_bounding_box(dict_csv, image_name)
        updated_image = crop_and_resize(bottom, right, left, top, image_array)
        x[count] = updated_image
        count += 1    
    x -= np.mean(x, dtype="float") # zero-mean
    return x


def get_y1(path):
    """Makes array of y labels.
    y.shape ==> (number_of_images, 6)
    where y[:,0] - number of digits for each image, maximun - 5 digits
    and y[:,1:6] - corresponding digits, 10 indicates blank digit
    """  
    if "test" in path:
        dict_csv = read_csv("data/digitStruct_test.csv")
    if "train" in path:
        dict_csv = read_csv("data/digitStruct_train.csv")
    count = 0    
    list_of_image_names = get_list_of_image(path)
    y = np.empty((len(dict_csv), 6), dtype=int)
    for image_name in list_of_image_names:
        digits = np.array(dict_csv[image_name]["DigitLabel"])
        number_of_digits = len(digits)
        if number_of_digits > 5:
            digits = digits[0:5]
        arr = np.empty((6))
        arr[:] = 10
        arr[0] = len(digits)
        arr[1:(len(digits)+1)] = digits
        y[count] = arr
        count += 1
    return y
