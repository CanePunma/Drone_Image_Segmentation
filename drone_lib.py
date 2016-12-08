from scipy import misc as smi
import numpy as np
import ogr, gdal
import matplotlib.pyplot as plt
import os
from PIL import Image, ImageDraw
import sys
import time
from scipy import misc as smi
import matplotlib.image as mpimg
from PIL import Image, ImageDraw
from osgeo import ogr
import numpy
from pyproj import Proj, transform
import caffe
import cv2
import csv
import random
from sklearn.cross_validation import train_test_split
import scipy.sparse
import math

tilesize = 256
"""Int: Default crop size.

"""
stride = 256
"""Int: Default strides for passing over an image

"""
DATASET_HOLDOUT_FLAG = False
"""Boolean: Leave False if you want to train on all of your datasets

"""

my_output_root = '' 
segmented_images_root = ''
cropped_images_root = ''
shapefile_images_root = ''
temp_shapefile_images_root =''
heatmap_images_root = ''
"""Strings: Root directories to save the preprocessed images

"""

def createDirectories(output_root,raw_data_root): 
    """Create the necessary folders in the drone working. This is where all the cropped images/shapefiles will be saved 
       and should contain all the necesary model files. E.g train/test/deploy.prototxt, solve.py,
       score.py, surgery.py
       
    Args:
        output_root: This current drone working directory where all the model files and images are.
        raw_data_root: The directory where all the .tif image files exist for .
        
    Returns:
        N/A
        
    """

    if (os.path.exists(output_root+'cropped_images') == False):
        os.mkdir(output_root+'cropped_images')

    if (os.path.exists(output_root+'output_segmented_images') == False):
        os.mkdir(output_root+'output_segmented_images')

    if (os.path.exists(output_root+'output_heatmap_images') == False):
        os.mkdir(output_root+'output_heatmap_images')

    if (os.path.exists(output_root+'shapefile_images') == False):
        os.mkdir(output_root+'shapefile_images')

    if (os.path.exists(output_root+'temp_shapefile_images') == False):
        os.mkdir(output_root+'temp_shapefile_images')

    output_files = os.listdir(output_root)

    tif_files = os.listdir(raw_data_root)
    for f1 in output_files:
        if os.path.isdir(output_root+f1) and f1 != 'weights':
            for f2 in tif_files:
                if (os.path.isdir(raw_data_root+f2) == False and os.path.exists(output_root + f1 + "/" + f2[0:len(f2)-4]) == False):
                    os.mkdir(output_root + f1 + '/' + f2[0:len(f2)-4])
                    
    global my_output_root,segmented_images_root,cropped_images_root,shapefile_images_root,heatmap_images_root,temp_shapefile_images_root
    my_output_root = output_root
    segmented_images_root = my_output_root + 'output_segmented_images/'
    cropped_images_root = my_output_root + 'cropped_images/'
    shapefile_images_root = my_output_root + 'shapefile_images/'
    temp_shapefile_images_root = my_output_root + 'temp_shapefile_images/'
    heatmap_images_root = my_output_root + 'output_heatmap_images/'
    
def createTestDirectories(output_root,raw_data_root,tif_nm): 
    """Same Function as above but just for the testing case
       
    Args:
        output_root: This current drone working directory where all the model files and images are.
        raw_data_root: The directory where all the .tif image files exist for .
        tif_nm: Name of image you are testing
        
    Returns:
        N/A
        
    """

    if (os.path.exists(output_root+'cropped_images') == False):
        os.mkdir(output_root+'cropped_images')

    if (os.path.exists(output_root+'output_segmented_images') == False):
        os.mkdir(output_root+'output_segmented_images')

    if (os.path.exists(output_root+'output_heatmap_images') == False):
        os.mkdir(output_root+'output_heatmap_images')
        
    if (os.path.exists(output_root+'final_results') == False):
        os.mkdir(output_root+'final_results')
    
    
    output_files = os.listdir(output_root)
    for f1 in output_files:
        if os.path.isdir(output_root+f1) and f1 != 'final_results':
            if (os.path.exists(output_root + f1 + "/" + tif_nm) == False):
                os.mkdir(output_root + f1 + '/' + tif_nm)
                    
    global my_output_root,segmented_images_root,cropped_images_root,heatmap_images_root
    my_output_root = output_root
    segmented_images_root = my_output_root + 'output_segmented_images/'
    cropped_images_root = my_output_root + 'cropped_images/'
    heatmap_images_root = my_output_root + 'output_heatmap_images/'
    
def flipDict(my_dict):
    """Flips the class dictionary so there is a reference from integer to actual class.
       
    Args:
        my_dict: Class Dictionary defined above.
        
    """
    new_dict = {}
    for key,value in my_dict.items():
        new_dict[value] = key
    return new_dict

def standardizeStingNames(my_dict):
    """Standardize the class string names.
       
    Args:
        my_dict: Class Dictionary defined above.
        
    """
    new_dict = {}
    for key,value in my_dict.items():
        new_dict[key.replace("_" , " ")] = value
    return new_dict


def loadTif(raw_data_root, tif_name):
    """Use GDAL Package to open up the raser tif file and save as an array.
       
    Args:
        tif_name: The name of the tif file for a partifcular construction site.
        raw_data_root: The directory where all the .tif image files exist for .
        
    Returns:
        Raster attributes
        
    """
    raster = gdal.Open(raw_data_root + tif_name)
    imarray = np.array(raster.ReadAsArray())
    imarray= imarray[0:3,:,:]
    channel,height,width = imarray.shape
    return width,height,channel,imarray,raster

def cropImage(tif_name,imarray,image_height,image_width):
    """Crop the large raster array into tilesize crops (256 x 256).
       
    Args:
        tif_name: The name of the tif file for a partifcular construction site.
        imarray: Actual array of raster.
        height: Height of the Raster.
        width: Width of the raster.
        
    Returns:
        Saves crops into cropped_images folder 
        
    """
    tif_nm = tif_name[:-4]
    count = 0
    if count == 0:
        print cropped_images_root + tif_nm + "/" 
    total_num_of_images = int(math.ceil(image_height * 1. / stride)) * int(math.ceil(image_width * 1. / stride))
    for height in xrange(0,image_height,stride):
        for width in xrange(0,image_width,stride):
            temp = imarray[:,height:height + tilesize, width: width + tilesize]
            temp = temp.transpose((1,2,0))
            smi.imsave(cropped_images_root + tif_nm + "/" + tif_nm + '_' + str(width) + "_" + str(height) + ".png", temp)
            if count % 50 == 0:
                sys.stdout.write('\r' + "Processed: " + str(count) + " images out of " + str(total_num_of_images))
            count += 1
    sys.stdout.write('\r' + "Done. Total images cropped: " + str(count))
    print

def degToGeo(coordinate):
    """Transform the (x,y) coordinate into Geocoordinates.
       
    Args:
        coordinate: (x,y).
        
    Returns:
        Coordinate in epsg:2178 or geocoordinates 
        
    """
    inProj = Proj(init='epsg:4326')
    outProj = Proj(init='epsg:2178')

    x = coordinate[0]
    y = coordinate[1]

    x2,y2 = transform(inProj,outProj,x,y)
    return ((x2,y2))

def standardize(coordinate,degXMin,degYMin,degWidth,degHeight,width,height):
    """Transform the (x,y) geocoordinate into standardized coordinate system from (0,0) to (xMax, yMax).
       
    Args:
        coordinate: (x,y) geocoordinate
        degXMin: X min geolocation of original raster
        degYMin: y min geolocation of original raster
        degWidth: width of original raster in geo length
        degHeight: height of original raster in geo length
        width: width of the raster
        height: height of the raster
        
    Returns:
        Coordinate in standardized coordinate system (0,0) - (xMax, yMax)
        
    """
    x = coordinate[0]
    y = coordinate[1]
    x_new = ((x - degXMin)* width)/degWidth
    y_new = ((y - degYMin) * height) / degHeight

    #y_new = height - y_new

    new_coordinate = (x_new,y_new)
    return new_coordinate

def getTopLeft(coordinate):
    """Of the Shape File object of interest, get the top left corner coordinate of the box surrounding it.
       
    Args:
        coordinate: (x,y) standardized format

    Returns:
        Top left corner coordinate of surrounding box.
        
    """
    x = coordinate[0]
    y = coordinate[1]

    x_new = x - (x % tilesize)
    y_new = y - (y % tilesize)

    return ((x_new,y_new))

def getBottomRight(coordinate):
    """Of the Shape File object of interest, get the bottom right corner coordinate of the box surrounding it.
       
    Args:
        coordinate: (x,y) standardized format

    Returns:
        Bottom right corner coordinate of surrounding box.
        
    """
    x = coordinate[0]
    y = coordinate[1]

    x_new = (x - (x % tilesize)) + tilesize
    y_new = (y - (y % tilesize))  + tilesize

    return ((x_new,y_new))

def getShape(geom, raster):
    """Read the Polygon from the shape geom file. Convert the polygons verticies into geocoordinates and then into
       into standardized formate. Finally calculate the surrounding box of the polygon and save the Top Left and
       Bottom right coordinates.
       
    Args:
        geom: (x,y) geom file that is pulled from the class shapefile

    Returns:
        minmaxpoints: Top left and bottom right standardized coordinate of the shape polygons surrounding box.
        standardized_verticies: The points of the polygon in standardized format.
        
    """
    ds = raster
    width = ds.RasterXSize
    height = ds.RasterYSize
    gt = ds.GetGeoTransform()
    minx = gt[0]
    #Flipped Min y and Max Y
    maxy = gt[3] + width*gt[4] + height*gt[5]
    maxx = gt[0] + width*gt[1] + height*gt[2]
    miny = gt[3]

    ##Pull the raw polygon verticies and change them into Degrees
    ring = geom.GetGeometryRef(0)
    totalvertices=ring.GetPointCount()
    verticies = []
    for vertices in range(totalvertices):
        x,y,z = (ring.GetPoint(vertices))

        if (x < 200 or y < 200): ##Temporary Hack to check if coordinate is in Degrees
            x,y = degToGeo((x,y))
        verticies.append((x,y))

    #Ignore naming, these coordinates are originally in geocoordinates when read from the shape geom file
    topLeftCorner = (minx,maxy)
    bottomRightCorner = (maxx,miny)

    degXMin = topLeftCorner[0]
    degXMax = bottomRightCorner[0]
    degYMin = bottomRightCorner[1]
    degYMax = topLeftCorner[1]

    degWidth = maxx-minx
    degHeight = maxy-miny

    #Standardize all the verticies into the pixel coordinate system
    standardized_verticies = []
    for vert in verticies:
        new_vert = standardize(vert,degXMin,degYMin,degWidth,degHeight,width,height)
        standardized_verticies.append(new_vert)

    #Get all the points/verticies of the polygon
    poly_Xs = []
    poly_Ys = []
    for vert in verticies:
        poly_Xs.append(vert[0])
        poly_Ys.append(vert[1])
    poly_xMin = min(poly_Xs)
    poly_yMax = max(poly_Ys)
    poly_xMax = max(poly_Xs)
    poly_yMin = min(poly_Ys)

    ploy_standardize_xMin_yMax = standardize((poly_xMin,poly_yMax),degXMin,degYMin,degWidth,degHeight,width,height)
    ploy_standardize_xMax_yMin = standardize((poly_xMax,poly_yMin),degXMin,degYMin,degWidth,degHeight,width,height)

    #If the polygon is outside the .tif dimensions, skip that polygon
    if(poly_xMin < degXMin or poly_xMax > degXMax or poly_yMax > degYMin or poly_yMin < degYMax):
        return None,None

    ##Get the points of interest of the cropped box that surrounds the shape file
    cropTopLeft = getTopLeft(ploy_standardize_xMin_yMax)
    cropBottonRight = getBottomRight(ploy_standardize_xMax_yMin)
    #Flipped bottom right and top left
    minmaxpoints = [int(cropTopLeft[0]),int(cropTopLeft[1]),int(cropBottonRight[0]),int(cropBottonRight[1])]
    #minmaxpoints = [int(cropTopLeft[0]),int(cropBottonRight[1]),int(cropBottonRight[0]),int(cropTopLeft[1])]

    return minmaxpoints,standardized_verticies

def readShapeFiles(dset,raster,dataset_shapes_root):
    """Loop through all the shapefiles and make a dictionary for each class which containes values for each polygon.
       
    Args:
        dset: The raster of the tif image 
        raster: The raster of the tif image in an array
        dataset_shapes_root: The location of folder that contains all the shape files

    Returns:
        shapes: Dictionary of all the classes which contains each polygon.
        
    """
    shapes = {}
    for shape in os.listdir(dataset_shapes_root):
        shape_name = shape[len(shape)-3:len(shape)]
        all_shapes = []
        if shape_name == 'shp':
            shape_file_dir = dataset_shapes_root + shape
            driver = ogr.GetDriverByName("ESRI Shapefile")

            dataSource = driver.Open(shape_file_dir, 0)
            layer = dataSource.GetLayer()

            all_shapes = []
            shape_index = 0
            for feature in layer:
                geom = feature.GetGeometryRef()
                if (geom != None):
                    #Get the polygon and its surounding box
                    minmax, poly = getShape(geom,raster)

                    if(minmax != None):
                        index = shape_file_dir.rfind('/')
                        shp_nm = shape_file_dir[index + 1 :len(shape_file_dir)-4]
                        all_shapes.append([shp_nm, shape_index, minmax, poly, tilesize])
                        shape_index = shape_index + 1
                    else:
                        #print('Object out of range')
                        pass
                else:
                    pass
            shape = shape.lower()
            shapes[shape] = all_shapes
    return shapes


def save_shapes(dset , shapefile_name, index_value, minmaxpoints, polygon, crop_size, classes):
    """Save the segmentation ground truth image file. This image is a 256x256 which contains values from 0 - 18 representing
       where the object pixel is located on the picture. If the surround box (minmaxpoints) is larger than 256x256 then
       it will be cropped into 256x256 images. These images are saved in the temp_shapefile_images where they will soon 
       after be argmaxed based on their cropped image location.
       
    Args:
        dset: The raster of the tif image 
        shapefile_name: Name of the class
        index_value: The # that representes that unique polygon shape within a class
        minmaxpoints: The box surround the polygon, Top Left and Bottom Right corners
        polygon: The actual verticies of the polygon
        crop_size: Same as tilesize
        classes: Dictionary that enumerates and assigns an integer value for each class

    Returns:
        Saves each class segmentation image in the temp_shapfile_images_root
        
    """
    dset = dset[:len(dset)-4]
    # calc min and max values for the geom (xminshape, xmaxshape, yminshape, ymaxshape)
    xmin, ymin, xmax, ymax = minmaxpoints
    image_height = ymax - ymin
    image_width = xmax - xmin
    im = Image.new('L', (image_width, image_height), 0)


    # Edit the polygon values to fit with image_width and image_height
    xvalues = []
    yvalues = []
    for p in polygon:
        x, y = p
        x = x - xmin
        y = y - ymin
        xvalues.append(x)
        yvalues.append(y)

    # if the shape file is too big, crop it. Take the min and max for the xvalues and yvalues and check if they are above a pre-set pixel value
    polygon1=[]
    for i in range(len(xvalues)):
        polygon1.append( (xvalues[i], yvalues[i]) )



    draw = ImageDraw.Draw(im)
    draw.polygon(polygon1, fill = classes.get(shapefile_name)) ## FILL CLASS LABELS HERE..USE A GLOBAL DICTIONARY HERE MAYBE?

    #im.save('bigimage.png', "PNG")
    if (image_height / crop_size > 1) or (image_width / crop_size > 1):
        for x in range(image_width / crop_size):
            for y in range(image_height / crop_size):
                bbox=(x * crop_size, y * crop_size, x * crop_size + crop_size, y * crop_size + crop_size)
                cropped_image=im.crop(bbox)
                cropped_image.save(temp_shapefile_images_root + dset + '/' + dset + '_' + shapefile_name + '_' + str(index_value) + '_' + str(xmin + (x * crop_size) ) + '_' + str(ymin + (y * crop_size) ) + '.png', "PNG")
    elif((image_height / crop_size == 1) and (image_width / crop_size == 1)):
        im.save(temp_shapefile_images_root + dset + '/' + dset + '_' + shapefile_name + '_' + str(index_value) + '_' + str(xmin) + '_' + str(ymin) + '.png', "PNG")
    else:
        a=1
        print('Did not create image')

def saveToUnmerged(dset, shapes,classes):
    """Loop through all the classes and save all their respective shapes that are present in the dictionary.
       
    Args:
        dset: The raster of the tif image 
        shapes: Dictionary of polygons for each class
        classes: Enumerated dictionary of all the classes 

    Returns:
        Saves each class segmentation image in the temp_shapfile_images_root
        
    """
    all_classes = classes.keys()
    for class_shape in all_classes:
        class_shape_name = class_shape + '.shp'
        if(class_shape_name != 'background.shp'):
            for i in range(len(shapes[class_shape_name])):

                ##Save the segmentation images
                shapename, index, minmax, poly, cropsize = shapes[class_shape_name][i]
                save_shapes(dset, shapename, index, minmax, poly, cropsize, classes)

def create_label_data(dset):
    """Take all the temp_shapefile images that exist at the same (x,y) crop location and argmax their images together.
        The higher priority class defined which take precedent over the lower ones and a single segmentation ground truth
        image with all the objects in the image will be created.
       
    Args:
        dset: The raster of the tif image 

    Returns:
        Saves each class argmax segmentation image in the shapfile_images_root
        
    """
    tif_name = dset[: -4]
    files = os.listdir(temp_shapefile_images_root + tif_name + '/')
    cropped_file_width = []
    cropped_file_height = []
    for f in files:
        dimensions = f.rsplit('_', 2)[1 : 3]
        cropped_file_width.append(int(dimensions[0]))
        cropped_file_height.append(int(dimensions[1].split('.')[0]))
    min_height, min_width, max_height, max_width = min(cropped_file_height), min(cropped_file_width), max(cropped_file_height), max(cropped_file_width)
    count = 0
    count2 = 0
    for width in range(min_width, max_width, tilesize ):
        for height in range(min_height, max_height, tilesize ):
            images = []
            for f in files:
                dimensions = f.rsplit('_', 2)[1 : 3]
                w = (int(dimensions[0]))
                h = (int(dimensions[1].split('.')[0]))
                if (w == width and h == height):
                    images.append(smi.imread(temp_shapefile_images_root + tif_name + '/' + f))
            if images != []:
                merged_image = np.dstack(images)
                merged_image=merged_image.max(axis=2)
                if (np.count_nonzero(merged_image) != 0 ):
                    count2 += 1
                    smi.imsave(shapefile_images_root + tif_name + '/' + tif_name + '_' + str(width) + '_' + str(height) + '.png', merged_image, 'PNG' )
                if count % 50 == 0:
                    sys.stdout.write('\r' + "Processed: " + str(count) + " images...")
                count += 1
    sys.stdout.write('\r' + "Out of " + str(count) + " images, " + str(count2) + " had shapes in them and were saved.")
    print

def create_train_test_files_V2(validation_ratio, holdout_ratio, use_all_dset_flag, train_dsets, test_dsets):
    """Split all the files in the shapefiles folder into train/(test or validation)/holdout.txt. You man either train
       on all the datasets or leave one out for use of testing.
       
    Args:
        validation_ratio: Percentage of images to test or validate during training after every X epochs
        holdout_ratio: Percentage of images to holdout 
        use_all_dset_flag: True if you want to train on all the datasets else False and then specify which to use
        train_dsets: If the use_all_dset_flag is False, specify which dataset is in the training
        test_dsets: If the use_all_dset_flag is False, specify which dataset is in the holdout

    Returns:
        Saves the train/test/holdout.txt to be used when running the model
        
    """
    file_list = []
    additional_list = []
    holdout_dset_list = []
    directories = os.listdir(shapefile_images_root)
    for directory in directories:
        if(directory[:directory.rfind('_')] in train_dsets or use_all_dset_flag or directory[:10].lower() == 'additional'):

            if (os.path.isdir(shapefile_images_root + directory) and directory[:10].lower() != 'additional'):
                files = os.listdir(shapefile_images_root + directory +'/')
                for f in files:
                    if os.path.exists(cropped_images_root + directory + '/' + f):
                        file_list.append( directory + '/' + f[:-4] )
            elif(os.path.isdir(shapefile_images_root + directory) and directory[:10].lower() == 'additional'):
                files = os.listdir(shapefile_images_root + directory +'/')
                for f in files:
                    if os.path.exists(cropped_images_root + directory + '/' + f):
                        #print(f)
                        additional_list.append( directory + '/' + f[:-4] )

        else:
            files = os.listdir(shapefile_images_root + directory +'/')
            for f in files:
                if os.path.exists(cropped_images_root + directory + '/' + f):
                    holdout_dset_list.append( directory + '/' + f[:-4] )

    train_file = open(my_output_root + "/train.txt", "w")
    holdout_file = open(my_output_root + "/holdout_cars.txt", "w")
    test_file = open(my_output_root + "/test.txt", "w")

    if(use_all_dset_flag):
        holdout_split = int( len(file_list) / (1.0/holdout_ratio) )
        holdout = file_list[0 : holdout_split ]
        file_list = file_list[holdout_split : len(file_list) ]
        for f in holdout:
            holdout_file.write("%s\n" % f)
        holdout_file.close()
    else:
        for f in holdout_dset_list:
            holdout_file.write("%s\n" % f)
        holdout_file.close()

    train, test = train_test_split(file_list, test_size = validation_ratio, random_state = 42)

    for f in train:
        train_file.write("%s\n" % f)
    for f in additional_list:
        #print('added ', f)
        train_file.write("%s\n" % f)
    train_file.close()

    for f in test:
        test_file.write("%s\n" % f)
    test_file.close()

def testMain(caffemodel,holdout,labels_dict_cleaned):
    """Pass all the images in the holdout file through the caffe segmentation model. Then calculate the per pixel
       accuries for each predicted image.
       
    Args:
        caffemodel: After you've trained the model. This should be saved in the Weights folder
        holdout: Txt file in the drone root that specifies which files to test
        labels_dict_cleaned: Dictionary of classes assigned to their values

    Returns:
        Saves the train/test/holdout.txt to be used when running the model
        
    """
    #caffe.set_device(1)
    caffe.set_mode_gpu()
    net = caffe.Net(my_output_root + 'deploy18.prototxt',
                    my_output_root + 'weights/' + caffemodel,
                    caffe.TEST)
    test_files = []
    with open(my_output_root + holdout, 'rb') as csvfile:
        spamreader = csv.reader(csvfile)
        for row in spamreader:
            test_files.append(row[0])
    print(len(test_files))
    class_dict = flipDict(labels_dict_cleaned)
    classify(test_files,net,my_output_root)
    getAccuracies(labels_dict_cleaned,class_dict,test_files)

def classify(test_files,net,output_root, threshold = 0.05):
    """Each image passed through will be forward passed through the caffe model and their segmentation and
       heatmap output will be save.
       
    Args:
        test_files: Images files to loop though
        net: Network created from the prototxt file
        output_root: Current working drone directory 

    Returns:
        Saves the segmentation argmax output in the segmented_images_root folder and the heatmap output in the
        heatmap_images_root folder
        
    """
    count = 0
    for image_name in test_files:
        if '.' in image_name: #Signifing Testing Phase
            image_folder_name = image_name[:image_name.find('_')]
            image_name = image_folder_name + '/' + image_name
            image_name = image_name[:-4]
            image = smi.imread(cropped_images_root + image_name + '.png')
        else:
            image = smi.imread(cropped_images_root + image_name + '.png')
            
        height,width,channel = image.shape

        in_ = np.array(image, dtype=np.float32)
        in_ = in_[:,:,0:3]
        in_ = in_[:,:,::-1]
        in_ -= np.array((113.67583, 104.88984, 98.37526))
        in_ = in_.transpose((2,0,1))
        net.blobs['data'].reshape(1, *in_.shape)
        net.blobs['data'].data[...] = in_
        net.forward()

        file_name = heatmap_images_root + image_name
        output = net.blobs['prob'].data[0]
        create_sparse_data(output, file_name, threshold)
        #output_classification = net.blobs['prob'].data[0].argmax(axis=0)
        smi.imsave(segmented_images_root + image_name + '.png', np.array(output.argmax(axis = 0), dtype = np.uint8))

        if count % 50 == 0:
            sys.stdout.write('\r' + "Processed: " + str(count) + " images...")
        count += 1

    sys.stdout.write('\r' + "Done. Total images cropped: " + str(count))
    print

def getAccuracies(labels_dict_cleaned,class_dict,test_files):
    """For each image loop through the pixels and check the ground truth dictionary to calculate 
       some of the key statistics in recall and pecision.
       
    Args:
        labels_dict_cleaned: Dictionary of Key: Class, Value: #
        class_dict: Key: #, Value: Class
        test_files: Holdout file images to loop through

    Returns:
        Printed per class recall and percision.
        
    """
    count = 0
    recall_denominator = {}
    precision_denominator = {}
    correct_count = {}
    for key in labels_dict_cleaned.keys():
        recall_denominator[key] = 0
        precision_denominator[key] = 0
        correct_count[key] = 0

    for shape_file_name in test_files:
        try:
            shapefile_dir =  shapefile_images_root + shape_file_name + '.png'
            classification_output_dir = segmented_images_root + shape_file_name + '.png'
            output1 = smi.imread(classification_output_dir)
            shape_file = smi.imread(shapefile_dir)

            for row in range(tilesize):
                for col in range(tilesize):
                    val = shape_file[row][col]
                    predicted = output1[row][col]
                    if(val != 0):
                        recall_denominator[class_dict[val]] = recall_denominator[class_dict[val]] + 1
                        if(predicted == val):
                            correct_count[class_dict[val]] = correct_count[class_dict[val]] + 1
                    if(predicted != 0):
                        precision_denominator[class_dict[predicted]] = precision_denominator[class_dict[predicted]] + 1
            if count % 50 == 0:
                sys.stdout.write('\r' + "Processed: " + str(count) + " images...")
            count += 1

        except:
            print('Could not find output for ' + shape_file_name)


    sys.stdout.write('\r' + "Done. Total images cropped: " + str(count))
    print

    print(recall_denominator)
    print(precision_denominator)
    print(correct_count)

    recall_accuracy = []
    precision_accuracy = []
    for cls in labels_dict_cleaned.keys():
        if(cls != 'background' and recall_denominator[cls] != 0 and precision_denominator[cls] != 0):
            recall_percentage = correct_count[cls]/float(recall_denominator[cls])
            recall_accuracy.append((cls,recall_percentage))

            precision_percentage = correct_count[cls]/float(precision_denominator[cls])
            precision_accuracy.append((cls,precision_percentage))

    recall_accuracy = sorted(recall_accuracy, key=lambda x: x[1])
    precision_accuracy = sorted(precision_accuracy, key=lambda x: x[1])
    results = {'Recall:':recall_accuracy, 'Precision:': precision_accuracy}
    print(results)
    
def save_sparse_csr(filename,array):
    """Save the sparse numpy.
       
    Args:
        filename: Location it is being saved
        array: Spare Array
        
    """
    np.savez(filename,data = array.data ,indices=array.indices,
             indptr =array.indptr, shape=array.shape)

def create_sparse_data(heatmap, file_name, threshold):
    """Checks if the probability is above a certain threshold, if not, that value converted to 0. Then a sparse
       matrix can be save for each heatmap.
       
    Args:
        filename: Location it is being saved
        heatmap: Actual numpy array
        threshold: Threshold probabilty to convert value to 0
        
    """
    channels, width, height = heatmap.shape
    sparse_array = []
    for i in range(0, channels):
        temp = np.copy(heatmap[i,:,:])
        temp[temp <= threshold] = 0
        sparse_array.append(scipy.sparse.csr_matrix(temp))
    sparse_heatmap = scipy.sparse.vstack(sparse_array)
    save_sparse_csr(file_name,sparse_heatmap)
    
def stichImage(output_root, tif_nm):
    """For all the images in the out_segmentation folder. Stitch them backtogether
       
    Args:
        output_root: Current working root for test image
        tif_nm: Name of image

    Returns:
        Full stitched back image.
        
    """
    images = os.listdir(output_root +  'output_segmented_images/' +  tif_nm)
    height =[]
    width=[]
    for f in images:
        try:
            splits = f.split('_')
            height.append(int(splits[len(splits)-1].split('.')[0]))
            width.append(int(splits[len(splits)-2]))
        except:
            pass

    max_height = max(height)
    max_width = max(width)
    new_im = Image.new('L',(max_width,max_height))

    count = 0
    for i in xrange(0,max_width+tilesize,tilesize):
        for j in xrange(0,max_height+tilesize,tilesize):

            name = tif_nm + '_' + str(i) + str('_') + str(j) + '.png'
            im = Image.open(output_root +  'output_segmented_images/' +  tif_nm + '/' + name)
            new_im.paste(im, (i,j))
    return new_im

    