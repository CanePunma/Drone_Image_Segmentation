#Setup

#For image processing
from PIL import Image
#For image visualization
import matplotlib.pyplot as plt
#For sklearn & caffe
from scipy import misc
import numpy as np
#For image processing
import cv2
#For clustering methods
from sklearn.cluster import KMeans
#For processing
import copy
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity
import math
import os
import time
plt.rcParams['image.interpolation'] = 'nearest' 

def to_matrix(l, n):
    """
    to_matrix creates a 2-d array from a list.

    argumens:
    l - 1-d list of pixels
    n - the number of rows & columns

    outputs:
    A 2-d array with n rows
    """
    return [l[i:i+n] for i in xrange(0, len(l), n)]
#------------------------------------------------------------------------------------------------------------------------
def pixcon(x,y):
    """
    pixcon takes an RGB value and scales it by y and converts it back into an RGB value

    arguments:
    x - RGB input value
    y - scaling value

    outputs:
    scaled RGB value
    """
    if y == 0:
        return 0
    else:
        return int((float(x)/y)*255)
#------------------------------------------------------------------------------------------------------------------------
def find_K(flat_pixels,penalization,method):
    """
    find_K tries several clustering solutions to arrive at an "optimal" number of clusters

    arguments:
    flat_pixels - a list of pixel values (pixel value = array(R,G,B))
    penalization - threshold for K1 method
    method - the method used to find the optimal number of clusters 
                K1 uses a threshold on the 1st derivative
                K2 refers to find_K2

    outputs:
    k - a value representing the optimal number of clusters
    """
    if method == "K2":
        return find_K2(flat_pixels)
    elif method == "K1":
        max_k = 5
        scores = []
        for k in range(2,max_k+1):
            print flat_pixels
            clust = KMeans(n_clusters=k,n_init=10).fit(flat_pixels)
            score = clust.inertia_
            scores.append(score)
        differences = []
        for ind in xrange(len(scores)-1):
            if ind == 0:
                first = float(scores[ind]-scores[ind+1])/scores[ind]
            dif = float(scores[ind]-scores[ind+1])/scores[ind]
            if dif <= first*penalization:
                return ind+1
            else:
                print "CLUSTERING SOLUTION DID NOT PASS THRESHOLD"
    elif method =="K3":
        max_k = 3
        scores = []
        for k in range(2,max_k+1):
            clust = KMeans(n_clusters=k,n_init=10).fit(flat_pixels)
            score = clust.inertia_
            scores.append(score)
            if len(scores)>1:
                if((scores[k-3]-scores[k-2])/scores[k-3])>0.48:
                    return 3
                else:
                    return 2
    else:
        print "NOT A VALID CLUSTERING OPTIMAL K METHOD"
#------------------------------------------------------------------------------------------------------------------------
def find_K2(flat_pixels):
    """
    find_K2 tries several clustering solutions and uses the 3rd derivative of the WSS
    to arrive at an "optimal" number of clusters

    arguments:
    flat_pixels - a list of pixel values (pixel value = array(R,G,B))

    outputs:
    k - a value representing the optimal number of clusters
    """
    max_k = 5
    scores = []
    for k in range(2,max_k+1):
        clust = KMeans(n_clusters=k,n_init=10).fit(flat_pixels)
        score = clust.inertia_
        scores.append(score)
    scores = np.asarray(scores)/sum(scores)
    diff_1 = []
    for ind in xrange(len(scores)-1):
        diff_1.append(float(scores[ind]-scores[ind+1]))
    diff_2 = []
    for ind in xrange(len(diff_1)-1):
        diff_2.append(float(diff_1[ind]-diff_1[ind+1]))
    diff_3 = []
    for ind in xrange(len(diff_2)-1):
        diff_3.append(float(diff_2[ind]-diff_2[ind+1]))
    return diff_3.index(max(diff_3))+2
#------------------------------------------------------------------------------------------------------------------------
def check_cluster_dist(labels):
    """
    check_cluster_dist looks at the distribution or frequency of labels

    arguments:
    labels - a list of cluster labels

    outputs:
    out - dictionary of label frequencies
    """
    out = dict()
    for clust in labels:
        if str(clust) in out.keys():
            out[str(clust)]+=1
        else:
            out[str(clust)]=1
    return out
#------------------------------------------------------------------------------------------------------------------------
def mean_pixel(pixels):
    """
    mean_pixel calculates the mean RGB value

    arguments:
    pixels - list of pixel values

    outputs:
    an array representing the average RGB values
    """
    remaining_pixels = []
    for pixel in pixels:
        pix_sum = sum(list(pixel))
        if (pix_sum != 255)|(pix_sum != 255*3):
            remaining_pixels.append(pixel)
    rem_pix_arr = np.array(remaining_pixels)
    return rem_pix_arr.mean(axis=0)
#------------------------------------------------------------------------------------------------------------------------
def NIOSC(new_image,penalization,method):#NEW IMAGE OF SAME CLASS
    """
    NIOSC or New Image of Same Class segments the image by normalizing the brightness and clustering the pixels

    arguments: 
    new_image - string containing the link to the to be segmented image
    penalization - threshold value for find_K
    method - method to be used in find_K

    outputs:
    dictionary containing the segments of the input image and the distribution of labels
    """
    im_new = cv2.cvtColor(cv2.imread(new_image), cv2.COLOR_BGR2RGB)
    flat_pixels_new = [item for sublist in im_new for item in sublist]
    flat_pixels_bland = []
    for pixel in flat_pixels_new:
        pix = list(pixel)
        p_sum = sum(pix)
    flat_pixels_bland.append([pixcon(pix[0],p_sum),pixcon(pix[1],p_sum),pixcon(pix[2],p_sum)])
    k = find_K(flat_pixels_bland,penalization,method)
    clustered_pixels = KMeans(n_clusters=k,n_init=10).fit_predict(flat_pixels_bland)
    check = check_cluster_dist(clustered_pixels)
    pixel_labels = list(clustered_pixels)
    images = []
    for cluster in check.keys():
        new_flat_pixels = copy.copy(flat_pixels_new)
        for ind in xrange(len(flat_pixels_new)):
            if (pixel_labels[ind]!=int(cluster)):
                new_flat_pixels[ind]=np.array([255, 255, 255],dtype='uint8')
        images.append(new_flat_pixels)
    return (dict({"images":images, "count":check})) #, "blur_images":blur_images}))
#------------------------------------------------------------------------------------------------------------------------
def NIOSC1_5(old_image,new_image,penalization,method):#NEW IMAGE OF SAME CLASS
    """
    NIOSC1_5 or New Image of Same Class version 1.5 segments the image by normalizing the 
    brightness and clustering the pixels. It goes one step further than NIOSC by calculating
    the mean pixel of the base image and comparing it to the mean pixel of each of the segments

    arguments: 
    old_image - list of base image links
    new_image - string containing the link to the to be segmented image
    penalization - threshold value for find_K
    method - method to be used in find_K

    outputs:
    2-d array of pixels that make up the correct image containing the correct segment
    """

    im_new = cv2.cvtColor(cv2.imread(new_image), cv2.COLOR_BGR2RGB)

    old_means = []
    for im in old_image:
        old_image_array = cv2.cvtColor(cv2.imread(im), cv2.COLOR_BGR2RGB)
    #plt.imshow(im_new)    
    #fig = plt.figure()
        old_image_array_flat = [item for sublist in old_image_array for item in sublist]
        flat_pixels_old = old_image_array_flat
        mean = mean_pixel(flat_pixels_old)
        old_means.append(mean)
    flat_pixels_new = [item for sublist in im_new for item in sublist]
    flat_pixels_bland = []
    for pixel in flat_pixels_new:
        pix = list(pixel)
        p_sum = sum(pix)
        flat_pixels_bland.append([pixcon(pix[0],p_sum),pixcon(pix[1],p_sum),pixcon(pix[2],p_sum)])
    k = find_K(flat_pixels_bland,penalization,method)
    clustered_pixels = KMeans(n_clusters=k,n_init=10).fit_predict(flat_pixels_bland)
    check = check_cluster_dist(clustered_pixels)
    pixel_labels = list(clustered_pixels)
    images = []
    old_mean = np.mean(old_means)
    distances = []
    for cluster in check.keys():
        new_flat_pixels = copy.copy(flat_pixels_new)
        for ind in xrange(len(new_flat_pixels)):
            if pixel_labels[ind]!=int(cluster):
                new_flat_pixels[ind]=np.array([255, 255, 255],dtype='uint8')
        images.append(new_flat_pixels)
        distances.append(euclidean(old_mean,mean_pixel(new_flat_pixels)))
        #distances.append(cosine(old_mean,mean_pixel(new_flat_pixels)))
    final_image = images[distances.index(min(distances))]
    new_image = np.asarray(to_matrix(final_image,256))
    return (new_image)
#------------------------------------------------------------------------------------------------------------------------
def getFeatureVector(image_array,transformer,net):
    """
    getFeatureVector pushes an image through a pretrained model and pulls the fc7 vector

    arguments:
    image_array - 2-d array of pixels
    transformer - caffe's image transformation object
    net - caffe's AlexNet object

    output
    """
    transformed_image = transformer.preprocess('data', image_array)
    net.blobs['data'].data[...] = transformed_image
    output = net.forward()
    #fc6 = net.blobs['fc6'].data[0]
    fc7 =net.blobs['fc7'].data[0]
    #fc8 = net.blobs['fc8'].data[0]
    return fc7
#------------------------------------------------------------------------------------------------------------------------
# def readImages(images_root, image_name):
#     image_dir = images_root + image_name
#     dataset = gdal.Open(image_dir, gdal.GA_ReadOnly)
#     width = dataset.RasterXSize
#     height = dataset.RasterYSize
#     image = []
#     for x in range(height):
#         a = []
#         for y in range(width):
#             b = []
#             for z in xrange(1, dataset.RasterCount + 1):
#                 band = dataset.GetRasterBand(z)
#                 array = band.ReadAsArray()
#                 b.append(array[x][y])
#             a.append(b)
#         #print(np.asarray(a).shape)
#         image.append(a)
#     image_array = np.asarray(image)
#     return image_array
#------------------------------------------------------------------------------------------------------------------------
def readImagesCaffe(images_root,image_name):
    """
    readImagesCaffe uses a caffe function to load an image from a link

    arguments:
    images_root - base folder containing the images to read (string)
    image_name - image file name (string)

    outputs:
    2-d array of image
    """
    image_array = caffe.io.load_image(images_root + image_name)
    return image_array
#------------------------------------------------------------------------------------------------------------------------
def toScale(L):
    """
    toScale takes a pixel and scales it the max color value

    arguments:
    L - pixel (array(R,G,B))

    outputs:
    scaled pixel array
    """
    new_l = []
    col_l = []
    L = np.asarray(L)
    max_l = L.max()
    min_l = L.min()
    temp = (L-min_l)
    temp2 = 255*temp/max(temp)
    temp3 = temp2.round(0)
    for i in temp3:
        new_l.append(i*100/255)#round((100*i)/max_l,1))
        col_l.append(np.array([i,0,0],dtype='uint8'))
    return (new_l,col_l)
#---------------------------------------------------------------------------------------
def NIOSC2_6(base_images,cluster_images,caffe_objects):
    """
    NIOSC2_6 or New Image of Same Class version 2.6 comapres the similarity of the fc7
    vectors of the base image with each of the segments and returns the final image

    arguments: 
    base_images - list of base image links
    cluster_images - the list of image segments
    caffe_objects - dictionary of caffe set up objects (transformer & net)

    outputs:
    2-d array of pixels that make up the correct image containing the correct segment
    """
    net = caffe_objects["net"]
    transformer = caffe_objects["transformer"]
    base_image_fvs = []
    for i in xrange(len(base_images)):
        fc7 = getFeatureVector(cv2.imread(base_images[i]),transformer,net)
        base_image_fvs.append(copy.copy(fc7))
    cluster_scores = []
    for cluster in cluster_images:
        cluster_2 = np.asarray(to_matrix(cluster,256))
        #plt.imshow(cluster_2)    
        #fig = plt.figure()
        new_fc7 = getFeatureVector(cluster_2,transformer,net)
        score = []
        for item in base_image_fvs:
            temp_score = cosine(new_fc7.tolist(),item.tolist())
            score.append(temp_score)
        #print score
        cluster_scores.append(np.mean(sorted(score)[:2]))
    out_cluster = cluster_images[cluster_scores.index(min(cluster_scores))]
    #plt.imshow(np.asarray(to_matrix(out_cluster,256)))    
    #fig = plt.figure()
    return (np.asarray(to_matrix(out_cluster,256)))
#------------------------------------------------------------------------------------------------------------------------
def NIOSC2(base_image,in_image,size,step):
    """
    NIOSC2 or New Image of Same Class version 2 comapres the similarity of the fc7
    vectors of the base image with each of the segments by creating a similarity heatmap
    over the image and determining which parts of the image are most similar to the base image.
    This turns the final image

    arguments: 
    base_image - list of base image links
    in_image - the list of image segments
    size - number representing the length of the box in pixels
    step - convolution step for the box


    outputs:
    out - a dictionary containing the heatmap, the pixel values for the heatmap, and the image indices
    """
    caffe_objects = setUpCaffe()
    net = caffe_objects["net"]
    transformer = caffe_objects["transformer"]
    base_image = cv2.imread(base_image)
    im = cv2.cvtColor(cv2.imread(in_image), cv2.COLOR_BGR2RGB)
    in_image = im
    flat_pixels = [item for sublist in im for item in sublist]
    #Determining indices
    step_size = step
    small_s = size
    large_s = len(im)
    ims = []
    im_inds = []
    for xstep in range(0,large_s-small_s+1,step_size):
        for ystep in xrange(0,large_s-small_s+1,step_size):
            inds = []
            xtab = im[xstep:xstep+small_s]
            for row in xrange(len(xtab)):
                inds.append(range(ystep+row*large_s+xstep*large_s,ystep+row*large_s+small_s+xstep*large_s))
            im_inds.append(inds)
    #Filling in indices
    images = []
    for blk_im in im_inds:
        inds = [item for sublist in blk_im for item in sublist]
        new_flat_pixels = copy.copy(flat_pixels)
        for ind in inds:
            new_flat_pixels[ind] = np.array([0, 0, 0],dtype='uint8')
        reshape_pixels = to_matrix(new_flat_pixels,256)
        black_image = np.asarray(reshape_pixels)
        images.append(black_image)
    distance_list = []
    orig_fc7 = getFeatureVector(base_image,transformer,net)
    for cur_im in images:
        new_fc7 = getFeatureVector(cur_im,transformer,net)
        distance = cosine(new_fc7.tolist(),orig_fc7.tolist())
        distance_list.append(distance)
    scaled = toScale(distance_list)
    heat_map_pixels = to_matrix(scaled[1],int(np.sqrt(len(distance_list))))
    heat_map = scaled[0]
    out = dict({"heat_map":heat_map,
                "heat_map_pixels":heat_map_pixels,
               "im_inds":im_inds})
    return (out)
#------------------------------------------------------------------------------------------------------------------------
def setUpCaffe(GPU):
    """
    setUpCaffe creates the caffe pre-trained AlexNet with image transformation

    arguments:
    GPU - the GPU (0 or 1) that will be used

    output:
    dictionary of caffe objects
    """
    caffe_root = '/home/ai2-jedi/Documents/code/caffe/'
    model_def = caffe_root + 'models/bvlc_reference_caffenet/deploy.prototxt'
    model_weights = caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'

    net = caffe.Net(model_def,      # defines the structure of the model
                    model_weights,  # contains the trained weights
                    caffe.TEST)     # use test mode (e.g., don't perform dropout)

    # load the mean ImageNet image (as distributed with Caffe) for subtraction
    mu = np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')
    mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values

    # Set up Transformer
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
    transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
    transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
    transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR

    # Caffe gpu set up
    caffe.set_device(GPU)  # if we have multiple GPUs, pick the first one
    caffe.set_mode_gpu()

    out = dict({"net":net,
               "transformer":transformer})
    return (out)
#------------------------------------------------------------------------------------------------------------------------
def which_section(image_cluster,size,step,percentage):
    """
    which_section creates a grid over the image_cluster, and assigns squares on the grid to that
    cluster if the percentage of pixels that are not white are over the percentage argument

    arguments:
    image_cluster - a 2-d array of pixels representing one of the segments
    size - number representing the length of the box in pixels
    step - convolution step for the box
    percentage - threshold for the minimum percentage of pixels needed to assign a square in the grid
                 to a given cluster

    output:
    image_map - 2-d array containing the segment assignments in the grid
    """
    step_size = step
    small_s = size
    large_s = int(np.sqrt(len(image_cluster)))
    ims = []
    #im_inds = []
    #flat_image_cluster = [item for sublist in image_cluster for item in sublist]
    image_map = []
    for xstep in range(0,large_s-small_s+1,step_size):
        for ystep in xrange(0,large_s-small_s+1,step_size):
            inds = []
            xtab = image_cluster[xstep:xstep+small_s]
            for row in xrange(len(xtab)):
                inds.extend(range(ystep+row*large_s+xstep*large_s,ystep+row*large_s+small_s+xstep*large_s))
            #section_inds_flat = [item for sublist in inds for item in sublist]
            section_inds_flat = inds
            section_pixels_f = map(image_cluster.__getitem__,section_inds_flat)
            blank_count = sum(1 for pixel in section_pixels_f if (sum(pixel)==255*3)) #####change to white by *3
            perc_blank = blank_count/float(len(section_pixels_f))
            if perc_blank>percentage:
                image_map.append(0)
            else:
                image_map.append(1)
    #return (to_matrix(image_map,int(np.sqrt(len(image_map)))))
    return (image_map)
#------------------------------------------------------------------------------------------------------------------------
def NIOSC3(base_image,in_image,size,step,sensitivity,penalization,method):
    """
    NIOSC3 or New Image of Same Class version 3 calls version 2 and determines the correct segment

    arguments: 
    base_image - list of base image links
    in_image - the image to be segmented and labeled
    size - number representing the length of the box in pixels
    step - convolution step for the box
    sensitivity - threshold for the minimum percentage of pixels needed to assign a square in the grid
                  to a given cluster
    penalization - threshold value for find_K
    method - method to be used in find_K

    outputs:
    none - it plots the image
    """
    NIOSC_outputs = NIOSC(in_image,penalization,method)
    image_clusters = NIOSC_outputs["images"]
    #blurred_clusters = NIOSC_outputs["blur_images"]
    cluster_counts = NIOSC_outputs["count"]
    NIOSC2_outputs = NIOSC2(base_image, in_image, size, step)
    im_inds = NIOSC2_outputs["im_inds"]
    heatmap_values = NIOSC2_outputs["heat_map"]
    heat_map_pixels = NIOSC2_outputs["heat_map_pixels"]
    clust_scores = []
    #print np.asarray(image_clusters[0]).shape
    for image_cluster in image_clusters:
        cluster_sections = which_section(np.asarray(image_cluster),size,step,sensitivity)
        #print cluster_sections
        clust_scores.append(np.mean(np.array(cluster_sections)*np.array(heatmap_values)))
    final_image = image_clusters[clust_scores.index(min(clust_scores))]
    plt.imshow(np.asarray(to_matrix(final_image,int(np.sqrt(len(final_image))))))
#------------------------------------------------------------------------------------------------------------------------
def NIOSC3_5(base_image,in_image,penalization,method,caffe_objects,con_color):
    """
    NIOSC3_5 or New Image of Same Class version 3.5 gives the option to use the cluster comparison method 
    in NIOSC2_6 or the mean pixel method from NIOSC1_5 to determine the final image

    arguments: 
    base_image - list of base image links
    in_image - the image to be segmented and labeled
    penalization - threshold value for find_K
    method - method to be used in find_K
    caffe_objects - dictionary of caffe set up objects (transformer & net)
    con_color - boolean that determines whether or not to use the mean pixel method (for consistently colored objects)

    outputs:
    final_image - 2-d array of pixels containing the correct segment
    """
    if con_color == True:
        return NIOSC1_5(base_image, in_image,penalization,method)
    else:
        NIOSC_outputs = NIOSC(in_image,penalization,method)
        image_clusters = NIOSC_outputs["images"]
        cluster_counts = NIOSC_outputs["count"]
        final_image = NIOSC2_6(base_image,image_clusters,caffe_objects)
        return (final_image)
#----------------------------------------------------------------------------------------------------------
def image_index(images,image_link):
    """
    image_index gives the index of the image link in a list of images

    arguments:
    images - a list of tuples containing an image link and a 2-d array
    image_link - string of image file name

    outputs:
    index - index of image link in image list
    """
    index = 0
    for image in images:
        if(image[0] == image_link):
            return(index)
        index = index + 1
#------------------------------------------------------------------------------------------------------------------------
def link_index(images,image):
    """
    image_index gives the index of the image in a list of images

    arguments:
    images - a list of tuples containing an image link and a 2-d array
    image - 2-d array of pixels

    outputs:
    index - index of image in image list
    """
    index = 0
    for image in image_feautrues:
        if(image[1] == image):
            return(index)
        index = index + 1
#------------------------------------------------------------------------------------------------------------------------
def calculating_image_vectors(raster_root,transformer,net):
    """
    calculate_image_vectors takes a folder of images and pushes all of them through a pre-trained model

    arguments: 
    raster_root - string address of the folder containing all of the images
    transformer - caffe's image transformation object
    net - caffe's AlexNet object

    output:
    image_features - list of tuples containing the image location and the fc7 vector
    """
    image_features = []
    count = 0
    for folder in os.listdir(raster_root):
        for image in os.listdir(raster_root+folder+"/"):
            image_dir = raster_root+ folder+"/" + image
            image_array = readImagesCaffe(raster_root+ folder+"/",image)
            fc7 = getFeatureVector(image_array,transformer,net)
            image_features.append((image_dir,fc7.tolist()))
            count = count + 1
            if(count % 100 == 0):
                print ("Pushed "+str(count)+" images through AlexNet to get the fc7 vector")
    return (image_features)
#------------------------------------------------------------------------------------------------------------------------
def similar_images(image_features,num_pictures,image_link):
    """
    similar_images uses the fc7 vectors to calculate distances in the multi-dimensional vector space
    It ultimately top closest pictures

    arguments:
    image_features - list of image locations & image vectors
    num_pictures - threshold for the top number of images to retain
    image_link - the base image to compare to

    output:
    a list of the most similar images
    """
    pic_index = image_index(image_features,image_link)
    cosines = []
    count = 0
    for i in range(len(image_features)):
        if(i != pic_index):
            cos = cosine(image_features[pic_index][1], image_features[i][1])
            cosines.append((image_features[i][0],cos))
            count = count + 1
            if(count%500 == 0):
                print("Calculated the similarity of "+str(count)+" images")           
    sorted_cosines = sorted(cosines, key=lambda x: x[1])
    top_pics = sorted_cosines[:num_pictures]
    out_pictures = []
    for pic in top_pics:
        #plt.figure()
        image = caffe.io.load_image(pic[0])
        #plt.imshow(image)
        out_pictures.append(pic[0])
    return out_pictures
#------------------------------------------------------------------------------------------------------------------------
def similar_images_2(image_features,image_links,sim_thresh):
    """
    similar_images_2 uses the fc7 vectors to calculate distances in the multi-dimensional vector space
    It ultimately top closest pictures

    arguments:
    image_features - list of image locations & image vectors
    sim_thresh - threshold for the top number of images to retain (given in standard deviations from the mean)
    image_links - the base images to compare to

    output:
    a list of the most similar images
    """
    image_features = copy.copy(image_features)
    pic_index = []
    for link in image_links:
        check_ind = image_index(image_features,image_link)
        if check_ind == None:
            image_array = readImagesCaffe("",link)
            image_array_flat = [item for sublist in image_array for item in sublist]
            counter = 0
            cont = False
            for pixel in image_array_flat:
                if (pixel.sum == 0):
                    counter+=1
                    if counter >= 10:
                        cont = True
            if cont==True:
                continue
            fc7 = getFeatureVector(image_array,transformer,net)
            image_features.append((link,fc7.tolist()))
            cur_ind = len(image_features)
            pic_index.append(cur_ind-1)
        else:
            pix_index.append(check_ind)
    cosines = []
    count = 0
    if (type(pic_index) == list):
        for i in range(len(image_features)):
            if(i not in pic_index):
                temp_cosines = []
                for j in xrange(len(pic_index)):
                    cos = cosine(image_features[pic_index[j]][1], image_features[i][1])
                    temp_cosines.append(cos)
                avg_cos = np.asarray(temp_cosines).mean()
                cosines.append((image_features[i][0],avg_cos))
                count = count + 1
                if(count%500 == 0):
                    print("Calculated the similarity of "+str(count)+" images")
            else:
                print "different image"
    else:
        for i in range(len(image_features)):
            if(i != pic_index):
                cos = cosine(image_features[pic_index][1], image_features[i][1])
                cosines.append((image_features[i][0],cos))
                count = count + 1
                if(count%500 == 0):
                    print("Calculated the similarity of "+str(count)+" images")

    sorted_cosines = sorted(cosines, key=lambda x: x[1])
    #plt.hist(sorted_cosines)
    cosine_values = [x[1] for x in sorted_cosines]
    mean = np.mean(cosine_values)
    std = np.std(cosine_values)
    num_pictures = list(np.asarray(cosine_values)<(mean-(sim_thresh)*std)).index(False)-1
    top_pics = sorted_cosines[:num_pictures]
    out_pictures = []
    for pic in top_pics:
        out_pictures.append(pic[0])
    print str(len(out_pictures)) + " out of " + str(len(cosine_values)) + " similar pictures found!"
    return out_pictures
#------------------------------------------------------------------------------------------------------------------------
#Make sure this labels dictionary is consistent across all processes
labels_dict = dict({'bridge': 17, 'container': 15, 'asphalt': 6, 'heap of sand': 7, 'cars': 13, 'pipes': 10,
 'rubble': 11, 'reinforcement': 8, 'water': 4, 'concrete': 1, 'trees': 14, 'bike lane': 12,
 'background': 0, 'wooden boards': 9, 'foundations': 2, 'grass': 5, 'heavy earthy equipment': 16,
 'concrete rings': 3})
#------------------------------------------------------------------------------------------------------------------------
def convert_images(final_images, original_images, class_name,output_root):
    """
    convert_images takes the images and converts them into labeled pairs and saves them in a directory

    arguments:
    final_images - list of images (2-d arrays of pixels) that have been segmented
    original_images - the original images that were pushed through the segmentation
    class_name - the class that is being replicated
    output_root - the string of the address of the directory where the images need to go

    output:
    the list of string addresses pointing to the shape files (labels)
    """
    shape_links = []
    if len(final_images) != len(original_images):
        print "Image Arrays are not the same length"
        return None
    if os.path.exists(output_root+"original_images/"):
        existing = os.listdir(output_root+"original_images/")
        numbers = []
        for item in existing:
            temp = item.split("/")
            if str(temp[len(temp)-1].split("_")[0]) == class_name:
                numbers.append(int(temp[len(temp)-1].split("_")[1][:-4]))
        placeholder = max(numbers)
        print "There are "+str(placeholder)+" images in this class - adding onto the class"
    else:
        os.makedirs(output_root+"original_images/")
        placeholder = 0
    if not os.path.exists(output_root+"shapes/"):
        os.makedirs(output_root+"shapes/")
    if not os.path.exists(output_root+"original_images_wgb/"):
        os.makedirs(output_root+"original_images_wgb/")
    for im_ind in xrange(len(final_images)):
        original_image =cv2.imread(original_images[im_ind])
        final_image = final_images[im_ind]
        fi_flat = [item for sublist in final_image for item in sublist]
        con_flat = []
        for pixel in fi_flat:
            if (pixel.sum()==3*255):
                con_flat.append(np.array(0,dtype='uint8'))
            else:
                con_flat.append(labels_dict[class_name])
        out = to_matrix(con_flat,len(final_images[im_ind]))
        image_path = "original_images/" + class_name + "_" + str(im_ind+placeholder) + ".png"
        image_path_w = "original_images_wbg/" + class_name + "_" + str(im_ind+placeholder) + ".png"
        shape_path = "shapes/" + class_name + "_" + str(im_ind+placeholder) + ".png" 
        shape_links.append(output_root+shape_path)
        cv2.imwrite(output_root+image_path_w,final_image)
        cv2.imwrite(output_root+image_path,original_image)
        cv2.imwrite(output_root+shape_path,np.asarray(out))
        if (im_ind % 10 == 0):
            print "Converted and Saved "+str(im_ind)+" images"
    print "Images are saved in "+output_root
    return shape_links
#-----------------------------------------------------------------------------------
def setUpPicture(image_link,k):
    """
    setUpPicture takes an image and segments it for a user to create base images

    arguments:
    image_link - string of address pointing to an image
    k - number of clusters to cluster the pixels into

    outputs:
    images - a list of the segments (each represented as 2-d arrays of pixels)
    """
    im = cv2.imread(image_link)
    plt.imshow(im)
    flat_pixels = [item for sublist in im for item in sublist]
    flat_pixels_2 = []
    for pixel in flat_pixels:
        pix = list(pixel)
        p_sum = sum(pix)
        p_max = max(pix)
        flat_pixels_2.append([pixcon(pix[0],p_sum),pixcon(pix[1],p_sum),pixcon(pix[2],p_sum)])
    if(k==0):
        k = find_K2(flat_pixels_2)
    clustered_pixels = KMeans(n_clusters=k,n_init=10).fit_predict(flat_pixels)#_2)
    check = check_cluster_dist(clustered_pixels)
    print check
    pixel_labels = list(clustered_pixels)
    fig = plt.figure()
    rp_o = to_matrix(flat_pixels_2,256)
    plt.imshow(np.asarray(rp_o))
    images =[]
    for cluster in check.keys():
        new_flat_pixels = copy.copy(flat_pixels)
        fig = plt.figure()
        for ind in xrange(len(new_flat_pixels)):
            if pixel_labels[ind]!=int(cluster):
                new_flat_pixels[ind]=np.array([255,255,255],dtype='uint8')
        reshape_pixels = to_matrix(new_flat_pixels,256)
        new_image = np.asarray(reshape_pixels)
        images.append(new_image)
        plt.imshow(new_image)
    return images
#------------------------------------------------------------------------------------
def setUpImage_WL(base_image_link,file_to_save_as=None):
    """
    setUpImage_WL automatically creates a base image from an image that has a label

    arguments:
    base_image_link - string of address pointing to an image (image needs to be in an
        image folder in a parent directory and must have the same name as the label. The
        label needs to be in a shape folder also in the parent directory)
    file_to_save_as - string of a file name

    output:
    none
    """
    link_split = base_image_link.split("/")
    image_file = link_split.pop()
    image_folder = link_split.pop()
    folders = os.listdir("/".join(link_split))
    #base_folder = link_split.pop()
    for folder in folders:
        if folder not in ['.ipynb_checkpoints', image_folder]:
            shape_folder = "/".join(link_split)+"/"+folder
    shape_file = shape_folder+"/"+image_file
    ##__________
    shape = cv2.imread(shape_file)
    image = cv2.imread(base_image_link)
    shape_flat = [item for sublist in shape for item in sublist]
    image_flat = [item for sublist in image for item in sublist]
    new_image_flat = copy.copy(image_flat)
    for pix_ind in xrange(len(shape_flat)):
        if shape_flat[pix_ind][0] == 0: 
            new_image_flat[pix_ind] = np.array([255,255,255],dtype='uint8')
    new_image = np.asarray(to_matrix(new_image_flat,len(image)))
    plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
    if file_to_save_as is None:
        print "Please enter a file name to save the image"
    else:
        cv2.imwrite(file_to_save_as,new_image)