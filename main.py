# ===============================================================================
# Author: Teodoro Valença de Souza Wacholski
# Universidade Tecnológica Federal do Paraná
# ===============================================================================

from math import sqrt
import numpy as np
import cv2
from matplotlib import pyplot as plt
import sys

np.set_printoptions(threshold=sys.maxsize)

# Feel free to change the image.
# There are some images in the project that you can test.
# They are 60.bmp, 82.bmp, 114.bmp, 150.bmp, 205.bmp
INPUT_IMAGE = '82.bmp'

# Set to True if you want to see the
# histograms of the component sizes
HISTOGRAM = True

pxBlob = -1

def flood(label, labelMatrix, y0, x0, n_pixels):
    """
    Flood fill algorithm to label connected components.
    Params:
        label: label value to assign to the connected component
        labelMatrix: matrix representing the labels of each pixel of the image
        y0: starting y-coordinate for flood fill
        x0: starting x-coordinate for flood fill
        n_pixels: current number of pixels in the connected component
    Returns:
        A dictionary containing information about the connected component:
            'T': top edge of the component
            'L': left edge of the component
            'B': bottom edge of the component
            'R': right edge of the component
            'n_pixels': number of pixels in the connected component
    """

    global pxBlob
    labelMatrix[y0,x0] = label
    rows, cols = labelMatrix.shape
    pxBlob += 1
    n_pixels += 1
    n = 0

    # Temporary storage of flood output to compare to info
    temp = {
        'T': y0,
        'L': x0,
        'B': y0,
        'R': x0,
        'n_pixels': 0
    }

    # Flood function output
    info = {
        'T': temp['T'],
        'L': temp['L'],
        'B': temp['B'],
        'R': temp['R'],
        'n_pixels': n_pixels
    }

    # Neighbors array to iterate, taking care to image bounds
    neighbors = [
        labelMatrix[y0 + 1, x0] if (y0 + 1) < rows else 0,
        labelMatrix[y0, x0 + 1] if (x0 + 1) < cols else 0,
        labelMatrix[y0, x0 - 1] if (x0 - 1) >= 0 else 0,
        labelMatrix[y0 - 1, x0] if (y0 - 1) >= 0 else 0,
    ]
    neighborsIndex = [[y0 + 1, x0], [y0, x0 + 1], [y0, x0 - 1], [y0 - 1, x0]]

    # For each neighbor
    for index in range(len(neighbors)):
        # Check for image bounds
        # if ((index == 0 and (y0+1) < rows) or (index == 1 and (x0+1) < cols) or (index == 2 and (x0-1) >= 0) or (index == 3 and (y0-1) >= 0)):
        #     se o vizinho é de interesse e não foi visitado...
        if (
            (index == 0 and (y0 + 1) < rows)
            or (index == 1 and (x0 + 1) < cols)
            or (index == 2 and (x0 - 1) >= 0)
            or (index == 3 and (y0 - 1) >= 0)
        ):
            # If the neighbor is a pixel of interest and was not visited
            if (neighbors[index] == -1):
                # Flood fill in the neighbor
                temp = flood(
                    label,
                    labelMatrix,
                    neighborsIndex[index][0],
                    neighborsIndex[index][1],
                    n_pixels
                )
        # Verify if the bounds have increased
        if (temp['T'] < info['T']):
            info['T'] = temp['T']
        if (temp['B'] > info['B']):
            info['B'] = temp['B']
        if (temp['L'] < info['L']):
            info['L'] = temp['L']
        if (temp['R'] > info['R']):
            info['R'] = temp['R']
        
        # Sum temp pixels to flood output
        n += temp['n_pixels']

    info['n_pixels'] = n_pixels + n

    return info

#-------------------------------------------------------------------------------

def labeling (img, min_pixels=0):
    """
    Labeling using Flood Fill.
    Params:
        img: input image
        min_pixels: discard components with less than min_pixels
    Returns:
        output_list: a list, where each item is a dictionary with the fields:
            'label': component label
            'n_pixels': component number of pixels
            'T', 'L', 'B', 'R': component edge coordinates (Top, Left, Bottom and Right)
        size_list: a list with component sizes
    """

    global pxBlob

    rows, cols = img.shape

    labelMatrix = np.empty((rows, cols))
    output_list = []
    size_list = []

    # Labels the pixels of interest as unvisited
    for row in range(rows):
        for col in range(cols):
            if (img[row, col] == 0):
                labelMatrix[row, col] = 0
            else:
                labelMatrix[row, col] = -1

    sys.setrecursionlimit(5000)
    label = 1

    # For each pixel
    for row in range(rows):
        for col in range(cols):
            # Flood fill if it's an interest unvisited pixel
            if (labelMatrix[row, col] == -1):
                pxBlob = -1
                n_pixels = 0
                info = flood(label, labelMatrix, row, col, n_pixels)
                component = {
                    "label": label,
                    "n_pixels": info['n_pixels'],
                    'T': info['T'],
                    'L': info['L'],
                    'B': info['B'],
                    'R': info['R']
                }
                if (component['n_pixels'] > min_pixels):
                    output_list.append(component)
                    label += 1
                    size_list.append(pxBlob)

    return output_list, size_list


def main():
    img = cv2.imread(INPUT_IMAGE, cv2.IMREAD_COLOR)

    # Preprocessing operations
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_255 = np.copy(img_gray)
    img_float = img_gray.astype(np.float32) / 255
    cv2.imshow('IMG', img_float)

    img_blur = cv2.medianBlur(img_255, 7)

    img_canny = cv2.Canny(img_blur, 100, 110)
    cv2.imshow('Canny', img_canny)

    img_median = cv2.medianBlur(img_float, 5)
    img_median = cv2.medianBlur(img_median, 5)
    cv2.imshow('median twice', img_median)

    img_gaussian = cv2.GaussianBlur(img_median, (301, 301), 0)

    blobs = img_median - img_gaussian
    blobs = np.where(blobs < 0, 0, blobs)

    cv2.imshow('Blobs', blobs)
    cv2.imshow('Gaussian', img_gaussian)

    _, otsu = cv2.threshold((blobs * 255).astype(np.uint8),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    cv2.imshow('otsu', otsu)

    kernel = np.ones((3, 3))

    erosion = cv2.erode(otsu, kernel, iterations=1)
    closing = cv2.dilate(erosion, kernel, iterations=1)
    cv2.imshow('closing', closing)

    components, size_list = labeling(closing)
    sizeListCopy = np.copy(size_list)
    size_list.sort()
    median_sizes = np.median(size_list)
    mean = np.mean(size_list)
    std = np.std(size_list)

    print("Median = %.2f" % median_sizes)
    print("Mean = %.2f" % mean)
    print("Standard deviation = %.2f" % std)

    # Draw rectangles around the labeled components
    for c in components:
        cv2.rectangle (img, (c ['L'], c ['T']), (c ['R'], c ['B']), (0, 255, 0))

    figure, axis = plt.subplots(1, 2)

    """
    BoxPlot of size_list
    Note that bp will be used to get some statistical information:
        - Lower Whisker
        - Lower Quartile
        - Median
        - Upper Quartile
        - Upper Whisker
    """
    bp = axis[0].boxplot(size_list)
    axis[0].set_title('Boxplot 1')

    axis[1].violinplot(size_list)
    axis[1].set_title('Violin 1')

    # Statistical analysis
    dict1 = {}
    dict1['lower_whisker'] = bp['whiskers'][0].get_ydata()[1]
    dict1['lower_quartile'] = bp['boxes'][0].get_ydata()[1]
    dict1['median'] = bp['medians'][0].get_ydata()[1]
    dict1['upper_quartile'] = bp['boxes'][0].get_ydata()[2]
    dict1['upper_whisker'] = bp['whiskers'][0].get_ydata()[1]
    print(dict1)

    avg_sum = 0
    avg_count = 0

    """
    Here the lower quartile and the median will be use to get the
    average rice size.
    We assume that each component with a size between the lower
    quartile and the median are a single rice.
    """
    # For each component size
    for i in size_list:
        # If the component size is greather than or equal to lower quartile
        # and less than or equal to median
        if (i >= dict1['lower_quartile'] and i <= dict1['median']):
            avg_sum += i
            avg_count += 1
    
    # Average rice size
    avg_rice = avg_sum / avg_count
    
    print('Average rice size: %.2f' % avg_rice)

    rice_count = 0
    not_remainder = 0
    remainders = []
    denominator = avg_rice

    # For each component/blob size
    for blob in range(len(size_list)):
        # If the component size is less than or equal to average rice size
        if size_list[blob] <= avg_rice:
            component = components[np.where(sizeListCopy == size_list[blob])[0][0]]
            cv2.putText(img, '1', (component['L'], component['T']), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            # Every component here will count as a rice
            rice_count += 1
            not_remainder += 1
        else:
            component = components[np.where(sizeListCopy == size_list[blob])[0][0]]
            # Components here will be divided by the average rice size,
            # to get the numbers of rice in the blob
            rice_count += size_list[blob] // denominator
            not_remainder += size_list[blob] // denominator
            # Remainder represents the rest of division between
            # the component and the average rice size.
            # This value will be divided by average rice size * 0.7
            # to verify if it representa a broken rice
            remainder = size_list[blob] % denominator
            cv2.putText(img, str(int(size_list[blob] // denominator) + int(remainder // (denominator * 0.7))), (component['L'], component['T']), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            remainders.append(remainder)
            rice_count += remainder // (denominator * 0.7)

    print('Rice Count: ', rice_count)

    cv2.imshow('Rectangles', img)

    if HISTOGRAM:
        figure, axis = plt.subplots(2, 1)
        figure.set_label('Histograms')
        axis[0].hist((img_float*255).ravel(),255,[0,255])
        axis[0].set_title('Original Image')
        axis[1].hist((img_median * 255).ravel(),255,[0,255])
        axis[1].set_title('Median')

    plt.show()
    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()