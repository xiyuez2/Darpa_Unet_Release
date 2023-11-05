import argparse
import math
import cv2
import os
import numpy as np
from PIL import Image
import rasterio
# import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
# from data_util import DataLoader
from h5Image import H5Image

from torchvision import transforms
import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl

from tqdm import tqdm
from models import SegmentationModel

h5_image = None




def save_plot_as_png(prediction_result, map_name, legend, outputPath):
    """
    This function visualizes and saves the True Segmentation, Predicted Segmentation, Full Map, 
    and the Legend in a single image.

    Parameters:
    - prediction_result: 2D numpy array representing the predicted segmentation.
    - map_name: string, the name of the map.
    - legend: string, the name of the legend.
    - outputPath: string, the directory where the output image will be saved.

    Returns:
    - None. The output image is saved in the specified directory.
    """

    global h5_image  # Using a global variable to access the h5 image object

    # Fetching the true segmentation layer
    true_seg = h5_image.get_layer(map_name, legend)
    
    # Fetching the full map
    full_map = h5_image.get_map(map_name)
    
    # Fetching the legend patch from h5 image
    legend_patch = h5_image.get_legend(map_name, legend)
    
    # Resize the legend to the specified dimensions
    legend_resized = cv2.resize(legend_patch, (256,256))

    # Convert the legend to uint8 range [0, 255] if its dtype is float32
    # if legend_resized == float32:
    # legend_resized = (legend_resized * 255).numpy().astype(np.uint8)

    # Construct the output image path
    output_image_path = os.path.join(outputPath, f"{map_name}_{legend}_visual.png")

    # Create a figure with 4 subplots: true segmentation, predicted segmentation, full map, and legend
    fig, axarr = plt.subplots(1, 4, figsize=(20,5))

    # Using GridSpec for custom sizing of the subplots
    gs = gridspec.GridSpec(1, 4, width_ratios=[1,1,1,1])

    # Display the true segmentation
    ax0 = plt.subplot(gs[0])
    ax0.imshow(true_seg)
    ax0.set_title('True segmentation')
    ax0.axis('off')

    # Display the predicted segmentation
    ax1 = plt.subplot(gs[1])
    ax1.imshow(prediction_result)
    ax1.set_title('Predicted segmentation')
    ax1.axis('off')

    # Display the full map
    ax2 = plt.subplot(gs[2])
    ax2.imshow(full_map)
    ax2.set_title('Map')
    ax2.axis('off')

    # Display the resized legend
    ax3 = plt.subplot(gs[3])
    ax3.imshow(legend_resized)
    ax3.set_title('Legend')
    ax3.axis('off')

    # Adjust layout to ensure there's no overlap
    plt.tight_layout()

    # Save the combined visualization to the specified path
    plt.savefig(output_image_path)

def prediction_mask(prediction_result, map_name, legend, outputPath):
    """
    Apply a mask to the prediction image to isolate the area of interest.

    Parameters:
    - prediction_result: numpy array, The output of the model after prediction.
    - map_name: str, The name of the map used for prediction.

    Returns:
    - masked_img: numpy array, The masked prediction image.
    """
    global h5_image

    # Get the map array corresponding to the given map name
    map_array = np.array(h5_image.get_map(map_name))
    print("map_array", map_array.shape)

    # Convert the RGB map array to grayscale for further processing
    gray = cv2.cvtColor(map_array, cv2.COLOR_BGR2GRAY)

    # Identify the most frequent pixel value, which will be used as the background pixel value
    pix_hist = cv2.calcHist([gray],[0],None,[256],[0,256])
    background_pix_value = np.argmax(pix_hist, axis=None)

    # Flood fill from the corners to identify and modify the background regions
    height, width = gray.shape[:2]
    corners = [[0,0],[0,height-1],[width-1, 0],[width-1, height-1]]
    for c in corners:
        cv2.floodFill(gray, None, (c[0],c[1]), 255)

    # Adaptive thresholding to remove small noise and artifacts
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 4)

    # Detect edges using the Canny edge detection method
    thresh_blur = cv2.GaussianBlur(thresh, (11, 11), 0)
    canny = cv2.Canny(thresh_blur, 0, 200)
    canny_dilate = cv2.dilate(canny, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))

    # Detect contours in the edge-detected image
    contours, hierarchy = cv2.findContours(canny_dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    # Retain only the largest contour
    contour = sorted(contours, key=cv2.contourArea, reverse=True)[0]
    
    # Create an empty mask of the same size as the prediction_result
    wid, hight = prediction_result.shape[0], prediction_result.shape[1]
    mask = np.zeros([wid, hight])
    mask = cv2.fillPoly(mask, pts=[contour], color=(1)).astype(np.uint8)

    # Convert prediction result to a binary format using a threshold
    prediction_result_int = (prediction_result > 0.5).astype(np.uint8)

    # Apply the mask to the thresholded prediction result
    masked_img = cv2.bitwise_and(prediction_result_int, mask)

    return masked_img

def perform_inference(legend_patch, map_patch, model):
    """
    Perform inference on a given map patch and legend patch using a trained model.

    Parameters:
    - legend_patch: numpy array, The legend patch from the map.
    - map_patch: numpy array, The map patch for inference.
    - model: tensorflow.keras Model, The trained deep learning model.

    Returns:
    - prediction: numpy array, The prediction result for the given map patch.
    """
    global h5_image
    # get im from h5
    
    legend_resized = legend_patch # cv2.resize(legend_patch, (256,256))
    map_patch_resize = map_patch#cv2.resize(map_patch, (256,256))
    # print("map_patch_resize:",np.max(map_patch_resize), np.min(map_patch_resize))
    # print("legend_resized:",np.max(legend_resized), np.min(legend_resized))
    # print("debug:", map_patch_resize.shape,legend_resized.shape)

    data_transforms = transforms.Compose([
        # transforms.ToPILImage(),
        transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
        # transforms.ColorJitter(brightness = 0.1, contrast = 0.1, saturation = 0.1, hue=0.5)                        
        ])
    mask_transforms = transforms.Compose([
        # transforms.ToPILImage(),
        transforms.Resize((h5_image.patch_size, h5_image.patch_size)),
        transforms.ToTensor()
        ])
    # Obtain the prediction from the trained model
    # print(np.mean(map_patch_resize),np.std(map_patch_resize),np.max(map_patch_resize),np.min(map_patch_resize))

    map_img = data_transforms(Image.fromarray(map_patch_resize))
    legend_img = data_transforms(Image.fromarray(legend_resized))

    # print("map_img:",np.max(map_img), np.min(map_img))
    # print("legend_img:",np.max(legend_img), np.min(legend_img))
    # print("debug:", map_img.shape,legend_img.shape)
    batch = {
            "map_img": map_img.unsqueeze(0),    # 3 H W
            "legend_img": legend_img.unsqueeze(0), # 1 3 H W - > 3 H W 
            "GT": map_img.unsqueeze(0),# fake img 
            'qry_ori_size': torch.tensor((256,256))
        }
    # for debug
    # for k in batch:
    #     try:
    #         print(batch[k].size())
    #     except:
    #         print(k)

    prediction = model.test_step(batch,0)

    return prediction.squeeze().detach().cpu().numpy() > 0.5

def save_results(prediction, map_name, legend, outputPath):
    """
    Save the prediction results to a specified output path.

    Parameters:
    - prediction: The prediction result (should be a 2D or 3D numpy array).
    - map_name: The name of the map.
    - legend: The legend associated with the prediction.
    - outputPath: The directory where the results should be saved.
    """
    output_image_path = os.path.join(outputPath, f"{map_name}_{legend}.tif")

    # Convert the prediction to an image
    # Note: The prediction array may need to be scaled or converted before saving as an image
    prediction_image = Image.fromarray((prediction*255).astype(np.uint8))

    # Save the prediction as a tiff image
    prediction_image.save(output_image_path, 'TIFF')

    ### Waiting for georeferencing data
    # This section will be used in future releases to save georeferenced images.

    ## Waiting for georeferencing data
    # with rasterio.open(map_img_path) as src:
    #     metadata = src.meta

    # metadata.update({
    #     'dtype': 'uint8',
    #     'count': 1,
    #     'height': reconstructed_image.shape[0],
    #     'width': reconstructed_image.shape[1],
    #     'compress': 'lzw',
    # })

    # with rasterio.open(output_image_path, 'w', **metadata) as dst:
    #     dst.write(reconstructed_image, 1)

def main(args):
    """
    Main function to orchestrate the map inference process.

    Parameters:
    - args: Command-line arguments.
    """
    global h5_image

    # Load the HDF5 file using the H5Image class
    print("Loading the HDF5 file.")
    h5_image = H5Image(args.mapPath, mode='r', patch_border=0)

    # Get map details
    print("Getting map details.")
    map_name = h5_image.get_maps()[0]
    print(f"Map Name: {map_name}")
    all_map_legends = h5_image.get_layers(map_name)
    print(f"All Map Legends: {all_map_legends}")
    
    # Filter the legends based on the feature type
    if args.featureType == "Polygon":
        map_legends = [legend for legend in all_map_legends if "_poly" in legend]
    elif args.featureType == "Point":
        map_legends = [legend for legend in all_map_legends if "_pt" in legend]
    elif args.featureType == "Line":
        map_legends = [legend for legend in all_map_legends if "_line" in legend]
    elif args.featureType == "All":
        map_legends = all_map_legends
    
    # Get the size of the map
    map_width, map_height, _ = h5_image.get_map_size(map_name)
    
    # Calculate the number of patches based on the patch size and border
    num_rows = math.ceil(map_width / h5_image.patch_size)
    num_cols = math.ceil(map_height / h5_image.patch_size)
    
    model = SegmentationModel.load_from_checkpoint(checkpoint_path = args.modelPath,args = args)
    model.eval()
    # Loop through the patches and perform inference
    for legend in (map_legends):
        print(f"Processing legend: {legend}")
        
        # Create an empty array to store the full prediction
        full_prediction = np.zeros((map_width, map_height))

        # Get the legend patch
        legend_patch = h5_image.get_legend(map_name, legend)

        # Iterate through rows and columns to get map patches
        for row in range(num_rows):
            for col in range(num_cols):

                map_patch = h5_image.get_patch(row, col, map_name)

                # Get the prediction for the current patch
                prediction = perform_inference(legend_patch, map_patch, model)
                # print(f"Prediction for patch ({row}, {col}) completed.")

                # Calculate starting indices for rows and columns
                x_start = row * h5_image.patch_size
                y_start = col * h5_image.patch_size

                # Calculate ending indices for rows and columns
                x_end = x_start + h5_image.patch_size
                y_end = y_start + h5_image.patch_size

                # Adjust the ending indices if they go beyond the image size
                x_end = min(x_end, map_width)
                y_end = min(y_end, map_height)

                # Adjust the shape of the prediction if necessary
                prediction_shape_adjusted = prediction[:x_end-x_start, :y_end-y_start]

                # Assign the prediction to the correct location in the full_prediction array
                full_prediction[x_start:x_end, y_start:y_end] = prediction_shape_adjusted
        
        # Mask out the map background pixels from the prediction
        print("Applying mask to the full prediction.")
        masked_prediction = prediction_mask(full_prediction, map_name, legend, args.outputPath)

        save_plot_as_png(masked_prediction, map_name, legend, args.outputPath)

        # Save the results
        print("Saving results.")
        save_results(masked_prediction, map_name, legend, args.outputPath)
    
    # Close the HDF5 file
    print("Inference process completed. Closing HDF5 file.")
    h5_image.close()

if __name__ == "__main__":
    # Command-line interface setup
    parser = argparse.ArgumentParser(description="Perform inference on a given map.")
    parser.add_argument("--mapPath", required=True, help="Path to the hdf5 file.")
    parser.add_argument("--featureType", choices=["Polygon", "Point", "Line", "All"], default="Polygon", help="Type of feature to detect. Three options, Polygon, Point, and, Line")
    parser.add_argument("--outputPath", required=True, help="Path to save the inference results. ")
    parser.add_argument("--modelPath", default="./inference_model/Unet-attentionUnet.h5", help="Path to the trained model. Default is './inference_model/Unet-attentionUnet.h5'.")
    parser.add_argument("--model", default="Unet", help="model name default to unet")
    
    args = parser.parse_args()
    main(args)

# Test command
# python inference.py --mapPath "/projects/bbym/shared/data/commonPatchData/256/OK_250K.hdf5" --featureType "Polygon" --outputPath "/projects/bbym/nathanj/attentionUnet/infer_results" --modelPath "/projects/bbym/nathanj/attentionUnet/inference_model/Unet-attentionUnet.h5"
# python inference.py --mapPath "/projects/bbym/shared/data/commonPatchData/256/OK_250K.hdf5" --outputPath res --modelPath jaccard_index_value\=0.9229.ckpt 

