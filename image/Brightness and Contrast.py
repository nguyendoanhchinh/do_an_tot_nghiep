import math
import imageio
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage import img_as_float
from skimage.exposure import rescale_intensity, adjust_gamma

plt.close('all')
def map_value(
        value, 
        range_in=(0,1), 
        range_out=(0,1), 
        invert=False, 
        non_lin_convex=None, 
        non_lin_concave=None):

    # truncate value to within input range limits
    if value > range_in[1]: value = range_in[1]
    if value < range_in[0]: value = range_in[0]
    
    # map values linearly to [0,1]
    value = (value - range_in[0]) / (range_in[1] - range_in[0])
    
    # invert values
    if invert is True: value = 1 - value
    
    # apply convex non-linearity 
    if non_lin_convex is not None:
        value = (value * non_lin_convex) / (1 + non_lin_convex - value)
         
    # apply concave non-linearity
    if non_lin_concave is not None:
        value = ((1 + non_lin_concave) * value) / (non_lin_concave + value)
    
    # mapping value to the output range in a linear way
    value = value * (range_out[1] - range_out[0]) + range_out[0]
    
    return value

def get_membership_luts(
        resolution=256, 
        lower_threshold=0.35, 
        upper_threshold=0.65,
        verbose=False):
    lut_lower = np.zeros(resolution, dtype='float')
    lut_mid = np.zeros(resolution, dtype='float')
    lut_upper = np.zeros(resolution, dtype='float')
    
    for i in range(resolution):
        
        i_float = i / (resolution - 1)
        
        # lower trapezoid membership function
        if i_float <= lower_threshold:
            lut_lower[i] = i_float / lower_threshold
        else:
            lut_lower[i] = 1
            
        # middle trapezoid membership function
        if i_float <= lower_threshold:
            lut_mid[i] = i_float / lower_threshold
        elif i_float <= upper_threshold:
            lut_mid[i] = 1
        else:
            lut_mid[i] = (1 - i_float) / (1 - upper_threshold)
            
        # upper trapezoid membership function
        if i_float <= upper_threshold:
            lut_upper[i] = 1
        else:
            lut_upper[i] = (1 - i_float) / (1 - upper_threshold)
        
    if verbose is True:
        plt.figure()
        
        plt.subplot(1,3,1)
        plt.plot(lut_lower)
        plt.title('Lower')
        plt.grid(True)
        
        plt.subplot(1,3,2)
        plt.plot(lut_mid)
        plt.title('Middle')
        plt.grid(True)
        
        plt.subplot(1,3,3)
        plt.plot(lut_upper)
        plt.title('Upper')
        plt.grid(True)
        
        plt.suptitle('Trapezoid membership functions')
        plt.show()
    return lut_lower, lut_mid, lut_upper
def get_sigmoid_lut(
        resolution=256, 
        threshold=0.2, 
        non_linearirty=0.2, 
        verbose=False):
    max_value = resolution - 1  # the maximum attainable value
    thr = threshold * max_value  # threshold in the range [0,resolution-1]
    alpha = non_linearirty * max_value  # controls non-linearity degree
    beta = max_value - thr
    if beta == 0: beta = 0.001
    lut = np.zeros(resolution, dtype='float')
    for i in range(resolution):
        i_comp = i - thr  # complement of i
        # upper part of the piece-wise sigmoid function
        if i >= thr:
            lut[i] = (((((alpha + beta) * i_comp) / (alpha + i_comp)) * 
                         (1 / (2 * beta))) + 0.5)
        else:
            lut[i] = (alpha * i) / (alpha - i_comp) * (1 / (2 * thr))
    
    if verbose is True:
        plt.figure()
        plt.plot(lut)
        plt.title('Sigmoid LUT | ' + 
                  'thr=' + str(int(thr)) + ' (' + str(round(threshold, 3)) + 
                  ') | nonlin=' + str(int(alpha)) + 
                  ' (' + str(round(non_linearirty, 3)) + ')')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    return lut
def get_photometric_mask(
        image, 
        smoothing=0.2, 
        grayscale_out=True, 
        verbose=False):
    # internal parameters
    THR_A = smoothing
    THR_B = 0.04  # ~10/255
    NON_LIN = 0.12  # ~30/255
    LUT_RES = 256
    # get sigmoid LUTs
    lut_a = get_sigmoid_lut(
            resolution=LUT_RES, 
            threshold=THR_A, 
            non_linearirty=NON_LIN, 
            verbose=verbose
            )
    lut_a_max = len(lut_a) -1
    lut_b = get_sigmoid_lut(
            resolution=LUT_RES, 
            threshold=THR_B, 
            non_linearirty=NON_LIN, 
            verbose=verbose
            )
    lut_b_max = len(lut_b) -1
    # dealing with different number of channels
    if len(image.shape) == 3:
        if grayscale_out is True:
            image_ph_mask = rgb2gray(image.copy())  # [0,1] 2D
        else:
            image_ph_mask = img_as_float(image.copy())  # [0,1] 3D
    elif len(image.shape) == 2:
        image_ph_mask = img_as_float(image.copy())  # [0,1] 2D
    else:
         image_ph_mask = img_as_float(image.copy())  # [0,1] ?D 
    # if image is 2D, expand dimensions to 3D for code compatibility
    # (filtering assumes a 3D image)
    if len(image_ph_mask.shape) == 2:
        image_ph_mask = np.expand_dims(image_ph_mask, axis=2)
    # up -> down
    for i in range(1, image_ph_mask.shape[0]-1):
        d = np.abs(image_ph_mask[i-1,:,:] - image_ph_mask[i+1,:,:])  # diff
        d = lut_a[(d * lut_a_max).astype(int)]
        image_ph_mask[i,:,:] = ((image_ph_mask[i,:,:] * d) + 
                              (image_ph_mask[i-1,:,:] * (1-d)))
    # left -> right
    for j in range(1, image_ph_mask.shape[1]-1):
        d = np.abs(image_ph_mask[:,j-1,:] - image_ph_mask[:,j+1,:])  # diff
        d = lut_a[(d * lut_a_max).astype(int)]
        image_ph_mask[:,j,:] = ((image_ph_mask[:,j,:] * d) + 
                              (image_ph_mask[:,j-1,:] * (1-d)))        
    # down -> up
    for i in range(image_ph_mask.shape[0]-2, 1, -1):
        d = np.abs(image_ph_mask[i-1,:,:] - image_ph_mask[i+1,:,:])  # diff
        d = lut_a[(d * lut_a_max).astype(int)]
        image_ph_mask[i,:,:] = ((image_ph_mask[i,:,:] * d) + 
                              (image_ph_mask[i+1,:,:] * (1-d)))        
    # right -> left
    for j in range(image_ph_mask.shape[1]-2, 1, -1):
        d = np.abs(image_ph_mask[:,j-1,:] - image_ph_mask[:,j+1,:])  # diff
        d = lut_b[(d * lut_b_max).astype(int)]
        image_ph_mask[:,j,:] = ((image_ph_mask[:,j,:] * d) + 
                              (image_ph_mask[:,j+1,:] * (1-d)))         
    # up -> down
    for i in range(1, image_ph_mask.shape[0]-1):
        d = np.abs(image_ph_mask[i-1,:,:] - image_ph_mask[i+1,:,:])  # diff
        d = lut_b[(d * lut_b_max).astype(int)]
        image_ph_mask[i,:,:] = ((image_ph_mask[i,:,:] * d) + 
                              (image_ph_mask[i-1,:,:] * (1-d)))
    # convert back to 2D if grayscale is needed
    if grayscale_out is True:
        image_ph_mask = np.squeeze(image_ph_mask)
    if verbose is True:   
        plt.figure()
        plt.subplot(1,2,1)
        plt.imshow(image)
        plt.title('Input image')
        plt.axis('off')
        plt.subplot(1,2,2)
        if grayscale_out is True:
            plt.imshow(image_ph_mask, cmap='gray', vmin=0, vmax=1)
        else:
            plt.imshow(image_ph_mask, vmin=0, vmax=1)
        plt.title('Photometric mask')
        plt.axis('off')
        
        plt.tight_layout(True)
        plt.suptitle('Estimation of photometric mask')
        plt.show()

    return image_ph_mask
def blend_expoures(
        exposure_list, 
        threshold_dark=0.35, 
        threshold_bright=0.65, 
        verbose=False
        ):
    # internal constants
    SCALE_COARSE = 0.6  # [0,1], 0->fine, 1->coarse
    SCALE_FINE = 0.2  # [0,1], 0->fine, 1->coarse
    LUMINANCE_MIDDLE = 0.5  # middle of the luminance scale in [0,1]
    GAMA_MAX = 2  # max gama to be used for darkening images
    GAMA_MIN = 0.2  # min gama to be used for brightening images
    LUT_RESOLUTION = 256
    total_exposures = len(exposure_list)
    # color or grayscale
    if len(exposure_list[0].shape) > 2:  # check the 1st image of the list
        color_exposures = True
    else:
        color_exposures = False
    #--- sort exposures from darkest to brightest
    exposure_list_gray = []
    mean_luminance_list = []
    
    if color_exposures is True:
        exposure_list_red = []
        exposure_list_green = []
        exposure_list_blue = []
    for image in exposure_list:
        image_gray = rgb2gray(image)
        exposure_list_gray.append(image_gray)  # grayscale
        mean_luminance_list.append(image_gray.mean())  # mean luminance
        if color_exposures is True:
            exposure_list_red.append(img_as_float(image[:,:,0]))  # red
            exposure_list_green.append(img_as_float(image[:,:,1]))  # green 
            exposure_list_blue.append(img_as_float(image[:,:,2]))  # blue
    # sort according to mean luminance
    indx_lum_ascending = sorted(
        range(len(mean_luminance_list)), 
        key=lambda i: mean_luminance_list[i]  
        )
    if verbose is True:
        print('Darkest to brightest exposure sequence:', indx_lum_ascending)
    # (the 3rd dimension has the separate grayscale or color exposures)
    exposure_array_gray = np.array(exposure_list_gray)
    exposure_array_gray = np.moveaxis(exposure_array_gray, 0, -1)
    exposure_array_gray = exposure_array_gray[:,:,indx_lum_ascending]
    if color_exposures is True:
        exposure_array_red = np.array(exposure_list_red)  
        exposure_array_red = np.moveaxis(exposure_array_red, 0, -1)
        exposure_array_red = exposure_array_red[:,:,indx_lum_ascending]
        
        exposure_array_green = np.array(exposure_list_green)  
        exposure_array_green = np.moveaxis(exposure_array_green, 0, -1)
        exposure_array_green = exposure_array_green[:,:,indx_lum_ascending]
        
        exposure_array_blue = np.array(exposure_list_blue)  
        exposure_array_blue = np.moveaxis(exposure_array_blue, 0, -1)
        exposure_array_blue = exposure_array_blue[:,:,indx_lum_ascending]

    illumination_coarse = get_photometric_mask(
        exposure_array_gray.copy(), 
        smoothing=SCALE_COARSE, 
        grayscale_out=False,  # estimaste each channel separately
        verbose=False)
    
    illumination_fine = get_photometric_mask(
        exposure_array_gray.copy(), 
        smoothing=SCALE_FINE, 
        grayscale_out=False,  # estimaste each channel separately
        verbose=False)
    for i in range(total_exposures):
        
        illumination_coarse[:,:,i] = rescale_intensity(
            illumination_coarse[:,:,i],
            in_range='image',
            out_range='dtype'
            )
        
        illumination_fine[:,:,i] = rescale_intensity(
            illumination_fine[:,:,i],
            in_range='image',
            out_range='dtype'
            )

    mean_lum = illumination_coarse[:,:,0].mean()
    if mean_lum > LUMINANCE_MIDDLE:
        gamma_new = map_value(
            mean_lum, 
            range_in=(LUMINANCE_MIDDLE,1), 
            range_out=(1,GAMA_MAX)
            )
        if verbose:
            print(
                'Darkest coarse exposure too bright! Applying gamma:', 
                gamma_new
                )
        illumination_coarse[:,:,0] = adjust_gamma(
        image = illumination_coarse[:,:,0], 
        gamma = gamma_new
        ) 
        
    mean_lum = illumination_fine[:,:,0].mean()
    if mean_lum > LUMINANCE_MIDDLE:
        gamma_new = map_value(
            mean_lum, 
            range_in=(LUMINANCE_MIDDLE,1), 
            range_out=(1,GAMA_MAX)
            )
        if verbose: 
            print(
                'Darkest fine exposure too bright! Applying gamma:', 
                gamma_new
                )
        illumination_fine[:,:,0] = adjust_gamma(
            image = illumination_fine[:,:,0], 
            gamma = gamma_new
            )
    mean_lum = illumination_coarse[:,:,-1].mean()
    if mean_lum < LUMINANCE_MIDDLE:
        gamma_new = map_value(
            mean_lum, 
            range_in=(0,LUMINANCE_MIDDLE), 
            range_out=(GAMA_MIN,1)
            )
        if verbose:
            print(
                'Brightest coarse exposure too dark! Applying gamma:', 
                gamma_new
                )
        illumination_coarse[:,:,-1] = adjust_gamma(
            image = illumination_coarse[:,:,-1], 
            gamma = gamma_new
            )
    
    mean_lum = illumination_fine[:,:,-1].mean()
    if mean_lum < LUMINANCE_MIDDLE:
        gamma_new = map_value(
            mean_lum, 
            range_in=(0,LUMINANCE_MIDDLE), 
            range_out=(GAMA_MIN,1)
            )
        if verbose:
            print(
                'Brightest fine exposure too dark! Applying gamma:', 
                gamma_new
                )
        illumination_fine[:,:,-1] = adjust_gamma(
            image = illumination_fine[:,:,-1], 
            gamma = gamma_new
            )
    weights_lower, weights_mid, weights_upper = get_membership_luts(
        resolution=LUT_RESOLUTION, 
        lower_threshold=threshold_dark,  # defines lower cutofd
        upper_threshold=threshold_bright,  # defines upper cutofd
        verbose=verbose
        )
    lut_resolution = len(weights_lower) - 1
        
    weights_coarse = np.zeros(illumination_coarse.shape, dtype=float)
    weights_coarse[:,:,0] = (weights_lower[(illumination_coarse[:,:,0] * 
                                            lut_resolution).astype(int)])
    weights_coarse[:,:,1:-1] = (weights_mid[(illumination_coarse[:,:,1:-1] * 
                                             lut_resolution).astype(int)])
    weights_coarse[:,:,-1] = (weights_upper[(illumination_coarse[:,:,-1] * 
                                             lut_resolution).astype(int)])
    
    weights_fine = np.zeros(illumination_fine.shape, dtype=float)
    weights_fine[:,:,0] = (weights_lower[(illumination_fine[:,:,0] * 
                                          lut_resolution).astype(int)])
    weights_fine[:,:,1:-1] = (weights_mid[(illumination_fine[:,:,1:-1] * 
                                           lut_resolution).astype(int)])
    weights_fine[:,:,-1] = (weights_upper[(illumination_fine[:,:,-1] * 
                                           lut_resolution).astype(int)])
    exposure_coarse = weights_coarse * exposure_array_gray
    exposure_coarse = (np.sum(exposure_coarse, axis=2) / 
                       np.sum(weights_coarse, axis=2))
    exposure_fine = weights_fine * exposure_array_gray
    exposure_fine = (np.sum(exposure_fine, axis=2) / 
                     np.sum(weights_fine, axis=2))
    exposure_out_gray = (exposure_coarse + exposure_fine) / 2
    exposure_out = exposure_out_gray
    if color_exposures is True:
        # red
        exposure_coarse_red = weights_coarse * exposure_array_red
        exposure_coarse_red = (np.sum(exposure_coarse_red, axis=2) / 
                               np.sum(weights_coarse, axis=2))
        exposure_fine_red = weights_fine * exposure_array_red
        exposure_fine_red = (np.sum(exposure_fine_red, axis=2) / 
                             np.sum(weights_fine, axis=2))
        exposure_out_red = (exposure_coarse_red + exposure_fine_red) / 2
        
        # green
        exposure_coarse_green = weights_coarse * exposure_array_green
        exposure_coarse_green = (np.sum(exposure_coarse_green, axis=2) / 
                               np.sum(weights_coarse, axis=2))
        exposure_fine_green = weights_fine * exposure_array_green
        exposure_fine_green = (np.sum(exposure_fine_green, axis=2) / 
                             np.sum(weights_fine, axis=2))
        exposure_out_green = (exposure_coarse_green + exposure_fine_green) / 2
        
        # blue
        exposure_coarse_blue = weights_coarse * exposure_array_blue
        exposure_coarse_blue = (np.sum(exposure_coarse_blue, axis=2) / 
                               np.sum(weights_coarse, axis=2))
        exposure_fine_blue = weights_fine * exposure_array_blue
        exposure_fine_blue = (np.sum(exposure_fine_blue, axis=2) / 
                             np.sum(weights_fine, axis=2))
        exposure_out_blue = (exposure_coarse_blue + exposure_fine_blue) / 2
    
        # combine all blended color channels to one image
        exposure_out_color = np.zeros(
            (exposure_out_gray.shape[0], exposure_out_gray.shape[1], 3), 
            dtype=float
            )
        exposure_out_color[:,:,0] = exposure_out_red
        exposure_out_color[:,:,1] = exposure_out_green
        exposure_out_color[:,:,2] = exposure_out_blue
        exposure_out = exposure_out_color
    #--- Visualizations
    if verbose is True:
        # display intermediate stages of the method
        plt.figure()
        for i in range(total_exposures):
            plt.subplot(6,total_exposures,i+1)
            plt.imshow(exposure_array_gray[:,:,i], cmap='gray')
            plt.title('Exposure ' + str(i))
            plt.axis('off')
            plt.subplot(6,total_exposures,i+1+total_exposures)
            plt.imshow(illumination_coarse[:,:,i], cmap='gray')
            plt.title('ill.coarse ' + str(i))
            plt.axis('off')
            
            plt.subplot(6,total_exposures,i+1+(total_exposures*2))
            plt.imshow(illumination_fine[:,:,i], cmap='gray')
            plt.title('ill.fine ' + str(i))
            plt.axis('off')
            
            plt.subplot(6,total_exposures,i+1+(total_exposures*3))
            plt.imshow(weights_coarse[:,:,i], cmap='gray')
            plt.title('W.coarse ' + str(i))
            plt.axis('off')
            plt.subplot(6,total_exposures,i+1+(total_exposures*4))
            plt.imshow(weights_fine[:,:,i], cmap='gray')
            plt.title('W.fine ' + str(i))
            plt.axis('off')

        plt.subplot(6,total_exposures,1+(total_exposures*5))
        plt.imshow(exposure_coarse, cmap='gray')
        plt.title('Coarse blended')
        plt.axis('off')
        
        plt.subplot(6,total_exposures,2+(total_exposures*5))
        plt.imshow(exposure_fine, cmap='gray')
        plt.title('Fine blended')
        plt.axis('off')
        
        plt.subplot(6,total_exposures,3+(total_exposures*5))
        plt.imshow(exposure_out_gray, cmap='gray')
        plt.title('Final blend')
        plt.axis('off')

        plt.suptitle('List of exposures')
        plt.tight_layout()
        plt.tight_layout()
        plt.show()
        plt.figure()
        grid = plt.GridSpec(total_exposures, total_exposures)
        if color_exposures is False: 
            cmap = 'gray'
        else:
            cmap = None
        
        for i in range(total_exposures):
            plt.subplot(grid[0,i])
            plt.imshow(exposure_list[indx_lum_ascending[i]], cmap=cmap)
            plt.title('Exposure ' + str(i))
            plt.axis('off')

        plt.subplot(grid[1:,:])
        plt.imshow(exposure_out, cmap=cmap)
        plt.title('Final blend')
        plt.axis('off')
        plt.tight_layout()
        plt.suptitle('Full color blend')
        plt.show()
    return exposure_out
def apply_local_contrast_enhancement(
        image, 
        image_ph_mask, 
        degree=1.5, 
        verbose=False):
    DARK_BOOST = 0.2
    THRESHOLD_DARK_TONES = 100 / 255
    detail_amplification_global = degree
    
    image_details = image - image_ph_mask  # image details
    
    # special treatment for dark regions
    detail_amplification_local = image_ph_mask / THRESHOLD_DARK_TONES
    detail_amplification_local[detail_amplification_local>1] = 1
    detail_amplification_local = ((1 - detail_amplification_local) * 
                                  DARK_BOOST) + 1  # [1, 1.2]
    # apply all detail adjustements
    image_details = (image_details * 
                     detail_amplification_global * 
                     detail_amplification_local)
    # add details back to the local neighborhood
    image_out = image_ph_mask + image_details  
    
    # stay within range
    image_out = np.clip(a=image_out, a_min=0, a_max=1, out=image_out)
    
    if verbose is True:
        plt.figure()
        plt.subplot(1,3,1)
        plt.imshow(image, cmap='gray', vmin=0, vmax=1)
        plt.title('Input image')
        plt.axis('off')
        plt.subplot(1,3,2)
        plt.imshow(image_ph_mask, cmap='gray', vmin=0, vmax=1)
        plt.title('Ph. mask')
        plt.axis('off')
        plt.subplot(1,3,3)
        plt.imshow(image_out, cmap='gray', vmin=0, vmax=1)
        plt.title('Output')
        plt.axis('off')
        
        plt.tight_layout(True)
        plt.suptitle('Local contrast enhancement [x' + str(degree) + ']')
        plt.show()
    
    return image_out

def apply_spatial_tonemapping(
        image, 
        image_ph_mask, 
        mid_tone=0.5,
        tonal_width=0.5,
        areas_dark=0.5,
        areas_bright=0.5,
        preserve_tones = True,
        verbose=True):
    # defining parameters
    EPSILON = 1 / 256
    # adjust range and non-linear response of parameters
    mid_tone = map_value(
            value=mid_tone, 
            range_in=(0,1), 
            range_out=(0,1), 
            invert=False, 
            non_lin_convex=None, 
            non_lin_concave=None
            )
    tonal_width = map_value(
            value=tonal_width, 
            range_in=(0,1), 
            range_out=(EPSILON,1), 
            invert=False, 
            non_lin_convex=0.1, 
            non_lin_concave=None
            )
    
    areas_dark = map_value(
            value=areas_dark, 
            range_in=(0,1), 
            range_out=(0,5), 
            invert=True, 
            non_lin_convex=0.05, 
            non_lin_concave=None
            )
    
    areas_bright = map_value(
            value=areas_bright, 
            range_in=(0,1), 
            range_out=(0,5), 
            invert=True, 
            non_lin_convex=0.05, 
            non_lin_concave=None
            )
    # lower tones (below mid_tone level)
    image_lower = image.copy()   
    image_lower[image_lower>=mid_tone] = 0
    alpha = (image_ph_mask ** 2) / tonal_width
    tone_continuation_factor = mid_tone / (mid_tone + EPSILON - image_ph_mask)
    alpha = alpha * tone_continuation_factor + areas_dark
    image_lower = (image_lower * (alpha + 1)) / (alpha + image_lower)
    
    # upper tones (above mid_tone level)
    image_upper = image.copy()
    image_upper[image_upper<mid_tone] = 0
    image_ph_mask_inv = 1 - image_ph_mask
    alpha = (image_ph_mask_inv ** 2) / tonal_width
    tone_continuation_factor = mid_tone / ((1 - mid_tone) - image_ph_mask_inv)
    alpha = alpha * tone_continuation_factor + areas_bright 
    image_upper = (image_upper * alpha) / (alpha + 1 - image_upper)
    image_tonemapped = image_lower + image_upper
    if preserve_tones is True:
        preservation_degree = np.abs(0.5 - image_ph_mask) / 0.5  # 0: near 0.5
#        preservation_degree = ((1 + 0.3) * preservation_degree) / (0.3 + preservation_degree)
        image_tonemapped = (preservation_degree * image_tonemapped + 
                           (1-preservation_degree) * image)
    if verbose is True:
        
        plt.figure()
        
        plt.subplot(2,2,1)
        plt.imshow(image, cmap='gray', vmin=0, vmax=1)
        plt.title('Input image')
        plt.axis('off')
        
        plt.subplot(2,2,3)
        plt.imshow(image_lower, cmap='gray', vmin=0, vmax=1)
        plt.title('Image lower')
        plt.axis('off')

        plt.subplot(2,2,4)
        plt.imshow(image_upper, cmap='gray', vmin=0, vmax=1)
        plt.title('Image upper')
        plt.axis('off')
        
        plt.subplot(2,2,2)
        plt.imshow(image_tonemapped, cmap='gray', vmin=0, vmax=1)
        plt.title('Image tonemapped')
        plt.axis('off')
        
        plt.tight_layout(True)
        plt.suptitle('Spatial tone mapping')
        plt.show()
    return image_tonemapped
def srgb_to_linear(image_srgb, verbose=False):
    # dealing with different input dimensions 
    dimensions = len(image_srgb.shape)
    if dimensions == 1:
        image_srgb = np.expand_dims(image_srgb, axis=2)  # make a 3rd dimension
    image_srgb = img_as_float(image_srgb)  # [0,1]
    # lower part of the piecewise formula
    image_lower = image_srgb.copy()
    image_lower[image_lower > 0.04045] = 0
    image_lower = image_lower / 12.92
    # upper part of the piecewise formula
    image_upper = image_srgb.copy()
    image_upper = image_upper + 0.055
    image_upper[image_upper <= (0.04045+0.055)] = 0
    image_upper = image_upper / 1.055
    image_upper = image_upper ** 2.4
    image_linear = image_lower + image_upper  # combine into the final result
    if verbose is True:
        plt.figure()
        plt.subplot(1,2,1)
        plt.imshow(image_srgb, vmin=0, vmax=1)
        plt.title('Image sRGB')
        plt.axis('off')
        plt.subplot(1,2,2)
        plt.imshow(image_linear, vmin=0, vmax=1)
        plt.title('Image linear')
        plt.axis('off')
        plt.tight_layout(True)
        plt.suptitle('sRGB -> linear space')
        plt.show()
    return image_linear
def linear_to_srgb(image_linear, verbose=False):
    # dealing with different input dimensions 
    dimensions = len(image_linear.shape)
    if dimensions == 1:
        image_linear = np.expand_dims(image_linear, axis=2)  # 3rd dimension
    
    image_linear = img_as_float(image_linear)  # [0,1] 
    # lower part of the piecewise formula
    image_lower = image_linear.copy()
    image_lower[image_lower > 0.0031308] = 0
    image_lower = image_lower * 12.92
    # upper part of the piecewise formula
    image_upper = image_linear.copy()
    image_upper[image_upper <= 0.0031308] = 0
    image_upper = image_upper ** (1/2.4)
    image_upper = image_upper * 1.055
    image_upper = image_upper - 0.055
    image_srgb = image_lower + image_upper
    image_srgb = np.clip(a=image_srgb, a_min=0, a_max=1, out=image_srgb)
    if verbose is True:
        plt.figure()
        plt.subplot(1,2,1)
        plt.imshow(image_linear, vmin=0, vmax=1)
        plt.title('Image linear')
        plt.axis('off')
        
        plt.subplot(1,2,2)
        plt.imshow(image_srgb, vmin=0, vmax=1)
        plt.title('Image sRGB')
        plt.axis('off')
        
        plt.tight_layout(True)
        plt.suptitle('Linear space -> sRGB')
        plt.show()
    
    return (image_srgb * 255).astype(np.uint8)
def transfer_graytone_to_color(image_color, image_graytone, verbose=False):
    EPSILON = 1 / 256
    # bring both color and graytone to linear space
    image_color_linear = srgb_to_linear(image_color.copy(), verbose=False)
    image_graytone_linear = srgb_to_linear(image_graytone.copy(),verbose=False)
    image_gray_linear = rgb2gray(image_color_linear.copy())
    image_gray_linear[image_gray_linear==0] = EPSILON  # for the division later
    # tone ratio of linear images: improved/original
    tone_ratio = image_graytone_linear / image_gray_linear
    # apply the tone ratios to the color image
    image_colortone_linear = image_color_linear * np.dstack([tone_ratio] * 3)
    # make sure it's within limits
    image_colortone_linear = np.clip(
        a=image_colortone_linear, 
        a_min=0, 
        a_max=1, 
        out=image_colortone_linear
        )
    # bring back to gamma-corrected sRGB space for visualization
    image_colortone = linear_to_srgb(image_colortone_linear, verbose=False)
    # display results
    if verbose is True:
        plt.figure()
        plt.subplot(2,4,1)
        plt.imshow(image_color, vmin=0, vmax=255)
        plt.title('Color')
        plt.axis('off')
        
        plt.subplot(2,4,5)
        plt.imshow(image_color_linear, vmin=0, vmax=1)
        plt.title('Color linear')
        plt.axis('off')
        
        plt.subplot(2,4,2)
        plt.imshow(image_graytone, cmap='gray', vmin=0, vmax=1)
        plt.title('Graytone')
        plt.axis('off')
        
        plt.subplot(2,4,6)
        plt.imshow(image_graytone_linear, cmap='gray', vmin=0, vmax=1)
        plt.title('Graytone linear')
        plt.axis('off')
        
        plt.subplot(2,4,7)
        plt.imshow(tone_ratio, cmap='gray')
        plt.title('Tone ratios')
        plt.axis('off')
        
        plt.subplot(2,4,4)
        plt.imshow(image_colortone, vmin=0, vmax=255)
        plt.title('Colortone')
        plt.axis('off')
        
        plt.subplot(2,4,8)
        plt.imshow(image_colortone_linear, vmin=0, vmax=1)
        plt.title('Colortone linear')
        plt.axis('off')
        
        plt.tight_layout(True)
        plt.suptitle('Transfering gray tones to color')
        plt.show()
    return image_colortone
def change_color_saturation(
        image_color, 
        image_ph_mask=None, 
        sat_degree=1.5, 
        verbose=False):
    
    LOCAL_BOOST = 0.2
    THRESHOLD_DARK_TONES = 100 / 255
    image_color = img_as_float(image_color)  # [0,1]
    
    # define gray scale
    image_gray = (image_color[:,:,0] + 
                  image_color[:,:,1] + 
                  image_color[:,:,2]) / 3
    image_gray = np.dstack([image_gray] * 3)  # grayscale with 3 channels
    
    image_delta = image_color - image_gray  # deviations from gray
    # defining local color amplification degree
    if image_ph_mask is not None:
        detail_amplification_local = image_ph_mask / THRESHOLD_DARK_TONES
        detail_amplification_local[detail_amplification_local>1] = 1
        detail_amplification_local = ((1 - detail_amplification_local) * 
                                      LOCAL_BOOST) + 1  # [1, 1.2]
        detail_amplification_local = np.dstack(
                [detail_amplification_local] * 3)  # 3 channels
    else:
        detail_amplification_local = 1
    
    image_new_sat = (image_gray + 
                     image_delta * sat_degree * detail_amplification_local)
    
    image_new_sat = np.clip(
            a=image_new_sat, 
            a_min=0, 
            a_max=1, 
            out=image_new_sat
            )
    if verbose is True:
        plt.figure()
        plt.subplot(1,2,1)
        plt.imshow(image_color, vmin=0, vmax=1)
        plt.title('Input image')
        plt.axis('off')

        plt.subplot(1,2,2)
        plt.imshow(image_new_sat, vmin=0, vmax=1)
        plt.title('New saturation [x' + str(sat_degree) + ']')
        plt.axis('off')

        plt.tight_layout(True)
        plt.suptitle('Color saturation adjustment')
        plt.show()
    return image_new_sat
def correct_colors(image, verbose):
    image_out = img_as_float(image.copy())  # [0,1]
    # mean of all channels
    image_mean = (image_out[:,:,0].mean() + 
                  image_out[:,:,1].mean() + 
                  image_out[:,:,2].mean()) / 3
    # logarithm base to which each channel will be raised
    base_r = image_out[:,:,0].mean() / image_out[:,:,0].max()
    base_g = image_out[:,:,1].mean() / image_out[:,:,1].max()
    base_b = image_out[:,:,2].mean() / image_out[:,:,2].max()
    # the power to which each channel will be raised
    power_r = math.log(image_mean, base_r)
    power_g = math.log(image_mean, base_g)
    power_b = math.log(image_mean, base_b)
    # separately applying different color correction powers to each channel
    image_out[:,:,0] = (image_out[:,:,0] / image_out[:,:,0].max()) ** power_r
    image_out[:,:,1] = (image_out[:,:,1] / image_out[:,:,1].max()) ** power_g
    image_out[:,:,2] = (image_out[:,:,2] / image_out[:,:,2].max()) ** power_b
    if verbose is True:
        plt.figure()
        plt.subplot(1,2,1)
        plt.imshow(image)
        plt.title('Input image')
        plt.axis('off')
        plt.subplot(1,2,2)
        plt.imshow(image_out, vmin=0, vmax=1)
        plt.title('Corrected colors')
        plt.axis('off')
        plt.tight_layout(True)
        plt.suptitle('Gray world color correction')
        plt.show()
    return image_out
def adjust_brightness(image, degree=0, verbose=False):
    EPSILON = 1 / 256  # what we consider minimum value
    if degree > 0:
        brighten = True
    else:
        brighten = False
    degree = abs(degree)  # [0,1]
    alpha = map_value(
            value=degree, 
            range_in=(0,1), 
            range_out=(0,5),   # from the paper: 5x brings close to linear
            invert=True,  # from the paper
            non_lin_convex=0.05,  # adding linearity to the response
            non_lin_concave=None
            )
    alpha = alpha + EPSILON  # to avoid division by zero
    # applying global tone-mapping
    if degree != 0:
        image_brightness = image.copy()
        
        if brighten is True:
            image_brightness = ((image_brightness * (alpha + 1)) / 
                                (alpha + image_brightness))
        else:
            image_brightness = ((image_brightness * alpha) / 
                                (alpha + 1 - image_brightness)) 
    else: image_brightness = image
    if verbose is True:
        plt.figure()
        plt.subplot(1,2,1)
        plt.imshow(image, cmap='gray', vmin=0, vmax=1)
        plt.title('Input image')
        plt.axis('off')
        
        plt.subplot(1,2,2)
        plt.imshow(image_brightness, cmap='gray', vmin=0, vmax=1)
        plt.title('Adjusted brightness image')
        plt.axis('off')
        
        plt.tight_layout(True)
        plt.suptitle('Adjusting brightness')
        plt.show()
    return image_brightness
def enhance_image(image, parameters, verbose=False):
    # sanity check for type, range and defaults
    if 'local_contrast' in parameters:
        parameters['local_contrast'] = float(parameters['local_contrast'])
        if parameters['local_contrast'] < 0: parameters['local_contrast'] = 0
    else: parameters['local_contrast'] = 1.2 # default: slight increase
    
    if 'mid_tones' in parameters:
        parameters['mid_tones'] = float(parameters['mid_tones'])
        if parameters['mid_tones'] > 1: parameters['mid_tones'] = 1
        if parameters['mid_tones'] < 0: parameters['mid_tones'] = 0
    else: parameters['mid_tones'] = 0.5  # default: middle of the range
    
    if 'tonal_width' in parameters:
        parameters['tonal_width'] = float(parameters['tonal_width'])
        if parameters['tonal_width'] > 1: parameters['tonal_width'] = 1
        if parameters['tonal_width'] < 0: parameters['tonal_width'] = 0
    else: parameters['tonal_width'] = 0.5  # default: middle of the range
    
    if 'areas_dark' in parameters:
        parameters['areas_dark'] = float(parameters['areas_dark'])
        if parameters['areas_dark'] > 1: parameters['areas_dark'] = 1
        if parameters['areas_dark'] < 0: parameters['areas_dark'] = 0
    else: parameters['areas_dark'] = 0.2  # default: gentle increase
    
    if 'areas_bright' in parameters:
        parameters['areas_bright'] = float(parameters['areas_bright'])
        if parameters['areas_bright'] > 1: parameters['areas_bright'] = 1
        if parameters['areas_bright'] < 0: parameters['areas_bright'] = 0
    else: parameters['areas_bright'] = 0.2  # default: gentle increase
    
    if 'brightness' in parameters:
        parameters['brightness'] = float(parameters['brightness'])
        if parameters['brightness'] > 1: parameters['brightness'] = 1
        if parameters['brightness'] < -1: parameters['brightness'] = -1
    else: parameters['brightness'] = 0.1  # default: gentle increase
    
    if 'preserve_tones' in parameters:
        parameters['preserve_tones'] = bool(parameters['preserve_tones'])
    else: parameters['preserve_tones'] = True  # default: preserve tones
    
    if 'color_correction' in parameters:
        parameters['color_correction'] = bool(parameters['color_correction'])
    else: parameters['color_correction'] = False  # default: no correction
    
    if 'saturation_degree' in parameters:
        parameters['saturation_degree'] = float(parameters['saturation_degree'])
        if parameters['saturation_degree'] < 0: parameters['saturation_degree'] = 0
    else: parameters['saturation_degree'] = 1.2 # default: slight increase
    # get photometric mask, as a guide for spatial-tone mapping
    image_ph_mask = get_photometric_mask(
        image=image,      
        verbose=verbose
        )
    # increase the local contrast of the grayscale image
    image_contrast = apply_local_contrast_enhancement(
            image=rgb2gray(image.copy()), 
            image_ph_mask=image_ph_mask, 
            degree=parameters['local_contrast'], 
            verbose=verbose
            )
    # apply spatial tonemapping on the previous stage
    image_tonemapped = apply_spatial_tonemapping(
            image=image_contrast, 
            image_ph_mask=image_ph_mask, 
            mid_tone=parameters['mid_tones'],
            tonal_width=parameters['tonal_width'],
            areas_dark=parameters['areas_dark'],
            areas_bright=parameters['areas_bright'],
            preserve_tones=parameters['preserve_tones'],
            verbose=verbose
            )
    image_brightness = adjust_brightness(
        image_tonemapped, 
        degree=parameters['brightness'], 
        verbose=verbose
        )
    # transfer the enhancement on the color image (in the linear color space)
    image_colortone = transfer_graytone_to_color(
            image_color=image, 
            image_graytone=image_brightness, 
            verbose=verbose
            )
    # apply color correction (if needed)
    if parameters['color_correction'] is True:
        image_colortone = correct_colors(
                image=image_colortone, 
                verbose=verbose
                )
    # adjust the color saturation
    image_colortone_saturation = change_color_saturation(
            image_color=image_colortone, 
            image_ph_mask=image_ph_mask, 
            sat_degree=parameters['saturation_degree'],
            verbose = verbose,
            )
    # display results
    if verbose is True:

        plt.figure()
        plt.subplot(2,3,1)
        plt.imshow(image, vmin=0, vmax=255)
        plt.title('Input image')
        plt.axis('off')
        plt.tight_layout()
        plt.subplot(2,3,4)
        plt.imshow(image_ph_mask, cmap='gray', vmin=0, vmax=1)
        plt.title('Photometric mask')
        plt.axis('off')
        plt.tight_layout()

        plt.subplot(2,3,5)
        plt.imshow(image_contrast, cmap='gray', vmin=0, vmax=1)
        plt.title('Local contrast enhancement')
        plt.axis('off')
        plt.tight_layout()

        plt.subplot(2,3,2)
        plt.imshow(image_colortone, vmin=0, vmax=255)
        plt.title('Spatial tone mapping')
        plt.axis('off')
        plt.tight_layout()

        plt.subplot(2,3,3)
        plt.imshow(image_colortone_saturation, vmin=0, vmax=255)
        plt.title('Increased saturation')
        plt.axis('off')
        plt.tight_layout()
    return image_colortone_saturation
if __name__=="__main__":

    filename = "../image/ashiba.jpg"
    image = imageio.v2.imread(filename)  # load image

    # setting up parameters
    parameters = {}
    parameters['local_contrast'] = 1.5  # 1.5x increase in details
    parameters['mid_tones'] = 0.5
    parameters['tonal_width'] = 0.5
    parameters['areas_dark'] = 0.7  # 70% improvement in dark areas
    parameters['areas_bright'] = 0.5  # 50% improvement in bright areas
    parameters['saturation_degree'] = 1.2  # 1.2x increase in color saturation
    parameters['brightness'] = 0.1  # slight increase in brightness
    parameters['preserve_tones'] = True
    parameters['color_correction'] = False
    image_enhanced = enhance_image(image, parameters, verbose=False)

    # display results
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(image, vmin=0, vmax=255)
    plt.title('Input image')
    plt.axis('off')
    plt.tight_layout()
    
    plt.subplot(1,2,2)
    plt.imshow(image_enhanced, vmin=0, vmax=255)
    plt.title('Enhanced image')
    plt.axis('off')
    plt.tight_layout()

    plt.show()
