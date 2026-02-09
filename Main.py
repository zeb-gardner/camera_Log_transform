
import numpy as np
import sys
import time
import math

from scipy import interpolate
from scipy.interpolate import Akima1DInterpolator, CubicHermiteSpline
from scipy.interpolate import PchipInterpolator

from matplotlib import pyplot as plt
from matplotlib import image as mpimg


import imageio.v3 as iio

import colour   #colour-science
from colour.models import RGB_COLOURSPACE_sRGB

from colour.models import RGB_COLOURSPACE_DAVINCI_WIDE_GAMUT
from colour.models import RGB_COLOURSPACE_BT709
from colour.models import RGB_COLOURSPACE_ACESCG
from colour.models import RGB_COLOURSPACE_ACES2065_1

import cv2 as cv
import png

from pathlib import Path
import argparse
import multiprocessing as mp

from CC_SG import *
#from CC_VIDEO_V1 import *  #Choose the specs for the right chart

from beautifultable import BeautifulTable



ERROR_PERCENTILE_LIMIT = 90  # Exclude chip error outliers above this percentile

CHART_UPPER_PERCENTILE = 90  #Keep pixels in this range, exlude outliers to reduce noise
CHART_LOWER_PERCENTILE = 10

SAVE_LUT = 1 #Save 1 = Shaper Lut, 2  = 3d Cube, 3 = CLF, 4 = all

CURVE_FIT_MODE_1D = "makima"   #Mode for Luma curve, pchip, spline, akima (not stable),makima, cubic, quintic
HSL_MODE = "cubic"   #mode for HSL curves; pchip, spline, akima (not stable),makima, cubic, quintic
CURVE_FIT_MODE_3D="cubic"  #“linear”, “nearest”, “slinear”, “cubic”, “quintic” and “pchip"
BEZ_SMOOTH = 0  #How smooth to force Luma Bez Curves for 1d spline  0 is none, 1 is max, small values may throw fit warning
HSL_SMOOTH = 0  #How smooth to force HSL Curves for 1d spline  0 is none, 1 is max, small values may throw fit warning

SLOPE_LIMIT = 0.1  #How much error to apply where Luma curve has negative slope
SLOPE_MIN = 0
SLOPE_MAX = 5
SMOOTHING = 1  #How much to increase error based on sharp curve in Luma Curve

MAX_ITER_LUM = 8
MAX_ITER_3X3 = 8
MAX_ITER_S_GAIN = 10
MAX_ITER_S_GAMMA = 10

PETURB_MAX_LUM = 0.1 #Perturb max for Binary Search algorithm
PETURB_MAX_LUM_GEN = 0.01  #Perturb max for Genetic algorithm
NUM_CHILDREN = 2000

NUM_THREADS = 12

PETURB_MAX_C = 0.25  # Color for 3x3 genetic algorithm
NUM_CHILDREN_C = 30
NUM_PARENT_C = 20

PERTURB_MAX_3X3 = 0.1  #for Binary Search Algorithms
PERTURB_MAX_HUE_HUE = 0.01
PERTURB_MAX_HUE_SAT = 0.02
PERTURB_MAX_S_GAIN = 0.1
PERTURB_MAX_S_GAMMA = 0.1


ILLUM_LED_B5 = (0.3118, 0.3236)  #6500k LED
ILLUM_D75 = (0.29902,0.31485)
ILLUM_D65 =(0.31272,0.32903)
ILLUM_D60 = (0.32168, 0.33767)
ILLUM_D55 = (0.33242, 0.34743)
ILLUM_D50 = (0.34567,0.35850)
ILLUMINANT = ILLUM_D65
CAT = 'CAT02'   # 'CAT16' or 'CAT02'
INPUT_ILLUMINANT = [-1,-1]  #Will be set to correct WB of input images


ILLUM_OPT_MAX = 0.01  #Max step in  output Illuminant optimization
ILLUM_OPT_ITER = 12   #How many iterations of  optimization
INPUT_ILLUM_OPT_MAX = 0.01
INPUT_ILLUM_OPT_ITER =  12

MIDDLE_GRAY_LIN = 0.18  #Value of MG in Lin Space

ERROR_EPS = 10**-3   #Epsilon for stop binary search on Luma Error

MANUAL_S_GAMMA = 0  #Manual aesthetic adjustments applied just to output lut
MANUAL_S_GAIN = 0
#MANUAL_HUE_HUE = [0,0,0.05,0.1,0.05,0,0]   #Red, Orange, Lime, Cyan, Blue, Magenta,Red
#MANUAL_HUE_SAT =[0.1,0,0.15,-0.25,-0.25,0.0,0.1]

MANUAL_HUE_HUE = [0,0,0,0,0,0,0]   #Red, Orange, Lime, Cyan, Blue, Magenta,Red
MANUAL_HUE_SAT =[0,0,0,0,0,0,0]

GAM_COMP_STRENGTH = 1  # scale parameters of Gammut Compression, Larger is more compression, 1.0 matches ACES recommended, Range 0.9 to 1.2

INT_THREE_BY_THREE = np.array([
    [0.6,0,0],
    [0.2,1,0],
    [0.2,0,1]])

#INT_THREE_BY_THREE =  colour.RGB_to_RGB(INT_THREE_BY_THREE,RGB_COLOURSPACE_BT709, RGB_COLOURSPACE_ACES2065_1,chromatic_adaptation_transform=None, apply_cctf_decoding=False, apply_cctf_encoding=False)


RGB_COLOURSPACE_LIN_CIEXYZ_SCENE = colour.RGB_Colourspace('CIE XYZ - Scene-referred',
                          [[ 1.,  0.], [ 0.,  1.], [ 0.,  0.]], (0.31272,0.32903),'D65',
                            [[ 1,  0. ,  0. ],
                            [ 0. ,  1. ,  0. ],
                            [ 0. ,  0. ,  1]],
                            [[ 1,  0. ,  0. ],
                            [ 0. ,  1. ,  0. ],
                            [ 0. ,  0. ,  1]],
                            None,None,False, False)


RGB_COLOURSPACE_LIN_sRGB = RGB_COLOURSPACE_sRGB.copy()
RGB_COLOURSPACE_LIN_sRGB.cctf_decoding= None
RGB_COLOURSPACE_LIN_sRGB.cctf_encoding= None

FLAT_RGB = None  #Array if flats provided

FIT_MODE = "CALC"

class log_interp:

    def __init__(self,m,input_mg):
        self.mode = m
        mg = -1

        if (m == "GP_LOG"):
            mg = (np.power(400,input_mg ) - 1) / 399
        elif (m == "GP_PROTUNE"):
            mg = colour.models.log_decoding_Protune(input_mg)
        elif (m == "CLOG"):
            mg = colour.models.log_decoding_CanonLog(input_mg)
        elif (m == "CLOG2"):
            mg = colour.models.log_decoding_CanonLog2(input_mg)
        elif (m== "CLOG3"):
            mg =  colour.models.log_decoding_CanonLog3(input_mg)
        elif (m == "FLOG"):
            mg = colour.models.log_decoding_FLog(input_mg)
        elif (m == "FLOG2"):
            mg = colour.models.log_decoding_FLog2(input_mg)
        elif (m == "SLOG"):
            mg = colour.models.log_decoding_SLog(input_mg)
        elif (m == "SLOG2"):
            mg = colour.models.log_decoding_SLog2(input_mg)
        elif (m == "SLOG3"):
            mg = colour.models.log_decoding_SLog3(input_mg)
        elif (m == "VLOG"):
            mg = colour.models.log_decoding_VLog(input_mg)
        elif (m == "REC709"):
            mg = colour.models.oetf_inverse_BT709(input_mg)
        elif (m == "REC2100_HLG"):
            mg = colour.models.ootf_BT2100_HLG(input_mg)
        self.gain = MIDDLE_GRAY_LIN / mg  #Calculate Gain to scale input middle gray to correct value

    def __call__(self,x):
        if (self.mode == "GP_LOG"):
            return self.gain * (np.power(400, x) - 1) / 399
        elif (self.mode == "GP_PROTUNE"):
            return self.gain * colour.models.log_decoding_Protune(x)
        elif (self.mode == "CLOG"):
            return self.gain * colour.models.log_decoding_CanonLog(x)
        elif (self.mode == "CLOG2"):
            return self.gain * colour.models.log_decoding_CanonLog2(x)
        elif (self.mode == "CLOG3"):
            return self.gain * colour.models.log_decoding_CanonLog3(x)
        elif (self.mode == "FLOG"):
            return self.gain * colour.models.log_decoding_FLog(x)
        elif (self.mode == "FLOG2"):
            return self.gain * colour.models.log_decoding_FLog2(x)
        elif (self.mode == "SLOG"):
            return self.gain * colour.models.log_decoding_SLog(x)
        elif (self.mode == "SLOG2"):
            return self.gain * colour.models.log_decoding_SLog2(x)
        elif (self.mode == "SLOG3"):
            return self.gain * colour.models.log_decoding_SLog3(x)
        elif (self.mode == "VLOG"):
            return self.gain * colour.models.log_decoding_VLog(x)
        elif (self.mode == "REC709"):
            return self.gain * colour.models.oetf_inverse_BT709(x)
        elif (self.mode == "REC2100_HLG"):
            return self.gain * colour.models.ootf_BT2100_HLG(x)
        else:
            print("Wrong FIT_MODE selected ", FIT_MODE)



def main(args):


    global ILLUMINANT
    global INPUT_ILLUMINANT
    global FIT_MODE


    np.set_printoptions(suppress=True)
    np.set_printoptions(precision=4)
    debevec_fit = args.debevec_fit
    enable_optimize_illum = args.enable_optimize_illum

    FIT_MODE = args.fit_mode

    print_input_charts = args.print_input_charts
    print_ref_chart= args.print_ref_chart
    print_proof_charts = args.print_proof_charts
    print_curves = args.print_curves
    disp_chip_pixel_loc  = args.disp_chip_pixel_loc

    chart_weight = args.chart_weight

    rec709_lut = args.rec709_lut
    DWG_I_lut = args.DWG_I_lut
    gammut_comp = args.gammut_comp
    expand_tone = args.expand_tone
    expand_shadow = args.expand_shadow
    inflection_percent = args.inflection_percent
    lut_name = args.lut_name
    norm_lut_max = 0  #Not used
    illum_flag = 0

    bez_count_lum = args.bez_count_lum+2  #add bez before 0 and after 1 to control shape
    bez_count_hue_hue = args.bez_count_hue_hue +1  #add 1 because 0 and 1 are same value wrapped around
    bez_count_hue_sat = args.bez_count_hue_sat +1
    lum_opt_iter = args.lum_opt_iter
    three_by_three_opt_iter = args.three_by_three_opt_iter
    WB_Control =0  # Not used
    hue_loop_iter = args.hue_loop_iter
    sat_opt_iter = args.sat_opt_iter
    hue_hue_opt_iter = args.hue_hue_opt_iter
    hue_sat_opt_iter = args.hue_sat_opt_iter
    seed_lut_name = args.seed_lut_name

    max_gen_lum = args.max_gen_lum

    white_clipping_point = args.white_clipping_point
    black_clipping_point = args.black_clipping_point

    #Set Smoothing values low and increase as we iterate
    global SLOPE_LIMIT
    global SMOOTHING
    SLOPE_LIMIT = SLOPE_LIMIT / 5 ** lum_opt_iter
    SMOOTHING = SMOOTHING / 5** lum_opt_iter


    image = mpimg.imread('data/0.png')
    image = image[:, :, 0:3]
    chip_pixel = int(image.shape[0] * CHIP_XDELTA * CHIP_SAFE_AREA)   #safe pixel dimensions of each color chip

    chart_name_high = 0
    chart_name_low = 0

    print(" Reading input charts ")
    print("Found file ", end=" ")
    for i in range (0,11):
        file = Path("data/" + str(i) + ".png")
        if file.exists() :
            print(file, end=" ")
            chart_name_high = i
        else: break
    for i in range(0,-11, -1 ):
        file = Path("data/" + str(i) + ".png")
        if file.exists():
            print(file, end=" ")
            chart_name_low = i
        else:
            break
    print("\n")
    num_chips = NUM_CHIP_X * NUM_CHIP_Y
    num_charts = chart_name_high - chart_name_low + 1

    test_chart_RGB = np.zeros(shape=(num_charts, NUM_CHIP_X, NUM_CHIP_Y, 3), dtype=float)
    flat_RGB = None
    bad_chips = np.zeros(shape=(num_charts, NUM_CHIP_X, NUM_CHIP_Y), dtype=float)  # matrix of 0 for each clipped chip
    chips_min = np.zeros(shape=(num_charts, NUM_CHIP_X, NUM_CHIP_Y), dtype=float)
    chips_max = np.zeros(shape=(num_charts, NUM_CHIP_X, NUM_CHIP_Y), dtype=float)
    chips_min.fill(sys.float_info.max )
    chips_max.fill(sys.float_info.min)
    print("Reading file", end=' ')
    for i in range(chart_name_low, chart_name_high + 1):
        print( i, end= ' ')
        read_chart(("data/" + str(i) + ".png"),test_chart_RGB[i],chip_pixel,disp_chip_pixel_loc,chips_min,chips_max,i)
    print("\nNumber of Samples in test charts = ", np.shape(test_chart_RGB))

    file = Path("data/flat.png")
    if file.exists():  #Read Flat if it is exists
        global FLAT_RGB
        FLAT_RGB = np.zeros(shape=(NUM_CHIP_X, NUM_CHIP_Y, 3), dtype=float)
        read_chart(("data/flat.png"), FLAT_RGB, chip_pixel, disp_chip_pixel_loc,None ,None, 0)
        FLAT_RGB = np.clip(FLAT_RGB, a_min=0.000001, a_max=None) #Make sure no zero or negatives
        temp = np.reshape(FLAT_RGB, shape = (-1,3))
        mean =  np.mean(temp, axis=0)
        max = np.max(temp, axis=0)
        min = np.min(temp, axis=0)
        print ("Flat mean = ", mean ," Flat Max = " , max," Flat min = ", min  , " Delta = ", (max-min)/mean)

    #input_blackpoint = np.min(chips_min)  #Find lowest/Highest individual pixel
    #input_whitepoint = np.max(chips_max)

    input_blackpoint = np.min(test_chart_RGB)  #Find lowest/ highest value of each chip average
    input_whitepoint = np.max(test_chart_RGB)


    exp_chip_count = np.sum(EXP_CHIPS)
    if (exp_chip_count==0): print ("Missing EXP_CHIPS")
    input_MG = np.sum((test_chart_RGB[0,:,:,0] * EXP_CHIPS,test_chart_RGB[0,:,:,1] * EXP_CHIPS,test_chart_RGB[0,:,:,2] * EXP_CHIPS))/(exp_chip_count*3)
    print("Input Blackpoint = ", input_blackpoint, " Whitepoint = ", input_whitepoint, " Middle Gray = ",input_MG )

    bad_chip_low = input_blackpoint+black_clipping_point  #Add margin provided by user
    bad_chip_high = input_whitepoint - (1- white_clipping_point)
    calc_bad_chips(chart_name_low,chart_name_high, chips_min,chips_max,bad_chip_high,bad_chip_low,bad_chips)  #Exlcude sample chips that are clipped low/high



    if(print_input_charts):
        for i in range(chart_name_low, chart_name_high + 1):

            plt.imshow(test_chart_RGB[i,:,:,:])
            plt.xlabel(None)
            plt.ylabel(None)
            plt.axis('off')
            plt.title("Chart " + str(i))
            plt.show()

    if (print_ref_chart):

        Ref_Chart_RGB = LAB_2_sRGB(REFCHIP_LAB)
        Ref_Chart_RGB = np.clip(Ref_Chart_RGB, a_min=0, a_max=1)
        plt.imshow(Ref_Chart_RGB)
        plt.xlabel(None)
        plt.ylabel(None)
        plt.axis('off')
        plt.title("Reference Chart")
        plt.show()

    seedLUT_values = []
    read_seed_lut(seed_lut_name, seedLUT_values)

    s_gain = 1
    s_gamma = 1

    x = interp_1d_x_vals(bez_count_lum, 0, input_blackpoint, input_whitepoint, input_MG)
    MG_bez = (np.abs(x - input_MG)).argmin()

    bez_values_lum = np.zeros(shape=(bez_count_lum, 2), dtype=float)
    bez_values_hue_hue =  np.zeros(shape=(bez_count_hue_hue, 2), dtype=float)
    best_bez_hue_hue = bez_values_hue_hue.copy()
    bez_values_hue_sat = np.full(shape=(bez_count_hue_sat, 2), fill_value=1, dtype=float)
    best_bez_hue_sat = bez_values_hue_sat.copy()
    gen_seed_bez_values(bez_values_lum, seedLUT_values, bez_count_lum,MG_bez, input_MG, input_blackpoint, input_whitepoint)
    if not debevec_fit:
        print("Initial Bez Values = ", bez_values_lum[:, 1])

    best_three_by_three = INT_THREE_BY_THREE.copy()

    time_start_lum = time.time()

    best_bez_lum = bez_values_lum.copy()

    debevec_curve = debevec(chart_name_low, chart_name_high, input_whitepoint, input_MG)
    if debevec_fit:
        best_bez_lum[:, 1] = debevec_curve(best_bez_lum[:, 0])
        best_bez_lum = set_bez_MG(bez_count_lum, best_bez_lum, MG_bez)
        print("\nDebevec Bez values = ", best_bez_lum[:,1])

    lum1_error = error_sum(bez_count_lum, best_bez_lum, test_chart_RGB, chart_name_low,
                           chart_name_high, best_three_by_three,
                           bad_chips, "Luma", chart_weight, final=0, input_blackpoint=input_blackpoint,
                           input_whitepoint=input_whitepoint, input_MG = input_MG )

    lum2_error = error_sum(bez_count_lum, best_bez_lum, test_chart_RGB, chart_name_low,
                           chart_name_high, best_three_by_three,
                           bad_chips, "Luma2", chart_weight, final=0, input_blackpoint=input_blackpoint,
                           input_whitepoint=input_whitepoint, input_MG = input_MG)

    print("\nLuma 1 Error = ", lum1_error , "Luma 2 Error = ",lum2_error)

    if three_by_three_opt_iter:
        bez_curve_lum = interp_1d_setup(bez_count_lum, best_bez_lum[:, 1], hue=0, input_blackpoint=input_blackpoint,
                                        input_whitepoint=input_whitepoint, input_MG=input_MG)
        best_three_by_three = main_color(test_chart_RGB, bez_curve_lum, best_three_by_three, bad_chips,
                                         chart_name_low, chart_name_high, chart_weight, three_by_three_opt_iter)
        print("\n")

    if FIT_MODE == "CALC":  #Skip if using calulated log formula
        # Genetic algorithm
        best_bez_lum = main_luma(best_bez_lum, num_charts, test_chart_RGB, input_blackpoint, input_whitepoint, bad_chips,
                                 1, best_three_by_three, bez_count_lum, chart_name_low, chart_name_high, chart_weight,
                                 max_gen_lum, input_MG, MG_bez, )

        print("\nBest Luma Bez  = ", best_bez_lum[:, 1])

        lum1_error = error_sum(bez_count_lum, best_bez_lum, test_chart_RGB, chart_name_low,
                               chart_name_high, best_three_by_three,
                               bad_chips, "Luma", chart_weight, final=0, input_blackpoint=input_blackpoint,
                               input_whitepoint=input_whitepoint, input_MG=input_MG)

        lum2_error = error_sum(bez_count_lum, best_bez_lum, test_chart_RGB, chart_name_low,
                               chart_name_high, best_three_by_three,
                               bad_chips, "Luma2", chart_weight, final=0, input_blackpoint=input_blackpoint,
                               input_whitepoint=input_whitepoint, input_MG=input_MG)

        print("Luma 1 Error = ", lum1_error,  "Luma 2 Error = ", lum2_error, "\n" )



    num_round_outer = 0
    while num_round_outer < lum_opt_iter:
        num_round_outer = num_round_outer +1

        if INPUT_ILLUMINANT == [-1,-1]: #We are not going through the second time with optimized ILLUM
            #Increase smoothing every round
            SLOPE_LIMIT = SLOPE_LIMIT * 5
            SMOOTHING =  SMOOTHING *5

        # Binary Search 3x3
        best_three_by_three = three_by_three_binary(PERTURB_MAX_3X3 / (1.5**num_round_outer), test_chart_RGB, chart_name_low,
                                                    chart_name_high, best_three_by_three, bad_chips,chart_weight,bez_count_lum,
                                                    best_bez_lum, bez_count_hue_hue,bez_count_hue_sat,best_bez_hue_hue,best_bez_hue_sat,
                                                    input_blackpoint,input_whitepoint, s_gain,s_gamma,input_MG, WB_Control)

        #Run Binary Search Luma Optimizations
        print ("\nBinary Search round ",num_round_outer, " of ", lum_opt_iter)

        lum1_error = error_sum(bez_count_lum, best_bez_lum, test_chart_RGB, chart_name_low,
                               chart_name_high, best_three_by_three,
                               bad_chips, "Luma", chart_weight, final=0, input_blackpoint=input_blackpoint,
                               input_whitepoint=input_whitepoint, input_MG=input_MG)

        lum2_error = error_sum(bez_count_lum, best_bez_lum, test_chart_RGB, chart_name_low,
                               chart_name_high, best_three_by_three,
                               bad_chips, "Luma2", chart_weight, final=0, input_blackpoint=input_blackpoint,
                               input_whitepoint=input_whitepoint, input_MG=input_MG)

        print("Luma 1 Error = ", lum1_error , "Luma 2 Error = ",lum2_error )

        if FIT_MODE == "CALC":  #Skip if using calculated log formula
            for num_round_inner in range(1,MAX_ITER_LUM):
                peturb_amount = PETURB_MAX_LUM / (1.5**num_round_outer) / (2**num_round_inner)
                error_old = sys.float_info.max
                error_new = -1
                flag = -1
                for n in range (0,bez_count_lum*5):
                    flag = n
                    best_bez_lum = perturb_curve(best_bez_lum, bez_values_hue_hue, bez_values_hue_sat, test_chart_RGB, bad_chips, num_round_inner, best_three_by_three,
                                         bez_count_lum, bez_count_hue_hue,bez_count_hue_sat, chart_name_low, chart_name_high, chart_weight,
                                         peturb_amount,"Lum",input_blackpoint,input_whitepoint, MG_bez = MG_bez, input_MG =input_MG)

                    error_new = error_sum(bez_count_lum, best_bez_lum, test_chart_RGB, chart_name_low,
                                   chart_name_high, best_three_by_three,
                                   bad_chips, "Luma2", chart_weight, final=0, input_blackpoint=input_blackpoint,
                                   input_whitepoint=input_whitepoint, input_MG=input_MG)

                    if (error_new + ERROR_EPS) < error_old:
                        error_old = error_new
                    else:
                        break

            print("Binary Search Best Luma Bez  = " ,best_bez_lum[:,1])

            lum1_error = error_sum(bez_count_lum, best_bez_lum, test_chart_RGB, chart_name_low,
                                   chart_name_high, best_three_by_three,
                                   bad_chips, "Luma", chart_weight, final=0, input_blackpoint=input_blackpoint,
                                   input_whitepoint=input_whitepoint, input_MG = input_MG)

            lum2_error = error_sum(bez_count_lum, best_bez_lum, test_chart_RGB, chart_name_low,
                                   chart_name_high, best_three_by_three,
                                   bad_chips, "Luma2", chart_weight, final=0, input_blackpoint=input_blackpoint,
                                   input_whitepoint=input_whitepoint ,input_MG = input_MG)

            print("Round ", num_round_outer, "Luma 1 Error = ",lum1_error ,"Luma 2 Error = ", lum2_error )


        # Run 3x3 Color genetic Optimization
        if three_by_three_opt_iter:
            bez_curve_lum = interp_1d_setup(bez_count_lum, best_bez_lum[:, 1], hue=0, input_blackpoint=input_blackpoint,
                                            input_whitepoint=input_whitepoint, input_MG=input_MG)
            best_three_by_three = main_color(test_chart_RGB, bez_curve_lum, best_three_by_three, bad_chips, chart_name_low, chart_name_high, chart_weight, three_by_three_opt_iter)
            #best_three_by_three = main_color_thread(test_chart_RGB, bez_curve_lum, best_three_by_three, num_charts, bad_chips,num_round_outer, chart_name_low, chart_name_high, chart_weight,three_by_three_opt_iter, WB_Control)
            print("\n")


        # Run Input Illuminant Optimization
        if (num_round_outer == lum_opt_iter)  and INPUT_ILLUMINANT == [-1, -1] and (enable_optimize_illum =="I" or enable_optimize_illum =="IO") :
            num_round_outer = 0
            illum_flag = 1
            bez_curve_lum = interp_1d_setup(bez_count_lum, best_bez_lum[:, 1], hue=0, input_blackpoint=input_blackpoint,
                                            input_whitepoint=input_whitepoint, input_MG=input_MG)
            optimize_input_illum(test_chart_RGB, chart_name_low, chart_name_high, bez_curve_lum, best_three_by_three)
            print ("\nRunning Optimization loop again with new input illuminant")


    print("\nTotal Time (s) = ",  time.time() - time_start_lum )

    color_error = error_sum(bez_count_lum, best_bez_lum, test_chart_RGB,
                               chart_name_low, chart_name_high, best_three_by_three, bad_chips, 'Color_all',
                               chart_weight, final=0,
                               bez_count_hue_hue=bez_count_hue_hue, bez_count_hue_sat=bez_count_hue_sat,
                               bez_values_hue_hue=best_bez_hue_hue, bez_values_hue_sat=best_bez_hue_sat,
                               input_blackpoint=input_blackpoint, input_whitepoint=input_whitepoint, input_MG = input_MG)

    combined_error = error_sum(bez_count_lum, best_bez_lum, test_chart_RGB,
                            chart_name_low, chart_name_high, best_three_by_three, bad_chips, 'Combined',chart_weight, final=0,
                            bez_count_hue_hue=bez_count_hue_hue, bez_count_hue_sat=bez_count_hue_sat,
                            bez_values_hue_hue=best_bez_hue_hue, bez_values_hue_sat=best_bez_hue_sat,
                               input_blackpoint=input_blackpoint,input_whitepoint=input_whitepoint, input_MG = input_MG)

    print("\nFinal Color-All Error W/3x3 = ",color_error, " Combined Error = ", combined_error)

    sat_max(bez_count_lum, best_bez_lum[:, 1], best_three_by_three,input_blackpoint,input_whitepoint,input_MG,test_chart_RGB, chart_name_low, chart_name_high)


    if hue_loop_iter:
        for hue_loop in range(1, hue_loop_iter + 1):

            for s_round in range(1, sat_opt_iter + 1):
                max_gamma = PERTURB_MAX_S_GAMMA / s_round / hue_loop
                max_gain = PERTURB_MAX_S_GAIN / s_round / hue_loop

                s_gamma = s_gamma_opt(s_gamma, max_gamma, bez_count_lum, best_bez_lum, test_chart_RGB, chart_name_low,
                                      chart_name_high,
                                      best_three_by_three, bad_chips, chart_weight, bez_count_hue_hue, bez_count_hue_sat,
                                      bez_values_hue_hue, bez_values_hue_sat,
                                      input_blackpoint, input_whitepoint, s_gain, input_MG)

                s_gain = s_gain_opt(s_gain, max_gain, bez_count_lum, best_bez_lum, test_chart_RGB, chart_name_low,
                                    chart_name_high,
                                    best_three_by_three, bad_chips, chart_weight, bez_count_hue_hue, bez_count_hue_sat,
                                    bez_values_hue_hue, bez_values_hue_sat,
                                    input_blackpoint, input_whitepoint, s_gamma, input_MG)

            color_error = error_sum(bez_count_lum, best_bez_lum, test_chart_RGB,
                                    chart_name_low, chart_name_high, best_three_by_three, bad_chips, 'Color',
                                    chart_weight, final=0,
                                    bez_count_hue_hue=bez_count_hue_hue, bez_count_hue_sat=bez_count_hue_sat,
                                    bez_values_hue_hue=best_bez_hue_hue, bez_values_hue_sat=best_bez_hue_sat,
                                    input_blackpoint=input_blackpoint, input_whitepoint=input_whitepoint, s_gain=s_gain,
                                    s_gamma=s_gamma, input_MG=input_MG)

            combined_error = error_sum(bez_count_lum, best_bez_lum, test_chart_RGB,
                                       chart_name_low, chart_name_high, best_three_by_three, bad_chips, 'Combined',
                                       chart_weight, final=0,
                                       bez_count_hue_hue=bez_count_hue_hue, bez_count_hue_sat=bez_count_hue_sat,
                                       bez_values_hue_hue=best_bez_hue_hue, bez_values_hue_sat=best_bez_hue_sat,
                                       input_blackpoint=input_blackpoint, input_whitepoint=input_whitepoint, s_gain=s_gain,
                                       s_gamma=s_gamma, input_MG=input_MG)

            print("\nSaturation Gain = ", s_gain, "Saturation Gamma = ", s_gamma)
            print("Color Error W S_gain = ", color_error , " Combined Error = ", combined_error, "\n")

            # Binary Search 3x3
            best_three_by_three = three_by_three_binary(PERTURB_MAX_3X3 / (hue_loop *10 ), test_chart_RGB,
                                                        chart_name_low, chart_name_high, best_three_by_three, bad_chips,
                                                        chart_weight, bez_count_lum, best_bez_lum,
                                                        bez_count_hue_hue, bez_count_hue_sat, best_bez_hue_hue,
                                                        best_bez_hue_sat, input_blackpoint, input_whitepoint, s_gain,
                                                        s_gamma, input_MG,WB_Control)



            print("\nHue Loop Iteration ", hue_loop, " of ", hue_loop_iter)
            if hue_hue_opt_iter:
                for num_round_outer in range(0, hue_hue_opt_iter ):
                    for num_round_inner in range(0, 10):
                        perturb_amount = PERTURB_MAX_HUE_HUE / (2**num_round_outer * 2**hue_loop) / 2**num_round_inner
                        for num_round_inner2 in range(5):
                            best_bez_hue_hue = perturb_curve(best_bez_lum, best_bez_hue_hue, best_bez_hue_sat, test_chart_RGB,
                                                             bad_chips,
                                                             num_round_inner, best_three_by_three,
                                                             bez_count_lum, bez_count_hue_hue, bez_count_hue_sat, chart_name_low,
                                                             chart_name_high, chart_weight,
                                                             perturb_amount, "Hue-Hue", input_blackpoint, input_whitepoint, s_gain,
                                                             s_gamma, MG_bez=MG_bez, input_MG=input_MG)

                    combined_error = error_sum(bez_count_lum, best_bez_lum, test_chart_RGB,
                                               chart_name_low, chart_name_high, best_three_by_three, bad_chips, 'Combined',
                                               chart_weight, final=0,
                                               bez_count_hue_hue=bez_count_hue_hue, bez_count_hue_sat=bez_count_hue_sat,
                                               bez_values_hue_hue=best_bez_hue_hue, bez_values_hue_sat=best_bez_hue_sat,
                                               input_blackpoint=input_blackpoint,
                                               input_whitepoint=input_whitepoint, s_gain=s_gain, s_gamma=s_gamma,
                                               input_MG=input_MG)

                    color_error = error_sum(bez_count_lum, best_bez_lum, test_chart_RGB,
                                            chart_name_low, chart_name_high, best_three_by_three, bad_chips, 'Color',
                                            chart_weight, final=0,
                                            bez_count_hue_hue=bez_count_hue_hue, bez_count_hue_sat=bez_count_hue_sat,
                                            bez_values_hue_hue=best_bez_hue_hue, bez_values_hue_sat=best_bez_hue_sat,
                                            input_blackpoint=input_blackpoint, input_whitepoint=input_whitepoint, s_gain=s_gain,
                                            s_gamma=s_gamma, input_MG=input_MG)
                    print("Hue-Hue Round ", num_round_outer+1, " of ", hue_hue_opt_iter," Color Error = ", color_error , " Combined Error = ",
                          combined_error , " Bez ", best_bez_hue_hue[:, 1])
            if hue_sat_opt_iter:
                for num_round_outer in range(0, hue_sat_opt_iter ):
                    for num_round_inner in range(0, 10):
                        perturb_amount = PERTURB_MAX_HUE_SAT / (2**num_round_outer * 2**hue_loop) / 2**num_round_inner
                        for num_round_inner2 in range(5):
                            best_bez_hue_sat = perturb_curve(best_bez_lum, best_bez_hue_hue, best_bez_hue_sat, test_chart_RGB,
                                                             bad_chips,
                                                             num_round_inner, best_three_by_three,
                                                             bez_count_lum, bez_count_hue_hue, bez_count_hue_sat, chart_name_low,
                                                             chart_name_high, chart_weight,
                                                             perturb_amount, "Hue-Sat", input_blackpoint, input_whitepoint, s_gain,
                                                             s_gamma, MG_bez=MG_bez, input_MG=input_MG )

                    combined_error = error_sum(bez_count_lum, best_bez_lum, test_chart_RGB,
                                               chart_name_low, chart_name_high, best_three_by_three, bad_chips, 'Combined',
                                               chart_weight, final=0,
                                               bez_count_hue_hue=bez_count_hue_hue, bez_count_hue_sat=bez_count_hue_sat,
                                               bez_values_hue_hue=best_bez_hue_hue, bez_values_hue_sat=best_bez_hue_sat,
                                               input_blackpoint=input_blackpoint, input_whitepoint=input_whitepoint,
                                               s_gain=s_gain, s_gamma=s_gamma, input_MG=input_MG)

                    color_error = error_sum(bez_count_lum, best_bez_lum, test_chart_RGB,
                                            chart_name_low, chart_name_high, best_three_by_three, bad_chips, 'Color',
                                            chart_weight, final=0,
                                            bez_count_hue_hue=bez_count_hue_hue, bez_count_hue_sat=bez_count_hue_sat,
                                            bez_values_hue_hue=best_bez_hue_hue, bez_values_hue_sat=best_bez_hue_sat,
                                            input_blackpoint=input_blackpoint, input_whitepoint=input_whitepoint, s_gain=s_gain,
                                            s_gamma=s_gamma, input_MG=input_MG)

                    print("Hue-Sat Round ", num_round_outer+1, " of ", hue_sat_opt_iter," Color Error = ", color_error , " Combined Error = ",
                          combined_error , " Bez ", best_bez_hue_sat[:, 1])


    bez_curve_lum = interp_1d_setup(bez_count_lum, best_bez_lum[range(bez_count_lum),1],hue=0,input_blackpoint=input_blackpoint,
                                    input_whitepoint=input_whitepoint, input_MG = input_MG)
    bez_curve_hue_hue = interp_1d_setup(bez_count_hue_hue, best_bez_hue_hue[range(bez_count_hue_hue), 1],hue=1,input_blackpoint=input_blackpoint,
                                        input_whitepoint=input_whitepoint,mode = HSL_MODE)
    bez_curve_hue_sat = interp_1d_setup(bez_count_hue_sat, best_bez_hue_sat[range(bez_count_hue_sat), 1],hue=1,input_blackpoint=input_blackpoint,
                                        input_whitepoint=input_whitepoint,mode = HSL_MODE)

    print("\nTotal Time (s) = ",  time.time() -time_start_lum )

    combined_error = error_sum(bez_count_lum, best_bez_lum, test_chart_RGB,
                               chart_name_low, chart_name_high, best_three_by_three, bad_chips, 'Combined',
                               chart_weight, final=1,
                               bez_count_hue_hue=bez_count_hue_hue, bez_count_hue_sat=bez_count_hue_sat,
                               bez_values_hue_hue=best_bez_hue_hue, bez_values_hue_sat=best_bez_hue_sat,
                               input_blackpoint=input_blackpoint, input_whitepoint=input_whitepoint,
                               s_gain=s_gain, s_gamma=s_gamma, input_MG=input_MG)

    color_error = error_sum(bez_count_lum, best_bez_lum, test_chart_RGB,
                            chart_name_low, chart_name_high, best_three_by_three, bad_chips, 'Color_all',
                            chart_weight, final=0,
                            bez_count_hue_hue=bez_count_hue_hue, bez_count_hue_sat=bez_count_hue_sat,
                            bez_values_hue_hue=best_bez_hue_hue, bez_values_hue_sat=best_bez_hue_sat,
                            input_blackpoint=input_blackpoint, input_whitepoint=input_whitepoint, s_gain=s_gain,
                            s_gamma=s_gamma, input_MG=input_MG)

    print("\nColor-ALL Error = ", color_error , " Final Combined Error = ",combined_error )

    test_chart_LAB_norm = calc_charts(test_chart_RGB, chart_name_low, chart_name_high, bez_curve_lum=bez_curve_lum,
                                      three_by_three=best_three_by_three, bez_curve_hue_hue=bez_curve_hue_hue,
                                      bez_curve_hue_sat=bez_curve_hue_sat, s_gain=s_gain,s_gamma =s_gamma)

    bez_curve_hue_hue = interp_1d_setup(bez_count_hue_hue, bez_values_hue_hue[range(bez_count_hue_hue), 1], hue=1,
                                        input_blackpoint=input_blackpoint, input_whitepoint=input_whitepoint,
                                        mode=HSL_MODE)
    bez_curve_hue_sat = interp_1d_setup(bez_count_hue_sat, bez_values_hue_sat[range(bez_count_hue_sat), 1], hue=1,
                                        input_blackpoint=input_blackpoint, input_whitepoint=input_whitepoint,
                                        mode=HSL_MODE)

    if (enable_optimize_illum =="O" or enable_optimize_illum =="IO"):       #Optimize output Illuminant
        ILLUMINANT = optimize_ouput_illum(best_three_by_three, s_gamma, s_gain, bez_curve_hue_hue, bez_curve_hue_sat)
    else:
        output_WB_error = wb_error(best_three_by_three, 1, s_gamma, s_gain, bez_curve_hue_hue, bez_curve_hue_sat)
        temperature = colour.xy_to_CCT(np.array(ILLUMINANT))
        print("Non-Optimized Output Illum = ", f'{ILLUMINANT[0]:.8f}', ",", f'{ILLUMINANT[1]:.8f}', " Temp = ", temperature,
              " Output WB Error = ", output_WB_error)

    calc_dr(test_chart_RGB, chart_name_low, chart_name_high, bez_curve_lum, input_whitepoint)

    if SAVE_LUT:

        if(expand_shadow):  #Set input black point to half of the next higher bez to retain more details in blacks
            print (best_bez_lum[1, 1], best_bez_lum[2, 1])
            best_bez_lum[1, 1] = 0.5 * best_bez_lum[2, 1]
            print(best_bez_lum[1, 1], best_bez_lum[2, 1])

        bez_curve_lum = interp_1d_setup(bez_count_lum, best_bez_lum[range(bez_count_lum), 1], hue=0,
                                        input_blackpoint=input_blackpoint,
                                        input_whitepoint=input_whitepoint, input_MG=input_MG)

        print ("\nWriting 1d LUT ", lut_name)
        write_lut_1D(lut_name, bez_curve_lum, input_whitepoint)

        if SAVE_LUT ==1 or SAVE_LUT == 4:  # Shaper
            print("\nWriting Shaper LUT ")
            write_lut_shaper(lut_name, bez_curve_lum, best_three_by_three,
                norm_lut_max, input_whitepoint,input_blackpoint, s_gain, s_gamma, gammut_comp, expand_tone,
                inflection_percent, bez_count_hue_hue, best_bez_hue_hue, bez_count_hue_sat, best_bez_hue_sat, enable_optimize_illum)

        if SAVE_LUT ==2 or SAVE_LUT == 4:  # Cube
            print("\nWriting Cube LUT ")
            write_lut_cube(lut_name, bez_curve_lum, best_three_by_three ,rec709_lut,
                           input_whitepoint,input_blackpoint,s_gain,s_gamma, DWG_I_lut,gammut_comp,expand_tone,inflection_percent,
                           best_bez_hue_hue,bez_count_hue_hue,best_bez_hue_sat,bez_count_hue_sat, enable_optimize_illum)

        if SAVE_LUT == 3 or SAVE_LUT == 4: #  CLF
            print ("\nWriting CLF LUT ")
            write_lut_clf(lut_name, bez_curve_lum, best_three_by_three, rec709_lut, bez_curve_hue_hue, bez_curve_hue_sat,
                          norm_lut_max, input_whitepoint, s_gain, s_gamma)
        print("Luts complete")

    plothisto(num_chips,test_chart_LAB_norm)


    if(print_curves):
        printcurves(bez_count_lum,best_bez_lum,best_three_by_three,input_blackpoint,input_whitepoint, input_MG,debevec_curve)
        printHSLcurves(bez_count_hue_hue, bez_count_hue_sat, best_bez_hue_hue, best_bez_hue_sat,input_blackpoint,input_whitepoint,s_gamma, s_gain)

    if(print_proof_charts):
        print("Displaying Test Charts")
        disp_chart_single(bez_curve_lum, best_three_by_three, test_chart_RGB,bez_curve_hue_hue,bez_curve_hue_sat,s_gain,s_gamma)
        disp_chart(bez_curve_lum, best_three_by_three,bez_curve_hue_hue,bez_curve_hue_sat,s_gain,s_gamma)

#end Main

def read_seed_lut(filename, seedLUT_values):
    with open(filename) as file_in:
        flag = 0
        for line in file_in:
            if flag == 1:
                temp = line.split(' ')

                t = (float(temp[0]) + float(temp[1])+ float(temp[2])) / 3
                seedLUT_values.append(t)
            if line[0] == '#':
                flag = 1

# end read_seed_lut


def gen_seed_bez_values(bez_values, seedLUT_values,bez_count_lum, MG_bez, input_MG, input_blackpoint, input_whitepoint):
    spacing = 1 / (bez_count_lum - 2)

    x = interp_1d_x_vals(bez_count_lum, 0, input_blackpoint, input_whitepoint, input_MG)

    for i in range(1, bez_count_lum - 1):
        j = int((len(seedLUT_values)-1) * (i-1)/(bez_count_lum-3))


        bez_values[i] = x[i], seedLUT_values[j]

    first = 0 - bez_values[2,1]
    last = (2 * bez_values[bez_count_lum - 2,1]) - bez_values[bez_count_lum - 3,1]

    bez_values[0] = (0- spacing ),first

    if input_whitepoint<1:
        max_x = input_whitepoint + spacing
    else:max_x = 1 + spacing
    bez_values[bez_count_lum-1] = max_x, last
    bez_values = set_bez_MG(bez_count_lum, bez_values, MG_bez)

#End gen_seed_bez_values


def read_chart(file_path,chip_RGB, chip_pixel,disp_chip_pixel_loc, chip_min, chip_max,chart_num):
    pngdata = png.Reader(file_path).read_flat()
    image = np.array(pngdata[2]).reshape((pngdata[1], pngdata[0], -1))
    image = (image/(2**16-1)).astype(float)

    chip_loc_x = []
    chip_loc_y = []

    for i in range(NUM_CHIP_X):
        chip_loc_x.append(int((CHIP_X0 + (i * CHIP_XDELTA)) * image.shape[0]))

    for i in range(NUM_CHIP_Y):
        chip_loc_y.append(int((CHIP_Y0 + (i * CHIP_YDELTA)) * image.shape[1]))

    if(disp_chip_pixel_loc):
        print(chip_loc_x)
        print(chip_loc_y)

    for i in range(NUM_CHIP_X):
        for j in range(NUM_CHIP_Y):
            x = chip_loc_x[i]
            y = chip_loc_y[j]

            t =image[(x-int(chip_pixel/2)):(x +int(chip_pixel/2)),(y -int(chip_pixel/2)):(y+int(chip_pixel/2))]
            t = t[:,:,0:3]
            t = np.reshape(t, shape = (-1,3))


            if chip_min is not None: #Not the Flat chart
                if np.max(t)> chip_max[chart_num,i,j]:
                    chip_max[chart_num,i,j] = np.max(t)

                if np.min(t) < chip_min[chart_num, i, j]:
                    chip_min[chart_num, i, j] = np.min(t)

            upper_limit = np.percentile(t,CHART_UPPER_PERCENTILE,axis=0)  #Filter out outliers
            lower_limit = np.percentile(t, CHART_LOWER_PERCENTILE, axis=0)

            mask = (
                    (t[:, 0] >= lower_limit[0]) & (t[:, 0] <= upper_limit[0]) &
                    (t[:, 1] >= lower_limit[1]) & (t[:, 1] <= upper_limit[1]) &
                    (t[:, 2] >= lower_limit[2]) & (t[:, 2] <= upper_limit[2])
            )
            chip_RGB[i, j] = t[mask].mean(axis=0)

# End Read Chart


def calc_bad_chips(chart_name_low,chart_name_high, chips_min,chips_max,white_point,black_point,bad_chips):  #Dectected clipped/crushed source clips

    for i in range(chart_name_low, chart_name_high + 1):
        for j in range(NUM_CHIP_X):
            for k in range(NUM_CHIP_Y):
                #print(chips_max[i,j,k],chips_min[i,j,k] )

                if (chips_max[i,j,k] <white_point) and (chips_min[i,j,k]>black_point):  #Check if chip is clipped or crushed
                    bad_chips [i, j, k] = 1

    good_chip_count = np.sum(bad_chips)
    print ("Total number of non-clipped chips = ", good_chip_count)
#end Bad_chips


def write_lut_cube(filename, bez_curve_lum, three_by_three ,rec709_lut,input_whitepoint,input_blackpoint,s_gain, s_gamma, DWG_I_lut,gammut_comp,
                   expand_tone,inflection_percent,best_bez_hue_hue,bez_count_hue_hue,best_bez_hue_sat,bez_count_hue_sat, enable_optimize_illum):


    bez_hue_hue = best_bez_hue_hue[range(bez_count_hue_hue), 1]
    bez_hue_hue = bez_hue_hue + MANUAL_HUE_HUE

    bez_hue_sat = best_bez_hue_sat[range(bez_count_hue_sat), 1]
    bez_hue_sat = bez_hue_sat + MANUAL_HUE_SAT

    bez_curve_hue_hue = interp_1d_setup(bez_count_hue_hue,bez_hue_hue, hue=1,
                                        input_blackpoint=input_blackpoint,
                                        input_whitepoint=input_whitepoint, mode=HSL_MODE)
    bez_curve_hue_sat = interp_1d_setup(bez_count_hue_sat, bez_hue_sat, hue=1,
                                        input_blackpoint=input_blackpoint,
                                        input_whitepoint=input_whitepoint, mode=HSL_MODE)
    lut_size = 65

    max_in = input_whitepoint
    if max_in< 1 :
        max_in = 1

    max_WP = interp_1d(input_whitepoint, bez_curve_lum)
    max_1 = interp_1d(1, bez_curve_lum)
    if not (expand_tone): print("Max Output Nits at input WP = ", max_WP * 100, " Max output at 1.0 input = ", max_1*100)

    with open(str("1d"+filename+".cube"), 'w', 100 * (2 ** 20)) as f:
        f.writelines("TITLE " + filename + "\n")
        f.writelines("LUT_1D_SIZE 1024\n")
        if (rec709_lut): f.writelines("#ZRG, OUTPUT COLOR SPACE = REC709\n")
        else: f.writelines("#ZRG, OUTPUT COLOR SPACE = sRGB Primaries, Linear Tone Curve\n")
        if not (expand_tone): f.writelines("#Max Output Nits = " + str(max_WP * 100) + " Max output at 1.0 input = "+ str(max_1 * 100) + "\n")

        out = np.zeros(shape=(1024), dtype=float)

        for i in range(1024):
            t = interp_1d(((i*input_whitepoint) / 1023), bez_curve_lum)
            out[i] = t

        #out = np.clip(out, a_min=0, a_max=None)

        if (rec709_lut): out = ACES2065_2_REC709(out)


        for i in range(1024):
            f.writelines(str(out[i]) + " " + str(out[i]) + " " + str(out[i]) + "\n")
        f.close()

    with open(str("3d"+filename+".cube"), 'w', 100 * (2 ** 20)) as f:
        f.writelines("TITLE " + filename + "\n")
        f.writelines("LUT_3D_SIZE "+str(lut_size)+""
                                                  "\n")
        f.writelines("LUT_3D_INPUT_RANGE 0 "+ str(max_in) +"\n")
        if (rec709_lut):
            f.writelines("#ZRG, OUTPUT COLOR SPACE = REC709\n")
        elif (DWG_I_lut):
            f.writelines("#ZRG, OUTPUT COLOR SPACE = Davinchi Wide Gammut Primaries, Intermediate Tone Curve\n")
            if not (expand_tone): f.writelines("#Max Output Nits = " + str(max_WP * 100) + " Max output at 1.0 input = " + str(max_1 * 100) + "\n")
        else:
            f.writelines("#ZRG, OUTPUT COLOR SPACE = sRGB Primaries, Linear Tone Curve\n")
            if not (expand_tone): f.writelines( "#Max Output Nits = " + str(max_WP * 100) + " Max output at 1.0 input = " + str(max_1 * 100) + "\n")

        out = np.zeros(shape=(lut_size, lut_size, lut_size, 3), dtype=float)
        for k in range(lut_size):
            for j in range(lut_size):
                for i in range(lut_size):
                    out[i,j,k] = interp_1d((i/(lut_size-1))*max_in, bez_curve_lum),interp_1d((j/(lut_size-1))*max_in, bez_curve_lum),interp_1d((k/(lut_size-1))*max_in, bez_curve_lum)

        out = np.matmul(out, three_by_three)  #Multiply by 3x3

        if (gammut_comp):
            out = gammut_compression(out, GAM_COMP_STRENGTH,gammut_comp)
        """else:
            out = np.clip(out, a_min=0, a_max=None)"""

        if not(s_gain ==1 and s_gamma ==1 ):  #No Need for HSL adjustments
            out_HSL = ACES2065_2_HSL(out)

            np.where(out_HSL[:, :, :, 0] < 0, out_HSL[:, :, :, 0],out_HSL[:, :, :, 0] + 1)  # Normalize Hue values that wrapped around
            np.where(out_HSL[:, :, :, 0] > 1, out_HSL[:, :, :, 0], out_HSL[:, :, :, 0] - 1)

            out_HSL = np.clip(out_HSL, a_min=0, a_max=None)
            out_HSL[:, :, :, 1] = out_HSL[:, :, :, 1]** (1 / s_gamma)  # Apply Saturation Gamma
            out_HSL[:, :, :, 1] = out_HSL[:, :, :, 1] * s_gain  # Apply saturation gain

            out_HSL[:, :,:, 0] = out_HSL[:, :,:, 0] + interp_1d(out_HSL[:, :,:, 0],bez_curve_hue_hue)  # Apply Hue_hue Curve
            np.where(out_HSL[:, :,:, 0] < 0, out_HSL[:, :, :,0], out_HSL[:, :,:, 0] + 1)  # Normalize Hue values that wrapped around
            np.where(out_HSL[:, :,:, 0] > 1, out_HSL[:, :, :,0], out_HSL[:, :,:, 0] - 1)

            out_HSL[:, :, :,1] = out_HSL[:, :,:, 1] * interp_1d(out_HSL[:, :,:, 0],bez_curve_hue_sat)  # Apply Hue_sat Curve
            #out_HSL = np.clip(out_HSL, a_min=0, a_max=None)

            out = HSL_2_ACES2065(out_HSL)  # Convert back to Aces2065

        if (expand_tone): #Do tone expansion in xyY space as OKlab was found to introduce artifacts.
            init_max = interp_1d(input_whitepoint, bez_curve_lum)
            t = [init_max, init_max, init_max]
            ACES2065_2_XYZ(t)
            init_max = colour.XYZ_to_xyY(t)[2]
            out = tone_map_xyY(out, inflection_percent, init_max)


        if (rec709_lut):
            out = ACES2065_2_REC709(out)
        elif (DWG_I_lut):
            if (enable_optimize_illum == "O" or enable_optimize_illum == "IO"):
                out = ACES2065_2_DWG_I_CAT(out)
            else:  out = ACES2065_2_DWG_I(out)

        for k in range(lut_size):
            for j in range(lut_size):
                for i in range(lut_size):
                    f.writelines(str(out[i, j, k,0]) + " " + str(out[i, j, k,1]) + " " + str(out[i, j, k,2]) + "\n")
        f.close()


def write_lut_1D(filename, bez_curve_lum, input_whitepoint ):

    lut_size = 1024


    with open(str("1d"+filename+".cube"), 'w', 100 * (2 ** 20)) as f:
        f.writelines("TITLE " + filename + "\n")
        f.writelines("LUT_1D_SIZE "+str(lut_size) +"\n")
        f.writelines("#ZRG, OUTPUT COLOR SPACE = Linear Tone Curve\n")

        out = np.zeros(shape=(lut_size), dtype=float)

        for i in range(lut_size):
            t = interp_1d(((i*input_whitepoint) / (lut_size-1)), bez_curve_lum)
            out[i] = t

        out = np.clip(out, a_min=0, a_max=None)

        for i in range(lut_size):
            f.writelines(str(out[i]) + " " + str(out[i]) + " " + str(out[i]) + "\n")
        f.close()



def write_lut_shaper(filename, bez_curve_lum, three_by_three,norm_lut_max,input_whitepoint,input_blackpoint,
                   s_gain, s_gamma,gammut_comp,expand_tone,inflection_percent,bez_count_hue_hue, best_bez_hue_hue, bez_count_hue_sat, best_bez_hue_sat ,enable_optimize_illum):


    bez_hue_hue = best_bez_hue_hue[range(bez_count_hue_hue), 1]
    bez_hue_hue = bez_hue_hue + MANUAL_HUE_HUE

    bez_hue_sat = best_bez_hue_sat[range(bez_count_hue_sat), 1]
    bez_hue_sat = bez_hue_sat + MANUAL_HUE_SAT

    bez_curve_hue_hue = interp_1d_setup(bez_count_hue_hue,bez_hue_hue, hue=1,
                                        input_blackpoint=input_blackpoint,
                                        input_whitepoint=input_whitepoint, mode=HSL_MODE)
    bez_curve_hue_sat = interp_1d_setup(bez_count_hue_sat, bez_hue_sat, hue=1,
                                        input_blackpoint=input_blackpoint,
                                        input_whitepoint=input_whitepoint, mode=HSL_MODE)

    lut_size_1d = 4096
    lut_size_3d = 65

    max_in_1d = input_whitepoint
    if max_in_1d < 1:
        max_in_1d = 1

    max_WP = interp_1d(input_whitepoint, bez_curve_lum)
    max_1 = interp_1d(1, bez_curve_lum)
    if not (expand_tone): print("Max Output Nits at input WP = ", max_WP * 100, " Max output at 1.0 input = ", max_1 * 100)


    with (open(str("Shaper_"+filename+".cube"), 'w', 100 * (2 ** 20)) as f):

        out = np.linspace(0,max_in_1d,lut_size_1d )
        out = interp_1d(out, bez_curve_lum)
        out = np.clip(out, a_min=0, a_max=None)

        out = colour.models.oetf_DaVinciIntermediate(out)

        max_in_3d = np.max(out)

        f.writelines("TITLE " + filename + "\n")
        f.writelines("LUT_1D_SIZE "+str(lut_size_1d) + "\n")
        f.writelines("LUT_1D_INPUT_RANGE 0 " + str(max_in_1d) + "\n")
        f.writelines("LUT_3D_SIZE " + str(lut_size_3d) + "\n")
        f.writelines("LUT_3D_INPUT_RANGE 0 " + str(max_in_3d) + "\n")
        f.writelines("#ZRG, OUTPUT COLOR SPACE = Davinchi Wide Gammut Intermediate \n")
        if not (expand_tone): f.writelines("#Max Output Nits = " + str(max_WP * 100) + " Max output at 1.0 input = "+ str(max_1 * 100) + "\n")

        for i in range(lut_size_1d):
            f.writelines(str(out[i]) + " " + str(out[i]) + " " + str(out[i]) + "\n")

        out = np.zeros(shape=(lut_size_3d, lut_size_3d, lut_size_3d, 3), dtype=float)
        for k in range(lut_size_3d):
            for j in range(lut_size_3d):
                for i in range(lut_size_3d):
                    out[i,j,k] = (i/(lut_size_3d-1) * max_in_3d), (j/(lut_size_3d-1) * max_in_3d), (k/(lut_size_3d-1) * max_in_3d),


        out = colour.models.oetf_inverse_DaVinciIntermediate(out)
        out = np.matmul(out, three_by_three)  #Multiply by 3x3

        if (gammut_comp):
            out = gammut_compression(out, GAM_COMP_STRENGTH,gammut_comp)


        if not (s_gain == 1 and s_gamma == 1):  # HSL adjustments
            out_HSL = ACES2065_2_HSL(out)
            np.where(out_HSL[:, :, :, 0] < 0, out_HSL[:, :, :, 0],out_HSL[:, :, :, 0] + 1)  # Normalize Hue values that wrapped around
            np.where(out_HSL[:, :, :, 0] > 1, out_HSL[:, :, :, 0], out_HSL[:, :, :, 0] - 1)

            out_HSL = np.clip(out_HSL, a_min=0, a_max=None)
            out_HSL[:, :, :, 1] = out_HSL[:, :, :, 1]** (1 / ( s_gamma + MANUAL_S_GAMMA))  # Apply Saturation Gamma
            out_HSL[:, :, :, 1] = out_HSL[:, :, :, 1] * (s_gain+MANUAL_S_GAIN)  # Apply saturation gain

            out_HSL[:, :,:, 0] = out_HSL[:, :,:, 0] + interp_1d(out_HSL[:, :,:, 0],bez_curve_hue_hue)  # Apply Hue_hue Curve
            np.where(out_HSL[:, :,:, 0] < 0, out_HSL[:, :, :,0], out_HSL[:, :,:, 0] + 1)  # Normalize Hue values that wrapped around
            np.where(out_HSL[:, :,:, 0] > 1, out_HSL[:, :, :,0], out_HSL[:, :,:, 0] - 1)
            out_HSL[:, :, :,1] = out_HSL[:, :,:, 1] * interp_1d(out_HSL[:, :,:, 0],bez_curve_hue_sat)  # Apply Hue_sat Curve
            out = HSL_2_ACES2065(out_HSL)  # Convert back to linSRGB


        if (expand_tone):  # Do tone expansion in xyY space as OKlab was found to introduce artifacts.
            init_max = interp_1d(input_whitepoint, bez_curve_lum)
            t = [init_max, init_max, init_max]
            ACES2065_2_XYZ(t)
            init_max = colour.XYZ_to_xyY(t)[2]
            out = tone_map_xyY(out, inflection_percent, init_max)

        if (enable_optimize_illum == "O" or enable_optimize_illum == "IO"):
            out = ACES2065_2_DWG_I_CAT(out)
        else:
            out = ACES2065_2_DWG_I(out)

        for k in range(lut_size_3d):
            for j in range(lut_size_3d):
                for i in range(lut_size_3d):
                    f.writelines(str(out[i, j, k,0]) + " " + str(out[i, j, k,1]) + " " + str(out[i, j, k,2]) + "\n")
        f.close()


def write_lut_clf(filename, bez_curve_lum, three_by_three ,rec709_lut,bez_curve_hue_hue,bez_curve_hue_sat,norm_lut_max,input_whitepoint, s_gain, s_gamma):

    max_in = input_whitepoint
    if max_in< 1 :
        max_in = 1

    with open(str("CLF"+filename+".clf"), 'w', 100 * (2 ** 20)) as f:

        f.writelines("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n")
        f.writelines("<ProcessList id=\"ZRG_Log2Linear\" compCLFversion=\"3.0\">\n")
        f.writelines("<Description>\"ZRG_Log2Linear\"</Description>\n")
        f.writelines("<InputDescriptor>Camera_Log</InputDescriptor>\n")
        f.writelines("<OutputDescriptor>Rec709, Linear</OutputDescriptor>\n")

        shaper_max = interp_1d(max_in, bez_curve_lum)

        f.writelines("<Range inBitDepth=\"32f\" outBitDepth=\"32f\">\n")
        f.writelines("<Description>Shaper Range</Description>\n")
        f.writelines("<minInValue>0</minInValue>\n")
        f.writelines("<maxInValue>" + str(max_in) + "</maxInValue>\n")
        f.writelines("<minOutValue>0</minOutValue>\n")
        f.writelines("<maxOutValue>1</maxOutValue>\n")
        f.writelines("</Range>\n")


        f.writelines("<LUT1D id=\"lut-1\" name=\"Shaper\" inBitDepth=\"32f\" outBitDepth=\"32f\">\n")
        f.writelines("<Description>Shaper</Description>\n")
        f.writelines("<Array dim=\"1024 1\">\n")

        shaper_out = np.zeros(shape=(1024), dtype=float)
        for i in range(1024):
            t = interp_1d(((i*max_in) / 1023), bez_curve_lum)
            shaper_out[i] = t
        shaper_out = np.clip(shaper_out, a_min=0, a_max=None)
        for i in range(1024):
            f.writelines(str(shaper_out[i]) + "\n")

        f.writelines("</Array>\n")
        f.writelines("</LUT1D>\n")

        f.writelines("<Matrix inBitDepth=\"32f\" outBitDepth=\"32f\">\n")
        f.writelines("<Description>Primaries</Description>\n")
        f.writelines("<Array dim=\"3 3\">\n")
        f.writelines(str(three_by_three[0,0])+" "+str(three_by_three[0,1])+" "+str(three_by_three[0,2])+"\n")
        f.writelines(str(three_by_three[1, 0]) + " "+str(three_by_three[1, 1]) + " "+str(three_by_three[1, 2]) + "\n")
        f.writelines(str(three_by_three[2, 0]) + " "+str(three_by_three[2, 1]) + " "+str(three_by_three[2, 2]) + "\n")
        f.writelines("</Array>\n")
        f.writelines("</Matrix>\n")

        """threeD = np.matmul([shaper_max,shaper_max,shaper_max], three_by_three)
        shaper_max_3d = np.max(threeD)

        out_HSL = np.zeros(shape=(65, 65, 65, 3), dtype=float)
        for k in range(65):
            for j in range(65):
                for i in range(65):
                    out_HSL[i, j, k] = ((i / 64) * shaper_max_3d),((j / 64) * shaper_max_3d),((k / 64) * shaper_max_3d)


        out_HSL = XYZ_2_HSL(out_HSL)

        out_HSL = np.clip(out_HSL, a_min=0, a_max=None)
        out_HSL[:, :, :, 1] = out_HSL[:, :, :, 1] ** (1 / s_gamma)  # Apply Saturation Gamma
        out_HSL[:, :, :, 1] = out_HSL[:, :, :, 1] * s_gain  # Apply saturation gain

        out_HSL[:, :, 0] = out_HSL[:, :, 0] + interp_1d(out_HSL[:, :, 0], bez_curve_hue_hue)  # Apply Hue_hue Curve
        np.where(out_HSL[:, :, 0] < 0, out_HSL[:, :, 0],out_HSL[:, :, 0] + 1)  # Normalize Hue values that wrapped around
        np.where(out_HSL[:, :, 0] > 1, out_HSL[:, :, 0], out_HSL[:, :, 0] - 1)

        out_HSL[:, :, 1] = out_HSL[:, :, 1] * interp_1d(out_HSL[:, :, 0], bez_curve_hue_sat)  # Apply Hue_sat Curve

        out_lin_XYZ = HSL_2_XYZ(out_HSL)  # Convert back to XYZ

        out = np.clip(out_lin_XYZ, a_min=0, a_max=None)

        f.writelines("<Range inBitDepth=\"32f\" outBitDepth=\"32f\">\n")
        f.writelines("<Description>3d Range</Description>\n")
        f.writelines("<minInValue>0</minInValue>\n")
        f.writelines("<maxInValue>" + str(shaper_max_3d) + "</maxInValue>\n")
        f.writelines("<minOutValue>0</minOutValue>\n")
        f.writelines("<maxOutValue>" + str(np.max(out)) + "</MaxOutValue>\n")
        f.writelines("</Range>\n")

        f.writelines("<LUT3D id=\"lut-2\" name=\"HSL Comp\" interpolation=\"tetrahedral\" inBitDepth=\"32f\" outBitDepth=\"32f\">\n")
        f.writelines("<Description>HSL Corrrections</Description>\n")
        f.writelines("<Array dim=\"65 65 65 3\">\n")
        for k in range(65):
            for j in range(65):
                for i in range(65):
                    f.writelines(str(out[i, j, k, 0]) + " " + str(out[i, j, k, 1]) + " " + str(out[i, j, k, 2]) + "\n")
        f.writelines("</Array>\n")
        f.writelines("</LUT3D>\n")"""


        f.writelines("</ProcessList>\n")

        f.close()




def set_bez_MG(bez_count_lum, bez_values_lum, MG_bez):

    bez_values_lum[MG_bez, 1] = MIDDLE_GRAY_LIN
    bez_values_lum[1, 1] = 0

    for j in range(MG_bez, bez_count_lum - 1):  # Force curve to be strictly increasing
        bez_values_lum[j + 1, 1] = np.clip(bez_values_lum[j + 1, 1], a_min=bez_values_lum[j, 1], a_max=None)

    for j in range(MG_bez, 1, -1):
        bez_values_lum[j -1, 1] =np.clip(bez_values_lum[j -1, 1], a_min = None, a_max=bez_values_lum[j, 1])

    bez_values_lum[1, 1] = 0

    return bez_values_lum


def LAB_2_XYZ (input):
    return colour.Lab_to_XYZ(input)



def XYZ_2_LAB(input):
   return colour.XYZ_to_Lab(input)



def XYZ_2_sRGB(input):
    return colour.RGB_to_RGB(input, RGB_COLOURSPACE_LIN_CIEXYZ_SCENE, RGB_COLOURSPACE_sRGB,
                            chromatic_adaptation_transform=None, apply_cctf_decoding=False, apply_cctf_encoding=True)


def ACES2065_2_DWG_I_CAT(input):

    RGB_COLOURSPACE_ACES2065_1_CUSTOM_WP = RGB_COLOURSPACE_ACES2065_1.copy()
    RGB_COLOURSPACE_ACES2065_1_CUSTOM_WP.whitepoint =  ILLUMINANT

    return colour.RGB_to_RGB(input, RGB_COLOURSPACE_ACES2065_1_CUSTOM_WP, RGB_COLOURSPACE_DAVINCI_WIDE_GAMUT,
                             chromatic_adaptation_transform=CAT, apply_cctf_decoding=False,
                             apply_cctf_encoding=True)


def ACES2065_2_DWG_I(input):

    return colour.RGB_to_RGB(input, RGB_COLOURSPACE_ACES2065_1, RGB_COLOURSPACE_DAVINCI_WIDE_GAMUT,
                             chromatic_adaptation_transform=None, apply_cctf_decoding=False,
                             apply_cctf_encoding=True)


def LAB_2_sRGB (input):
    return colour.XYZ_to_sRGB(LAB_2_XYZ(input))

def LAB_2_HSL(input):

    return ACES2065_2_HSL (LAB_2_ACEC2065(input))

def LAB_2_ACEC2065(input):
    return XYZ_2_ACES2065(LAB_2_XYZ(input))

def ACES2065_2_HSL(input):
    #aces = np.clip(input, a_min=0, a_max=None)
    #return colour.RGB_to_HSL(input)
    return XYZ_2_HSL(ACES2065_2_XYZ(input))



def HSL_2_ACES2065(input):
    #return colour.HSL_to_RGB(input)
    return XYZ_2_ACES2065(HSL_2_XYZ(input))

def HSL_2_XYZ(input):

    temp = np.reshape(input, shape=(-1, 3))
    lch = temp.copy()
    lch[:, 0] = temp[:, 2]
    lch[:, 2] = temp[:, 0] * 360
    lch = np.reshape(lch, shape=np.shape(input))
    okLAB = colour.Oklch_to_Oklab(lch)
    xyz = colour.Oklab_to_XYZ(okLAB)
    return xyz

def XYZ_2_HSL(input):

    okLAB = colour.XYZ_to_Oklab(input)
    lch = colour.Oklab_to_Oklch(okLAB)
    lch = np.reshape(lch, shape=(-1, 3))
    temp2 = lch.copy()
    temp2[:, 0] = lch[:, 2] / 360
    temp2[:, 2] = lch[:, 0]
    return np.reshape(temp2, shape=np.shape(input))


def HSL_2_LAB(input):

    #return ACES2065_2_LAB(colour.HSL_to_RGB(input))
    return XYZ_2_LAB(HSL_2_XYZ(input))

def HSL_2_sRGB(input):

   #return ACES2065_2_REC709 (colour.HSL_to_RGB(input))
    return XYZ_2_sRGB(HSL_2_XYZ(input))

def ACES2065_2_REC709 (input):
    return colour.RGB_to_RGB(input, RGB_COLOURSPACE_ACES2065_1, RGB_COLOURSPACE_BT709,
                             chromatic_adaptation_transform=None, apply_cctf_decoding=False,apply_cctf_encoding=True)

def ACES2065_2_XYZ(input):
    return colour.RGB_to_RGB(input, RGB_COLOURSPACE_ACES2065_1, RGB_COLOURSPACE_LIN_CIEXYZ_SCENE,
                             chromatic_adaptation_transform=None, apply_cctf_decoding=False,apply_cctf_encoding=False)

def ACES2065_2_LAB(input):
    return XYZ_2_LAB(ACES2065_2_XYZ(input))

def XYZ_2_ACES2065 (input):
    return colour.RGB_to_RGB(input,RGB_COLOURSPACE_LIN_CIEXYZ_SCENE, RGB_COLOURSPACE_ACES2065_1,
                             chromatic_adaptation_transform=None, apply_cctf_decoding=False,apply_cctf_encoding=False)


def interp_3d_setup(xyz, w):
    return interpolate.RegularGridInterpolator(points=xyz, values=w, method=CURVE_FIT_MODE_3D)



def interp_1d(x, curve, mode=None):
    if (FIT_MODE=="CALC"):
        if mode==None:  #use compile time mode
            if (CURVE_FIT_MODE_1D=="spline"):
                return interpolate.splev(x,curve)
            elif (CURVE_FIT_MODE_1D=="akima"):
                return curve(x)
            elif (CURVE_FIT_MODE_1D == "makima"):
                return curve(x)
            elif (CURVE_FIT_MODE_1D == "pchip"):
                return curve(x)
            elif (CURVE_FIT_MODE_1D == "cubic"):
                return curve(x)
            elif (CURVE_FIT_MODE_1D == "quintic"):
                return curve(x)
            else: print("Failed ot select 1d interp mode")
        elif (mode == "spline"):   #use run time modes
            return interpolate.splev(x, curve)
        elif (mode == "akima"):
            return curve(x)
        elif (mode == "makima"):
            return curve(x)
        elif (mode == "pchip"):
            return curve(x)
        elif (mode == "cubic"):
            return curve(x)
        elif (mode == "quintic"):
            return curve(x)
        else:
            print("Failed to select 1d interp mode")
    else: return curve(x)  #Evaluate log_interp Class

def interp_1d_x_vals(x_max,hue,input_blackpoint,input_whitepoint, input_MG):

    if hue:  #add additional values before zero and after 1 to smoothly wrap around ends
        x = np.linspace(0, 1, x_max)
        spaceing = 1/ (x_max-1)

        new_x = np.zeros(shape=(x_max+2), dtype=float)
        new_x[0]=(-1 * spaceing)
        new_x[x_max+1] = 1+spaceing

        for i in range (1,x_max+1):
            new_x[i]=x[i-1]
        x = new_x.copy()

        return x

    else:
        if(input_blackpoint==None or input_whitepoint==None or input_MG==None): print("Missing White/Black Point/MG")
        spaceing = 1 / (x_max - 2)
        x1 = np.linspace(input_blackpoint, input_whitepoint, x_max-2)  #first bez point before zero
        x2 = input_whitepoint + spaceing  #Last bez point after 1


        x = 0 - spaceing
        x= np.append(x, x1)
        x= np.append(x, x2)


        idx = (np.abs(x - input_MG)).argmin()
        x[idx] = input_MG  #Set bez closest to input_MG to be exactly input MG

        return x



def interp_1d_setup(x_max,bez_values,hue=0,input_blackpoint=None,input_whitepoint=None,input_MG=None, mode = None):
    if not hue and (FIT_MODE!="CALC"):
        return log_interp(FIT_MODE,input_MG)

    x = interp_1d_x_vals(x_max, hue, input_blackpoint, input_whitepoint, input_MG)

    if hue:
        new_bez_values = np.zeros(shape=(x_max + 2), dtype=float)

        new_bez_values[0] = bez_values[x_max - 2]
        new_bez_values[x_max + 1] = bez_values[1]

        for i in range(1, x_max + 1):
            new_bez_values[i] = bez_values[i - 1]

        bez_values = new_bez_values.copy()
    if mode == None:
        if (CURVE_FIT_MODE_1D=="spline"):
            return interpolate.splrep(x, bez_values, s=BEZ_SMOOTH)
        elif(CURVE_FIT_MODE_1D=="akima") :
            return Akima1DInterpolator(x, bez_values, method="akima", extrapolate=1)
        elif(CURVE_FIT_MODE_1D=="makima") :
            return Akima1DInterpolator(x, bez_values, method="makima", extrapolate=1)
        elif (CURVE_FIT_MODE_1D == "pchip"):
            return PchipInterpolator(x, bez_values,extrapolate=1)
        elif (CURVE_FIT_MODE_1D == "cubic"):
            return interpolate.UnivariateSpline(x, bez_values,k=3,ext=0,s=BEZ_SMOOTH)
        elif (CURVE_FIT_MODE_1D == "quintic"):
            return interpolate.UnivariateSpline(x, bez_values,k=5,ext=0,s=BEZ_SMOOTH)
        else:
            print("Failed to select 1d interp mode")

    elif (mode == "spline"):
        return interpolate.splrep(x, bez_values, s=HSL_SMOOTH)
    elif (mode == "akima"):
        return Akima1DInterpolator(x, bez_values, method="akima", extrapolate=1)
    elif (mode == "makima"):
        return Akima1DInterpolator(x, bez_values, method="makima", extrapolate=1)
    elif (mode == "pchip"):
        return PchipInterpolator(x, bez_values, extrapolate=1)
    elif (mode == "cubic"):
        return interpolate.UnivariateSpline(x, bez_values, k=3, ext=0, s=HSL_SMOOTH)
    elif (mode == "quintic"):
        return interpolate.UnivariateSpline(x, bez_values, k=5, ext=0, s=HSL_SMOOTH)
    else:
        print("Failed to select 1d interp mode")


def perturb_curve (bez_values_lum, bez_values_hue_hue, bez_values_hue_sat  ,test_chart_RGB, bad_chips, num_round,three_by_three,
                   bez_count_lum, bez_count_hue_hue, bez_count_hue_sat,chart_name_low, chart_name_high,chart_weight,perturb_step,mode,
                   input_blackpoint,input_whitepoint, s_gain =1, s_gamma=1, MG_bez=0, input_MG= 0 ):

    init_error = -1
    init_bez_values = []
    test_bez_count = 0
    test_type = "Luma2"
    max_iter=0
    error_zero = sys.float_info.max

    if (mode == "Lum"):
        init_bez_values = bez_values_lum.copy()
        test_bez_count = bez_count_lum
        test_type = "Luma2"
        error_zero = error_sum(bez_count_lum, init_bez_values, test_chart_RGB, chart_name_low,
                               chart_name_high, three_by_three, bad_chips, test_type, chart_weight, final=0,
                               input_blackpoint=input_blackpoint, input_whitepoint=input_whitepoint, input_MG=input_MG)

    elif (mode == "Hue-Hue"):
        init_bez_values = bez_values_hue_hue.copy()
        test_bez_count = bez_count_hue_hue
        test_type = "Color"

    elif (mode == "Hue-Sat"):
        init_bez_values = bez_values_hue_sat.copy()
        test_bez_count = bez_count_hue_sat
        test_type = "Color"

    else: print ("Invalid mode sent to Perturb Curve")


    test_bez_values_up = init_bez_values.copy()
    test_bez_values_down = init_bez_values.copy()
    best_bez_values = init_bez_values.copy()


    for bez in range(0, test_bez_count):  #Determin if up or down for each bez

        if (mode == "Hue-Hue" or mode == "Hue-Sat"):  # Force first and last values to be the same so wrapes smoothly around zero
            test_bez_values_up[bez, 1] = best_bez_values[bez, 1] + perturb_step
            test_bez_values_down[bez, 1] = best_bez_values[bez, 1] - perturb_step
            test_bez_values_up[test_bez_count - 1, 1] = test_bez_values_up[0, 1]
            test_bez_values_down[test_bez_count - 1, 1] = test_bez_values_down[0, 1]
        if (mode != "Hue-Hue") and bez > 0: test_bez_values_down[bez, 1] = np.clip(test_bez_values_down[bez, 1], a_min= 0, a_max= None)

        if (mode == "Lum"):  #For first and last bez point to zero and one
            scalar = abs(init_bez_values[bez, 1]) + 0.1  #Scale perturb amound by beginning amount
            test_bez_values_up[bez, 1] = best_bez_values[bez, 1] + (perturb_step * scalar)
            test_bez_values_down[bez, 1] = best_bez_values[bez, 1] - (perturb_step * scalar)

            test_bez_values_up[0, 1] = np.clip(test_bez_values_up[0, 1], a_min=None, a_max=0)
            test_bez_values_up[1, 1] = 0

            test_bez_values_down[0, 1] = np.clip(test_bez_values_down[0, 1], a_min=None, a_max=0)
            test_bez_values_down[1, 1] = 0

            test_bez_values_up = set_bez_MG(bez_count_lum, test_bez_values_up, MG_bez)
            test_bez_values_down = set_bez_MG(bez_count_lum, test_bez_values_down, MG_bez)

        if (mode == "Hue-Hue"):  #Force sum to total zero to prevent general hue rotation from WB error
            test_bez_values_up[:,1] = test_bez_values_up[:,1] - np.sum(test_bez_values_up[1:test_bez_count,1])/(test_bez_count-1)
            test_bez_values_down[:, 1] = test_bez_values_down[:, 1] - np.sum(test_bez_values_down[1:test_bez_count, 1]) /(test_bez_count-1)


        init_error_up = -1
        init_error_down = -1



        if (mode == "Lum"):

            init_error_up = error_sum(bez_count_lum, test_bez_values_up, test_chart_RGB, chart_name_low,
                                       chart_name_high, three_by_three, bad_chips, test_type, chart_weight, final=0,
                                      input_blackpoint=input_blackpoint,input_whitepoint=input_whitepoint,input_MG = input_MG )

            init_error_down = error_sum(bez_count_lum, test_bez_values_down, test_chart_RGB, chart_name_low,
                                          chart_name_high, three_by_three, bad_chips, test_type, chart_weight, final=0,
                                        input_blackpoint=input_blackpoint,input_whitepoint=input_whitepoint,input_MG = input_MG)

            error_zero =min((error_zero, error_sum(bez_count_lum, best_bez_values, test_chart_RGB, chart_name_low,
                                          chart_name_high, three_by_three, bad_chips, test_type, chart_weight, final=0,
                                        input_blackpoint=input_blackpoint,input_whitepoint=input_whitepoint,input_MG = input_MG)))


        elif (mode == "Hue-Hue"):
            init_error_up = error_sum(bez_count_lum, bez_values_lum, test_chart_RGB, chart_name_low,
                                      chart_name_high, three_by_three, bad_chips, test_type, chart_weight, final=0,
                                      bez_count_hue_hue=bez_count_hue_hue, bez_count_hue_sat=bez_count_hue_sat,
                                      bez_values_hue_hue=test_bez_values_up, bez_values_hue_sat=bez_values_hue_sat,input_blackpoint=input_blackpoint,
                                      input_whitepoint=input_whitepoint,s_gain = s_gain,s_gamma = s_gamma,input_MG = input_MG)

            init_error_down = error_sum(bez_count_lum, bez_values_lum, test_chart_RGB,
                                        chart_name_low,
                                        chart_name_high, three_by_three, bad_chips, test_type, chart_weight, final=0,
                                        bez_count_hue_hue=bez_count_hue_hue, bez_count_hue_sat=bez_count_hue_sat,
                                        bez_values_hue_hue=test_bez_values_down, bez_values_hue_sat=bez_values_hue_sat,input_blackpoint=input_blackpoint,
                                        input_whitepoint=input_whitepoint,s_gain = s_gain,s_gamma = s_gamma, input_MG = input_MG)

            error_zero = error_sum(bez_count_lum, bez_values_lum, test_chart_RGB,
                                        chart_name_low,
                                        chart_name_high, three_by_three, bad_chips, test_type, chart_weight, final=0,
                                        bez_count_hue_hue=bez_count_hue_hue, bez_count_hue_sat=bez_count_hue_sat,
                                        bez_values_hue_hue=best_bez_values, bez_values_hue_sat=bez_values_hue_sat,
                                        input_blackpoint=input_blackpoint,
                                        input_whitepoint=input_whitepoint, s_gain=s_gain, s_gamma=s_gamma,
                                        input_MG=input_MG)


        elif (mode == "Hue-Sat"):
            init_error_up = error_sum(bez_count_lum, bez_values_lum, test_chart_RGB, chart_name_low,
                                      chart_name_high, three_by_three, bad_chips, test_type, chart_weight, final=0,
                                      bez_count_hue_hue=bez_count_hue_hue, bez_count_hue_sat=bez_count_hue_sat,
                                      bez_values_hue_hue=bez_values_hue_hue, bez_values_hue_sat=test_bez_values_up,input_blackpoint=input_blackpoint,
                                      input_whitepoint=input_whitepoint,s_gain = s_gain,s_gamma = s_gamma,input_MG = input_MG)

            init_error_down = error_sum(bez_count_lum, bez_values_lum, test_chart_RGB,chart_name_low,
                                        chart_name_high, three_by_three, bad_chips, test_type, chart_weight, final=0,
                                        bez_count_hue_hue=bez_count_hue_hue, bez_count_hue_sat=bez_count_hue_sat,
                                        bez_values_hue_hue=bez_values_hue_hue, bez_values_hue_sat=test_bez_values_down,input_blackpoint=input_blackpoint,
                                        input_whitepoint=input_whitepoint,s_gain = s_gain,s_gamma = s_gamma,input_MG = input_MG)

            error_zero = error_sum(bez_count_lum, bez_values_lum, test_chart_RGB,chart_name_low,
                                        chart_name_high, three_by_three, bad_chips, test_type, chart_weight, final=0,
                                        bez_count_hue_hue=bez_count_hue_hue, bez_count_hue_sat=bez_count_hue_sat,
                                        bez_values_hue_hue=bez_values_hue_hue, bez_values_hue_sat=best_bez_values,input_blackpoint=input_blackpoint,
                                        input_whitepoint=input_whitepoint,s_gain = s_gain,s_gamma = s_gamma,input_MG = input_MG)


        if ((init_error_up<init_error_down) and (init_error_up < error_zero)):
            best_bez_values = test_bez_values_up.copy()
        elif ((init_error_down<init_error_up) and (init_error_down <error_zero)):
            best_bez_values = test_bez_values_down.copy()
        else:
            pass
    return best_bez_values


def calc_error_RGB(test, ref, chip_weight, bad_chips, luma_test,chart_name_low,chart_name_high,chart_weight, final = 0):


    if (luma_test == "Luma3"):
        luma_1 = calc_error_RGB(test, ref, chip_weight, bad_chips, "Luma",chart_name_low,chart_name_high,chart_weight, final)
        luma_2= calc_error_RGB(test, ref, chip_weight, bad_chips, "Luma2", chart_name_low, chart_name_high, chart_weight,
                           final)
        return (luma_1 + luma_2)/2

    error = 0.0
    num_chip_x = NUM_CHIP_X
    num_chip_y = NUM_CHIP_Y

    num_charts = chart_name_high - chart_name_low + 1

    test_1 = test.copy()
    ref_1 = ref.copy()

    error_list = np.zeros(shape=(num_charts, num_chip_x, num_chip_y), dtype=float)
    error_list_weighted = np.zeros(shape=(num_charts, num_chip_x, num_chip_y), dtype=float)

    mean_weight = 0

    for i in range(chart_name_low, chart_name_high+1):
        if (chart_weight==0): weight = 1
        else: weight = (1 / (abs(i / chart_weight) + 1)) #Make charts further from zero exposure have less impact

        mean_weight =  mean_weight +weight
        if (luma_test == "Luma"):  # Error is relative to LAB reference chart values
            del_L = bad_chips[i] * chip_weight * (test[i, :, :, 0] - ref[:, :, 0])
            t = np.abs(del_L)  * weight
            error_list[i] = del_L
            error_list_weighted[i] = t

        elif (luma_test == "Luma2"):  # Compare against Test chart 0, not reference.
            if (i != 0):
                del_L = bad_chips[i] * chip_weight * (test[i, :, :, 0] - test[0, :, :, 0])
                t = np.abs(del_L)  * weight
                error_list[i] = del_L
                error_list_weighted[i] = t

        elif (luma_test == "Color" or luma_test == "Color_all"):  # Color Test exclude luma error
            mean_lum = (test[i, :, :, 0] + ref[:, :, 0]) / 2  #set both chips to mean luma
            test_1[i, :, :, 0] = mean_lum
            ref_1[:, :, 0] = mean_lum
            deltae = colour.difference.delta_E_CIE2000(test_1[i], ref_1) #** 2
            t = deltae * bad_chips[i] * chip_weight * weight
            error_list[i] = deltae
            error_list_weighted[i] = t


        elif (luma_test == "Combined"):  #Combined Luma and Color error
            deltae = (colour.difference.delta_E_CIE2000(test[i], ref)) #** 2
            t = deltae * bad_chips[i] * chip_weight * weight
            error_list[i] = deltae
            error_list_weighted[i] = t

        else: print("Failed to send mode to calc_error_RGB")

    mean_weight = mean_weight / num_charts
    error_list_weighted = error_list_weighted / mean_weight

    mask = error_list_weighted>0
    error_list_weighted = error_list_weighted[mask]
    upper_limit = np.percentile(error_list_weighted, ERROR_PERCENTILE_LIMIT)
    mask = error_list_weighted <upper_limit
    error_list_weighted = error_list_weighted[mask]
    error = np.mean(error_list_weighted)

    if (final):

        print("Error Table")

        table = BeautifulTable()

        for j in range(num_chip_x):
            row = []
            for k in range(num_chip_y):
                row = np.append(row,error_list[0,j,k])
            table.rows.append([row])

        print(table)


    return error

#End calc_error


def calc_charts(chip_RGB, chart_name_low,chart_name_high, bez_curve_lum=None, three_by_three=None,bez_curve_hue_hue=None, bez_curve_hue_sat = None,
                s_gain = 1, s_gamma =1, ):

    num_chip_x=NUM_CHIP_X
    num_chip_y = NUM_CHIP_Y
    num_charts =chart_name_high-chart_name_low + 1

    chip_RGB_LIN = np.zeros(shape=(num_charts, num_chip_x, num_chip_y, 3), dtype=float)
    chip_RGB_LIN_norm = np.zeros(shape=(num_charts, num_chip_x, num_chip_y, 3), dtype=float)
    chip_LAB_norm = np.zeros(shape=(num_charts, num_chip_x, num_chip_y, 3), dtype=float)

    chip_RGB_LIN[:, :, :, 0] = interp_1d(chip_RGB[:, :, :, 0], bez_curve_lum)  # use luma bez curve first
    chip_RGB_LIN[:, :, :, 1] = interp_1d(chip_RGB[:, :, :, 1], bez_curve_lum)
    chip_RGB_LIN[:, :, :, 2] = interp_1d(chip_RGB[:, :, :, 2], bez_curve_lum)

    if FLAT_RGB is not None:
        flat_lin = interp_1d(FLAT_RGB, bez_curve_lum)  # Linearize flats

        flat_lin = np.reshape(flat_lin, shape=(-1, 3))
        flat_lin_mean = [0, 0, 0]  # calc geo mean for each R, G, and B
        r = np.log(flat_lin[:, 0])
        g = np.log(flat_lin[:, 1])
        b = np.log(flat_lin[:, 2])
        flat_lin_mean[0] = np.exp(r.mean())
        flat_lin_mean[1] = np.exp(g.mean())
        flat_lin_mean[2] = np.exp(b.mean())

        flat_lin_delta = flat_lin / flat_lin_mean  #How far each flat chip is from mean
        flat_lin_delta = np.reshape(flat_lin_delta, shape=(num_chip_x, num_chip_y, 3))


    for i in range(chart_name_low, chart_name_high + 1):  # Normalize exposure
        if FLAT_RGB is not None:  # Apply Flat Compensation
            chip_RGB_LIN[i, :, :, 0] = chip_RGB_LIN[i, :, :, 0] / flat_lin_delta[:, :, 0]
            chip_RGB_LIN[i, :, :, 1] = chip_RGB_LIN[i, :, :, 1] / flat_lin_delta[:, :, 1]
            chip_RGB_LIN[i, :, :, 2] = chip_RGB_LIN[i, :, :, 2] / flat_lin_delta[:, :, 2]

        chip_RGB_LIN_norm[i, :, :] = chip_RGB_LIN[i, :, :] * (2 ** (-1 * i))

    chip_ACES_2065_lin_norm = np.matmul(chip_RGB_LIN_norm, three_by_three)  # Then multiply by three by three

    if not INPUT_ILLUMINANT == [-1,-1]:  #Apply input illuminat compensation

        RGB_COLOURSPACE_ACES2065_1_CUSTOM_WP = RGB_COLOURSPACE_ACES2065_1.copy()  # Apply chromatic adaption
        RGB_COLOURSPACE_ACES2065_1_CUSTOM_WP.whitepoint = INPUT_ILLUMINANT
        chip_ACES_2065_lin_norm = colour.RGB_to_RGB(chip_ACES_2065_lin_norm,RGB_COLOURSPACE_ACES2065_1, RGB_COLOURSPACE_ACES2065_1_CUSTOM_WP,
                 chromatic_adaptation_transform=CAT, apply_cctf_decoding=False, apply_cctf_encoding=False)

    if (bez_curve_hue_hue is None)  and (bez_curve_hue_sat is None):  # Use 3x3 not hue_hue,Hue_sat
        chip_LAB_norm = ACES2065_2_LAB(chip_ACES_2065_lin_norm)

    elif (bez_curve_hue_hue is not None) and (bez_curve_hue_sat is not None):  # Use hue_hue,Hue_sat

        if (s_gamma ==1 and s_gain ==1 ):  #No need to run HSL conversion as we are still at defaults
            chip_LAB_norm = ACES2065_2_LAB(chip_ACES_2065_lin_norm)
        else:
            chip_HSL_norm = ACES2065_2_HSL(chip_ACES_2065_lin_norm)
            chip_HSL_norm[:, :, :, 1] = chip_HSL_norm[:, :, :, 1] **(1/ s_gamma) #Apply Saturation Gamma
            chip_HSL_norm[:, :, :, 1] = chip_HSL_norm[:, :, :, 1] * s_gain   #Apply Saturation gain

            chip_HSL_norm[:, :, :, 0] = chip_HSL_norm[:, :, :, 0] + interp_1d(chip_HSL_norm[:, :, :, 0], bez_curve_hue_hue)   #Apply Hue_hue Curve
            np.where(chip_HSL_norm[:, :, :, 0]<0, chip_HSL_norm[:, :, :, 0], chip_HSL_norm[:, :, :, 0]+1)  #Normalize Hue values that wrapped around
            np.where(chip_HSL_norm[:, :, :, 0] >1, chip_HSL_norm[:, :, :, 0], chip_HSL_norm[:, :, :, 0] - 1)
            chip_HSL_norm[:, :, :, 1] = chip_HSL_norm[:, :, :, 1] * interp_1d(chip_HSL_norm[:, :, :, 0], bez_curve_hue_sat)  # Apply Hue_sat Curve
            chip_LAB_norm = HSL_2_LAB(chip_HSL_norm)  # Convert back to LAB

    else: print ("You messed up, chart calc selection")
    


    return  chip_LAB_norm

#End Calc Charts

def main_color_thread(chip_RGB, bez_curve_lum, start_three_by_three, num_charts, bad_chips, num_round, chart_name_low,
                                    chart_name_high, chart_weight,three_by_three_opt_iter,WB_Control ):

    time_start = time.time()
    new_chip_LAB_norm = calc_charts(chip_RGB, chart_name_low, chart_name_high, bez_curve_lum=bez_curve_lum, three_by_three=start_three_by_three)
    init_color_error = calc_error_RGB(new_chip_LAB_norm, REFCHIP_LAB, ALL_CHIPS, bad_chips, "Color_all", chart_name_low,
                                    chart_name_high, chart_weight, final=0)

    child_three_by_three_queue =mp.Queue()
    min_color_error = init_color_error
    child_color_error_list = np.zeros(shape=(NUM_THREADS), dtype=float)
    child_best_three_by_three = np.zeros(shape=(NUM_THREADS, 3,3), dtype=float)
    parent_three_by_three = np.zeros(shape=(NUM_THREADS, 3,3), dtype=float)

    for i in range(NUM_THREADS):  # setup original children
        parent_three_by_three[i] = start_three_by_three.copy()

    for gen in range(0, three_by_three_opt_iter):
        processes = []

        chip_LAB_norm = calc_charts(chip_RGB, chart_name_low, chart_name_high, bez_curve_lum=bez_curve_lum,
                                        three_by_three=parent_three_by_three[0] )

        color_error = calc_error_RGB(chip_LAB_norm, REFCHIP_LAB, ALL_CHIPS, bad_chips, "Color_all",
                                          chart_name_low,chart_name_high, chart_weight, final=0)

        print ("\n 3x3 Color Generation # " + str(gen+1) + " of " + str(three_by_three_opt_iter) +"   Color-All Error " + str(color_error))


        for i in range(NUM_THREADS):
            p = mp.Process(target=peturb_color_three_by_three_thread, args=(i, (num_round-1)*(gen+1), num_charts, chip_RGB, REFCHIP_LAB, bad_chips, parent_three_by_three[i],
                                                                     bez_curve_lum, child_three_by_three_queue,chart_name_low, chart_name_high,chart_weight,WB_Control ))
            processes.append(p)
            processes[i].start()

        for i in range(NUM_THREADS):  # Unpack queue of best results
            t = child_three_by_three_queue.get()
            child_color_error_list[i] = t[0, 0]
            child_best_three_by_three[i] = t[1:4,:]

        for i in range(NUM_THREADS):
            processes[i].join()

        if np.min(child_color_error_list) < min_color_error:
            min_color_error = np.min(child_color_error_list)

        min_color_error_pos = np.argpartition(child_color_error_list, int(NUM_THREADS / 2))

        for i in range(NUM_THREADS):
            if (i < int(NUM_THREADS / 2)):
                parent_three_by_three[i] =  child_best_three_by_three[min_color_error_pos[i]]  # choose the best half to carry on
            else:
                t = child_best_three_by_three[i - int(NUM_THREADS / 2), :, :] + child_best_three_by_three[(2 * NUM_THREADS) - i - int(NUM_THREADS / 2) - 1, :, :]
                parent_three_by_three[i] = t / 2

        best_three_by_three = child_best_three_by_three[child_color_error_list.argmin()]
        print(best_three_by_three)

    print("\n Round "+ str(num_round)+ " Final Color-All error = ", end="")

    new_chip_LAB_norm = calc_charts(chip_RGB, chart_name_low, chart_name_high, bez_curve_lum=bez_curve_lum,three_by_three=best_three_by_three)

    final_color_error = calc_error_RGB(new_chip_LAB_norm, REFCHIP_LAB, ALL_CHIPS, bad_chips, "Color_all", chart_name_low,
                                      chart_name_high, chart_weight, final=0)


    print(final_color_error)
    print("Duration (s)= ", time.time()-time_start)

    return best_three_by_three

#End Main Color


def main_color(chip_RGB, bez_curve_lum, start_three_by_three, bad_chips, chart_name_low,
                                    chart_name_high, chart_weight,three_by_three_opt_iter):

    child_color_error_list = np.zeros(shape=(NUM_PARENT_C), dtype=float)
    child_best_three_by_three = np.zeros(shape=(NUM_PARENT_C, 3,3), dtype=float)
    parent_three_by_three = np.zeros(shape=(NUM_PARENT_C, 3,3), dtype=float)

    new_chip_LAB_norm = calc_charts(chip_RGB, chart_name_low, chart_name_high, bez_curve_lum=bez_curve_lum,three_by_three=start_three_by_three)

    init_color_error  = calc_error_RGB(new_chip_LAB_norm, REFCHIP_LAB, ALL_CHIPS, bad_chips, "Color_all", chart_name_low,
                                      chart_name_high, chart_weight, final=0)
    print("\nGenetic Initial Color-All error = ", init_color_error)


    for i in range(NUM_PARENT_C):  # setup original children
        parent_three_by_three[i] = start_three_by_three.copy()

    for gen in range(0, three_by_three_opt_iter):
        for i in range(NUM_PARENT_C):
            child_best_three_by_three[i], child_color_error_list[i]= peturb_color_three_by_three(i, gen, chip_RGB, REFCHIP_LAB, bad_chips,
                                   parent_three_by_three[i],bez_curve_lum,chart_name_low, chart_name_high,chart_weight)

        min_color_error_pos = np.argpartition(child_color_error_list, int(NUM_PARENT_C / 2))

        for i in range(NUM_PARENT_C):
            if (i < int(NUM_PARENT_C / 2)):
                parent_three_by_three[i] =  child_best_three_by_three[min_color_error_pos[i]]  # choose the best half to carry on
            else: #average others together
                t = child_best_three_by_three[i - int(NUM_PARENT_C / 2), :, :] + child_best_three_by_three[(2 * NUM_PARENT_C) - i - int(NUM_PARENT_C / 2) - 1, :, :]
                parent_three_by_three[i] = t / 2

        best_three_by_three = child_best_three_by_three[child_color_error_list.argmin()]

        chip_LAB_norm = calc_charts(chip_RGB, chart_name_low, chart_name_high, bez_curve_lum=bez_curve_lum,
                                        three_by_three=best_three_by_three)

    new_chip_LAB_norm = calc_charts(chip_RGB, chart_name_low, chart_name_high, bez_curve_lum=bez_curve_lum,three_by_three=best_three_by_three)

    final_color_error = calc_error_RGB(new_chip_LAB_norm, REFCHIP_LAB, ALL_CHIPS, bad_chips, "Color_all", chart_name_low,
                                      chart_name_high, chart_weight, final=0)
    print("Final Color-All error = ", final_color_error)
    print(best_three_by_three)
    return best_three_by_three

#End Main Color

def three_by_three_binary(perturb_max, test_chart_RGB, chart_name_low,chart_name_high, init_three_by_three, bad_chips, chart_weight,bez_count_lum, best_bez_lum,
                          bez_count_hue_hue,bez_count_hue_sat,best_bez_hue_hue,best_bez_hue_sat,input_blackpoint,input_whitepoint,s_gain,s_gamma,input_MG, WB_Control):

    three_by_three_zero = init_three_by_three.copy()

    error_init = error_sum(bez_count_lum, best_bez_lum, test_chart_RGB,chart_name_low, chart_name_high, three_by_three_zero, bad_chips, 'Color_all',
                            chart_weight, final=0,bez_count_hue_hue=bez_count_hue_hue, bez_count_hue_sat=bez_count_hue_sat,
                            bez_values_hue_hue=best_bez_hue_hue, bez_values_hue_sat=best_bez_hue_sat, input_blackpoint=input_blackpoint,
                           input_whitepoint=input_whitepoint, s_gain=s_gain,s_gamma=s_gamma, input_MG=input_MG)

    error_zero = error_init

    print("Binary Search Initial Color-All Error = ",error_init)

    for k in range (MAX_ITER_3X3):
        for i in range(MAX_ITER_3X3):
            perturb_step = perturb_max / (2 ** i)/(1.5**k)
            for j in range (0,3):  #run 3 times for each column of 3x3
                for l in range (0,3):  # run 3 times for each row of 3x3
                    three_by_three_up = three_by_three_zero.copy()
                    three_by_three_down = three_by_three_zero.copy()
                    t = [0,0,0]
                    match l:
                        case 0:
                            t[j] = perturb_step
                            t[(j + 1) % 3] = (perturb_step / - 2)
                            t[(j - 1) % 3] =  (perturb_step / - 2)

                        case 1:
                            t[j] = (perturb_step / - 2)
                            t[(j + 1) % 3] = perturb_step
                            t[(j - 1) % 3] = (perturb_step / - 2)

                        case 2:
                            t[j] = (perturb_step / - 2)
                            t[(j + 1) % 3] = (perturb_step / - 2)
                            t[(j - 1) % 3] = perturb_step

                    three_by_three_up[:, j] = three_by_three_up[:, j] + t
                    #if np.sum(three_by_three_up) != 0: three_by_three_up = three_by_three_up / (np.sum(three_by_three_up) /3)  #force sum to 1

                    three_by_three_down[:, j] = three_by_three_down[:, j] - t
                    #if np.sum(three_by_three_down) != 0: three_by_three_down = three_by_three_down / (np.sum(three_by_three_down) / 3)  # force sum to 1

                    error_up = error_sum(bez_count_lum, best_bez_lum, test_chart_RGB, chart_name_low, chart_name_high,
                                           three_by_three_up, bad_chips, 'Color_all',
                                           chart_weight, final=0, bez_count_hue_hue=bez_count_hue_hue,
                                           bez_count_hue_sat=bez_count_hue_sat,
                                           bez_values_hue_hue=best_bez_hue_hue, bez_values_hue_sat=best_bez_hue_sat,
                                           input_blackpoint=input_blackpoint,
                                           input_whitepoint=input_whitepoint, s_gain=s_gain, s_gamma=s_gamma, input_MG=input_MG)

                    error_down  = error_sum(bez_count_lum, best_bez_lum, test_chart_RGB, chart_name_low, chart_name_high,
                                           three_by_three_down, bad_chips, 'Color_all',
                                           chart_weight, final=0, bez_count_hue_hue=bez_count_hue_hue,
                                           bez_count_hue_sat=bez_count_hue_sat,
                                           bez_values_hue_hue=best_bez_hue_hue, bez_values_hue_sat=best_bez_hue_sat,
                                           input_blackpoint=input_blackpoint,
                                           input_whitepoint=input_whitepoint, s_gain=s_gain, s_gamma=s_gamma, input_MG=input_MG)


                    if (error_up > error_zero) and (error_down > error_zero):
                        pass
                    elif error_up < error_down:
                        three_by_three_zero = three_by_three_up.copy()
                        error_zero = error_up
                    else:
                        three_by_three_zero = three_by_three_down.copy()
                        error_zero = error_down


    error_final = error_sum(bez_count_lum, best_bez_lum, test_chart_RGB, chart_name_low, chart_name_high,
                           three_by_three_zero, bad_chips, 'Color_all',
                           chart_weight, final=0, bez_count_hue_hue=bez_count_hue_hue,
                           bez_count_hue_sat=bez_count_hue_sat,
                           bez_values_hue_hue=best_bez_hue_hue, bez_values_hue_sat=best_bez_hue_sat,
                           input_blackpoint=input_blackpoint,
                           input_whitepoint=input_whitepoint, s_gain=s_gain, s_gamma=s_gamma, input_MG=input_MG)
    print ("Binary Search Final Color-All Error = ", error_final)
    print(three_by_three_zero)

    return three_by_three_zero
    

def peturb_color_three_by_three_thread(thread_num, generation, num_charts, chip_RGB, refchip_lab, bad_chips, parent_three_by_three,
                                bez_curve, best_three_by_three_queue,chart_name_low, chart_name_high,chart_weight,WB_Control):

    peturb_max = PETURB_MAX_C / (1.5**generation)
    if(thread_num == 0): peturb_max = peturb_max * 0.1   # Child zero is more conservative
    elif (thread_num == NUM_THREADS - 2):
        peturb_max = peturb_max * 2  # 2nd to last child is more extreme
    elif (thread_num == NUM_THREADS - 1):
        peturb_max = peturb_max * 5  # Last child is most extreme
    num_children = NUM_CHILDREN_C
    color_chips = ALL_CHIPS

    best_three_by_three = parent_three_by_three.copy()

    new_chip_LAB_norm = calc_charts(chip_RGB, chart_name_low, chart_name_high, bez_curve_lum=bez_curve,
                                    three_by_three=parent_three_by_three)
    min_error = calc_error_RGB(new_chip_LAB_norm, refchip_lab, color_chips, bad_chips, "Color_all",
                           chart_name_low, chart_name_high, chart_weight, final=0)


    seed = 593 + (thread_num * 1847) + (generation *  	2111)
    rng = np.random.default_rng(seed=seed)

    for c in range(num_children):

        new_three_by_three = parent_three_by_three.copy()

        rand=np.zeros(shape=9)

        if (c> 0):
            rand = rng.normal(loc=0.0, scale=peturb_max, size=9)
            rand = np.reshape(rand, newshape=(3,3))
            #rand = rand * [[1,0.5,0.5],[0.5,1,0.5],[0.5,0.5,1]]   #off axis values get smaller effect
            new_three_by_three = best_three_by_three + rand
            sum = np.sum(new_three_by_three)
            if (sum != 0): new_three_by_three = new_three_by_three / (sum/3)


        new_chip_LAB_norm = calc_charts(chip_RGB,  chart_name_low, chart_name_high,bez_curve_lum=bez_curve, three_by_three=new_three_by_three)
        error = calc_error_RGB(new_chip_LAB_norm, refchip_lab, color_chips, bad_chips, "Color_all",
                                          chart_name_low,chart_name_high, chart_weight, final=0)

        if (error<min_error):
            min_error= error
            best_three_by_three = new_three_by_three.copy()

            if (c>0):
                test_three_by_three = new_three_by_three.copy()
                for n in range (10):  # Apply the same perterb values again and see if it is better
                    test_three_by_three = new_three_by_three + rand

                    sum = np.sum(test_three_by_three)
                    test_three_by_three = test_three_by_three / (sum/3) #Force sum to 1


                    new_chip_LAB_norm = calc_charts(chip_RGB, chart_name_low, chart_name_high, bez_curve_lum=bez_curve,
                                                    three_by_three=test_three_by_three)

                    test_error = calc_error_RGB(new_chip_LAB_norm, refchip_lab, color_chips, bad_chips, "Color_all",
                                           chart_name_low,
                                           chart_name_high, chart_weight, final=0)

                    if (test_error <  error):  #We found a better solution
                        new_three_by_three = test_three_by_three.copy()
                        error = test_error
                    else: break

                if (error < min_error):  #save these good new values
                    #print("Better results found, new Error ", error, " vs ", min_error)
                    min_error = error
                    best_three_by_three = new_three_by_three.copy()

    t = np.zeros(shape=(4, 3), dtype=float)   #Pack up array to return up
    t[0,0] = min_error
    t[1:4,:] = best_three_by_three

    best_three_by_three_queue.put(t)

#End Peturb Color


def peturb_color_three_by_three(parent_num, generation, chip_RGB, refchip_lab, bad_chips, parent_three_by_three,
                                bez_curve,chart_name_low, chart_name_high,chart_weight):

    peturb_max = PETURB_MAX_C / (2**generation)
    peturb_max = peturb_max * ((parent_num+1) / NUM_PARENT_C)

    num_children = NUM_CHILDREN_C
    color_chips = ALL_CHIPS

    best_three_by_three = parent_three_by_three.copy()

    new_chip_LAB_norm = calc_charts(chip_RGB, chart_name_low, chart_name_high, bez_curve_lum=bez_curve,
                                    three_by_three=parent_three_by_three)
    min_error = calc_error_RGB(new_chip_LAB_norm, refchip_lab, color_chips, bad_chips, "Color_all",
                           chart_name_low, chart_name_high, chart_weight, final=0)

    seed = 593 + (parent_num * 1847) + (generation * 2111)
    rng = np.random.default_rng(seed=seed)

    for c in range(num_children):

        rand = rng.normal(loc=0.0, scale=peturb_max, size=9)
        rand = np.reshape(rand, shape=(3,3))
        new_three_by_three = best_three_by_three + rand
        sum = np.sum(new_three_by_three)
        if (sum != 0): new_three_by_three = new_three_by_three / (sum/3)  #Force Sum to 1


        new_chip_LAB_norm = calc_charts(chip_RGB,  chart_name_low, chart_name_high,bez_curve_lum=bez_curve, three_by_three=new_three_by_three)
        error = calc_error_RGB(new_chip_LAB_norm, refchip_lab, color_chips, bad_chips, "Color_all",
                                          chart_name_low,chart_name_high, chart_weight, final=0)

        if (error<min_error):
            min_error= error
            best_three_by_three = new_three_by_three.copy()

    return best_three_by_three, min_error

#End Peturb Color

def disp_chart(bez_curve,three_by_three,bez_curve_hue_hue,bez_curve_hue_sat,s_gain,s_gamma):
    print("Found file ", end=' ')
    for exp_shift in range(-10, 11):
        filename = Path("data/test_" + str(exp_shift) + ".png")
        if filename.exists():
            print(filename,end=' ')

            image = mpimg.imread(filename)
            image = image[:, :, 0:3]
            image = np.clip(image, a_min=0, a_max=1)

            image_lin = image.copy()

            image_lin[:, :, 0] = interp_1d(image[:, :, 0], bez_curve)  # use luma bez curve first
            image_lin[:, :, 1] = interp_1d(image[:, :, 1], bez_curve)
            image_lin[:, :, 2] = interp_1d(image[:, :, 2], bez_curve)

            image_lin_norm = image_lin * (2 ** (-1 * exp_shift))
            image_ACES2065_lin_norm = np.matmul(image_lin_norm, three_by_three)  # Then multiply by three by three

            if not INPUT_ILLUMINANT == [-1, -1]:  # Apply input illuminat compensation

                RGB_COLOURSPACE_ACES2065_1_CUSTOM_WP = RGB_COLOURSPACE_ACES2065_1.copy()  # Apply chromatic adaption
                RGB_COLOURSPACE_ACES2065_1_CUSTOM_WP.whitepoint = INPUT_ILLUMINANT
                image_ACES2065_lin_norm = colour.RGB_to_RGB(image_ACES2065_lin_norm, RGB_COLOURSPACE_ACES2065_1,
                                                       RGB_COLOURSPACE_ACES2065_1_CUSTOM_WP,
                                                       chromatic_adaptation_transform=CAT, apply_cctf_decoding=False,
                                                       apply_cctf_encoding=False)

            image_ACES2065_lin_norm = np.clip(image_ACES2065_lin_norm, a_min=0, a_max=1)
            image_HSL_Norm = ACES2065_2_HSL(image_ACES2065_lin_norm)

            image_HSL_Norm = np.clip(image_HSL_Norm, a_min=0, a_max=None)
            image_HSL_Norm[:, :, 1] = image_HSL_Norm[:, :, 1] ** (1 / s_gamma)  # Apply Saturation Gamma
            image_HSL_Norm[:, :, 1] = image_HSL_Norm[ :, :, 1] * s_gain  #Apply Saturation Gain

            image_HSL_Norm[ :, :, 0] = image_HSL_Norm[ :, :, 0] + interp_1d(image_HSL_Norm[ :, :, 0], bez_curve_hue_hue)  # Apply Hue_hue Curve
            np.where(image_HSL_Norm[ :, :, 0] < 0, image_HSL_Norm[ :, :, 0],image_HSL_Norm[ :, :, 0] + 1)  # Normalize Hue values that wrapped around
            np.where(image_HSL_Norm[ :, :, 0] > 1, image_HSL_Norm[ :, :, 0], image_HSL_Norm[ :, :, 0] - 1)

            image_HSL_Norm[ :, :, 1] = image_HSL_Norm[ :, :, 1] * interp_1d(image_HSL_Norm[ :, :, 0], bez_curve_hue_sat)  # Apply Hue_sat Curve

            image_SRGB_Norm = HSL_2_sRGB(image_HSL_Norm)  # Convert back to sRGB

            image_SRGB_Norm = np.clip(image_SRGB_Norm, a_min=0, a_max=1)
            export_filename = (str(filename).removesuffix('.png') + "_corrected.png")
            export_img = (image_SRGB_Norm * 255).astype(np.uint8)
            iio.imwrite(export_filename, export_img)

            plt.imshow(image)
            plt.xlabel(None)
            plt.ylabel(None)
            plt.axis('off')
            plt.title(("Chart ",exp_shift," before"))
            plt.show()

            plt.imshow(image_SRGB_Norm)
            plt.xlabel(None)
            plt.ylabel(None)
            plt.axis('off')
            plt.title(("Chart ",exp_shift," After"))
            plt.show()
    print("\n")

def disp_chart_single(bez_curve,three_by_three,test_chart_RGB,bez_curve_hue_hue,bez_curve_hue_sat, s_gain,s_gamma):

        image_lin = test_chart_RGB[0,:,:,:].copy()

        image_lin[:, :, 0] = interp_1d(test_chart_RGB[0,:, :, 0], bez_curve)  # use luma bez curve first
        image_lin[:, :, 1] = interp_1d(test_chart_RGB[0,:, :, 1], bez_curve)
        image_lin[:, :, 2] = interp_1d(test_chart_RGB[0,:, :, 2], bez_curve)

        image_ACES2065_lin = np.matmul(image_lin, three_by_three) # Then multiply by three by three

        if not INPUT_ILLUMINANT == [-1, -1]:  # Apply input illuminat compensation

            RGB_COLOURSPACE_ACES2065_1_CUSTOM_WP = RGB_COLOURSPACE_ACES2065_1.copy()  # Apply chromatic adaption
            RGB_COLOURSPACE_ACES2065_1_CUSTOM_WP.whitepoint = INPUT_ILLUMINANT
            image_ACES2065_lin = colour.RGB_to_RGB(image_ACES2065_lin, RGB_COLOURSPACE_ACES2065_1,
                                                    RGB_COLOURSPACE_ACES2065_1_CUSTOM_WP,
                                                    chromatic_adaptation_transform=CAT, apply_cctf_decoding=False,
                                                    apply_cctf_encoding=False)

        image_ACES2065_lin = np.clip(image_ACES2065_lin, a_min=0, a_max=1)

        image_HSL_Norm = ACES2065_2_HSL(image_ACES2065_lin)

        image_HSL_Norm = np.clip(image_HSL_Norm, a_min=0, a_max=None)
        image_HSL_Norm[:, :, 1] = image_HSL_Norm[:, :, 1] ** (1 / s_gamma)  # Apply Saturation Gamma
        image_HSL_Norm[:, :, 1] = image_HSL_Norm[:, :, 1] * s_gain  # Apply Saturation Gain

        image_HSL_Norm[:, :, 0] = image_HSL_Norm[:, :, 0] + interp_1d(image_HSL_Norm[:, :, 0],bez_curve_hue_hue)  # Apply Hue_hue Curve
        np.where(image_HSL_Norm[:, :, 0] < 0, image_HSL_Norm[:, :, 0],image_HSL_Norm[:, :, 0] + 1)  # Normalize Hue values that wrapped around
        np.where(image_HSL_Norm[:, :, 0] > 1, image_HSL_Norm[:, :, 0], image_HSL_Norm[:, :, 0] - 1)

        image_HSL_Norm[ :, :, 1] = image_HSL_Norm[ :, :, 1] * interp_1d(image_HSL_Norm[ :, :, 0],bez_curve_hue_sat)  # Apply Hue_sat Curve

        image_SRGB_Norm = HSL_2_sRGB(image_HSL_Norm)  # Convert back to sRGB

        image_SRGB_Norm = np.clip(image_SRGB_Norm, a_min=0, a_max=1)
        export_img = (image_SRGB_Norm * 255).astype(np.uint8)
        iio.imwrite("data/results/ref_chart_corrected.png", export_img)

        plt.imshow(test_chart_RGB[0,:,:,:])
        plt.xlabel(None)
        plt.ylabel(None)
        plt.axis('off')
        plt.title(("Chart before"))
        plt.show()

        plt.imshow(image_SRGB_Norm)
        plt.xlabel(None)
        plt.ylabel(None)
        plt.axis('off')
        plt.title(("Chart After"))
        plt.show()


def plothisto(num_chips,test_chart_LAB_norm):
    deltae_combined = np.zeros(shape=(num_chips), dtype=float)
    deltae_color = np.zeros(shape=(num_chips), dtype=float)
    ref_chip_LAB_list = np.reshape(REFCHIP_LAB, (-1, 3))
    test_chart_LAB_norm_list = np.reshape(test_chart_LAB_norm[0,:,:], (-1, 3))

    for i in range(num_chips):
        deltae_combined[i] = colour.difference.delta_E_CIE2000(ref_chip_LAB_list[i], test_chart_LAB_norm_list[i])
    print("\nMean Combined Error = ", np.mean(deltae_combined), " Median error = ", np.median(deltae_combined))
    print("Max error = ", np.max(deltae_combined), " Min Error = ", np.min(deltae_combined))

    test_chart_LAB_norm_list[:, 0] = ref_chip_LAB_list[:,0] #Set luma to same, to only test color error
    for i in range(num_chips):
        deltae_color[i] = colour.difference.delta_E_CIE2000(ref_chip_LAB_list[i], test_chart_LAB_norm_list[i])
    print("\nMean Color Error = ", np.mean(deltae_color), " Median error = ", np.median(deltae_color))
    print("Max error = ", np.max(deltae_color), " Min Error = ", np.min(deltae_color))

    plt.hist(deltae_combined, bins=20)
    plt.title("Histogram of Combined Error")
    plt.xlabel("Magnitude of Error")
    plt.ylabel("Count of Chips")
    plt.savefig('data/results/Combined_Error_histogram.png')
    plt.show()

    plt.hist(deltae_color, bins=20)
    plt.title("Histogram of Color Error")
    plt.xlabel("Magnitude of Error")
    plt.ylabel("Count of Chips")
    plt.savefig('data/results/Color_Error_histogram.png')
    plt.show()



def printcurves(bez_count_lum, best_bez_lum, best_three_by_three,input_blackpoint,input_whitepoint, input_MG ,debevec_curve):
    bez_curve_lum = interp_1d_setup(bez_count_lum, best_bez_lum[:, 1],hue=0,input_blackpoint=input_blackpoint,input_whitepoint=input_whitepoint, input_MG = input_MG)

    y = np.linspace(0, input_whitepoint, 500)

    debevec_values= debevec_curve(y)
    debevec_values = np.power(debevec_values, 1/2.4)  #Apply gamma 2.4 to compare with REC709

    l = interp_1d(y, bez_curve_lum)
    l = np.clip(l, a_min=0, a_max=None)
    l = np.power(l, 1/2.4)  #Apply gamma 2.4 to compare with REC709


    plt.title("Bez curves, Cyan is Debevec fit, Black custom fit")
    plt.plot(y, l, 'k')
    plt.plot(y, debevec_values, 'c')
    plt.savefig('data/results/luma_curve.png')
    plt.show()

def printHSLcurves( bez_count_hue_hue, bez_count_hue_sat,bez_values_hue_hue, bez_values_hue_sat,input_blackpoint,input_whitepoint, s_gamma, s_gain):
    bez_hue_hue = bez_values_hue_hue[range(bez_count_hue_hue), 1]
    bez_hue_hue = bez_hue_hue + MANUAL_HUE_HUE

    bez_hue_sat = bez_values_hue_sat[range(bez_count_hue_sat), 1]
    bez_hue_sat = bez_hue_sat + MANUAL_HUE_SAT

    bez_curve_hue_hue = interp_1d_setup(bez_count_hue_hue, bez_hue_hue, hue=1,
                                        input_blackpoint=input_blackpoint,
                                        input_whitepoint=input_whitepoint, mode=HSL_MODE)
    bez_curve_hue_sat = interp_1d_setup(bez_count_hue_sat, bez_hue_sat, hue=1,
                                        input_blackpoint=input_blackpoint,
                                        input_whitepoint=input_whitepoint, mode=HSL_MODE)


    y = np.linspace(0, 1, 500)
    hue_hue = interp_1d(y, bez_curve_hue_hue)
    hue_sat = interp_1d(y, bez_curve_hue_sat)

    figure, axis = plt.subplots(1, 2)
    axis[ 0].plot(y, hue_hue)
    axis[0].set_title("Hue-Hue")
    axis[1].plot(y, hue_sat)
    axis[1].set_title("Hue-Sat")

    plt.savefig('data/results/HSL_curves.png')
    plt.show()

    chart = np.zeros(shape = (100,300,3), dtype=float)
    lum = 0.5
    for x in range (0,300):
        hue = (x - 50) / 200
        for y in range(0, 50):
            sat = y/300 + 0.1

            hsl_orig = [hue, sat, lum]
            rgb_orig = HSL_2_sRGB(hsl_orig)

            hsl_after =[0,0,lum]
            hsl_after[1] = (sat ** (1 / (s_gamma + MANUAL_S_GAMMA))) * (s_gain + MANUAL_S_GAIN)
            hsl_after[0]  = hue + interp_1d(hue, bez_curve_hue_hue)
            hsl_after[1] = hsl_after[1] * interp_1d(hsl_after[0],bez_curve_hue_sat)

            rgb_after = HSL_2_sRGB(hsl_after)
            chart[y,x] = rgb_orig
            chart[99-y, x] =rgb_after

    chart = np.clip(chart, a_min=0, a_max=1)
    plt.imshow(chart)
    plt.xlabel(None)
    plt.ylabel(None)
    plt.axis('off')
    plt.title(("HSL Changes, Top Before, Bottom After"))
    plt.savefig('data/results/HSL_chart.png')
    plt.show()


def error_sum(bez_count_lum, best_bez_lum, test_chart_RGB, chart_name_low, chart_name_high,
              best_three_by_three, bad_chips, luma_test, chart_weight, final,bez_count_hue_hue=None,bez_count_hue_sat=None,
              bez_values_hue_hue=None, bez_values_hue_sat=None,input_blackpoint=None,input_whitepoint=None, s_gain =1, s_gamma=1,
              input_MG = 0 ):

    combined_error = -1
    chip_selec = COLOR_CHIPS
    slope_error = 1
    second_d_error = 1

    bez_curve_lum = interp_1d_setup(bez_count_lum, best_bez_lum[range(bez_count_lum),1],hue=0,input_blackpoint=input_blackpoint,input_whitepoint=input_whitepoint,input_MG = input_MG)

    if SLOPE_LIMIT and (luma_test == "Luma" or luma_test == "Luma2" or luma_test == "Luma3"):
        x = np.linspace(input_blackpoint,input_whitepoint, 101 )
        lum = interp_1d(x, bez_curve_lum)


        slope = np.subtract(lum[1:100], lum [0:99])
        num_slope_low = np.where(slope<SLOPE_MIN, 1, 0)
        num_slope_high = np.where(slope > SLOPE_MAX, 1, 0)
        slope_error = 1 +(np.sum(num_slope_low) +  np.sum(num_slope_high))*SLOPE_LIMIT

        if SMOOTHING>0:    #do second derivative error
            second_d = np.subtract(slope[1:99], slope [0:98])
            second_d = np.power(second_d, 2)
            second_d_error = 1 + (np.sum(second_d) * SMOOTHING * 100)

    if (luma_test=="Luma" or luma_test=="Luma2" or luma_test=="Luma3"):
        chip_selec = LUMA_CHIPS
    elif luma_test == "Color_all":
        chip_selec = ALL_CHIPS
    else: chip_selec =COLOR_CHIPS


    if (bez_count_hue_hue==None or (s_gamma == 1 and s_gain ==1)):

        test_chart_LAB_norm = calc_charts(test_chart_RGB, chart_name_low, chart_name_high, bez_curve_lum=bez_curve_lum,
                                          three_by_three=best_three_by_three)
        combined_error = calc_error_RGB(test_chart_LAB_norm, REFCHIP_LAB, chip_selec, bad_chips, luma_test, chart_name_low,
                                   chart_name_high, chart_weight, final=final)

    else:
        bez_curve_hue_hue = interp_1d_setup(bez_count_hue_hue, bez_values_hue_hue[range(bez_count_hue_hue), 1],hue=1,
                                            input_blackpoint=input_blackpoint,input_whitepoint=input_whitepoint, mode=HSL_MODE)
        bez_curve_hue_sat = interp_1d_setup(bez_count_hue_sat, bez_values_hue_sat[range(bez_count_hue_sat), 1], hue=1,
                                            input_blackpoint=input_blackpoint,input_whitepoint=input_whitepoint, mode=HSL_MODE)

        test_chart_LAB_norm = calc_charts(test_chart_RGB, chart_name_low, chart_name_high, bez_curve_lum=bez_curve_lum,
                                          three_by_three=best_three_by_three,bez_curve_hue_hue=bez_curve_hue_hue,
                                          bez_curve_hue_sat = bez_curve_hue_sat,s_gain=s_gain,s_gamma = s_gamma )
        combined_error = calc_error_RGB(test_chart_LAB_norm, REFCHIP_LAB, chip_selec, bad_chips, luma_test,
                                        chart_name_low,chart_name_high, chart_weight, final=final)

    return combined_error * slope_error * second_d_error



def main_luma(init_bez_values,num_charts,chip_RGB,input_blackpoint, input_whitepoint, bad_chips, num_round,three_by_three,bez_count_lum,
              chart_name_low, chart_name_high,chart_weight,max_gen_lum, input_MG, MG_Bez):

    bez_curve_lum = interp_1d_setup(bez_count_lum, init_bez_values[range(bez_count_lum),1],hue=0,input_blackpoint=input_blackpoint,
                                    input_whitepoint=input_whitepoint, input_MG = input_MG)

    test_chart_LAB_norm = calc_charts(chip_RGB, chart_name_low, chart_name_high, bez_curve_lum=bez_curve_lum,
                                      three_by_three=three_by_three)
    lum_error = calc_error_RGB(test_chart_LAB_norm, REFCHIP_LAB, LUMA_CHIPS, bad_chips, "Luma2", chart_name_low,
                                    chart_name_high, chart_weight, final=0)

    best_bez_queue  = mp.Queue()
    min_error = lum_error
    child_error_list = np.zeros(shape=(NUM_THREADS), dtype=float)
    child_bez_list = np.zeros(shape=(NUM_THREADS,bez_count_lum, 2), dtype=float)
    new_child_bez_list = np.zeros(shape=(NUM_THREADS, bez_count_lum, 2), dtype=float)

    for i in range(NUM_THREADS):  #setup orginal children
        new_child_bez_list [i] = init_bez_values.copy()

    last_time = time.time()

    for gen in range(0,max_gen_lum):
        print ("Generation = "+str(gen+1) +" of " +str(max_gen_lum) + " Min Luma Error = " +str(min_error ))

        t= time.time()
        print("Time Remaining " + str(((t-last_time)*(max_gen_lum-gen))/60) + " mins")
        last_time = t

        processes = []

        for i in range(NUM_THREADS):
            p = mp.Process(target=peturb_luma, args=(i, gen*num_round, best_bez_queue, new_child_bez_list[i,:,:], num_charts,
                                                     chip_RGB, REFCHIP_LAB, input_blackpoint, input_whitepoint, bad_chips,three_by_three,
                                                     chart_name_low,chart_name_high,bez_count_lum,chart_weight,MG_Bez, input_MG))
            processes.append(p)
            processes[i].start()

        for i in range(NUM_THREADS):

            t = best_bez_queue.get()#Unpack queue of best results
            child_error_list[i] = t[0, 0]
            child_bez_list[i] = t.copy()
            child_bez_list[i, 0, 0] = 0

        for i in range(NUM_THREADS):
            processes[i].join()

        if np.min(child_error_list) < min_error:
            min_error = np.min(child_error_list)

        min_error_pos = np.argpartition(child_error_list, int(NUM_THREADS/2))

        for i in range(NUM_THREADS) :
            if (i<int(NUM_THREADS/2)) :new_child_bez_list [i,:,1] = child_bez_list [min_error_pos[i],:,1]   #choose the best half to carry on
            else:
                for  j in range (bez_count_lum):   #average the best parents
                    t = child_bez_list [i - int(NUM_THREADS/2), j, 1] + child_bez_list [(2*NUM_THREADS) - i - int(NUM_THREADS/2) -1 , j, 1]
                    new_child_bez_list[i,j,1] = t / 2

                for n in range(1, bez_count_lum):
                    if (new_child_bez_list[i,n,1] < new_child_bez_list[i,n-1,1]):
                        ave = (new_child_bez_list[i,n,1] + new_child_bez_list[i,n-1,1]) / 2
                        new_child_bez_list[i,n,1] = ave
                        new_child_bez_list[i,n-1,1] = ave

    lowest_error_POS = np.argmin(child_error_list)
    best_bez_values = child_bez_list[lowest_error_POS, :, :]

    return best_bez_values

#end Main Luma


def peturb_luma (thread_num, generation, best_bez_list, parent_bez_values, num_charts, chip_RGB, refchip_lab, input_blackpoint, input_whitepoint,
                 bad_chips,three_by_three,chart_name_low,chart_name_high,bez_count,chart_weight, MG_bez, input_MG):
    peturb_max = PETURB_MAX_LUM_GEN/(2**generation)

    luma_test = "Luma2"

    if (thread_num == 0):
        peturb_max = peturb_max * 0.1  # Child zero is more conservative
    elif (thread_num == NUM_THREADS - 2): peturb_max = peturb_max * 5  # 2nd to last child is more extreme
    elif (thread_num == NUM_THREADS - 1): peturb_max = peturb_max * 10  # Last child is most extreme
    if (generation == 0): peturb_max = peturb_max * 2  # start first round more extreme

    num_children = NUM_CHILDREN

    min_error = sys.float_info.max

    new_bez_values = parent_bez_values.copy()
    bez_value = parent_bez_values.copy()

    t = np.zeros(shape=(bez_count,2), dtype=float)
    seed = 1733 + (thread_num * 107) + (generation * 2237)
    rng = np.random.default_rng(seed = seed)


    for i in range(num_children):
        if (i>0):  #retain parent
            rand = rng.normal(loc=0.0, scale=peturb_max, size=bez_count)
            scalar = np.abs(bez_value [:,1]) + 0.1  #Reduce/Increase adjustment by magnitude of existing values
            new_bez_values[:,1]= bez_value [:,1] + (rand*scalar)

        new_bez_values[0, 1] = np.clip(new_bez_values[0, 1], a_min=None, a_max=0)
        new_bez_values[1, 1] = 0
        if new_bez_values[bez_count-2, 1] < .5:new_bez_values[bez_count - 2, 1] = .5  #Constraint to prevent forcing all bez to zero and getting zero error
        new_bez_values[bez_count - 1, 1] = np.clip(new_bez_values[bez_count - 1, 1], a_min=new_bez_values[bez_count - 2, 1], a_max=None)
        new_bez_values[1:,1]= np.clip(new_bez_values[1:,1], 0, a_max=None)

        new_bez_values = set_bez_MG(bez_count, new_bez_values, MG_bez)

        error = error_sum(bez_count, new_bez_values, chip_RGB, chart_name_low, chart_name_high, three_by_three,bad_chips,
                          luma_test, chart_weight, final=0, input_blackpoint=input_blackpoint,input_whitepoint=input_whitepoint,
                          input_MG=input_MG)

        if (error < min_error):
            min_error = error
            bez_value = new_bez_values.copy()
            if (i>0):
                test_bez = new_bez_values.copy()
                for n in range (4):  # Apply the same perturb values again and see if it is better
                    test_bez[:,1]= new_bez_values [:,1] + rand

                    test_bez[0, 1] = np.clip(test_bez[0, 1], a_min=None, a_max=0)
                    test_bez[1, 1] = 0
                    if test_bez[bez_count-2, 1] < .5:test_bez[bez_count - 2, 1] = .5 #Constraint to prevent forcing all bez to zero and getting zero error
                    test_bez[bez_count - 1, 1] = np.clip(test_bez[bez_count - 1, 1],a_min=test_bez[bez_count - 2, 1], a_max=None)
                    test_bez[1:, 1] = np.clip(test_bez[1:, 1], 0, a_max=None)

                    test_bez = set_bez_MG(bez_count, test_bez, MG_bez)

                    test_error = error_sum(bez_count, test_bez, chip_RGB, chart_name_low, chart_name_high,three_by_three, bad_chips,
                                      luma_test, chart_weight, final=0, input_blackpoint=input_blackpoint,input_whitepoint=input_whitepoint, input_MG=input_MG)

                    if (test_error <  error):  #We found a better solution
                        new_bez_values = test_bez.copy()
                        error = test_error
                    else: break

                if (error < min_error):  #save these good new values
                    min_error = error
                    bez_value = new_bez_values.copy()

    bez_value[0, 0] = min_error
    best_bez_list.put(bez_value)


def s_gain_opt(s_gain_init,perturb_max, bez_count_lum, bez_values_lum, test_chart_RGB,chart_name_low, chart_name_high, three_by_three, bad_chips,
               chart_weight,bez_count_hue_hue,bez_count_hue_sat,bez_values_hue_hue, bez_values_hue_sat,input_blackpoint,input_whitepoint,s_gamma ,input_MG):

    test_s_gain = s_gain_init

    for i in range (MAX_ITER_S_GAIN):

        perturb_step = perturb_max / 2**i

        error_up = error_sum(bez_count_lum, bez_values_lum, test_chart_RGB,
                                   chart_name_low, chart_name_high, three_by_three, bad_chips, "Color", chart_weight,
                                   final=0,
                                   bez_count_hue_hue=bez_count_hue_hue, bez_count_hue_sat=bez_count_hue_sat,
                                   bez_values_hue_hue=bez_values_hue_hue, bez_values_hue_sat=bez_values_hue_sat,input_blackpoint=input_blackpoint,
                                  input_whitepoint=input_whitepoint, s_gain = test_s_gain+perturb_step,s_gamma = s_gamma,input_MG = input_MG)
        error_down = error_sum(bez_count_lum, bez_values_lum, test_chart_RGB,
                                   chart_name_low, chart_name_high, three_by_three, bad_chips, "Color", chart_weight,
                                   final=0,
                                   bez_count_hue_hue=bez_count_hue_hue, bez_count_hue_sat=bez_count_hue_sat,
                                   bez_values_hue_hue=bez_values_hue_hue, bez_values_hue_sat=bez_values_hue_sat,input_blackpoint=input_blackpoint,
                                  input_whitepoint=input_whitepoint, s_gain = test_s_gain-perturb_step,s_gamma = s_gamma,input_MG = input_MG)
        error_zero = error_sum(bez_count_lum, bez_values_lum, test_chart_RGB,
                                   chart_name_low, chart_name_high, three_by_three, bad_chips, "Color", chart_weight,
                                   final=0,
                                   bez_count_hue_hue=bez_count_hue_hue, bez_count_hue_sat=bez_count_hue_sat,
                                   bez_values_hue_hue=bez_values_hue_hue, bez_values_hue_sat=bez_values_hue_sat,input_blackpoint=input_blackpoint,
                                  input_whitepoint=input_whitepoint, s_gain = test_s_gain,s_gamma = s_gamma,input_MG = input_MG)

        if (error_up>error_zero) and (error_down>error_zero):
            pass
        elif error_up < error_down :
            test_s_gain = test_s_gain + perturb_step
        else:
            test_s_gain = test_s_gain - perturb_step

    return test_s_gain


def s_gamma_opt(s_gamma_init,perturb_max, bez_count_lum, bez_values_lum, test_chart_RGB,chart_name_low, chart_name_high, three_by_three, bad_chips,
               chart_weight,bez_count_hue_hue,bez_count_hue_sat,bez_values_hue_hue, bez_values_hue_sat,input_blackpoint,input_whitepoint,s_gain ,input_MG):

    test_s_gamma = s_gamma_init
    error_down = -1

    for i in range (MAX_ITER_S_GAMMA):

        perturb_step = perturb_max / 2**i

        error_up = error_sum(bez_count_lum, bez_values_lum, test_chart_RGB,
                                   chart_name_low, chart_name_high, three_by_three, bad_chips, "Color", chart_weight,
                                   final=0,
                                   bez_count_hue_hue=bez_count_hue_hue, bez_count_hue_sat=bez_count_hue_sat,
                                   bez_values_hue_hue=bez_values_hue_hue, bez_values_hue_sat=bez_values_hue_sat,input_blackpoint=input_blackpoint,
                                  input_whitepoint=input_whitepoint, s_gain =s_gain, s_gamma = test_s_gamma+perturb_step, input_MG = input_MG)
        if (test_s_gamma-perturb_step)>0:
            error_down = error_sum(bez_count_lum, bez_values_lum, test_chart_RGB,
                                       chart_name_low, chart_name_high, three_by_three, bad_chips, "Color", chart_weight,
                                       final=0,
                                       bez_count_hue_hue=bez_count_hue_hue, bez_count_hue_sat=bez_count_hue_sat,
                                       bez_values_hue_hue=bez_values_hue_hue, bez_values_hue_sat=bez_values_hue_sat,input_blackpoint=input_blackpoint,
                                      input_whitepoint=input_whitepoint, s_gain =s_gain, s_gamma = test_s_gamma-perturb_step, input_MG = input_MG)
        else: error_down = sys.float_info.max
        error_zero = error_sum(bez_count_lum, bez_values_lum, test_chart_RGB,
                                   chart_name_low, chart_name_high, three_by_three, bad_chips, "Color", chart_weight,
                                   final=0,
                                   bez_count_hue_hue=bez_count_hue_hue, bez_count_hue_sat=bez_count_hue_sat,
                                   bez_values_hue_hue=bez_values_hue_hue, bez_values_hue_sat=bez_values_hue_sat,input_blackpoint=input_blackpoint,
                                  input_whitepoint=input_whitepoint, s_gain =s_gain, s_gamma = test_s_gamma, input_MG = input_MG)


        if (error_up>error_zero) and (error_down>error_zero):
            pass
        elif error_up < error_down :
            test_s_gamma = test_s_gamma + perturb_step
        else:
            test_s_gamma = test_s_gamma - perturb_step

    return test_s_gamma

def sat_max(bez_count_lum, best_bez_lum, three_by_three,input_blackpoint,input_whitepoint, input_MG, test_chart_RGB, chart_name_low, chart_name_high):
    steps = 25

    bez_curve_lum = interp_1d_setup(bez_count_lum, best_bez_lum, hue=0, input_blackpoint=input_blackpoint, input_whitepoint=input_whitepoint, input_MG=input_MG)

    rgb = np.zeros(shape= (steps,steps,steps,3), dtype=float)
    s = np.linspace(input_blackpoint, input_whitepoint, steps)
    l = interp_1d(s, bez_curve_lum)

    for r in range (steps):
        for g in range(steps):
            for b in range(steps):
                rgb[r,g,b] = (l[r] ,l[g], l[b])

    rgb = np.matmul (rgb, three_by_three)
    rgb = np.clip(rgb, a_min=-0, a_max=None)

    hsl = ACES2065_2_HSL(rgb)
    max = np.max(hsl[:,:,:,1])

    location =np.unravel_index(np.argmax(hsl[:,:,:,1]), hsl[:,:,:,1].shape )

    print ("Sat Max Theoretical = ", max, " at (", s[location[0]], ",", s[location[1]], ",", s[location[2]],')')


    test_chart_LAB_norm = calc_charts(test_chart_RGB, chart_name_low, chart_name_high, bez_curve_lum=bez_curve_lum,
                                      three_by_three=three_by_three )

    test_chart_HSL =  LAB_2_HSL(test_chart_LAB_norm)
    max = np.max(test_chart_HSL[:, :, :, 1])
    location = np.unravel_index(np.argmax(test_chart_HSL[:, :, :, 1]), test_chart_HSL[:, :, :, 1].shape)
    print("Sat Max Actual = ", max, " at ",location)

# Calculate compressed distance
def compress( dist,  lim,  thr,  pwr):

    comprDist=-1

    if (dist < thr) :
        comprDist = dist #No compression below threshold
    else:
        # Calculate scale factor for y = 1 intersect
        scl = (lim - thr) /  np.power(np.power((1.0 - thr) / (lim - thr), -pwr) - 1.0, 1.0 / pwr)

        #Normalize distance outside threshold by scale factor
        nd = (dist - thr) / scl
        p = np.power(nd, pwr)

        comprDist = thr + scl * nd / (np.power(1.0 + p, 1.0 / pwr)) # Compress

    return comprDist

def gammut_compression(input, strength, mode):  #Take sRGB/lin in, apply aces gammut compression, and return sRGB/lin

    if (mode ==1):  #Aces

        if strength< 0.8: strength = 0.8
        elif strength > 1: strength = 1

        """/* --- Gamut Compress Parameters --- */
        // Distance from achromatic which will be compressed to the gamut boundary
        // Values calculated to encompass the encoding gamuts of common digital cinema cameras"""
        LIM_CYAN =  1.147*strength
        LIM_MAGENTA = 1.264*strength
        LIM_YELLOW = 1.312*strength

        """// Percentage of the core gamut to protect
        // Values calculated to protect all the colors of the ColorChecker Classic 24 as given by
        // ISO 17321-1 and Ohta (1997)"""
        THR_CYAN = 0.815/strength
        THR_MAGENTA = 0.803/strength
        THR_YELLOW = 0.880/strength

        #// Aggressiveness of the compression curve
        PWR = 1.2

        input = colour.RGB_to_RGB(input, RGB_COLOURSPACE_ACES2065_1,RGB_COLOURSPACE_ACESCG,
                                        chromatic_adaptation_transform=None, apply_cctf_decoding=False, apply_cctf_encoding=False)

    elif (mode == 2):  # DWG

        if strength< 0.9: strength = 0.9
        elif strength > 1: strength = 1

        LIM_CYAN =  1.4 *strength
        LIM_MAGENTA = 1.2 *strength
        LIM_YELLOW = 1.1 *strength

        THR_CYAN = .8 /strength
        THR_MAGENTA = .9 /strength
        THR_YELLOW = .9 /strength

        PWR = 1.2

        input = colour.RGB_to_RGB(input, RGB_COLOURSPACE_ACES2065_1, RGB_COLOURSPACE_DAVINCI_WIDE_GAMUT,
                                  chromatic_adaptation_transform=None, apply_cctf_decoding=False,
                                  apply_cctf_encoding=False)

    output = input.copy()

    for i in (range(np.shape(input)[0])):
        for j in (range(np.shape(input)[1])):
            for k in (range(np.shape(input)[2])):
                rIn = input[i,j,k,0]
                gIn = input[i, j, k,1]
                bIn = input[i, j, k,2]

                #Achromatic axis
                ach = np.max((rIn, gIn, bIn))

                #Distance from the achromatic axis for each color component aka inverse RGB ratios
                dist = [0,0,0]
                if (ach == 0.0):
                    dist[0] = 0.0
                    dist[1] = 0.0
                    dist[2] = 0.0

                else:
                    dist[0] = (ach - rIn) / abs(ach)
                    dist[1] = (ach - gIn) / abs(ach)
                    dist[2] = (ach - bIn) / abs(ach)


                # Compress distance with parameterized shaper function
                comprDist = (
                    compress(dist[0], LIM_CYAN, THR_CYAN, PWR),
                    compress(dist[1], LIM_MAGENTA, THR_MAGENTA, PWR),
                    compress(dist[2], LIM_YELLOW, THR_YELLOW, PWR)
                )

                #Recalculate RGB from compressed distance and achromatic
                output[i,j,k,0] = ach - comprDist[0] * abs(ach)
                output[i,j,k,1] = ach - comprDist[1] * abs(ach)
                output[i,j,k,2] = ach - comprDist[2] * abs(ach)

    if (mode == 1):  # ACES
        output = colour.RGB_to_RGB(output, RGB_COLOURSPACE_ACESCG, RGB_COLOURSPACE_ACES2065_1,
                                   chromatic_adaptation_transform=None, apply_cctf_decoding=False,
                                   apply_cctf_encoding=False)
    elif (mode == 2):  # DWG
        output = colour.RGB_to_RGB(output, RGB_COLOURSPACE_DAVINCI_WIDE_GAMUT,RGB_COLOURSPACE_ACES2065_1,
                              chromatic_adaptation_transform=None, apply_cctf_decoding=False, apply_cctf_encoding=False)

    return output


def tone_map_xyY(input,inflection_percent,init_max):

    target_max = [1,1,1]
    middle_gray = [.18,.18,.18]

    target_max = colour.RGB_to_RGB(target_max, RGB_COLOURSPACE_DAVINCI_WIDE_GAMUT, RGB_COLOURSPACE_LIN_CIEXYZ_SCENE,
                      chromatic_adaptation_transform=None, apply_cctf_decoding=True, apply_cctf_encoding=False)
    target_max = colour.XYZ_to_xyY(target_max)[2]
    middle_gray = colour.XYZ_to_xyY(middle_gray)[2]

    input = colour.RGB_to_RGB(input, RGB_COLOURSPACE_ACES2065_1, RGB_COLOURSPACE_LIN_CIEXYZ_SCENE,
                             chromatic_adaptation_transform=None, apply_cctf_decoding=False,apply_cctf_encoding=False)
    input = colour.XYZ_to_xyY(input) #Convert to xxY colorspace
    out = input.copy()

    input_clipped = np.clip(input[:,:,:,2], a_min=0, a_max=None)

    gain = target_max / init_max
    l = input_clipped / np.max(input[:,:,:,2])
    a = (32 * inflection_percent**4) + 1 #adjusts how soon the gain applies
    out[:, :, :, 2]  = (input[:,:,:,2] * gain * l**a) + ((1-l**a) * input[:,:,:,2])

    print("Tone Mapping : Init Max = " , init_max, "Target Max = ", target_max, " Scale Amount = ",gain )

    out = colour.xyY_to_XYZ(out)  #Return back to Lin-SRGB
    out = colour.RGB_to_RGB(out,  RGB_COLOURSPACE_LIN_CIEXYZ_SCENE,RGB_COLOURSPACE_ACES2065_1,
                      chromatic_adaptation_transform=None, apply_cctf_decoding=False, apply_cctf_encoding=False)

    return out


def wb_error(three_by_three,print_results, s_gamma,s_gain, bez_curve_hue_hue,bez_curve_hue_sat):
    input = [MIDDLE_GRAY_LIN,MIDDLE_GRAY_LIN,MIDDLE_GRAY_LIN]
    ACES2065_rgb = np.matmul(input,three_by_three)
    if  not (s_gamma ==1 and s_gain ==1):
        HSL = ACES2065_2_HSL(ACES2065_rgb)
        HSL[ 1] = HSL[ 1] ** (1 / s_gamma)  # Apply Saturation Gamma
        HSL[1] = HSL[1] * s_gain  # Apply Saturation Gain
        HSL[ 0] = HSL[ 0] + interp_1d(HSL[0],bez_curve_hue_hue)  # Apply Hue_hue Curve
        HSL[ 1] = HSL[1] * interp_1d(HSL[ 0], bez_curve_hue_sat)  # Apply Hue_sat Curve
        ACES2065_rgb = HSL_2_ACES2065(HSL)  # Convert back to sRGB

    DWG = ACES2065_2_DWG_I_CAT(ACES2065_rgb)

    if np.min(DWG) <= 0: error = 100
    else: error = (np.max(DWG)**2/np.min(DWG)**2) - 1
    if print_results : print ( "White point error, DWG values = ", DWG)

    return error

def debevec(chart_name_low, chart_name_high, input_whitepoint, input_MG ):

    num_charts = chart_name_high - chart_name_low + 1

    times = np.zeros(shape = num_charts, dtype=np.float32)
    n = 0
    for i in range (chart_name_low, chart_name_high+1):
        times[n] = 2**i
        n+=1

    img_fn = []

    for i in range(chart_name_low, chart_name_high + 1):
        file_path = ("data/" + str(i) + ".png")
        img_fn = np.append(img_fn, file_path)

    img_list = [cv.imread(fn) for fn in img_fn]

    cv.AlignExposures(img_list, img_list, times)

    # Obtain Camera Response Function (CRF)
    calibrateDebevec = cv.createCalibrateDebevec()
    responseDebevec = calibrateDebevec.process(img_list, times)
    responseDebevec = responseDebevec.reshape(256, 3)

    weights = np.array([1, 2, 1])  # More weight to Green
    weighted_average = np.average(responseDebevec, axis=1, weights=weights)
    weighted_average = weighted_average

    existing_MG = weighted_average[int(input_MG * 256)]
    weighted_average = weighted_average * (MIDDLE_GRAY_LIN / existing_MG)  #Scale so input middle gray = 0.18
    responseDebevec = responseDebevec * (MIDDLE_GRAY_LIN / existing_MG)

    # Generate new x values for a smoother curve
    x_new = np.linspace(0, 1, 256)
    derivatives = np.gradient(weighted_average, x_new)
    hermite_spline = CubicHermiteSpline(x_new, weighted_average, derivatives)

    return hermite_spline

#Binary search to find best illuminant for Output CST
def optimize_ouput_illum( three_by_three, s_gamma,s_gain, bez_curve_hue_hue,bez_curve_hue_sat):
    global ILLUMINANT

    init_color_error = wb_error(three_by_three, 0, s_gamma,s_gain, bez_curve_hue_hue,bez_curve_hue_sat)
    temperature = colour.xy_to_CCT(np.array(ILLUMINANT))
    print("\nExisting Output Illum = ", f'{ILLUMINANT[0]:.8f}', ",",f'{ILLUMINANT[1]:.8f}', " Temp = ", temperature, " Starting Error = ", init_color_error)

    illum_zero = ILLUMINANT
    illum_up = illum_zero
    illum_down =illum_zero
    for n in range (3):
        iter_max = ILLUM_OPT_MAX / (n+1)
        for x in range(ILLUM_OPT_ITER):

            illum_up = (illum_zero[0] + iter_max / 2**x ,illum_zero[1])
            illum_down = (illum_zero[0] - iter_max / 2 ** x, illum_zero[1])

            ILLUMINANT = illum_zero
            error_zero = wb_error(three_by_three, 0, s_gamma,s_gain, bez_curve_hue_hue,bez_curve_hue_sat)

            ILLUMINANT = illum_up
            error_up = wb_error(three_by_three, 0, s_gamma,s_gain, bez_curve_hue_hue,bez_curve_hue_sat)

            ILLUMINANT = illum_down
            error_down = wb_error(three_by_three, 0, s_gamma,s_gain, bez_curve_hue_hue,bez_curve_hue_sat)


            if (error_down< error_up) and (error_down< error_zero):
                illum_zero = illum_down
            elif (error_up< error_down) and (error_up<error_zero):
                illum_zero = illum_up

        for y in range(ILLUM_OPT_ITER):

            illum_up = (illum_zero[0], illum_zero[1] + iter_max / 2 ** y, )
            illum_down = (illum_zero[0], illum_zero[1] - iter_max / 2 ** y)

            ILLUMINANT = illum_zero
            error_zero = wb_error(three_by_three, 0, s_gamma,s_gain, bez_curve_hue_hue,bez_curve_hue_sat)

            ILLUMINANT = illum_up
            error_up = wb_error(three_by_three, 0, s_gamma,s_gain, bez_curve_hue_hue,bez_curve_hue_sat)

            ILLUMINANT = illum_down
            error_down = wb_error(three_by_three, 0, s_gamma,s_gain, bez_curve_hue_hue,bez_curve_hue_sat)

            if (error_down < error_up) and (error_down < error_zero):
                illum_zero = illum_down
            elif (error_up < error_down) and (error_up < error_zero):
                illum_zero = illum_up

    ILLUMINANT = illum_zero

    final_error = wb_error(three_by_three, 1, s_gamma,s_gain, bez_curve_hue_hue,bez_curve_hue_sat)
    temperature = colour.xy_to_CCT(np.array(ILLUMINANT))
    print("Optimized Output Illum = ", f'{ILLUMINANT[0]:.8f}', ",",f'{ILLUMINANT[1]:.8f}'," Temp = ", temperature,  " Final Error = ",final_error)
    return illum_zero

def optimize_input_illum(test_chart_RGB, chart_name_low, chart_name_high,bez_curve_lum, three_by_three):
    global INPUT_ILLUMINANT

    if INPUT_ILLUMINANT == [-1,-1]:
        INPUT_ILLUMINANT = ILLUMINANT

    inv_matrix = np.linalg.inv(three_by_three)
    test_chart_lin =test_chart_RGB.copy()
    test_chart_lin[:, :, :, 0] = interp_1d(test_chart_RGB[:, :, :, 0], bez_curve_lum)  # use luma bez curve first
    test_chart_lin[:, :, :, 1] = interp_1d(test_chart_RGB[:, :, :, 1], bez_curve_lum)
    test_chart_lin[:, :, :, 2] = interp_1d(test_chart_RGB[:, :, :, 2], bez_curve_lum)

    illum_zero = INPUT_ILLUMINANT

    error = input_illum_error(test_chart_lin, three_by_three, INPUT_ILLUMINANT, inv_matrix)
    temperature = colour.xy_to_CCT(np.array(INPUT_ILLUMINANT))
    print("\nInitial Input Illuminat = ", INPUT_ILLUMINANT, " Temp = ", temperature,   " Error = ", error)

    for n in range(3):
        iter_max = INPUT_ILLUM_OPT_MAX / (n + 1)
        for x in range(INPUT_ILLUM_OPT_ITER):

            illum_up = (illum_zero[0] + iter_max / 2 ** x, illum_zero[1])
            illum_down = (illum_zero[0] - iter_max / 2 ** x, illum_zero[1])

            INPUT_ILLUMINANT = illum_zero
            error_zero = input_illum_error(test_chart_lin, three_by_three, illum_zero, inv_matrix)

            INPUT_ILLUMINANT = illum_up
            error_up = input_illum_error(test_chart_lin, three_by_three, illum_up, inv_matrix)

            INPUT_ILLUMINANT = illum_down
            error_down = input_illum_error(test_chart_lin, three_by_three, illum_down, inv_matrix)


            if (error_down < error_up) and (error_down < error_zero):
                illum_zero = illum_down
            elif (error_up < error_down) and (error_up < error_zero):
                illum_zero = illum_up

        for y in range(INPUT_ILLUM_OPT_ITER):

            illum_up = (illum_zero[0], illum_zero[1] + iter_max / 2 ** y,)
            illum_down = (illum_zero[0], illum_zero[1] - iter_max / 2 ** y)

            INPUT_ILLUMINANT = illum_zero
            error_zero = input_illum_error(test_chart_lin, three_by_three, illum_zero, inv_matrix)

            INPUT_ILLUMINANT = illum_up
            error_up = input_illum_error(test_chart_lin, three_by_three, illum_up, inv_matrix)

            INPUT_ILLUMINANT = illum_down
            error_down = input_illum_error(test_chart_lin, three_by_three, illum_down, inv_matrix)

            if (error_down < error_up) and (error_down < error_zero):
                illum_zero = illum_down
            elif (error_up < error_down) and (error_up < error_zero):
                illum_zero = illum_up

    INPUT_ILLUMINANT = illum_zero
    error = input_illum_error(test_chart_lin, three_by_three, INPUT_ILLUMINANT, inv_matrix)
    temperature = colour.xy_to_CCT(np.array(INPUT_ILLUMINANT))
    print("Final Input Illuminat = ",INPUT_ILLUMINANT," Temp = ", temperature, " Error = ",error )

    return INPUT_ILLUMINANT


def input_illum_error(test_chart_lin,three_by_three, test_illum, inv_matrix):


    test_chart_ACES2065_lin = np.matmul(test_chart_lin[0], three_by_three)  # Then multiply by three by three

    RGB_COLOURSPACE_ACES2065_1_CUSTOM_WP = RGB_COLOURSPACE_ACES2065_1.copy()  #Apply chromatic adaption
    RGB_COLOURSPACE_ACES2065_1_CUSTOM_WP.whitepoint = test_illum
    test_chart_ACES2065_lin = colour.RGB_to_RGB(test_chart_ACES2065_lin, RGB_COLOURSPACE_ACES2065_1,RGB_COLOURSPACE_ACES2065_1_CUSTOM_WP
                                            ,chromatic_adaptation_transform=CAT, apply_cctf_decoding=False,
                                            apply_cctf_encoding=False)
    invert_ACES2065_lin = np.matmul(test_chart_ACES2065_lin, inv_matrix)

    test = ACES2065_2_LAB(invert_ACES2065_lin)

    A = test[:,:,1] * WB_CHIPS #Filter out to only neutral chips
    B = test[:,:,2] *  WB_CHIPS
    return np.mean(A ** 2 + B ** 2)  #Calculate the mean Chroma


def calc_dr(test_chart, chart_name_low, chart_name_high, bez_curve_lum, input_whitepoint):

    darkest_chip = np.unravel_index(np.argmin(test_chart, axis=None), test_chart.shape)
    darkest_value = test_chart[darkest_chip]
    darkest_value_lin = interp_1d(darkest_value, bez_curve_lum)
    darkest_value_lin = np.clip(darkest_value_lin, a_min=2 ** -15, a_max=None)
    value = -1
    value_lin = -1
    for n in range (chart_name_low, chart_name_high):  #Find what chart is slightly brighter
        value = test_chart[n,darkest_chip[1],darkest_chip[2],darkest_chip[3] ]
        value_lin = interp_1d(value, bez_curve_lum)
        value_lin = np.clip(value_lin, a_min=2 ** -15, a_max=None)
        if value_lin> darkest_value_lin *2:  #This chip is 1 stop brighter than darkest
            break

    abovestops =  math.log((interp_1d(input_whitepoint, bez_curve_lum) / .18), 2)
    belowstops = math.log((value_lin / .18), 2) - 1
    print("\nDynamic Range = ", abovestops-belowstops, " stops. ", abovestops, " above MG, ", belowstops, " below MG" )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--fit_mode",
        default="CALC",
        type=str,
        help="If CALC, calculate luma curve based on test images, otherwise use calculation in Color Library,"
             " following curves supported; GP_LOG, GP_PROTUNE, CLOG, CLOG2, CLOG3, FLOG, FLOG2, SLOG,SLOG2,SLOG3, VLOG, REC709, REC2100_HLG ",
    )


    parser.add_argument(
        "--rec709_lut",
        default=0,
        required=False,
        type=int,
        help="Set to 1 for final lut to covert to rec709 instead of linear, default = 0",
    )
    parser.add_argument(
        "--DWG_I_lut",
        default=1,
        required=False,
        type=int,
        help="Set to 1 for final lut to covert to Davinchi Wide Gammut, Intermediate, else linear, default = 1",
    )
    parser.add_argument(
        "--debevec_fit",
        default=1,
        required=False,
        type=int,
        help="Set to 1 to use CV2 Debevec fit for tone curve, 0 for DIY algorithm, default = 1",
    )
    parser.add_argument(
        "--enable_optimize_illum",
        default="IO",
        required=False,
        type=str,
        help="Set to \"I\" perform input Illuminate optimization only, \"O\" to perform output optimization only, or \"IO\" for both,  default = \"IO\"",
    )
    parser.add_argument(
        "--gammut_comp",
        default=0,
        required=False,
        type=int,
        help="Set to 1 to apply ACES gammut compression on resulting lut, 2 to use DWG color Space, 0 off",
    )
    parser.add_argument(
        "--expand_tone",
        default=0,
        required=False,
        type=int,
        help="0 = off, 1 = expand so Input_whitepoint is DWG/I max",
    )
    parser.add_argument(
        "--inflection_percent",
        default=0,
        required=False,
        type=float,
        help="when expand_tone is on, where to start expansion, 0 even across tone range (pure gain), 1.0 at nearly max luma (harsh), Default = 0.5",
    )
    parser.add_argument(
        "--expand_shadow",
        default=0,
        required=False,
        type=int,
        help="0 = off, 1 = to raise shadows slightly to reduce negative values sent to LUT",
    )
    parser.add_argument(
        "--print_input_charts",
        default=1,
        required=False,
        type=int,
        help="Set to 1 for input charts to be displayed back to confirm they were read correctly",
    )
    parser.add_argument(
        "--print_ref_chart",
        default=1,
        required=False,
        type=int,
        help="Set to 1 for ground truth chart to be displayed back to confirm they were read correctly",
    )
    parser.add_argument(
        "--print_proof_charts",
        default=1,
        required=False,
        type=int,
        help="Set to 1 for final conversion to be applied to proof charts of naming format 'test_n.png', where n is the exposure offset of the chart",
    )
    parser.add_argument(
        "--print_curves",
        default=1,
        required=False,
        type=int,
        help="Set to 1 for plots of curves",
    )
    parser.add_argument(
        "--chart_weight",
        default=4,
        required=False,
        type=int,
        help="Apply less weight to under/over exposed charts/. 0= all charts equal weight, 1= chart +/-1ev, half weight,  w/w+1...",
    )
    parser.add_argument(
        "--disp_chip_pixel_loc",
        default=0,
        required=False,
        type=int,
        help="1 to print test chart pixel locations to console to debug test chart reads",
    )
    parser.add_argument(
        "--lut_name",
        default='final_lut',
        type=str,
        help="Specify filename of output lut.",
    )

    parser.add_argument(
        "--bez_count_lum",
        default=8,
        required=False,
        type=int,
        help="number of bezier points in luma curve model, default 8",
    )
    parser.add_argument(
        "--bez_count_hue_hue",
        default=6,
        required=False,
        type=int,
        help="number of bezier points in hue-hue curve model, default 6",
    )
    parser.add_argument(
        "--bez_count_hue_sat",
        default=6,
        required=False,
        type=int,
        help="number of bezier points in hue-sat curve model, default 6",
    )
    parser.add_argument(
        "--max_gen_lum",
        default=4,
        required=False,
        type=int,
        help="how many iterations run the genetic algorithm, default 4",
    )
    parser.add_argument(
        "--lum_opt_iter",
        default=4,
        required=False,
        type=int,
        help="how many iterations on lum curve optimizations",
    )
    parser.add_argument(
        "--three_by_three_opt_iter",
        default=5,
        required=False,
        type=int,
        help="how many iterations on  genetic three_by_three optimizations",
    )
    parser.add_argument(
        "--sat_opt_iter",
        default=5,
        required=False,
        type=int,
        help="how many iterations on saturation optimizations, error based",
    )
    parser.add_argument(
        "--hue_loop_iter",
        default=0,
        required=False,
        type=int,
        help="how many iterations on repeat the entire Hue-Hue, Hue-Sat loop",
    )
    parser.add_argument(
        "--hue_hue_opt_iter",
        default=0,
        required=False,
        type=int,
        help="how many iterations on hue_hue curve optimizations",
    )
    parser.add_argument(
        "--hue_sat_opt_iter",
        default=0,
        required=False,
        type=int,
        help="how many iterations on Hue-sat curve optimizations",
    )
    parser.add_argument(
        "--seed_lut_name",
        default='initial_LUT.cube',
        type=str,
        help="Specify filename of seed 1d  lut.",
    )
    parser.add_argument(
        "--white_clipping_point",
        default=0.97,
        required=False,
        type=float,
        help="How high to accept input chips as not clipped, as ratio of actual maximum seen, default = 0.97",
    )
    parser.add_argument(
        "--black_clipping_point",
        default=0.03,
        required=False,
        type=float,
        help="How low to accept input chips as not clipped, as ratio of actual minimum seen, default = 0.03",
    )

    args = parser.parse_args()
    print(args)

    main(args)

