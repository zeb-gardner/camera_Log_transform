# Premise
Generate a LUT to convert from any specific camera color space to Davinchi Wide Gammut / Intermediate to produce and accurate starting point for color grading. Useful where camera manufacture has not documented the camera space, or it provided inaccurate conversions.

# Workflow

## Prepare test chart PNGs.
Light a test chart with a high CRI, even light. Care must be taken that camera will not see reflections. Ideally the light should be the same a will be used for real shoots.

White balance camera to the lighting, ideally with a gray card. The --enable_optimize_illum = “I” flag will attempt to reverse out incorrect camera WB, but best practice is to get it as accurate as possible in camera.

Set camera to base ISO, no ND, auto exposure, note what it thinks 'correct exposure' is and write this down. Eg ISO 800, F2.8, 1/100s.  

We now need to shoot the chart at different exposures by changing only shutter time.  Enough different exposures are needed to cover entire dynamic range of camera. Ideally using in camera exposure tools like false color to take a test shot where multiple chips of test chart are clipped low, all the way up to where multiple chips are clipped high. This is done by only changing the shutter time, not adjusting light, iris or ND.

In Canon R5 example this was 5 stops over exposed to 2 stops under exposed. 1/3s to 1/400s.  Best practice is to note settings and show to camera, so when we later pull the frames we know the settings used.

If you wish to use flat field correction, at the +0Ev exposure, cover the chart with a sheet of white paper and capture this as well. We will later use this to attempt to correct for uneven chart illumination.

After recording test clip of the multiple exposures, open the file in your video editor and export a still of each of the different exposures. Don't use any color management or grade in the video editor, stills should be 'straight out of camera'. Export as 16 bit tiff (Resolve seems to limit PNGs stills to 8 bit)

Open the stills as a stack in Photoshop and auto-align.  File > Scripts > Load Files into Stack. Select all the stills and select 'Attempt to automatically Align Source Images"

Select all layers, Using Crop tool and edit > transform > Skew, set edges of test chart to match the image canvas edges. See example images included in data folder. The program doesn't care about the resulting aspect ratio or pixel dimensions, but it is hard coded to assume each test chip is at certain % of full frame from the top left corner.  When the program reads in the charts it will display the colors it found back to the user, make sure this is what you expect, if not you didn't crop the picture correctly. The orientation of the chart must also match what was coded, so rotate image if needed.

![example chart](/img/example_chart.jpg)

Export each exposure as an individual 16 bit PNG.  File > Save a Copy, and change type to PNG. (Export as PNG seems to only be 8 bit.)  Files must be named with their exposure, eg the middle exposure file is '0.png', the file over exposed 1 stop is '1.png' the file underexposed is '-1.png'.   You must have a file for each stop, so the range 5 over to 2 under needs 8 individual files

For the flat field correction image, name “flat.png”

Place these files in the 'data' folder.  You can also place verification images that will get the final calculated conversion applied to them. They are named 'test_0.png' if the file is middle exposure, 'test_1.png' if the file is over exposed 1 stop, 'test_-2.png' if the file is under exposed 2 stops.  These files can be a simple copy of test charts created above or any image you want to see the correction applied to with calculated exposure correction applied.

# Python Program

### Dependencies;
	scipy
	matplotlib
	scikit-image
	opencv-python
	pypng
	imageio
	colour-science
	numpy
	argparse
	beautifultable

'main.py' is the program itself. Default arguments should produce reasonable results, but can be tweaked by user if desired. Run program with –help flag to see documetaion of command line arguments.

## Input Data

Program will read in all the input test charts found in the data folder and display them back to user. Confirm they look correct.

Program reads in minimum and maximum pixel values found and displays as 'Input Black Point' and 'Input White Point'. Verify these are what you expect. EG, if you did not expose a  chart bright enough, white point may be read as 0.9.  If this is the case the resulting LUT will not have accurate information for the entire tone curve.

Middle Gray IRE value is also displayed. This should be as close as possible to the middle gray IRE value specified by the camera manufacturer. This value gets set to 0.18 linear in the curve fitting process.  If you find the resulting lut is making your image to dark/bright, the file you labeled as '0.png' was not actually neutral exposure. Relabel '-1.png' as '0.png' and '0.png' as '1.png', etc and run program again to see if results are better.  If results are worse, you relabeled the wrong direction, the original '0.png' needs to be '-1.png'.

The LAB values for test charts as provided by the manufacture are listed in CC-SG.py (Xrite Color Checker SG) and CC_VIDEO_V1.py (Xrite Color Checker Video Passport V1). If user has a different chart, they can create a new file for their chart with the correct values. The correct chart file needs to be selected in the imports section, SG selected by default.

from CC_SG import *
	'#from CC_VIDEO_V1 import *  #Choose the specs for the right chart

## Luma Fit

Next process is genetic fitting of curve to convert camera log to linear.  You will see resulting luma error decrease as better fit is found.  Final value of 1.0 here would be excellent, greater than 5 is probably a problem.

## 3x3 Color Fit

Color error is minimized next by optimizing values of 3x3 conversion matrix. This converts from camera color space to ACES AP0 space. Error of around 5.0 is reasonable here.

This process then repeats, optimizing linearization (minimize luma error) and optimizing conversion matrix. Default setting is 4 rounds of optimization.

## HSL Fit

Optional are next are HSL space optimizations, default is disabled.   A Saturation Gamma curve and Saturation Gain are applied to minimize color error.  This optimization ignores the neutral color chips, where 3x3 optimization did not, so resulting error numbers will be lower than seen in above optimization. (This is done because a saturation of 0 will produce best error for neutral chips, but is obviously not the solution we want)

Hue-Loop optimizations apply a Hue-Hue curve and a Hue-Saturation curve to again try to better fit the camera space to reference charts.

These HSL optimizations are unlikely to work well with charts with limited hue chips, such as Color Checker Video Passport V1. As the algorithm does not have multiple tones of each hue to optimize too. This feature should be disabled in this case, command line;
--sat_opt_iter 0 --hue_loop_iter 0 
	
## Results

A plot of final linearization curve is displayed to user. If process works correctly curve should be smooth, such as following. Note, this curve has a gamma 2.4 curve baked into it to make it display nicer, but this is not in final LUT. As such a perfect REC709 camera would produce a straight line curve here. The debevec fit is displayed in cyan and the program's fit in black

![example curve](/img/example_luma_curve.png)

Test_0.png files and others will have correction applied to them and displayed back to user. 

Resulting LUT default name is 'Shaper_final_lut.cube'.  This file can be imported into resolve lut folder, for windows  "C:\ProgramData\Blackmagic Design\DaVinci Resolve\Support\LUT". You must then update LUT list in Resolve, Project Settings > Color Managment > Lookup Tables >Update lists. Or reboot Resolve.

This LUT converts from camera color space to DWG/Intermediate This LUT will be the first node in your grade. You will need to apply a CST node to covert from  DWG/Intermediate to your desired output space such as Rec709 Gamma 2.2. See following screenshot example;
 
LUT is a Shaper LUT and may not be compatible with programs other than Resolve. A 4096 element 1d LUT first converts from camera log ton curve to Resolve Intermediate A 65 element 3d LUT then performs the 3x3 matrix and HSL optimizations that converts color space to Davinchi Wide Gammut

# Trouble Shooting

__LUT produces significant white balance change in Resolve.__ 

If camera white balance when shooting test charts was not perfect, an opposite correction is baked into the LUT.  Experiment by changing;
	--enable_optimize_illum
“I” is input only, “O” is output only, “IO” is both, “None” is none

__Error Message while reading input charts__

Charts must be 16 bit tiff files. Other formats will error

__Error “ RuntimeWarning: invalid value encountered in log2 DI_C * (np.log2(L + DI_A) + DI_B),”__

The color-science library doesn't seem to gracefully handle negative values fed to the conversion to DWG/I. These negative values are produced when a camera gammut is larger than DWG (eg Canon Cinema Gammut). So the camera gammut will be clipped outside DWG, but DWG was designed that no real image should actually clip. So this error can be ignored

