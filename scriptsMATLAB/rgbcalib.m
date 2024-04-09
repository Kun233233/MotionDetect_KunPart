% Auto-generated by cameraCalibrator app on 27-Mar-2024
%-------------------------------------------------------


% Define images to process
imageFileNames = {'D:\aaaLab\aaagraduate\SaveVideo\source\Calibration\RGB\0.png',...
    'D:\aaaLab\aaagraduate\SaveVideo\source\Calibration\RGB\1.png',...
    'D:\aaaLab\aaagraduate\SaveVideo\source\Calibration\RGB\2.png',...
    'D:\aaaLab\aaagraduate\SaveVideo\source\Calibration\RGB\3.png',...
    'D:\aaaLab\aaagraduate\SaveVideo\source\Calibration\RGB\4.png',...
    'D:\aaaLab\aaagraduate\SaveVideo\source\Calibration\RGB\5.png',...
    'D:\aaaLab\aaagraduate\SaveVideo\source\Calibration\RGB\6.png',...
    'D:\aaaLab\aaagraduate\SaveVideo\source\Calibration\RGB\7.png',...
    'D:\aaaLab\aaagraduate\SaveVideo\source\Calibration\RGB\8.png',...
    'D:\aaaLab\aaagraduate\SaveVideo\source\Calibration\RGB\9.png',...
    'D:\aaaLab\aaagraduate\SaveVideo\source\Calibration\RGB\10.png',...
    'D:\aaaLab\aaagraduate\SaveVideo\source\Calibration\RGB\11.png',...
    'D:\aaaLab\aaagraduate\SaveVideo\source\Calibration\RGB\12.png',...
    'D:\aaaLab\aaagraduate\SaveVideo\source\Calibration\RGB\13.png',...
    'D:\aaaLab\aaagraduate\SaveVideo\source\Calibration\RGB\14.png',...
    'D:\aaaLab\aaagraduate\SaveVideo\source\Calibration\RGB\16.png',...
    'D:\aaaLab\aaagraduate\SaveVideo\source\Calibration\RGB\17.png',...
    'D:\aaaLab\aaagraduate\SaveVideo\source\Calibration\RGB\18.png',...
    'D:\aaaLab\aaagraduate\SaveVideo\source\Calibration\RGB\19.png',...
    };
% Detect calibration pattern in images
detector = vision.calibration.monocular.CheckerboardDetector();
[imagePoints, imagesUsed] = detectPatternPoints(detector, imageFileNames);
imageFileNames = imageFileNames(imagesUsed);

% Read the first image to obtain image size
originalImage = imread(imageFileNames{1});
[mrows, ncols, ~] = size(originalImage);

% Generate world coordinates for the planar pattern keypoints
squareSize = 10;  % in units of 'millimeters'
worldPoints = generateWorldPoints(detector, 'SquareSize', squareSize);

% Calibrate the camera
[cameraParams, imagesUsed, estimationErrors] = estimateCameraParameters(imagePoints, worldPoints, ...
    'EstimateSkew', false, 'EstimateTangentialDistortion', true, ...
    'NumRadialDistortionCoefficients', 3, 'WorldUnits', 'millimeters', ...
    'InitialIntrinsicMatrix', [], 'InitialRadialDistortion', [], ...
    'ImageSize', [mrows, ncols]);

% View reprojection errors
h1=figure; showReprojectionErrors(cameraParams);

% Visualize pattern locations
h2=figure; showExtrinsics(cameraParams, 'CameraCentric');

% Display parameter estimation errors
displayErrors(estimationErrors, cameraParams);

% For example, you can use the calibration data to remove effects of lens distortion.
undistortedImage = undistortImage(originalImage, cameraParams);

% See additional examples of how to use the calibration data.  At the prompt type:
% showdemo('MeasuringPlanarObjectsExample')
% showdemo('StructureFromMotionExample')
