echo This is my first shell script
folder=./
if test -d "$folder"; then 
    rm -rf ./output_folder
fi
mkdir output_folder

# <<<<<<<<<<<<<< BUILING PERFORMANCE UNITTEST >>>>>>>>>>>  

sudo rm -rvf build*
mkdir build
cd build || exit
cmake ..
make

# <<<<<<<<<<<<<< INITIALIZATION >>>>>>>>>>>  

cd build
INPUTPATH=$1
width=$2
height=$3
testcase=0
batch_size=$4
device=$5
rgb=1
shard_count=1
shuffle=0

# <<<<<<<<<<<<<< EXECUTION OF ALL FUNCTIONALITIES >>>>>>>>>>>>>>

echo arguments $INPUTPATH $width $height $testcase $batch_size $device $shard_count $shuffle

while [ $testcase -le 28 ]
do
    case "$testcase" in
   "0") file_name=rocalResize.txt
   ;;
   "1") file_name=rocalColorCast.txt
   ;;
   "2") file_name=rocalRotate.txt 
   ;;
   "3") file_name=rocalBrightness.txt
   ;;
   "4") file_name=rocalGamma.txt
   ;;
   "5") file_name=rocalContrast.txt 
   ;;
   "6") file_name=rocalFlip.txt
   ;;
   "7") file_name=rocalBlur.txt
   ;;
   "8") file_name=rocalBlend.txt 
   ;;
   "9") file_name=rocalWarpAffine.txt
   ;;
   "10") file_name=rocalFishEye.txt
   ;;
   "11") file_name=rocalVignette.txt 
   ;;
   "12") file_name=rocalJitter.txt
   ;;
   "13") file_name=rocalSnPNoise.txt
   ;;
   "14") file_name=rocalSnow.txt 
   ;;
   "15") file_name=rocalRain.txt
   ;;
   "16") file_name=rocalColorTemp.txt
   ;;
   "17") file_name=rocalFog.txt 
   ;;
   "18") file_name=rocalLensCorrection.txt
   ;;
   "19") file_name=rocalPixelate.txt
   ;;
   "20") file_name=rocalExposure.txt 
   ;;
   "21") file_name=rocalHue.txt
   ;;
   "22") file_name=rocalSaturation.txt
   ;;
   "23") file_name=rocalSpatter.txt 
   ;;
   "24") file_name=rocalColorTwist.txt
   ;;
   "25") file_name=rocalCropMirrorNormalize.txt
   ;;
   "26") file_name=rocalCrop.txt 
   ;;
   "27") file_name=rocalResizeMirrorNormalize.txt
   ;;
   "28") file_name=GridMask.txt
   ;;
   "29") file_name=ColorJitter.txt
   ;;
esac
    echo arguments $INPUTPATH $width $height $testcase $batch_size $device $rgb $shard_count $shuffle >> ./../output_folder/$file_name
    ./rocAL_performance_tests $INPUTPATH $width $height $testcase $batch_size $device $rgb $shard_count $shuffle >> ./../output_folder/$file_name

    testcase=$(($testcase+1))
done
echo End of my shell script