import os
import csv

path=os. getcwd()
print(path)

# #<<<<<<<<<<<<<<<<<<<RPP<<<<<<<<<<<<<<<<<<<<<<<
os.system("git clone https://github.com/sampath1117/rpp.git")
os.chdir('./rpp/')
os.system("git checkout sr/ibs_cmn_fix")
os.system('mkdir build')
os.chdir('./build/')
os.system('cmake ..')
os.system('sudo make -j21 install')
os.chdir(path)

# <<<<<<<<<<<<<<<<<<<<<image mivisionx<<<<<<<<<<<<<<<<<<

os.system("mkdir MivisionxTOT")
os.chdir('./MivisionxTOT/')
os.system("pwd")
os.system("git clone https://github.com/fiona-gladwin/MIVisionX.git")
os.chdir('./MIVisionX/')
os.system("git checkout rocal_image_perf")
os.system('mkdir build')
os.chdir('./build/')
TOT_build1= os.getcwd()
os.system('cmake ..')
os.system('sudo make -j21 install')
os.chdir("./../utilities/rocAL/rocAL_performance_tests/")
os.system('mkdir build')
os.chdir('./build/')
os.system('cmake ..')
os.system('make')
TOT_build2 = os.getcwd()
os.chdir("./../")
TOT_path= os.getcwd()
os.chdir(path)

# >>>>>>>>tensor branch <<<<<<<<<<<<<<<<<<<<<<

os.system("mkdir MivisionxTensor")
os.chdir('./MivisionxTensor/')
os.system("pwd")
os.system("git clone https://github.com/fiona-gladwin/MIVisionX.git")
os.chdir('./MIVisionX/')
os.system("git checkout rocal_tensor_perf")
os.system('mkdir build')
os.chdir('./build/')
Tensor_build1=os.getcwd()
os.system('cmake ..')
os.system('sudo make -j21 install')
os.chdir("./../utilities/rocAL/rocAL_performance_tests/")
os.system('mkdir build')
os.chdir('./build/')
os.system('cmake ..')
os.system(' make ')
Tensor_build2 = os.getcwd()
os.chdir("./../")
Tensor_path = os.getcwd()
os.chdir(path)

bs=[32,64,128,256]
device_type=[0,1]

for image_type in range(2):
    for batch_size in bs:
        for d_type in device_type:
            PIPELINE="image"
            
            if(image_type==0):
                os.chdir(TOT_build1)
                os.system('sudo make -j21 install')
                os.chdir(TOT_build2)
                os.system('make')
                os.chdir(TOT_path)
                os.system("./performance_testing.sh '/media/sample_test/coco/val2017_10_images/' 300 300 "+str(batch_size) +" "+str(d_type))
                print("./performance_testing.sh '/media/sample_test/coco/val2017_10_images/' 300 300 "+str(batch_size)+" " +str(d_type))
                print(TOT_path)

            if(image_type==1):
                os.chdir(Tensor_build1)
                os.system('sudo make -j21 install')
                os.chdir(Tensor_build2)
                os.system('make')
                os.chdir(Tensor_path)
                os.system("./performance_tensor.sh '/media/sample_test/coco/val2017_10_images/' 300 300 "+ str(batch_size)+ " "+str(d_type))
                PIPELINE="tensor"

            os.chdir(path)

            header = ['test case', 'Load     time', 'Decode   time','Process  time', 'Transfer time', 'Total time']
            new_list, new_list1, new_list2, new_list3, new_list4, new_list5, tot_list = [], [], [], [], [], [], []
            aug_list = ["rocalResize", "rocalCropResize", "rocalRotate", "rocalBrightness", "rocalGamma", "rocalContrast", "rocalFlip", "rocalBlur", "rocalBlend", "rocalWarpAffine", "rocalFishEye", "rocalVignette", "rocalVignette", "rocalSnPNoise", "rocalSnow", "rocalRain", "rocalColorTemp", "rocalFog", "rocalLensCorrection", "rocalPixelate", "rocalExposure", "rocalHue", "rocalSaturation", "rocalCopy", "rocalColorTwist", "rocalCropMirrorNormalize", "rocalCrop", "rocalResizeCropMirror", "No-Op"]
            aug_list1 = ["rocalResize", "rocalBrightness", "rocalGamma", "rocalContrast", "rocalFlip",  "rocalBlend",  "rocalSnPNoise",  "rocalExposure",  "rocalColorTwist", "rocalCropMirrorNormalize"]
            op_paths = [path+"/MivisionxTOT/MIVisionX/utilities/rocAL/rocAL_performance_tests//output_folder/",path+"/MivisionxTensor/MIVisionX/utilities/rocAL/rocAL_performance_tests//output_folder/"]
            print(op_paths[0])
            for file in aug_list:
                file_name = op_paths[image_type]+file +".txt"
                if(os.path.exists(file_name)):
                    # opening and reading the file
                    file_read = open(file_name, "r")
                    lines = file_read.readlines()
                    
                    for line in lines:
                        
                        if header[0] in line:
                            words= line.split()
                            a=int(words[-1])
                            new_list.append(aug_list[a])

                        if header[1] in line:
                            words= line.split()
                            new_list1.append( words[-1])
                        
                        if header[2] in line:
                            words= line.split()
                            new_list2.append( words[-1])


                        if header[3] in line:
                            words= line.split()
                            new_list3.append( words[-1])

                        if header[4] in line:
                            words= line.split()
                            new_list4.append( words[-1])

                        if header[5] in line:
                            words= line.split()
                            words[-1]=int(words[-1])/1000000
                            new_list5.append( words[-1])

                    # closing file after reading

                    file_read.close()
                    if len(new_list)==0:
                        print( "\ not found in \"" +file_name+ "\"!")
                    else:
                        lineLen = len(new_list)
                        if new_list[-1] in aug_list1:
                            tot_list.append([new_list[-1],new_list1[-1],new_list2[-1],new_list3[-1],new_list4[-1],new_list5[-1]])
                        print(tot_list)
                        
                with open('tensor_performance_'+str(batch_size)+'_'+str(d_type)+'_'+$PIPELINE+'.csv', 'w', encoding='UTF8') as f:
                    writer = csv.writer(f)

                    # write the header
                    writer.writerow(header)

                    # write the data
                    for i in range(len(tot_list)):
                        writer.writerow(tot_list[i])