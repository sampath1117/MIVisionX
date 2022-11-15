
import csv

from re import A  

header = ['test case', 'Load     time', 'Decode   time','Process  time', 'Transfer time', 'Total time']
new_list, new_list1, new_list2, new_list3, new_list4, new_list5, tot_list = [], [], [], [], [], [], []
aug_list = ["rocalResize", "rocalCropResize", "rocalRotate", "rocalBrightness", "rocalGamma", "rocalContrast", "rocalFlip", "rocalBlur", "rocalBlend", "rocalWarpAffine", "rocalFishEye", "rocalVignette", "rocalVignette", "rocalSnPNoise", "rocalSnow", "rocalRain", "rocalColorTemp", "rocalFog", "rocalLensCorrection", "rocalPixelate", "rocalExposure", "rocalHue", "rocalSaturation", "rocalCopy", "rocalColorTwist", "rocalCropMirrorNormalize", "rocalCrop", "rocalResizeCropMirror", "No-Op"]
idx =0
aug_list1 = ["rocalResize", "rocalBrightness", "rocalGamma", "rocalContrast", "rocalFlip",  "rocalBlend", "rocalColorCast",  "rocalSnPNoise", "rocalSnow",  "rocalExposure",  "rocalColorTwist", "rocalCropMirrorNormalize", "rocalResizeMirror","ColorJitter", "rocalSpatter"]
print("length", len(aug_list1))
# input file name with extension
for file in aug_list:
	file_name = "./output_folder/"+file +".txt"
	try:

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
				new_list5.append( words[-1])

		# closing file after reading

		file_read.close()
		print("PKD3   ")
		# if length of new list is 0 that means
		# the input string doesn't
		# found in the text file
		if len(new_list)==0:
			print( "\ not found in \"" +file_name+ "\"!")
		else:

			# displaying the lines
			# containing given string
			lineLen = len(new_list)
			# print("\n**** Lines containing \"" +text+ "\" ****\n")
			# for i in range(lineLen):
			if new_list[-1] in aug_list1:
				tot_list.append([new_list[-1],new_list1[-1],new_list2[-1],new_list3[-1],new_list4[-1],new_list5[-1]])
			
		with open('tensor_check1.csv', 'w', encoding='UTF8') as f:
			writer = csv.writer(f)

			# write the header
			writer.writerow(header)

			# write the data
			for i in range(29):
				writer.writerow(tot_list[i])
	# entering except block
	# if input file doesn't exist



		# Names=[]
		# Values=[]

		# with open('aaa.csv','r') as csvfile:
		# 	lines = csv.reader(csvfile, delimiter=',')
		# 	for row in lines:
		# 		Names.append(row[0])
		# 		Values.append(int(row[-1]))
            
		# plt.scatter(Names, Values, color = 'g',s = 100)
		# plt.xticks(rotation = 25)
		# plt.xlabel('Augmentation')
		# plt.ylabel('Time in ms')
		# plt.title('Timing', fontsize = 20)
		# plt.show()

	except :
		print("\nThe file doesn't exist!")

