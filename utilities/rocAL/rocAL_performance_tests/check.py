import matplotlib.pyplot as plt 
import csv
import numpy as np
Names=[]
Values=[]
# Names
Values1 =[]
with open('/media/MivisionXTOT/MIVisionX/utilities/rocAL/rocAL_performance_tests/aaa.csv','r') as csvfile:
    lines = csv.reader(csvfile, delimiter=',')
    header = next(lines)
    if header != None:
        for row in lines:
            Names.append(row[0])
            Values.append(int(row[-1]))
    
with open('bbb.csv','r') as csvfile:
    lines = csv.reader(csvfile, delimiter=',')
    header = next(lines)
    if header != None:
        for row in lines:
            # Names.append(row[0])
            Values1.append(int(row[-1]))
print(Values)
print()
# print(Values1)
     
# plt.scatter(Names, Values, color = 'g',s = 100)
X_axis = np.arange(len(Names))

plt.bar(X_axis-0.2,Values, 0.4, label = 'Girls')
plt.bar(X_axis+0.2,Values1, 0.4, label = 'Girls')


# print("hello", values)
plt.xticks(X_axis, Names,rotation=90)

plt.xlabel('Augmentation')
plt.ylabel('Time in ms')
plt.title('Timing', fontsize = 20)
plt.savefig("/media/tensor/MIVisionX/utilities/rocAL/rocAL_performance_tests/img.jpg")