import matplotlib.pyplot as plt

x1=[]
y1=[]
x2=[]
y2=[]
y10 = [-2.6080591333000003, -2.440175935721985, -0.07536794459536811, 3.448264940110335, 7.014653885634058, 5.697399611806679, 3.6242626105719773]
y4 = [0.9140763166, 0.9717117628848145, 1.0350896183344427, 0.8795531414533548, 0.7196120994312158, 0.6789606488598479, 0.6837474343505701]
kind = 'flavor11'
path = '/Users/arlex/Documents/课程/初赛文档/statistics/TrainData_2015_summay.txt';
with open(path,'r') as data:
    lines = data.readlines()
for line in lines:
    params = line.split(' ')
    if params[1] == kind:
        x1.append(params[0][5:])
        y1.append(int(params[2]))

path = '/Users/arlex/Documents/课程/初赛文档/statistics/TestData_2015_summay.txt';
with open(path,'r') as data:
    lines = data.readlines()
for line in lines:
    params = line.split(' ')
    if params[1] == kind:
        x2.append(params[0][5:])
        y2.append(int(params[2]))

ax = plt.subplot()

ax.set_title(kind+' Test Diagram')# give plot a title
ax.set_xlabel('Date')# make axis labels
ax.set_ylabel('number')
ax.set_ylim([-5,10])
ax.set_xticklabels(x1+x2,rotation=90)
ax.plot(x1,y1,c='r')
ax.plot(x2,y2,c='g')
ax.plot(x2,y4,c='b')
ax.plot(x2,y10,c='y')

plt.plot(x1, y1,label='train data')
plt.plot(x2, y2,label='test data')
plt.plot(x2, y4,label='4 params data')
plt.plot(x2, y10,label='10 params data')

plt.legend(loc='upper left')

plt.show()