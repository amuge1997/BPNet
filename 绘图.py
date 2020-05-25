from numpy import *
import matplotlib.pyplot as plt


# 给定样本
from 样本 import data,label



# 给定权重
weight_h_new = array([

[-1.1314743011148067, -2.00473547032765, -6.173965739518393, 5.176959309831678, 3.8476828026219403] ,

[-6.42835494458836, 3.4246904267168503, -0.40609694628722975, 1.0366853620973617, 0.04386732127863438] ,

[-5.779386616397211, -3.7019109489746373, 6.119532550470353, 14.496243340533393, -9.612628080258462] ,


])

weight_o_new=array([

    [10.686170491991662, -1.5591774506218077],

    [14.339683259458772, -2.300874621699876],

    [-13.814924505689811, -2.5538081525905123],

    [15.37019387335392, -3.2545577014723035],

    [-11.892742027783754, -1.8255159429493928],

    [-9.645912799173324, -3.7378402455011988],

])





def sigmoid(z):
    return 1/(1+exp(-z))



line_x_s=-7
line_x_e=7
line_y_s=-5
line_y_e=5

data_x=array([6,-1,1])
m1=dot(data_x,weight_h_new)
m1_o=sigmoid(m1)
m1_o=array([m1_o])
data_x2=column_stack((m1_o,[1.]))

m3=dot(data_x2,weight_o_new)
m3_o=sigmoid(m3)
print('point \'m3(%s,%s)\'\nprobability:\n1:%s\n2:%s' %(data_x[0],data_x[1],m3_o[0,0],m3_o[0,1]) )







size1=4
size2=50

stp_x=(line_x_e-line_y_s)/200
stp_y=(line_y_e-line_y_s)/(line_x_e-line_y_s)*stp_x

test=[arange(line_x_s,line_x_e,stp_x),arange(line_y_s,line_y_e,stp_y)]


color=[
    'red',
    'blue',
    'brown',
    'gray',
    'yellow',
    'black',
    'yellow',
    'green',

       ]







red=[[],[]]
blue=[[],[]]
brown=[[],[]]
gray=[[],[]]
for i in range(len(test[1])):
    for j in range(len(test[0])):
        
        m=array([test[0][j],test[1][i],1])
        m1=dot(m,weight_h_new)
        m1_o=sigmoid(m1)
        m1_o=array([m1_o])
        
        data_x2=column_stack((m1_o,[1.]))
        m3=dot(data_x2,weight_o_new)
        m3_o=sigmoid(m3)

        
        if m3_o[0,0]>=0.5 and m3_o[0,1]>=0.5:
            red[0].append(float(test[0][j]))
            red[1].append(float(test[1][i]))
        if m3_o[0,0]<=0.5 and m3_o[0,1]>=0.5:
            blue[0].append(float(test[0][j]))
            blue[1].append(float(test[1][i]))
        

        if m3_o[0,0]<=0.5 and m3_o[0,1]<=0.5:
            brown[0].append(float(test[0][j]))
            brown[1].append(float(test[1][i]))

            
        if m3_o[0,0]>=0.5 and m3_o[0,1]<=0.5:
            gray[0].append(float(test[0][j]))
            gray[1].append(float(test[1][i]))


plt.scatter(red[0],red[1],s=size1,marker='.',c='red')
plt.scatter(blue[0],blue[1],s=size1,marker='.',c='green')
plt.scatter(brown[0],brown[1],s=size1,marker='.',c='brown')
plt.scatter(gray[0],gray[1],s=size1,marker='.',c='gray')


yellow=[[],[]]
black=[[],[],]
white=[[],[]]
green=[[],[]]

for i in range(len(data)):
    if label[i,0]==label[i,1]==1:
        yellow[0].append(data[i,0])
        yellow[1].append(data[i,1])
    if label[i,0]==0 and label[i,1]==1:
        black[0].append(data[i,0])
        black[1].append(data[i,1])
    if label[i,0]==label[i,1]==0:
        white[0].append(data[i,0])
        white[1].append(data[i,1])
    if label[i,0]==1 and label[i,1]==0:
        green[0].append(data[i,0])
        green[1].append(data[i,1])

plt.scatter(yellow[0],yellow[1],s=size2,marker='.',c='yellow')
plt.scatter(black[0],black[1],s=size2,marker='.',c='black')
plt.scatter(white[0],white[1],s=size2,marker='.',c='white')
plt.scatter(green[0],green[1],s=size2,marker='.',c='blue')


'''
for i in range(len(data)):
    if label[i,0]==label[i,1]==1:
        plt.plot(data[i,0],data[i,1],'ro',label="point",c='yellow',marker='.')
    if label[i,0]==0 and label[i,1]==1:
        plt.plot(data[i,0],data[i,1],'ro',label="point",c='black',marker='.')
    if label[i,0]==label[i,1]==0:
        plt.plot(data[i,0],data[i,1],'ro',label="point",c='white',marker='.')
    if label[i,0]==1 and label[i,1]==0:
        plt.plot(data[i,0],data[i,1],'ro',label="point",c='green',marker='.')
'''

'''
#图像边界设定
plt.xlim(line_x_s-0.5,line_x_e+0.5)
plt.ylim(y_lim_l,y_lim_h)
'''
plt.show()



















