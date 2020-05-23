from numpy import *
import matplotlib.pyplot as plt


# 给定样本
from 样本 import data,label



# 给定权重
weight_h_new = array([
[3.195813606702313, 1.4249719537029764, -4.127983181837761, 3.7872028348131783, 4.164677343989234] ,

[-0.06506926202983794, 6.271413041509171, -0.36074735776525696, 0.6342170847768597, -3.8916480699450084] ,

[-8.766133694591428, 6.534590811229829, 3.1055866596108976, 10.022649923089597, 5.754547968909948] ,
])

weight_o_new=array([
    [-6.620641481323172, -1.723886310029632] ,

    [-6.421045660505503, -2.14673063715942] ,

    [-5.417403201280019, -1.1180828643520453] ,

    [9.893148863737832, -2.668834819675197] ,

    [-7.029647068203273, -1.1986979312071484] ,

    [5.356108696833903, -4.977391890148404] ,
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



















