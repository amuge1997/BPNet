from numpy import *

def sigmoid(z):
    return 1/(1+exp(-z))

def parfunction_hiden(data,weight_h,weight_o,label):

    #样本个数
    dtlen=len(data)

    net_all_h=dot(data,weight_h)
    out_all_h=sigmoid(net_all_h)
    out_all_h=column_stack((out_all_h,[1. for i in range(dtlen)])    )

    net_all_o=dot(out_all_h,weight_o)
    out_all_o=sigmoid(net_all_o)

    #定义输入层到隐藏层的权重增量的个数
    par=array([[0. for i in range(len(weight_h[0]))]for i in range(len(weight_h))])

    for k in range(len(weight_h[0])):#k表示某行的第几列
         #隐藏层的第一个神经元
         for j in range(len(weight_h)):
            for data_x in range(dtlen):
                sum1=0
                for out_num in range(len(label[0])):
                    sum1+=-(label[data_x,out_num]-out_all_o[data_x,out_num])*weight_o[k,out_num]
                    
                par[j,k]+=   sum1     *out_all_h[data_x,k]*(1-out_all_h[data_x,k])*data[data_x,j]
        
    return par


def parfunction_out(data,weight_h,weight_o,label):

    dtlen=len(data)
    net_all_h=dot(data,weight_h)
    out_all_h=sigmoid(net_all_h)

    out_all_h=column_stack((out_all_h,[1. for i in range(dtlen)])    )

    net_all_o=dot(out_all_h,weight_o)
    out_all_o=sigmoid(net_all_o)
    
    #定义隐藏层到输出层的权重增量的个数
    par=array([[0.for i in range(len(weight_o[0]))]for i in range(len(weight_o))])

    for out_num in range(len(label[0])):#第out_num个输出神经元
        for j in range(len(weight_o)):
            for data_x in range(dtlen):
                par[j,out_num]+=-(label[data_x,out_num]-out_all_o[data_x,out_num])*out_all_h[data_x,j]
    return par
    

def cal(data,label,lr,epochs):
    #隐藏层神经元个数
    t1=5
    #输出层神经元个数
    t2=len(label[0])
    
    #增加偏置节点
    data=column_stack((data,[1 for i in range(len(data))])    )

    weight_h_new=array([[random.uniform(-1,1) for i in range(t1)]for i in range(len(data[0]))])
    weight_o_new=array([[random.uniform(-1,1) for i in range(t2)]for i in range(t1+1)])#加1是因为偏置节点
    
    stp=lr
    for k in range(epochs):
        weight_h=weight_h_new
        weight_o=weight_o_new
        par_h=parfunction_hiden(data,weight_h,weight_o,label)
        par_o=parfunction_out(data,weight_h,weight_o,label)
        weight_h_new=weight_h-stp*par_h
        weight_o_new=weight_o-stp*par_o

    # 输出隐藏层权重
    for i in range(len(weight_h_new)):
        print(list(weight_h_new[i]),',','\n')
    # 输出输出层权重
    for i in range(len(weight_o_new)):
        print(list(weight_o_new[i]),',','\n')

    return

if __name__ == '__main__':

    from 样本 import data,label
    cal(data,label,lr=0.1,epochs=1000)

































