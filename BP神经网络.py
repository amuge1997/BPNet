import numpy as np

def sigmoid(z):
    return 1/(1+np.exp(-z))

def parfunction_hiden(data,weight_h,weight_o,label):

    #样本个数
    dtlen=len(data)

    net_all_h=np.dot(data,weight_h)
    out_all_h=sigmoid(net_all_h)
    out_all_h=np.column_stack((out_all_h,[1. for i in range(dtlen)])    )

    net_all_o=np.dot(out_all_h,weight_o)
    out_all_o=sigmoid(net_all_o)

    #定义输入层到隐藏层的权重增量的个数
    par=np.array([[0. for i in range(len(weight_h[0]))]for i in range(len(weight_h))])

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
    net_all_h=np.dot(data,weight_h)
    out_all_h=sigmoid(net_all_h)

    out_all_h=np.column_stack((out_all_h,[1. for i in range(dtlen)])    )

    net_all_o=np.dot(out_all_h,weight_o)
    out_all_o=sigmoid(net_all_o)
    
    #定义隐藏层到输出层的权重增量的个数
    par=np.array([[0.for i in range(len(weight_o[0]))]for i in range(len(weight_o))])

    for out_num in range(len(label[0])):#第out_num个输出神经元
        for j in range(len(weight_o)):
            for data_x in range(dtlen):
                par[j,out_num]+=-(label[data_x,out_num]-out_all_o[data_x,out_num])*out_all_h[data_x,j]
    return par
    

def cal(data,label,hiden,lr,epochs):
    #隐藏层神经元个数
    t1=hiden
    #输出层神经元个数
    t2=len(label[0])
    
    #增加偏置节点
    data=np.column_stack((data,[1 for i in range(len(data))])    )

    weight_h_new=np.array([[np.random.uniform(-1,1) for i in range(t1)]for i in range(len(data[0]))])
    weight_o_new=np.array([[np.random.uniform(-1,1) for i in range(t2)]for i in range(t1+1)])#加1是因为偏置节点
    
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

    weight_h_final = weight_h_new
    weight_o_final = weight_o_new
    return weight_h_final, weight_o_final

def predict_pro(X,h,o):
    # X.shape = ( n_sampes,n_dim )
    n_samples = X.shape[0]
    bais = np.ones((n_samples,1))
    Xbais = np.column_stack((X,bais))
    ho = sigmoid(np.dot(Xbais,h))
    hobais = np.column_stack((ho,bais))
    oo = sigmoid(np.dot(hobais,o))
    return oo

def predict(X,h,o):
    # X.shape = ( n_sampes,n_dim )
    preY_pro = predict_pro(X,h,o)
    preY = np.zeros(preY_pro.shape)
    for i,row in enumerate(preY_pro):
        preY[i,np.argmax(row)] = 1
    preY = preY.astype(np.uint8)
    return preY

if __name__ == '__main__':

    from 样本 import data,label
    h,o = cal(data=data,label=label,hiden=10,lr=0.1,epochs=1000)
    X = np.array([
        [-4,0],
        [-4,1],

        [-2,0],
        [-4.5,-0.5]
    ])
    oo = predict_pro(X,h,o)
    print(oo)
    oo = predict(X,h,o)
    print(oo)































