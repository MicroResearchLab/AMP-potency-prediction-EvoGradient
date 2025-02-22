
import os
import torch
import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.autograd import Variable
import math
from matplotlib.colors import Normalize
import warnings
warnings.filterwarnings("ignore")
import re



mydict = {'A':0,'C':1,'D':2,'E':3,'F':4,'G':5,'H':6,'I':7,'K':8,'L':9,'M':10,'N':11,'P':12,'Q':13,'R':14,'S':15,'T':16,'V':17,'W':18,'Y':19}
myInvDict = dict([val, key] for key, val in mydict.items())
sigmoid = torch.sigmoid


NBTdict = {'A':1,'C':2,'D':3,'E':4,'F':5,'G':6,'H':7,'I':8,'K':9,'L':10,'M':11,'N':12,'P':13,'Q':14,'R':15,'S':16,'T':17,'V':18,'W':19,'Y':20}



MAX_MIC = math.log10(8192)
max_mic_buffer = 0.1
My_MAX_MIC = math.log10(600)


def get_substring_indices(input_string):
    indices = []
    length = len(input_string)
    
    for i in range(length):
        for j in range(i+4,  min(i+10, length+1)):
            indices.append([i, j])
    
    return indices


def filterSubstring(grad:list, alpha:float, indexLs:list, seq:str):
    alpha_grad = [v if v>=alpha else 0 for v in grad]
    finalLs = []
    for i in indexLs:
        subGrad = alpha_grad[i[0]:i[1]]
        flag = subGrad.count(0)
        if flag <= int((i[1]-i[0])/2):
            subSeq = list(seq[i[0]:i[1]])
            for j in range(len(subSeq)):
                if subGrad[j] == 0:
                    subSeq[j] = 'X'
            if subSeq[0]!= 'X' and subSeq[-1]!= 'X':
                finalLs.append(''.join(subSeq))


    return finalLs


def filterSubstring2(grad:list, alpha:float, indexLs:list, seq:str):
    alpha_grad = [v if v>=alpha else 0 for v in grad]
    finalLs = []
    for i in indexLs:
        subGrad = alpha_grad[i[0]:i[1]]
        flag = subGrad.count(0)
        if flag <= int((i[1]-i[0])/2):
            subSeq = list(seq[i[0]:i[1]])
            for j in range(len(subSeq)):
                if subGrad[j] == 0:
                    subSeq[j] = 'X'
            if subSeq[0]!= 'X' and subSeq[-1]!= 'X':
                finalLs.append(''.join(subSeq))


    return finalLs


def find_motif(input_string,grad,alpha):
    result = get_substring_indices(input_string)
    ls = filterSubstring(grad,alpha=alpha, indexLs = result, seq=input_string)

    return ls


# # example
# input_string = "abcdesdkjlgf"
# grad = [.4,.6,.4,0,.4,.6,.4,0,0,.4,.6,.4,]
# result = get_substring_indices(input_string)
# ls = filterSubstring(grad,alpha = 0.4, indexLs = result, seq=input_string)
# print(ls)
# ls1 = find_motif(input_string=input_string,grad = grad,alpha=0.4)
# print(ls1)







def CosineSimilarity(tensor_1, tensor_2):
    tensor_1 = tensor_1.squeeze()
    tensor_2 = tensor_2.squeeze()

    normalized_tensor_1 = tensor_1 / tensor_1.norm(dim=-1, keepdim=True)
    normalized_tensor_2 = tensor_2 / tensor_2.norm(dim=-1, keepdim=True)
    return (normalized_tensor_1 * normalized_tensor_2).sum()
    
def seq2num(seq):
    
    seqlist = list(seq)

    length = len(seq)
    result = re.findall(r'[BJOUXZ]',seq)

    if result:
        return 

    else:
        numlist = [NBTdict[char.upper()] for char in seqlist]
        
        zeroPad = [0 for i in range(300-length)]
        zeroPad.extend(numlist)
        zeroPad = np.array(zeroPad)
        
        return zeroPad


def dataProcessPipeline(seq):
    # This function first converts the sequence into a sequence consisting of values from 0 to 19,
    # then applies one-hot encoding, followed by padding.
    # It also returns the padded sequence and the mask.
    #print('ori seq', seq)

    testest = seq
    num_seq = [mydict[character.upper()] for character in seq]

    seq = np.array(num_seq,dtype=int)
    len = seq.shape[0]
    torch_seq = torch.tensor(seq)

    if torch.sum(torch_seq[torch_seq<0])!=0:
        print(torch_seq[torch_seq<0])
        print('wrong seq:',seq)
        print(testest)

    onehotSeq = torch.nn.functional.one_hot(torch_seq,num_classes=20)
    pad = torch.nn.ZeroPad2d(padding=(0,0,0,100-len))
    mask = np.zeros(100,dtype = int)
    mask[len:]=1
    mask = torch.tensor(mask)
    pad_seq = pad(onehotSeq) 
    
    
    return pad_seq,mask


def num2onehot(array2d):
    result = torch.zeros_like(array2d)
    index = torch.argmax(array2d,dim = -1)
    for i in range(index.shape[0]):
        result[i,index[i]] = 1

    return result


class TrainDataset(Dataset):
    def __init__(self,data_path,transform = dataProcessPipeline):

        df = pd.read_csv(data_path,header=0)
        self.df = df
        self.seqs = list(self.df['sequence'])

        #print(self.seqs.shape)
        self.values = self.df['value']

        self.values[self.values>MAX_MIC] = MAX_MIC
        self.values = list(self.values)
        #print(self.labels.shape)
        self.transform = transform


    def __getitem__(self,idex):
        seq = self.seqs[idex]
        num_seq, mask = self.transform(seq)
        label = self.values[idex]


        return num_seq, mask, label, seq

    def __len__(self):
        return len(self.seqs)


class TestDataset(Dataset):
    def __init__(self,data_path,transform = dataProcessPipeline):
        self.df = pd.read_csv(data_path,header=0)
        self.seqs = self.df['Sequence']

        self.transform = transform


    def __getitem__(self,idex):
        seq = self.seqs[idex]
        num_seq, mask = self.transform(seq)

        return num_seq, mask, seq

    def __len__(self):
        return len(self.seqs)



class PositionalEncoding(nn.Module):
     def __init__(self, len, d_model=20, dropout=0):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(len, d_model)
        position = torch.arange(0, len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float()
                                * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        #pe = pe.unsqueeze(0).transpose(0, 1)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

     def forward(self, x):
        x = x + self.pe
        #x = x + self.pe[:,:x.size(0), :]
        return x






class AttentionNetwork(nn.Module):
    
    def __init__(self,batch_size=128,embedding_size=20,num_tokens=100,num_classes=1,num_heads=4):
        super(AttentionNetwork,self).__init__()
        self.pe = PositionalEncoding(len=num_tokens,d_model = embedding_size)
        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.num_tokens = num_tokens
        self.num_classes = num_classes
        self.num_heads = num_heads
        # self.hidden1 = 20
        self.hidden1 = 20
        self.hidden2 = 60
        self.hidden3 = 20
        self.dropout = 0.2


        self.relu = nn.ReLU()

        self.LN = nn.LayerNorm(normalized_shape = self.hidden1)
        self.fc1 = nn.Linear(self.embedding_size,self.hidden1)



        self.multihead_att = nn.MultiheadAttention(embed_dim=self.hidden1,num_heads = self.num_heads,batch_first=1,dropout=self.dropout)

        self.flatten = nn.Flatten()
        self.fc2 = nn.Linear(self.hidden1*self.num_tokens,self.hidden2)
        self.fc3 = nn.Linear(self.hidden2,self.hidden3)
        self.new_fc4 = nn.Linear(self.hidden3,self.num_classes)

        self.dropout = nn.Dropout(self.dropout)
        self.softmax = nn.functional.softmax



    def forward(self,x,mask):
        x = self.pe(x)
        x = self.fc1(x)


        mask = mask.to(torch.bool)
        x, w1= self.multihead_att.forward(x,x,x,key_padding_mask=mask)

        x = self.flatten(x)
        x = self.fc2(x)
        x = self.dropout(x)
        
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.new_fc4(x)


        return x


pe = PositionalEncoding(len=100,d_model = 20)
model_list = {
    # change the path to the your model path
    'myAttention': './model/regression/Attention.pth',

}
test_model_list = model_list

opt_seqls = [
# put your sequence there
'LWWRKAKWKRKIAKRMIRVIGAAKI',                                                                                                                            
'KWLGAFGKMRKIAIRLRLKRKKAF',                                                                                                                             
'WWRLWKTLLKAPKKLTGLRRW',                                                                                                                             
'RKLKKLRWRAGMMYKYVKLK',
'MRFPWKHWWKKWKWWWKKKR',
'MKKARFWWWVAWKKLLRKKA'
]


df_ls = []

norm = Normalize(vmin=-0.2, vmax=0.2)
for seq in opt_seqls:
    tseq = seq

    ModelNameList = model_list.keys()

    gradient = {}
    

    oriseq = tseq


    df = pd.DataFrame(columns = ['Sequence','Length','label'])
    items = [{'Sequence':oriseq,'Length':len(oriseq)}]
    # df = df.append(items,ignore_index = 1)
    df = df._append(items,ignore_index = 1) # try df._append() if your pandas version does not support append()
    df.to_csv(oriseq+'.csv',index = False)


    SeqPath = oriseq+'.csv'




    for modelName in ModelNameList:

        iternum = 0

        testData = TestDataset(data_path = SeqPath)
        test_loader = DataLoader(dataset=testData, batch_size=1)
        attmodel = torch.load(model_list[modelName])


        attmodel.cuda()
        attmodel.zero_grad()



            
        for data in test_loader: 
            resultList = []
            # ensamble_values = []
            resultSeq = [oriseq]
            outMIC = []
            # attmodel.zero_grad()
            inputs,masks, seqs = data

            inputs = inputs.float()
            masks = masks.float()
            

            masks = masks.cuda()
            print(seqs[0])


            attmodel.eval()

            attmodel.zero_grad()
            stdev_spread = 0.1
            n_samples = 25
            x = inputs[0].detach()
            stdev = stdev_spread * (torch.max(x)-torch.min(x))
            x = x.numpy() 
            total_grad = np.zeros_like(x)
            for i in range(n_samples):

                    # final.append(xx)
                noise = np.random.normal(0,stdev,x.shape).astype(np.float32)
                x_plus_noise = x + noise
                
                x_plus_noise = torch.from_numpy(x_plus_noise).cuda()
                # x_plus_noise = x_plus_noise + 0.1
                x_plus_noise[masks[0]==1] = 0
                x_plus_noise = Variable(torch.unsqueeze(x_plus_noise,dim=0), requires_grad = True)
                # final.append(x_plus_noise)
                if i==0:
                    x_plus_noise = Variable(inputs.cuda(), requires_grad = True)
            
                if modelName == 'lstm_att' or modelName == 'RCNN'or modelName == 'CNN' or modelName == 'Transformer':
                    out = attmodel(x_plus_noise)
                else:
                    out = attmodel(x_plus_noise,masks)

                out = out.cpu()
                if 'RCNN' in modelName:
                    out = out.unsqueeze(0)

                conloss = -out

                conloss.backward()
                grad = x_plus_noise.grad
                # if i==0:
                #     print(grad)
                colindex = masks[0]==1
                grad[0][masks[0]==1] = 0
                grad = grad[0].cpu().numpy()
                
                total_grad += grad


            avg_grad = total_grad/n_samples
            myIndex = [mydict[v] for v in seqs[0]]

            mylen = 100-colindex.sum()
            grad = avg_grad[:mylen]
            mag = np.sum(abs(grad),axis=-1).tolist()


            value = [grad[k,myIndex[k]] for k in range(len(seqs[0]))]
 
            grad = [mag[k]*value[k] for k in range(len(value))]

            absGrad = np.sum(abs(np.array(grad)))
            grad = [v/absGrad for v in grad]

            gradient[modelName] = grad

    
    # trying different alpha
    # seq_dict = {'sequence': seq,'model':label_dict[seq]}
    # for k in range(11):
    #     alpha = k/100
    #     ls = find_motif(tseq,grad,alpha)
    #     print(tseq,':',ls)
    #     print(sum(abs(np.array(grad))))
    #     seq_dict[alpha] = ls

    # df_ls.append(seq_dict) # 
    
    os.remove(SeqPath) # clean temp files

    print(gradient)
    grad = gradient['myAttention']
    alpha = 0.02
    ls = find_motif(tseq,grad,alpha)
    print(tseq,':',ls)
    # print(sum(abs(np.array(grad))))
