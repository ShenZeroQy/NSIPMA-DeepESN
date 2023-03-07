from matplotlib.axis import Axis
#from matplotlib import projections
from numpy import *
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec

from enum import Enum
import os

# align these pram from runner before analying

REP=20
Node_numb=100
N_num=50
G_num=4

Me='ss'
PreEVAdir='./SS/EVA/exp0/'
PreREPdir='./SS/REP/exp0/'
Outdir='./SS/OUT/exp0/'
#for matplot GUI
def inverse(temp, position):
    pass


#def ce

#for  fname


class SingleLineFName(Enum): 
    ASE=0
    NRMSE_test=1
    NRMSE_train=2
    MAX_corr=3
class MultiLineFName(Enum):    
    EVANodeStack=0
class EVAFName(Enum): 
    EVAASE=0
    EVANRMSE_test=1
    EVANRMSE_train=2
class REPFName(Enum): 
    REPASE=0
    REPNRMSE_test=1
    REPNRMSE_train=2

class MethodGUIName(Enum): 
    ED =0
    PC =1
    SC =2
    KC =3
    # Random=4    
#https://zhuanlan.zhihu.com/p/65220518
class PlotColor(Enum): 
    blue=0
    deeppink=1
    firebrick=5
    darkgray=2
    aquamarine=4
    gold=3
    lemonchiffon=105
    beige=6
    bisque=7
    black=8
    blanchedalmond=9

__font_size=20


def AverlizeOVerall(ldir):
    #overall

    outdir=ldir+'aveg/'  
    try:
        os.makedirs(outdir)
        print (outdir+'创建成功')
    except:
        print(outdir+'already exist!')
    
    for fname in EVAFName:
        Lines=list()
        for i in range(REP):
            indir=ldir+str(i)+'/'
            inname=fname.name+'.txt'
            L=loadtxt(indir+inname,dtype=double, delimiter=',')
            Lines.append(L)
        avegLines=mean(Lines,axis=0)
        stdLines=std(Lines,axis=0)
        savetxt(outdir+'aveg'+inname,avegLines,fmt='%f', delimiter=',')
        savetxt(outdir+'std'+inname,stdLines,fmt='%f', delimiter=',')
        print(outdir+'a-s'+inname+'wrote')
    for fname in MultiLineFName:
        Lines=list()
        for i in range(REP):
            indir=ldir+str(i)+'/'
            inname=fname.name+'.txt'
            L=loadtxt(indir+inname,dtype=int, delimiter=',')
            Lines.append(L)
        avegLines=mean(Lines,axis=0)
        stdLines=std(Lines,axis=0)
        savetxt(outdir+'aveg'+inname,avegLines,fmt='%f', delimiter=',')
        savetxt(outdir+'std'+inname,stdLines,fmt='%f', delimiter=',')
        print(outdir+'a-s'+inname+'wrote')

def LayerMerge(ldir):
    txtoutdir=ldir+'aveg/'
    for fname in SingleLineFName:
        inname=fname.name+'.txt'
        SDA=list()
        for i in range(REP):    
            figoutdir=ldir+str(i)+'/out/'
            try:
                os.makedirs(figoutdir)
                print (figoutdir+'创建成功')
            except:
               pass
                
            
            #layer by layer  
            Lines=list()    
            for l in range(9):
                indir=ldir+str(i)+'/'+'layer'+str(l)+'/'
                L=loadtxt(indir+inname,dtype=double, delimiter=',')
                Lines.append(L)
            #all layer done
        
            #To One Lines
            Lc=Lines[0]
            for k in range(1,len(Lines)):
                Lc=concatenate((Lc,Lines[k]),axis=0)
            SDA.append(Lc)
            #draw Lc
            
            # fig=plt.figure(1,figsize=(10,5))
            # fig.clear()
            # axia=fig.gca()
            # xoffset=0
            # pi=0
            # limmax=L.max()
            # limmin=L.min()
            # tick=list()
            # ticklabel=list()
            # tick.append(0)
            # ticklabel.append(str(Node_numb))
            # ticklayer=['']+ [j+1 for  j in range(G_num)]
            # for L in Lines:
            #     xt= [xoffset+j for  j in range(len(L))]
            #     tick.append(xt[-1])
            #     if(pi==G_num-1):
            #         ticklabel.append(str(xoffset+Node_numb-len(L)))
            #     else:
            #         ticklabel.append(str(xoffset+Node_numb-len(L))+'\n(+'+str(Node_numb)+')')
            #     axia.plot(xt,L,label='layer'+str(pi))
            #     # axia.set_ylim(limmin,limmax)
                
            #     xoffset=xoffset+Node_numb-len(L)
            #     pi=pi+1
            # axia.set_xticks(tick)
            # axia.set_xticklabels(ticklabel)
            # axia.set_xlabel('Total neuron number')
            # axia.set_ylabel(fname.name)

            # secax = axia.secondary_xaxis('top')
            # secax.set_xticklabels(ticklayer)
            # secax.set_xlabel('Layer number')
            # plt.savefig(figoutdir+fname.name+'.eps')
            # fig.show()
            # plt.close(fig)
        avegSDA=mean(SDA,axis=0)
        stdSDA=std(SDA,axis=0)
        savetxt(txtoutdir+'aveg'+inname,avegSDA,fmt='%f', delimiter=',')
        savetxt(txtoutdir+'std'+inname,stdSDA,fmt='%f', delimiter=',')
    
def DrawNodeStack4(ldir,fz=(3.8,10),fs=12):
    #allocate storage 
    #assign name deallina ASE
    rin= MultiLineFName(0).name
    rifn=rin+'.txt'

    plt.rcParams['font.size'] = fs
    #compare different methods
    fig=plt.figure(1,figsize=fz)
    #help for drowing:https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.errorbar.html?highlight=errorbar#matplotlib.axes.Axes.errorbar
    fig.clear()
    #generage alabar
    #draw 10%
    alabarx=[str(i+1) for i in range(G_num)]
    alabary=['0%','10%','20%','30%','40%','50%','60%','70%','80%','90%']
    alabary=['0','10','20','30','40','50','60','70','80','90']
    #alabary=['0%','20%','40%','60%','80%']
    # for i in range(G_num):
    #     alabarx.append(str(i+1))
    #alabarx=alabarx.to_array()
    
    for rj in MethodGUIName:

        fname=ldir+'m'+str(rj.value)+'/aveg/aveg'+rifn
        M=loadtxt(fname,dtype=double, delimiter=',').T

        axes = fig.add_subplot(4,1,rj.value+1)
        
        im=axes.matshow(M,cmap=plt.cm.Reds)
        if(rj.value==2):
            rm=im
        axes.xaxis.set_major_locator(plt.MaxNLocator(10))
        axes.set_xticklabels(['']+alabary) 
        axes.yaxis.get_major_locator()
        axes.set_ylabel('('+MethodGUIName(rj.value).name+')      ',rotation=0, x=-0.2,y=0.5,size=fs)
        axes.set_yticklabels(['']+alabarx) 
        axes.tick_params(labelsize= fs)
        # divider = make_axes_locatable(axes)
        # cax = divider.append_axes("right", size="5%", pad=0.05)
        # plt.colorbar(im, cax=cax)
    plt.subplots_adjust(bottom=0.05, left=0,right=0.96, top=0.95,hspace=0.35)
    cax = plt.axes([0.82, 0.05, 0.025, 0.9])
    plt.colorbar(rm ,cax=cax)
    plt.savefig(ldir+'/'+'_Evaluation'+rin+'.eps',bbox_inches ='tight') 
    plt.savefig(Outdir+'/'+Me+'_Evaluation'+rin+'.eps',bbox_inches ='tight') 
    plt.show()
    plt.close(plt.figure(1))

def AverlizeRep(ldir):
    #only aveg
    outdir=ldir+'aveg/'  
    try:
        os.makedirs(outdir)
        print (outdir+'创建成功')
    except:
        print(outdir+'already exist!')
    
    for ri in REPFName:
        rin=ri.name+'.txt'
        eva_mean=zeros((10,))
        eva_std=zeros((10,))

        for p in range(10):
            dir=ldir+str(p)+'/'
            print('loading:'+dir)
            repval=loadtxt(dir+rin,dtype=double, delimiter=',')
            eva_mean[p]=mean(repval)
            eva_std[p]=std(repval)
        savetxt(outdir+'aveg'+rin,eva_mean,fmt='%f', delimiter=',')
        savetxt(outdir+'std'+rin,eva_std,fmt='%f', delimiter=',')
        print(outdir+'a-s'+rin+'wrote')
    return  


def RoughPare():
    pass
def Load_eva_aveg(ldir,meind:int):
    dir=ldir+'m'+str(meind)+'/aveg/'        
    print('loading:'+dir)
    ase=loadtxt(dir+'aveg'+EVAFName(0).name+'.txt',dtype=double, delimiter=',').T 
    test=loadtxt(dir+'aveg'+EVAFName(1).name+'.txt',dtype=double, delimiter=',').T  
    train=loadtxt(dir+'aveg'+EVAFName(2).name+'.txt',dtype=double, delimiter=',').T    
    return  test,train,ase
def Load_rep_aveg(ldir):
    
    dir=ldir+'/aveg/'        
    print('loading:'+dir)
    ase=loadtxt(dir+'aveg'+REPFName(0).name+'.txt',dtype=double, delimiter=',').T 
    test=loadtxt(dir+'aveg'+REPFName(1).name+'.txt',dtype=double, delimiter=',').T  
    train=loadtxt(dir+'aveg'+REPFName(2).name+'.txt',dtype=double, delimiter=',').T    
    return  test,train,ase
def photo():
    pind=PecPreindex(Node_num)
    return
def CmpNRMSE_ASE():
    fs=20
    rtr,rte,ra=Load_rep_aveg(PreREPdir)
    ete0,etr0,ea0=Load_eva_aveg(PreEVAdir,0)
    ete1,etr1,ea1=Load_eva_aveg(PreEVAdir,1)
    ete2,etr2,ea2=Load_eva_aveg(PreEVAdir,2)
    ete3,etr3,ea3=Load_eva_aveg(PreEVAdir,3)
    ete4,etr4,ea4=Load_eva_aveg(PreEVAdir,4)
    #pare NrMse te
    fig=plt.figure(1,  figsize=(12, 8)) 
    axia=plt.gca()
    axia.plot(range(10),rtr,color='red', linewidth=1.0, linestyle='-', label='Unpruned DeepESN',marker='^',markersize=10)
    axia.plot(range(10),ete0,color='blue', linewidth=1.0, linestyle='-', label='ED-IPMA',marker='*',markersize=10)
    axia.plot(range(10),ete1,color='orange', linewidth=1.0, linestyle='-', label='PC-IPMA',marker='h',markersize=10)
    axia.plot(range(10),ete2,color='gray', linewidth=1.0, linestyle='-', label='SC-IPMA',marker='X',markersize=10)
    axia.plot(range(10),ete3,color='pink', linewidth=1.0, linestyle='-', label='KC-IPMA',marker='p',markersize=10)
    axia.plot(range(10),ete4,color='indigo', linewidth=1.0, linestyle='-', label='IPMA',marker='s',markersize=10)
    axia.set_xlabel('Neuron number',fontsize= fs)
    axia.set_ylabel('$NRMSE_{test}$',fontsize= fs)
    axia.set_ylim(0.12,0.42)
    
    #set font
    
    #axia.set_title('$NRMSE_{test}$ comparation with M='+str(Node_num),fontsize= fs)
    axia.xaxis.set_major_locator(plt.MaxNLocator(10))
    axia.set_xticklabels(['','400' ,'360','320','280','240','200','160','120','80','40'])
    axia.tick_params(labelsize= fs) # 设置坐标轴上刻度的字体大小
    axia.legend(loc=0, fontsize =  fs) # 显示图例，loc=0表示图例会根据图片情况自动摆放
    plt.savefig(Outdir+'/'+Me+'_Compare_NRMSE_test.eps',bbox_inches ='tight')
    fig.show()

    #pare NrMse tr
    fig=plt.figure(2,figsize=(12, 8)) 
    axia=plt.gca()
    axia.plot(range(10),rte,color='red', linewidth=1.0, linestyle='-', label='Unpruned DeepESN',marker='^',markersize=10)
    axia.plot(range(10),etr0,color='blue', linewidth=1.0, linestyle='-', label='ED-IPMA',marker='*',markersize=10)
    axia.plot(range(10),etr1,color='orange', linewidth=1.0, linestyle='-', label='PC-IPMA',marker='h',markersize=10)
    axia.plot(range(10),etr2,color='gray', linewidth=1.0, linestyle='-', label='SC-IPMA',marker='X',markersize=10)
    axia.plot(range(10),etr3,color='pink', linewidth=1.0, linestyle='-', label='KC-IPMA',marker='P',markersize=10)
    axia.plot(range(10),etr4,color='indigo', linewidth=1.0, linestyle='-', label='IPMA',marker='s',markersize=10)
    axia.set_xlabel('Neuron number',fontsize= fs)
    axia.set_ylabel('$NRMSE_{train}$',fontsize= fs)
    axia.set_ylim(0,0.4)

    #axia.set_title('$NRMSE_{train}$ comparation with M='+str(Node_num),fontsize= fs)
    axia.tick_params(labelsize= fs) # 设置坐标轴上刻度的字体大小
    axia.xaxis.set_major_locator(plt.MaxNLocator(10))
    axia.set_xticklabels(['','400' ,'360','320','280','240','200','160','120','80','40'])
    axia.legend(loc=0, fontsize =  fs) # 显示图例，loc=0表示图例会根据图片情况自动摆放
    plt.savefig(Outdir+'/'+Me+'_Compare_NRMSE_train.eps',bbox_inches ='tight')
    fig.show()

    #pare ASE

    fig=plt.figure(3,figsize=(12, 8)) 
    axia=plt.gca()
    axia.plot(range(10),ra,color='red', linewidth=1.0, linestyle='-', label='Unpruned DeepESN',marker='^',markersize=10)
    axia.plot(range(10),ea0,color='blue', linewidth=1.0, linestyle='-', label='ED-IPMA',marker='*',markersize=10)
    axia.plot(range(10),ea1,color='orange', linewidth=1.0, linestyle='-', label='PC-IPMA',marker='h',markersize=10)
    axia.plot(range(10),ea2,color='gray', linewidth=1.0, linestyle='-', label='SC-IPMA',marker='X',markersize=10)
    axia.plot(range(10),ea3,color='pink', linewidth=1.0, linestyle='-', label='KC-IPMA',marker='p',markersize=10)
    axia.plot(range(10),ea4,color='indigo', linewidth=1.0, linestyle='-', label='IPMA',marker='s',markersize=10)
    axia.set_xlabel('Neuron number',fontsize= fs)
    axia.set_ylabel('ASE',fontsize= fs)
    axia.set_ylim(-0.1,0.4)
    #axia.set_title('ASE comparation with M='+str(Node_num),fontsize= fs)
    axia.xaxis.set_major_locator(plt.MaxNLocator(10))
    axia.set_xticklabels(['','400' ,'360','320','280','240','200','160','120','80','40'])
    axia.tick_params(labelsize= fs) # 设置坐标轴上刻度的字体大小
    axia.legend(loc=0, fontsize =  fs) # 显示图例，loc=0表示图例会根据图片情况自动摆放
    plt.savefig(Outdir+'/'+Me+'_Compare_ASE.eps',bbox_inches ='tight')
    plt.show()
    ens=1

if __name__ == '__main__':
    for m in range(5):
        Lredir=PreEVAdir+'m'+str(m)+'/'
        AverlizeOVerall(Lredir)
        LayerMerge(Lredir)
    DrawNodeStack4(PreEVAdir,fz=(5.6,8),fs=16)
    # DrawNodeStack(PreEVAdir,fz=(4.2,10))
    AverlizeRep(PreREPdir)
    # RoughPare()
    CmpNRMSE_ASE()
    end=1
   