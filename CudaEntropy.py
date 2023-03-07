#for cuda
#always use float32 to calculate
from pycuda import driver
import pycuda
import pycuda.autoinit
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
import pycuda.cumath as gpumath
#end cuda

import numpy as np 



#defination of GPU function
GPUMod = SourceModule("""
//cuda
  // #include <thrust/sort.h>
  // #define _N1 3000
  // struct R{
  //   int ind;
  //   float val;
  // };
  // __host__ int comf (const void* a,const void* b)
  // {
  //   struct R* aa=(struct R*) a;
  //   struct R* bb=(struct R*) a;
  //   float d=aa->val-bb->val;
  //   return d>0?1:-1;
  // }
  // __device__ void Sort(float* Data, int N , int* R)
  // {
  //   thrust::sort_by_key(Data,Data+N,R)
  //   return;
  // }
__global__ void GPU_corr_s(int N_num,int T_num,int*Xtrain,float* result){
    //Xtrain:N_num*T_num(rank)
    //call block=N_num,N_num,1
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;
    if(idx>=N_num||idy>=N_num)
      return;
    if(idx>=idy)
      return;
    else 
    {
      int i;

      int  s=0;
      for (i = 0; i < T_num; i++)
      {
        //int d=(Ix+i)->ind-(Iy+i)->ind;
        int d=Xtrain[idx*T_num+i]-Xtrain[idy*T_num+i];
        s+= d*d;
      }
      float g=1.0;
      g-=(float)s*6/T_num/(T_num*T_num-1);
      result[idx*N_num+idy]=g;
    } 
    return;
  }

  __global__ void GPU_distance(int N_num,int T_num,float*Xtrain, float* result){
    //Xtrain:N_num*T_num
    //call block=N_num,N_num,1
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;
    if(idx>=N_num||idy>=N_num)
      return;
    if(idx>=idy)
      return;
    else
    {
        float sum=0;
        float dif;
        for(int i =0 ; i<T_num;i++){
            dif=Xtrain[idx*T_num+i]-Xtrain[idy*T_num+i];
            dif=dif*dif;
            sum+=dif;
        }
        sum=1/(sum+1);
        result[idx*N_num+idy]=sum;
        } 
    return;
  }
  __global__ void GPU_corr_k(int N_num,int T_num,float*Xtrain,float* result){
    //Xtrain:N_num*T_num
    //call block=N_num,N_num,1
    //https://blog.csdn.net/chenxy_bwave/article/details/126919019
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;
    if(idx>=N_num||idy>=N_num)
      return;
    if(idx>=idy)
      return;
    else 
    {
      int i;
      int j;
      //dealing Xtrain[idx,:] & Xtrain[idy,:]
      int c=0;
      int d=0;
      int tx=0;
      int ty=0;
      for (i = 0; i < T_num; i++)
      {
        for (j = i+1; j < T_num; j++)
        {
          float dx=Xtrain[idx*T_num+i]-Xtrain[idx*T_num+j];
          float dy=Xtrain[idy*T_num+i]-Xtrain[idy*T_num+j];
          if((dx*dy)>0)
           c+=1;
          else
          {
            if(dx*dy<0)
              d+=1;
            else
            {
              if(dx==0 && dy!=0)
                tx+=1;
              if(dx!=0 && dy==0)
                ty+=1;
            }
          }
        }
      }
      int fenm=(c+d+tx)*(c+d+ty);
      if(fenm<=0)        //div o is not allowed
         result[idx*N_num+idy]=1;
      else{
        float fenz=(float)(c-d);
        float div=(float)fenm;
        div=sqrt(div);
        float taob=fenz/div;
        result[idx*N_num+idy]=taob;
      }
    return;
    }
  }


    __global__ void GPU_corr_p(int N_num,int T_num,float*Xtrain, float* result){
    //Xtrain:N_num*T_num
    //call block=N_num,N_num,1
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;
    if(idx>=N_num||idy>=N_num)
      return;
    if(idx>=idy)
      return;
    else 
    {
      int i;
      //dealing Xtrain[idx,:] & Xtrain[idy,:]
      float fenzi   = 0;
      float fenmu_X = 0;
      float fenmu_Y = 0;

      float average_X = 0;
      float average_Y = 0;
      for (i = 0; i < T_num; i++)
      {
        average_X += Xtrain[idx*T_num+i];
        average_Y += Xtrain[idy*T_num+i];
      }
      average_X /= (float)T_num;
      average_Y /= (float)T_num;

      for (i = 0; i < T_num; i++)
      {
        fenzi   += (Xtrain[idx*T_num+i] - average_X)*(Xtrain[idy*T_num+i] - average_Y);
        fenmu_X += (Xtrain[idx*T_num+i] - average_X)*(Xtrain[idx*T_num+i] - average_X);
        fenmu_Y += (Xtrain[idy*T_num+i] - average_Y)*(Xtrain[idy*T_num+i] - average_Y);
      }
      //div o is not allowed
      float Pearson=0;
      if(fenmu_X==0 ||fenmu_Y==0)
        result[idx*N_num+idy]=1;
      else{
        Pearson = fenzi/(sqrt(fenmu_X)*sqrt(fenmu_Y));
        //Pearson=abs(Pearson);
        result[idx*N_num+idy]=Pearson;
        } 
    return;
    }
  }
  __device__ float GuassKernal(float lamda ,float x ,float y)
  {
    float e=x-y;
    float res=exp(-e*e/2/lamda/lamda)/lamda/2.506628274631000502415765284811;
    return res;
  }
  __global__  void GPU_ASE_t(int T_num,int X_num,int G_num,float* X_train,int* NodeExist, float* result){
    //tensor_X_train:Nnum*TNum
    //result Gnum+1 *T_num
    //block(T_num*G_num+1)
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;
    //int tid=idx*+idy;
    if (idx>=T_num)
        return;

    int N_num=X_num*G_num;
    if(idy<G_num){
        //dealing lev idy

        //first scaning all
        //STD
        int cnt=0;
        float sum = 0;
        float average = 0;
        float var = 0; 
        float standard = 0; 
        for (int i = idy*X_num; i < (idy+1)*X_num;i++){
          if(NodeExist[i]){
            //sum += X_train[i][idx];//求和
            sum += X_train[i*T_num+idx];//求和
            cnt++;
          }
        }
        average = sum / cnt;//求平均值
        for (int i = idy*X_num; i < (idy+1)*X_num;i++){
        if(NodeExist[i]){
            standard=X_train[i*T_num+idx]-average;
            var += standard*standard/cnt;//求方差
        }
        }
        standard =sqrt(var);//求标准差
        if(standard<=0 || cnt<=1)
            result[idy*T_num+idx]=0;
        else{
          sum=0;
          standard*=pow((float)cnt/4*3,-0.2);
          standard=standard*1.4142135623730950488016887242097;
          for (int i = idy*X_num; i < (idy+1)*X_num;i++){
          if(NodeExist[i]){
              for (int j = idy*X_num; j < (idy+1)*X_num;j++){
              if(NodeExist[j]){
                  sum+=GuassKernal(standard,X_train[i*T_num+idx],X_train[j*T_num+idx]);
              }
              }
          }
          }
          sum/=cnt*cnt;
          result[idy*T_num+idx]=-log(sum);
        }
    }
    else
    {
        //compute ASE_all
        //first scaning all
        //STD
        int cnt=0;
        float sum = 0;
        float average = 0;
        float var = 0; 
        float standard = 0; 
        for (int i = 0; i < N_num;i++){
          if(NodeExist[i]){
            //sum += X_train[i][idx];//求和
            sum += X_train[i*T_num+idx];//求和
            cnt++;
          }
        }
        average = sum / cnt;//求平均值
        for (int i = 0; i < N_num;i++){
        if(NodeExist[i]){

            standard=X_train[i*T_num+idx]-average;
            var += standard*standard/cnt;//求方差
        }
        }
        standard =sqrt(var);//求标准差
        if(standard<=0 | cnt<=1)
            result[G_num*T_num+idx]=0;
        else{
          sum=0;
          standard*=pow((float)cnt/4*3,-0.2);
          standard=standard*1.4142135623730950488016887242097;
          for (int i = 0; i < N_num;i++){
          if(NodeExist[i]){
              for (int j = 0; j < N_num;j++){
              if(NodeExist[j]){
                  sum+=GuassKernal(standard,X_train[i*T_num+idx],X_train[j*T_num+idx]);
              }
              }
          }
          }
          sum/=cnt*cnt;
          result[G_num*T_num+idx]=-log(sum);
        }
        //result[G_num*T_num+idx];
    }
    return;
}
__global__ void GPU_ASE(int T_num,int N_num,float* X_train, float* result){
    //tensor_X_train:Nnum*TNum
    //result   T_num
    //block(T_num*G_num+1)
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;
    idx=idx*idy;
    //int tid=idx*+idy;
    if (idx>=T_num)
        return;
    {
        //compute ASE_all
        //first scaning all
        //STD
        int cnt=0;
        float sum = 0;
        float average = 0;
        float var = 0; 
        float standard = 0; 
        for (int i = 0; i < N_num;i++){
            //sum += X_train[i][idx];//求和
            sum += X_train[i*T_num+idx];//求和
            cnt++;
        }
        average = sum / cnt;//求平均值
        for (int i = 0; i < N_num;i++){
            standard=X_train[i*T_num+idx]-average;
            var += standard*standard/cnt;//求方差
        }
        standard =sqrt(var);//求标准差
        if(standard<=0 | cnt<=1)
            result[idx]=0;
        else{
          sum=0;
          standard*=pow((float)cnt/4*3,-0.2);
          standard=standard*1.4142135623730950488016887242097;
          for (int i = 0; i < N_num;i++){
              for (int j = 0; j < N_num;j++){

                  sum+=GuassKernal(standard,X_train[i*T_num+idx],X_train[j*T_num+idx]);
              }
          }
          sum/=cnt*cnt;
          result[idx]=-log(sum);
        }

    }
    return;
}
//!cuda
  """)


def EntropyEstimater(X_train,X_num=50,T_num=3000):
  #X_train ndarrray[N_num*T_num]
  N_num=X_num
  X_train=X_train.astype(np.float32)
  X_train_gpu=gpuarray.to_gpu(X_train)


  result=np.zeros((T_num,1))
  result=result.astype(np.float32)
  GPUASE=GPUMod.get_function("GPU_ASE")
  #void GPU_ASE_t(int T_num,int X_num,int G_num,float* X_train,int* NodeExist, float* result)
  GPUASE(np.int32(T_num),np.int32(X_num),X_train_gpu,driver.Out(result),block=(16,4,1),grid=(int(T_num/256)+1,4,1))
  #block<=1024
  #print(result)
  result=np.mean(result,axis=0)
  #print(result)
  return result.astype(np.float64)

def SMEstimater(X_train,X_num=50,T_num=3000,method=0):
  #X_train ndarrray[N_num*T_num]
  N_num=X_num

  X_train=X_train.astype(np.float32)
  

  result=np.zeros((N_num,N_num))
  result=result.astype(np.float32)    
  #void GPU_corr_p(int N_num,int T_num,float*Xtrain, int* NodeExist,float* result)
  #void GPU_corr_s(int N_num,int T_num,float*Xtrain, int* NodeExist,float* result)
  #void GPU_distance(int N_num,int T_num,float*Xtrain,int* NodeExist, float* result)
  if(method==0):
    X_train_gpu=gpuarray.to_gpu(X_train)
    GPUfunc=GPUMod.get_function("GPU_distance")
    GPUfunc(np.int32(N_num),np.int32(T_num),X_train_gpu,driver.Out(result),block=(32,32,1),grid=(int(N_num/32)+1,int(N_num/32)+1,1))
  if(method==1):
    X_train_gpu=gpuarray.to_gpu(X_train)
    GPUfunc=GPUMod.get_function("GPU_corr_p")
    GPUfunc(np.int32(N_num),np.int32(T_num),X_train_gpu,driver.Out(result),block=(32,32,1),grid=(int(N_num/32)+1,int(N_num/32)+1,1))
  if(method==2):
    X_train_rank=np.zeros((N_num,T_num)).astype(np.int32)
    for i in range(N_num):
      X_train_rank[i:i+1,:]=np.argsort(X_train[i:i+1,:])
    X_train_gpu=gpuarray.to_gpu(X_train_rank)
    GPUfunc=GPUMod.get_function("GPU_corr_s")
    GPUfunc(np.int32(N_num),np.int32(T_num),X_train_gpu,driver.Out(result),block=(32,32,1),grid=(int(N_num/32)+1,int(N_num/32)+1,1))
  if(method==3):
    X_train_gpu=gpuarray.to_gpu(X_train)
    GPUfunc=GPUMod.get_function("GPU_corr_k")
    GPUfunc(np.int32(N_num),np.int32(T_num),X_train_gpu,driver.Out(result),block=(32,32,1),grid=(int(N_num/32)+1,int(N_num/32)+1,1))
  # else:
  #   print('err from GPU corr'+str(method))
  #GPUfunc(np.int32(N_num),np.int32(T_num),X_train_gpu,Node_exist_gpu,driver.Out(result),block=(32,32,1),grid=(int(N_num/32)+1,int(N_num/2)+1,1))
  #block<=1024
  #print(result)
  # for i in range(N_num):      
  #   for j in range(i+1,N_num):
  #     if(Node_exist[i] and Node_exist[j]):
  #                   if(isnan(Kc_array[i][j])):
  #                       Kc_array[i][j]=1
  #                       Kc_array[j][i]=1
  #               else:
  #                   Kc_array[i][j]=1e-10
  #                   Kc_array[j][i]=1e-10
  #           #matshow(Kc_array)
  #       return Kc_array  
  
  return result.astype(np.float64)
