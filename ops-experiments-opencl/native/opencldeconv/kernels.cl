#pragma OPENCL EXTENSION cl_khr_fp64 : enable                    
__kernel void vecAdd(  __global float *a,                       
                       __global float *b,                       
                       __global float *c,                       
                       const unsigned long n)                    
{                                                               
    //Get our global thread ID                                  
    int id = get_global_id(0);                                  
                                                                
    //Make sure we do not go out of bounds                      
    if (id < n)                                                 
        c[id] = a[id] + b[id];                                  
}                                                               
                                                                 
                                      
#pragma OPENCL EXTENSION cl_khr_fp64 : enable                    
__kernel void vecComplexMultiply(  __global float *a,                       
                       __global float *b,                       
                       __global float *c,                       
                       const unsigned long n)                    
{                                                               
    //Get our global thread ID                                  
    int id = get_global_id(0);                                  
                                                                
    //Make sure we do not go out of bounds                      
    if (id < n)  {                                               
        float real = a[2*id] * b[2*id]-a[2*id+1]*b[2*id+1];                                  
        float imag = a[2*id]*b[2*id+1] + a[2*id+1]*b[2*id];                            
        c[2*id]=real; 
        c[2*id+1]=imag; 
        }                           
}                                                               
#pragma OPENCL EXTENSION cl_khr_fp64 : enable                    
__kernel void vecComplexConjugateMultiply(  __global float *a,                       
                       __global float *b,                       
                       __global float *c,                       
                       const unsigned long n)                    
{                                                               
    //Get our global thread ID                                  
    int id = get_global_id(0);                                  
                                                                
    //Make sure we do not go out of bounds                      
    if (id < n)  {                                               
        float real= a[2*id] * b[2*id]+a[2*id+1]*b[2*id+1];                                  
        float imag = -a[2*id]*b[2*id+1] + a[2*id+1]*b[2*id];                            
        c[2*id]=real; 
        c[2*id+1]=imag; 
     }                           
}                                                               
#pragma OPENCL EXTENSION cl_khr_fp64 : enable                    
__kernel void vecDiv(  __global float *a,                       
                       __global float *b,                       
                       __global float *c,                       
                       const unsigned long n)                    
{                                                               
    //Get our global thread ID                                  
    int id = get_global_id(0);                                  
                                                                
    //Make sure we do not go out of bounds                      
    if (id < n)  {                                               
     if (true)  {                                               
        c[id] = a[id]/b[id];        
       }                           
       else {                           
        c[id]=0;                        
     }                           
     }                           
}                                                               
#pragma OPENCL EXTENSION cl_khr_fp64 : enable                    
__kernel void vecMul(  __global float *a,                       
                       __global float *b,                       
                       __global float *c,                       
                       const unsigned long n)                    
{                                                               
    //Get our global thread ID                                  
    int id = get_global_id(0);                                  
                                                                
    //Make sure we do not go out of bounds                      
    if (id < n)  {                                               
        c[id] = a[id]*b[id];        
        }                           
}                                                               
 

