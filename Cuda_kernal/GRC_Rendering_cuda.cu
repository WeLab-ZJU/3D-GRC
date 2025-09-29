#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <THC/THCAtomics.cuh>
#include <cuda_runtime.h>
#include <curand_kernel.h>


#define CORD3D(i,j,k) \
        int i = 0,j = 0,k = 0; \

#define INDEX_TO_CORD4D(d,l,i,j,k,G,H,W,S) \
        int l = (d)/((H)*(W)*(S));   \
        int i = (((d)/((W)*(S))) - (l*(H)));\
        int j = (((d)/(S)) - (l*(H)*(W)) - (i*(W)));\
        int k = (d)%(S);         

#define INDEX_TO_CORD3D(d,i,j,k,H,W,S) \
        int i = (d)/((W)*(S));   \
        int j = (((d)/(S)) - (i*W));\
        int k = (d)%(S);         

#define INDEX_TO_CORD2D(d,i,j,H,W) \
        int i = (d)/(W),    \
            j = (d)%(W);    

#define CORD3D_TO_INDEX(d,i,j,k,H,W,S) \
        int d = ((i)*(W)*(S)) + ((j)*(S)) + (k);      \

#define CORD2D_TO_INDEX(d,i,j,H,W) \
        int d =  (i)*(W) + (j);          \

#define CORD4D_TO_INDEX(d,l,i,j,k,G,H,W,S) \
        int d = ((l)*(W)*(S)*(H))  + ((i)*(W)*(S)) + ((j)*(S)) + (k);   \

#define CORD4D_GAP_L(GAP,G,H,W,S)\
        int GAP = (H)*(W)*(S);

#define CORD4D_GAP_I(GAP,G,H,W,S)\
        int GAP = (W)*(S);

#define CORD4D_GAP_J(GAP,G,H,W,S)\
        int GAP = (S);

#define CORD3D_GAP_I(GAP,H,W,S)\
        int GAP = (W)*(S);

#define CORD3D_GAP_J(GAP,H,W,S)\
        int GAP = (S);

#define CORD2D_GAP_I(GAP,H,W,S)\
        int GAP = (S);

#define TEADS_PER_BLOCK 1024
#define TEADS_PER_WARP  32
#define WARP_PER_BLOCK  (TEADS_PER_BLOCK/TEADS_PER_WARP)

#define T_ID    (blockIdx.x*blockDim.x + threadIdx.x)
#define T_NUM   (gridDim.x*blockDim.x)
#define B_T_ID  (threadIdx.x)
#define B_T_NUM (blockDim.x)
#define W_ID    (T_ID/32)
#define W_T_ID  (B_T_ID%32)
#define B_W_ID  (B_T_ID/32)
#define W_T_NUM (B_T_NUM/32)

#define CUDA_KERNEL_LOOP(d,TASK_NUM)       \
    for (int d  = T_ID;    \
             d  < TASK_NUM;\
             d += T_NUM)

#define FLOOR(a,b) ((int)((a)/(b))+(((a)%(b)!=0)?1:0))

#define CUDA_KERNEL_WARP_LOOP(d,TASK_NUM)       \
    for (int d  = T_ID;    \
            (d  < FLOOR(TASK_NUM,TEADS_PER_WARP)*TEADS_PER_WARP);\
             d += T_NUM)

#define CUDA_ALLOCATION(TASK_NUM)       \
     if((T_ID<TASK_NUM))

#define CROSS_WRITE(c,CHANNCEL) \
        for( int c  = (T_ID%(CHANNCEL)),CROSS_WRITE_COUNT = 0; \
                 CROSS_WRITE_COUNT < CHANNCEL; \
                 c += ((c!=(CHANNCEL-1))?1:(1-(CHANNCEL))),CROSS_WRITE_COUNT++)



inline int GET_BLOCKS(int N){
    const int BLOCK = (N + TEADS_PER_BLOCK - 1) / TEADS_PER_BLOCK;
    return BLOCK;}

inline int GET_TEADS(int N){
    if(N < TEADS_PER_BLOCK) return N;
    else return TEADS_PER_BLOCK;}

template <typename scalar_t>
__global__ void GetInvMatrix(const scalar_t *pA,   // 6 GN
                                   scalar_t *pM,   // 6 GN
                                   const int GN){  // 
    scalar_t xx,yy,zz,xy,xz,yz;
    scalar_t a11,a12,a13,a22,a23,a33;
    scalar_t xx_yy,xy_yz,xz_xz,xx_yz,xy_xy,det;
    CUDA_KERNEL_LOOP(d,GN)
    {     
        a11                   = *(pA+(GN*0)+d);
        a12                   = *(pA+(GN*1)+d);
        a13                   = *(pA+(GN*2)+d);
        a22                   = *(pA+(GN*3)+d);
        a23                   = *(pA+(GN*4)+d);
        a33                   = *(pA+(GN*5)+d);
        xx                    = fmaxf((a11*a11)+(a12*a12)+(a13*a13),1.0f);
        xy                    = (a12*a22)+(a13*a23);
        xz                    = (a13*a33);
        yy                    = fmaxf((a22*a22)+(a23*a23),1.0f);
        yz                    = (a23*a33);
        zz                    = fmaxf((a33*a33),1.0f);
        xx_yy                 = xx*yy;
        xy_yz                 = xy*yz; 
        xz_xz                 = xz*xz;
        xx_yz                 = xx*yz;
        xy_xy                 = xy*xy;
        det                   = (xx_yy*zz)+((xz*xy_yz)*2)-(xz_xz*yy)-(xx_yz*yz)-(xy_xy*zz);
        det                   = fmaxf(det,0.001f);
        *(pM+(GN*0)+d)        = ((yy*zz)-(yz*yz))/det;
        *(pM+(GN*1)+d)        = ((yz*xz)-(xy*zz))/det;
        *(pM+(GN*2)+d)        = ((xy_yz)-(xz*yy))/det;
        *(pM+(GN*3)+d)        = ((xx*zz)-(xz_xz))/det;
        *(pM+(GN*4)+d)        = ((xy*xz)-(xx_yz))/det;
        *(pM+(GN*5)+d)        = ((xx_yy)-(xy_xy))/det;
    }
}

template <typename scalar_t>
__global__ void AccRendering_Kernel(const scalar_t *pP,     //3 GN
                                    const scalar_t *pV, //1 GN
                                    const scalar_t *pM, //6 GN
                                    scalar_t       *pR, //H W S
                                    const int PN,
                                    const int GN,
                                    const float RT,
                                    const float GT,
                                    const int H,
                                    const int W,
                                    const int S
                                    ){
    scalar_t r;
    scalar_t xi,xj,xk,si,sj,sk,flag;
    scalar_t S00,S01,S02,S11,S12,S22;
    CUDA_KERNEL_LOOP(d,PN)  
    {      
    INDEX_TO_CORD3D(d,i,j,k,H,W,S)
    r = xi = xj = xk = si = sj = sk = flag = 0;
    for(int n=0;n<GN;n++)
    {
        xi              = (scalar_t)i - *(pP+(GN*0)+n);
        xj              = (scalar_t)j - *(pP+(GN*1)+n);
        xk              = (scalar_t)k - *(pP+(GN*2)+n);
        flag            = (xi*xi) + (xj*xj) + (xk*xk);
        if(flag<RT)
        {
            S00 = *(pM+(GN*0)+n);
            S01 = *(pM+(GN*1)+n);
            S02 = *(pM+(GN*2)+n);
            S11 = *(pM+(GN*3)+n);
            S12 = *(pM+(GN*4)+n);
            S22 = *(pM+(GN*5)+n);
            si  = (xi*S00) + (xj*S01) + (xk*S02);
            sj  = (xi*S01) + (xj*S11) + (xk*S12);
            sk  = (xi*S02) + (xj*S12) + (xk*S22);
            scalar_t fc = ((si*xi) + (sj*xj) + (sk*xk))*(-0.5);

            r  += (scalar_t)exp((float)fc)*(*(pV+n));
        
            
        }
    }
    if(r>GT)*(pR+d) = r;
    }
} 

at::Tensor AccRendering_KFunc(const int H,const int W,const int S,
                                        const float RT,   
                                        const float GT,   
                                        const at::Tensor P,  //3 n
                                        const at::Tensor V,  //1 n 
                                        const at::Tensor A){ //6 n

    const int        PN = H*W*S;     // pixel num
    const int        GN = P.size(1); // guess num
    at::Tensor        M = at::zeros({6,GN},V.options());
    at::Tensor        R = at::zeros({H,W,S},V.options());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        V.scalar_type(),       //数据类型
        "Rendering_KFunc", //
        [&]{
            const scalar_t *pP      = P.data<scalar_t>();
            const scalar_t *pV      = V.data<scalar_t>();
            const scalar_t *pA      = A.data<scalar_t>();
                  scalar_t *pM      = M.data<scalar_t>();
                  scalar_t *pR      = R.data<scalar_t>();
            GetInvMatrix<scalar_t>
                <<<GET_BLOCKS(GN),GET_TEADS(GN),0,stream>>>
                (pA,pM,GN);
            AccRendering_Kernel<scalar_t>
                <<<GET_BLOCKS(PN),GET_TEADS(PN),0,stream>>>
                (pP,pV,(const scalar_t*)pM,pR,PN,GN,RT,GT,H,W,S);
            cudaStreamSynchronize(stream);
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess)
            {
                printf("error in Rendering_KFunc:%s:%s\n", 
                        cudaGetErrorName(err),
                        cudaGetErrorString(err));
            }
        }             
    );
    return R;
}

template <typename scalar_t>
__global__ void AccRendering_Gradient_Kernel(   const scalar_t *pGR, //H W S
                                                const scalar_t *pP,  //3 GN
                                                const scalar_t *pV,  //1 GN
                                                const scalar_t *pA,  //6 GN
                                                const scalar_t *pM,  //6 GN
                                                scalar_t       *pG,  //10 GN P V C
                                                const int H,
                                                const int W,
                                                const int S,
                                                const float RT,
                                                const float GT,
                                                const int GN,
                                                const int PN){
        scalar_t rg,flag;
        scalar_t a1,a2,a3;
        scalar_t v,b11,b12,b13,b22,b23,b33;
        scalar_t a11,a12,a13,a22,a23,a33;
        scalar_t fc,t1,t2,t3;
        scalar_t dgr[10];
        CUDA_KERNEL_LOOP(d,PN)  
        {      
        if(d<PN)
        {
        INDEX_TO_CORD3D(d,i,j,k,H,W,S)
        rg = *(pGR+d);
        for(int n=0;n<GN;n++)
        {
                a1              = (scalar_t)i - *(pP+(GN*0)+n);
                a2              = (scalar_t)j - *(pP+(GN*1)+n);
                a3              = (scalar_t)k - *(pP+(GN*2)+n);
                flag            = (a1*a1) + (a2*a2) + (a3*a3);
                if(flag<RT)
                {
                        b11 = *(pM+(GN*0)+n);b12 = *(pM+(GN*1)+n);b13 = *(pM+(GN*2)+n);
                        b22 = *(pM+(GN*3)+n);b23 = *(pM+(GN*4)+n);b33 = *(pM+(GN*5)+n);
                        a11 = *(pA+(GN*0)+n);a12 = *(pA+(GN*1)+n);a13 = *(pA+(GN*2)+n);
                        a22 = *(pA+(GN*3)+n);a23 = *(pA+(GN*4)+n);a33 = *(pA+(GN*5)+n);
                        //----------
                        v   = *(pV+n);
                        //==========
                        t1  = (a1*b11) + (a2*b12) + (a3*b13);
                        t2  = (a1*b12) + (a2*b22) + (a3*b23);
                        t3  = (a1*b13) + (a2*b23) + (a3*b33);
                        fc  = exp(((t1*a1) +(t2*a2) +(t3*a3))*(-0.5));
                        //==========
                        dgr[3]   = rg*fc;
                        fc       = -0.5*rg*v*fc;
                        //----------
                        dgr[0]   = -2*fc*t1;
                        dgr[1]   = -2*fc*t2;
                        dgr[2]   = -2*fc*t3;
                        //----------
                        scalar_t f11,f22,f33,f12,f13,f23;
                        f11      = -fc*t1*t1;   //11
                        f22      = -fc*t2*t2;   //22
                        f33      = -fc*t3*t3;   //33
                        f12      = -2*fc*t1*t2; //12
                        f13      = -2*fc*t3*t1; //13
                        f23      = -2*fc*t2*t3; //23
                        //----------
                        dgr[4]   =  (f11*a11)*2;   
                        dgr[5]   = ((f11*a12)*2+(f12*a22));  
                        dgr[6]   = ((f11*a13)*2+(f12*a23)+(f13*a33));
                        dgr[7]   = ((f12*a12)+(f22*a22)*2);
                        dgr[8]   = ((f12*a13)+(f22*a23)*2+(f23*a33));
                        dgr[9]   = ((f13*a13)+(f23*a23)+(f33*a33)*2);
                        CROSS_WRITE(c,10)
                                if((dgr[c]>GT)||(dgr[c]<-GT))
                                        atomicAdd((scalar_t*)(pG+(GN*c)+n),(scalar_t)dgr[c]);
                }
        }
        }
    }
} 



at::Tensor AccRendering_Gradient_KFunc( const at::Tensor GR,     //H W S
                                        const at::Tensor P,      //3 n
                                        const at::Tensor V,      //1 n
                                        const at::Tensor A,      //6 n 
                                        float T1,float T2
                                        ){ //6 n
    const int        H  = GR.size(0);
    const int        W  = GR.size(1);
    const int        S  = GR.size(2);
    const int        PN = H*W*S;     // pixel num
    const int        GN = P.size(1); // guess num
    at::Tensor        M = at::zeros({6 ,GN},V.options());
    at::Tensor        G = at::zeros({10,GN},V.options());// P V C
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        V.scalar_type(),       //数据类型
        "AccRendering_Gradient_KFunc", //
        [&]{
            const scalar_t *pGR     = GR.data<scalar_t>();
            const scalar_t *pP      = P.data<scalar_t>();
            const scalar_t *pV      = V.data<scalar_t>();
            const scalar_t *pA      = A.data<scalar_t>();
                  scalar_t *pM      = M.data<scalar_t>();
                  scalar_t *pG      = G.data<scalar_t>();
            GetInvMatrix<scalar_t>
                <<<GET_BLOCKS(GN),GET_TEADS(GN),0,stream>>>
                (pA,pM,GN);
            AccRendering_Gradient_Kernel<scalar_t>
                <<<GET_BLOCKS(PN),GET_TEADS(PN),0,stream>>>
                (pGR,pP,pV,pA,(const scalar_t*)pM,pG,H,W,S,T1,T2,GN,PN);
            cudaStreamSynchronize(stream);
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess)
            {
                printf("error in AccRendering_Gradient_KFunc:%s:%s\n", 
                        cudaGetErrorName(err),
                        cudaGetErrorString(err));
            }
        }             
    );
    return G;
}



template <typename scalar_t>
__global__ void AccRendering_e2e_Kernel(        const scalar_t *pGT, //H W S
                                                const scalar_t *pR, //H W S
                                                const scalar_t *pP,  //3 GN
                                                const scalar_t *pV,  //1 GN
                                                const scalar_t *pA,  //6 GN
                                                const scalar_t *pM,  //6 GN
                                                scalar_t       *pG,  //10 GN P V S
                                                const int   H,
                                                const int   W,
                                                const int   S,
                                                const float RT,
                                                const float GT,
                                                const int   GN,
                                                const int   PN){
        scalar_t rg,gt,flag;
        scalar_t a1,a2,a3;
        scalar_t v,b11,b12,b13,b22,b23,b33;
        scalar_t a11,a12,a13,a22,a23,a33;
        scalar_t fc_t1,fc_t2,fc_t3;
        scalar_t fc,t1,t2,t3;
        scalar_t f11,f22,f33,f12,f13,f23;
        scalar_t dgr[10];
        CUDA_KERNEL_LOOP(d,PN)  
        {      
        INDEX_TO_CORD3D(d,i,j,k,H,W,S)
        gt = *(pGT+d);
        rg = *(pR+d);
        rg = 2*(rg-gt);
        for(int n=0;n<GN;n++)
        {
                a1              = (scalar_t)i - *(pP+(GN*0)+n);
                a2              = (scalar_t)j - *(pP+(GN*1)+n);
                a3              = (scalar_t)k - *(pP+(GN*2)+n);
                flag            = (a1*a1) + (a2*a2) + (a3*a3);
                if(flag<RT)
                {
                        //==========
                        b11      = *(pM+(GN*0)+n);b12 = *(pM+(GN*1)+n);
                        b13      = *(pM+(GN*2)+n);b22 = *(pM+(GN*3)+n);
                        b23      = *(pM+(GN*4)+n);b33 = *(pM+(GN*5)+n);
                        //----------
                        a11      = *(pA+(GN*0)+n);a12 = *(pA+(GN*1)+n);
                        a13      = *(pA+(GN*2)+n);a22 = *(pA+(GN*3)+n);
                        a23      = *(pA+(GN*4)+n);a33 = *(pA+(GN*5)+n);
                        //----------
                        v        = *(pV+n);
                        //----------
                        t1       = (a1*b11) + (a2*b12) + (a3*b13);
                        t2       = (a1*b12) + (a2*b22) + (a3*b23);
                        t3       = (a1*b13) + (a2*b23) + (a3*b33);
                        scalar_t temp = ((t1*a1) +(t2*a2) +(t3*a3))*(-0.5);
                        fc       = (scalar_t)exp((float)temp);
                        //==========
                        dgr[3]   = rg*fc;
                        fc       = -0.5*rg*v*fc;
                        //----------
                        fc_t1    = fc*t1;
                        fc_t2    = fc*t2;
                        fc_t3    = fc*t3;
                        //----------
                        dgr[0]   = -2*fc_t1; 
                        dgr[1]   = -2*fc_t2; 
                        dgr[2]   = -2*fc_t3;
                        //----------
                        f11      = -fc_t1*t1;   //11
                        f22      = -fc_t2*t2;   //22
                        f33      = -fc_t3*t3;   //33
                        f12      = -2*fc_t1*t2; //12
                        f13      = -2*fc_t3*t1; //13
                        f23      = -2*fc_t2*t3; //23
                        //----------
                        dgr[4]   =  (f11*a11)*2;   
                        dgr[5]   = ((f11*a12)*2+(f12*a22));  
                        dgr[6]   = ((f11*a13)*2+(f12*a23)+(f13*a33));
                        dgr[7]   = ((f12*a12)+(f22*a22)*2);
                        dgr[8]   = ((f12*a13)+(f22*a23)*2+(f23*a33));
                        dgr[9]   = ((f13*a13)+(f23*a23)+(f33*a33)*2);
                        //==========
                        CROSS_WRITE(c,10)if((dgr[c]>GT)||(dgr[c]<-GT))
                            atomicAdd((scalar_t*)(pG+(GN*c)+n),(scalar_t)dgr[c]);
                        //==========
                }
        }
}
}

at::Tensor AccRendering_e2e_KFunc(const at::Tensor GT,     //H W S
                                  const at::Tensor P,      //3 n
                                  const at::Tensor V,      //1 n
                                  const at::Tensor A,      //6 n 
                                  float T1,float T2,float T3
                                  ){ //6 n
    const int        H  = GT.size(0);
    const int        W  = GT.size(1);
    const int        S  = GT.size(2);
    const int        PN = H*W*S;     // pixel num
    const int        GN = P.size(1); // guess num
    at::Tensor        M = at::zeros({6 ,GN},V.options());
    at::Tensor        R = at::zeros({H,W,S},V.options());
    at::Tensor        G = at::zeros({10,GN},V.options());// P V C
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        V.scalar_type(),       //数据类型
        "AccRendering_Gradient_KFunc", //
        [&]{
            const scalar_t *pGT     = GT.data<scalar_t>();
            const scalar_t *pP      = P .data<scalar_t>();
            const scalar_t *pV      = V .data<scalar_t>();
            const scalar_t *pA      = A .data<scalar_t>();
                  scalar_t *pM      = M .data<scalar_t>();
                  scalar_t *pR      = R .data<scalar_t>();
                  scalar_t *pG      = G .data<scalar_t>();
            GetInvMatrix<scalar_t>
                <<<GET_BLOCKS(GN),GET_TEADS(GN),0,stream>>>
                (pA,pM,GN);
            AccRendering_Kernel<scalar_t>
                <<<GET_BLOCKS(PN),GET_TEADS(PN),0,stream>>>
                (pP,pV,(const scalar_t*)pM,pR,PN,GN,T1,T3,H,W,S);
            AccRendering_e2e_Kernel<scalar_t>
                <<<GET_BLOCKS(PN),GET_TEADS(PN),0,stream>>>
                (pGT,(const scalar_t*)pR,pP,pV,(const scalar_t*)pA,\
                        (const scalar_t*)pM,pG,H,W,S,T1,T2,GN,PN);
            cudaStreamSynchronize(stream);
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess)
            {
                printf("error in AccRendering_e2e_KFunc:%s:%s\n", 
                        cudaGetErrorName(err),
                        cudaGetErrorString(err));
            }
        }             
    );
    return G;
}

template <typename scalar_t>
__global__ void AccRendering_sparse_Kernel(const scalar_t *pP,     //3 GN
                                            const scalar_t *pV, //1 GN
                                            const scalar_t *pM, //6 GN
                                            scalar_t       *pR, //H W S
                                            const int PN,
                                            const int GN,
                                            const float RT,
                                            const float GT,
                                            const int   RS,
                                            const int   OF,
                                            const int H,
                                            const int W,
                                            const int S
                                            ){
    scalar_t r;
    scalar_t xi,xj,xk,si,sj,sk,flag;
    scalar_t S00,S01,S02,S11,S12,S22;
    CUDA_KERNEL_LOOP(od,PN/RS) 
    {      
    int d = (od*RS) + OF;
    INDEX_TO_CORD3D(d,i,j,k,H,W,S)
    r = xi = xj = xk = si = sj = sk = flag = 0;
    for(int n=0;n<GN;n++)
    {
        xi              = (scalar_t)i - *(pP+(GN*0)+n);
        xj              = (scalar_t)j - *(pP+(GN*1)+n);
        xk              = (scalar_t)k - *(pP+(GN*2)+n);
        flag            = (xi*xi) + (xj*xj) + (xk*xk);
        if(flag<RT)
        {
            S00 = *(pM+(GN*0)+n);
            S01 = *(pM+(GN*1)+n);
            S02 = *(pM+(GN*2)+n);
            S11 = *(pM+(GN*3)+n);
            S12 = *(pM+(GN*4)+n);
            S22 = *(pM+(GN*5)+n);
            si  = (xi*S00) + (xj*S01) + (xk*S02);
            sj  = (xi*S01) + (xj*S11) + (xk*S12);
            sk  = (xi*S02) + (xj*S12) + (xk*S22);
            scalar_t fc = ((si*xi) + (sj*xj) + (sk*xk))*(-0.5);
            r  += (scalar_t)exp((float)fc)*(*(pV+n));
        }
    }
    if(r>GT)*(pR+d) = r;
    }
} 

template <typename scalar_t>
__global__ void AccRendering_e2e_sparse_Kernel(const scalar_t *pGT, //H W S
                                                const scalar_t *pR,  //H W S
                                                const scalar_t *pP,  //3 GN
                                                const scalar_t *pV,  //1 GN
                                                const scalar_t *pA,  //6 GN
                                                const scalar_t *pM,  //6 GN
                                                scalar_t       *pG,  //10 GN P V S
                                                const int   H,
                                                const int   W,
                                                const int   S,
                                                const float RT,
                                                const float GT,
                                                const int   RS,
                                                const int   OF,
                                                const int   GN,
                                                const int   PN){
        scalar_t rg,gt,flag;
        scalar_t a1,a2,a3;
        scalar_t v,b11,b12,b13,b22,b23,b33;
        scalar_t a11,a12,a13,a22,a23,a33;
        scalar_t fc_t1,fc_t2,fc_t3;
        scalar_t fc,t1,t2,t3;
        scalar_t f11,f22,f33,f12,f13,f23;
        scalar_t dgr[10];
        CUDA_KERNEL_LOOP(od,PN/RS)  
        {     
        int d = (od*RS) + OF;
        INDEX_TO_CORD3D(d,i,j,k,H,W,S)
        gt = *(pGT+d);
        rg = *(pR+d);
        rg = 2*(rg-gt);
        for(int n=0;n<GN;n++)
        {
                a1              = (scalar_t)i - *(pP+(GN*0)+n);
                a2              = (scalar_t)j - *(pP+(GN*1)+n);
                a3              = (scalar_t)k - *(pP+(GN*2)+n);
                flag            = (a1*a1) + (a2*a2) + (a3*a3);
                if(flag<RT)
                {
                        //==========
                        b11      = *(pM+(GN*0)+n);b12 = *(pM+(GN*1)+n);
                        b13      = *(pM+(GN*2)+n);b22 = *(pM+(GN*3)+n);
                        b23      = *(pM+(GN*4)+n);b33 = *(pM+(GN*5)+n);
                        //----------
                        a11      = *(pA+(GN*0)+n);a12 = *(pA+(GN*1)+n);
                        a13      = *(pA+(GN*2)+n);a22 = *(pA+(GN*3)+n);
                        a23      = *(pA+(GN*4)+n);a33 = *(pA+(GN*5)+n);
                        //----------
                        v        = *(pV+n);
                        //----------
                        t1       = (a1*b11) + (a2*b12) + (a3*b13);
                        t2       = (a1*b12) + (a2*b22) + (a3*b23);
                        t3       = (a1*b13) + (a2*b23) + (a3*b33);
                        scalar_t temp = ((t1*a1) +(t2*a2) +(t3*a3))*(-0.5);

                        fc       = (scalar_t)exp((float)temp);
                        
                        //==========
                        dgr[3]   = rg*fc;
                        fc       = -0.5*rg*v*fc;
                        //----------
                        fc_t1    = fc*t1;
                        fc_t2    = fc*t2;
                        fc_t3    = fc*t3;
                        //----------
                        dgr[0]   = -2*fc_t1; 
                        dgr[1]   = -2*fc_t2; 
                        dgr[2]   = -2*fc_t3;
                        //----------
                        f11      = -fc_t1*t1;   //11
                        f22      = -fc_t2*t2;   //22
                        f33      = -fc_t3*t3;   //33
                        f12      = -2*fc_t1*t2; //12
                        f13      = -2*fc_t3*t1; //13
                        f23      = -2*fc_t2*t3; //23
                        //----------
                        dgr[4]   =  (f11*a11)*2;   
                        dgr[5]   = ((f11*a12)*2+(f12*a22));  
                        dgr[6]   = ((f11*a13)*2+(f12*a23)+(f13*a33));
                        dgr[7]   = ((f12*a12)+(f22*a22)*2);
                        dgr[8]   = ((f12*a13)+(f22*a23)*2+(f23*a33));
                        dgr[9]   = ((f13*a13)+(f23*a23)+(f33*a33)*2);
                        //==========
                        CROSS_WRITE(c,10)if((dgr[c]>GT)||(dgr[c]<-GT))
                            atomicAdd((scalar_t*)(pG+(GN*c)+n),(scalar_t)dgr[c]);
                        //==========
                }
        }
        }
}

at::Tensor AccRendering_e2e_sparse_KFunc(const at::Tensor GT,     //H W S
                                         const at::Tensor P,      //3 n
                                         const at::Tensor V,      //1 n
                                         const at::Tensor A,      //6 n 
                                         float T1,float T2,float T3,
                                         int RS,int OF
                                  ){ //6 n
    const int        H  = GT.size(0);
    const int        W  = GT.size(1);
    const int        S  = GT.size(2);
    const int        PN = H*W*S;     // pixel num
    const int        GN = P.size(1); // guess num
    at::Tensor        M = at::zeros({6 ,GN},V.options());
    at::Tensor        R = at::zeros({H,W,S},V.options());
    at::Tensor        G = at::zeros({10,GN},V.options());// P V C
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        V.scalar_type(),       //数据类型
        "AccRendering_Gradient_KFunc", //
        [&]{
            const scalar_t *pGT     = GT.data<scalar_t>();
            const scalar_t *pP      = P .data<scalar_t>();
            const scalar_t *pV      = V .data<scalar_t>();
            const scalar_t *pA      = A .data<scalar_t>();
                  scalar_t *pM      = M .data<scalar_t>();
                  scalar_t *pR      = R .data<scalar_t>();
                  scalar_t *pG      = G .data<scalar_t>();
            GetInvMatrix<scalar_t>
                <<<GET_BLOCKS(GN),GET_TEADS(GN),0,stream>>>
                (pA,pM,GN);
            AccRendering_sparse_Kernel<scalar_t>
                <<<GET_BLOCKS(PN/RS),GET_TEADS(PN/RS),0,stream>>>
                (pP,pV,(const scalar_t*)pM,pR,PN,GN,T1,T3,RS,OF,H,W,S);
            AccRendering_e2e_sparse_Kernel<scalar_t>
                <<<GET_BLOCKS(PN/RS),GET_TEADS(PN/RS),0,stream>>>
                (pGT,(const scalar_t*)pR,pP,pV,(const scalar_t*)pA,\
                        (const scalar_t*)pM,pG,H,W,S,T1,T2,RS,OF,GN,PN);
            cudaStreamSynchronize(stream);
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess)
            {
                printf("error in AccRendering_e2e_KFunc:%s:%s\n", 
                        cudaGetErrorName(err),
                        cudaGetErrorString(err));
            }
        }             
    );
    return G;
}

