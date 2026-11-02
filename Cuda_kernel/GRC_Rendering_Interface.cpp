#include <torch/extension.h>

// CUDA函数声明
at::Tensor AccRendering_KFunc(const int H,const int W,const int S,
                              const float RT,const float GT,
                                        const at::Tensor position,
                                        const at::Tensor value,
                                        const at::Tensor covariance) ;

at::Tensor AccRendering(int H,int W,int S,
                                const at::Tensor position,
                                const at::Tensor value,
                                const at::Tensor covariance,
                                float R_Threshold,
                                float G_Threshold) ;

at::Tensor AccRendering_Gradient_KFunc(const at::Tensor grad ,       //H W S
                                       const at::Tensor position,    //3 n
                                       const at::Tensor value,       //1 n
                                       const at::Tensor covariance,  //6 n 
                                       float R_Threshold,
                                       float G_Threshold);
                                        
at::Tensor AccRendering_Gradient(const at::Tensor grad ,       //H W S
                                 const at::Tensor position,    //3 n
                                 const at::Tensor value,       //1 n
                                 const at::Tensor covariance,  //6 n 
                                 float R_Threshold,
                                 float G_Threshold);

at::Tensor AccRendering_e2e_KFunc(const at::Tensor groundtruth,//H W S
                                  const at::Tensor position,    //3 n
                                  const at::Tensor value,       //1 n
                                  const at::Tensor A_matrix,  //6 n 
                                  float R_Threshold,
                                  float G_Threshold,
                                  float I_Threshold);

at::Tensor AccRendering_e2e     (const at::Tensor groundtruth, //H W S
                                 const at::Tensor position,    //3 n
                                 const at::Tensor value,       //1 n
                                 const at::Tensor A_matrix,  //6 n 
                                 float R_Threshold,
                                 float G_Threshold,
                                 float I_Threshold);

at::Tensor AccRendering_e2e_sparse_KFunc(const at::Tensor groundtruth,//H W S
                                         const at::Tensor position,    //3 n
                                         const at::Tensor value,       //1 n
                                         const at::Tensor A_matrix,  //6 n 
                                         float R_Threshold,
                                         float G_Threshold,
                                         float I_Threshold,
                                         int   Ramdom_select,
                                         int   offset
                                         );

at::Tensor AccRendering_sparse_e2e     (const at::Tensor groundtruth, //H W S
                                        const at::Tensor position,    //3 n
                                        const at::Tensor value,       //1 n
                                        const at::Tensor A_matrix,  //6 n 
                                        float R_Threshold,
                                        float G_Threshold,
                                        float I_Threshold,
                                        int   Ramdom_select,
                                        int   offset
                                        );



#define CHECK_CUDA(x)        TORCH_CHECK(x.device().is_cuda(), #x, " must be a CUDAtensor ")
#define CHECK_CONTIGUOUS(x)  TORCH_CHECK(x.is_contiguous()   , #x, " must be contiguous ")
#define CHECK_INPUT(x)       CHECK_CUDA(x);CHECK_CONTIGUOUS(x);

// C++函数包装

at::Tensor AccRendering(int H,int W,int S,
                        const at::Tensor position,    //3 n
                        const at::Tensor value,       //1 n
                        const at::Tensor covariance,  //6 n 
                        float R_Threshold,
                        float G_Threshold
                        ){ 
    CHECK_INPUT(position);
    CHECK_INPUT(value);
    CHECK_INPUT(covariance);
    return AccRendering_KFunc((const int)H,(const int)W,(const int)S,\
                            (const float)(R_Threshold*R_Threshold),
                            (const float)(G_Threshold),
                            position,value,covariance);
}

at::Tensor AccRendering_Gradient(const at::Tensor grad,        //H W S
                                 const at::Tensor position,    //3 n
                                 const at::Tensor value,       //1 n
                                 const at::Tensor covariance,  //6 n 
                                 float R_Threshold,
                                 float G_Threshold
                                ){ 
    CHECK_INPUT(grad);
    CHECK_INPUT(position);
    CHECK_INPUT(value);
    CHECK_INPUT(covariance);
    return AccRendering_Gradient_KFunc(grad,position,value,covariance,
                                      (const float)(R_Threshold*R_Threshold),
                                      (const float)(G_Threshold));
}

at::Tensor AccRendering_e2e     (const at::Tensor groundtruth,  //H W S
                                 const at::Tensor position,     //3 n
                                 const at::Tensor value,        //1 n
                                 const at::Tensor A_matrix,     //6 n 
                                 float R_Threshold,
                                 float G_Threshold,
                                 float I_Threshold
                                ){ 
    CHECK_INPUT(groundtruth);
    CHECK_INPUT(position);
    CHECK_INPUT(value);
    CHECK_INPUT(A_matrix);
    return AccRendering_e2e_KFunc(groundtruth,position,value,A_matrix,
                                      (const float)(R_Threshold*R_Threshold),
                                      (const float)(G_Threshold),
                                      (const float)(I_Threshold));
}

at::Tensor AccRendering_sparse_e2e(const at::Tensor groundtruth,  //H W S
                                 const at::Tensor position,     //3 n
                                 const at::Tensor value,        //1 n
                                 const at::Tensor A_matrix,     //6 n 
                                 float R_Threshold,
                                 float G_Threshold,
                                 float I_Threshold,
                                 int   Ramdom_select,
                                 int   offset
                                ){ 
    CHECK_INPUT(groundtruth);
    CHECK_INPUT(position);
    CHECK_INPUT(value);
    CHECK_INPUT(A_matrix);
    return AccRendering_e2e_sparse_KFunc(groundtruth,position,value,A_matrix,
                                      (const float)(R_Threshold*R_Threshold),
                                      (const float)(G_Threshold),
                                      (const float)(I_Threshold),
                                      (const int)  (Ramdom_select),
                                      (const int)  (offset)
                                      );
}





// 绑定
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("AccRendering"         ,&AccRendering         ,"AccRendering(cuda)");
    m.def("AccRendering_Gradient",&AccRendering_Gradient,"AccRendering_Gradient(cuda)");
    m.def("AccRendering_e2e"     ,&AccRendering_e2e     ,"AccRendering_e2e(cuda)");
    m.def("AccRendering_sparse_e2e",&AccRendering_sparse_e2e  ,"AccRendering_sparse_e2e(cuda)");
}
