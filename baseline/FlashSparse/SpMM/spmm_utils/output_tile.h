
#ifndef SPMM_OUTPUT_Tile_H
#define SPMM_OUTPUT_Tile_H

#include <mma.h>


    // 4 warps Tile_N = 128 8-bit v=2 4 8
    struct mmaOutputTile_fp16{
        //
        // Member variables
        //
        // int lane_id_;
        // int valid_thread;
        int warp_id;
        int warpin_id;
        int wrow_offset;
        int wcol_offset;
        // The register file fragment with the results to store
        const half* output_fragment_;
        half* output_matrix_;

        // Constructor
        __device__ __forceinline__ mmaOutputTile_fp16(
            int lane_id, 
            half* output_fragment)
        {
            output_fragment_ = output_fragment;
            //有数据的线程，因为当vec_len小于8时，一个warp内并不是所有线程都参与运算
            //由于C的转置问题，所以在C转置矩阵中找前vec_le列
            // lane_id_ = lane_id;
            warp_id = lane_id>>5;
            warpin_id=lane_id&31;
            // valid_thread = warpin_id &3;
            wrow_offset = warpin_id>>2;
            wcol_offset = (warpin_id &3) <<1;
        }

        // Store
        __device__ __forceinline__ void Store(long m_index_vec, long column_offset,
            long cols, half* output_matrix,int rowEdge, int colEdge){
        /*将线程需要搬运的数据定位在全局的行位置,
        m_index_vec = blockIdx.x;
        column_offset=dimN_index= blockIdx.y * Tile_N;
        m_index_vec * vec_length：block所在全局的行偏移;
        (m_index_vec * vec_length + wcol_offset) * cols：当前线程在block内的行偏移
        column_offset：block的全局列偏移
        */
        //转置矩阵当前的列偏移实际为结果矩阵的行偏移
      
            #pragma unroll
            for(int i=0;i<2;i++)
            {
                long row=((m_index_vec << 3)+ wcol_offset + i) ;
                long col=column_offset+wrow_offset+ (warp_id<<4);
                // const long output_offset = ((m_index_vec << 3)+ wcol_offset + i) * cols + column_offset;
                output_matrix_ = output_matrix +(row*cols)+col;
                //结果矩阵的块内列偏移为转置矩阵的行偏移
                //c(i), c(i+2)
                if(row<rowEdge)
                {
                    if(col<colEdge)
                    *output_matrix_ = output_fragment_[i];
                    if((col+8)<colEdge)
                    *(output_matrix_+8) = output_fragment_[i+2];
                }
            }
        
        }
    };

    //fp16 16
    struct mmaOutputTile_fp16_16{
        //
        // Member variables
        //
        // int lane_id_;
        // int valid_thread;
        int warp_id;
        int warpin_id;
        // int wrow_offset;
        // int wcol_offset;
        // The register file fragment with the results to store
        const half* output_fragment_;
        half* output_matrix_;

        // Constructor
        __device__ __forceinline__ mmaOutputTile_fp16_16(
            int lane_id, 
            half* output_fragment)
        {
            output_fragment_ = output_fragment;
            //有数据的线程，因为当vec_len小于8时，一个warp内并不是所有线程都参与运算
            //由于C的转置问题，所以在C转置矩阵中找前vec_le列
            // lane_id_ = lane_id;
            warp_id = lane_id>>5;
            warpin_id=lane_id&31;
            // valid_thread = warpin_id &3;
            // wrow_offset = warpin_id>>2;
            // wcol_offset = (warpin_id &3) <<1;
        }

        // Store
        __device__ __forceinline__ void Store(long m_index_vec, long column_offset,
            long cols, half* output_matrix,int rowEdge, int colEdge){
        /*将线程需要搬运的数据定位在全局的行位置,
        m_index_vec = blockIdx.x;
        column_offset=dimN_index= blockIdx.y * Tile_N;
        m_index_vec * vec_length：block所在全局的行偏移;
        (m_index_vec * vec_length + wcol_offset) * cols：当前线程在block内的行偏移
        column_offset：block的全局列偏移
        */
        //转置矩阵当前的列偏移实际为结果矩阵的行偏移
      
            #pragma unroll
            for(int i=0;i<2;i++)
            {
                long row=((m_index_vec * 16)+ warpin_id/4 + i*8) ;
                long col=column_offset + warp_id*8 + (warpin_id%4)*2;
                // const long output_offset = ((m_index_vec << 3)+ wcol_offset + i) * cols + column_offset;
                output_matrix_ = output_matrix +(row*cols)+col;
                //结果矩阵的块内列偏移为转置矩阵的行偏移
                //c(i), c(i+2)
                if(row<rowEdge)
                {
                    if(col<colEdge)
                    *output_matrix_ = output_fragment_[i*2];
                    if((col+1)<colEdge)
                    *(output_matrix_+1) = output_fragment_[i*2+1];

                }

            }
        
        }
    };



     // 4 warps Tile_N = 128 8-bit v=2 4 8
    struct mmaOutputTile_tf32{
        //
        // Member variables
        //
        // int lane_id_;
        // int valid_thread;
        int warp_id;
        int warpin_id;
        int wrow_offset;
        int wcol_offset;
        // The register file fragment with the results to store
        const float* output_fragment_;
        float* output_matrix_;

        // Constructor
        __device__ __forceinline__ mmaOutputTile_tf32(
            int lane_id, 
            float* output_fragment)
        {
            output_fragment_ = output_fragment;
            //有数据的线程，因为当vec_len小于8时，一个warp内并不是所有线程都参与运算
            //由于C的转置问题，所以在C转置矩阵中找前vec_le列
            // lane_id_ = lane_id;
            warp_id = lane_id>>5;
            warpin_id=lane_id&31;
            // valid_thread = warpin_id &3;
            wrow_offset = warpin_id%4;
            wcol_offset = warpin_id/4;
        }

        // Store
        __device__ __forceinline__ void Store(long m_index_vec, long column_offset,
            long cols, float* output_matrix,int rowEdge, int colEdge){
        /*将线程需要搬运的数据定位在全局的行位置,
        m_index_vec = blockIdx.x;
        column_offset=dimN_index= blockIdx.y * Tile_N;
        m_index_vec * vec_length：block所在全局的行偏移;
        (m_index_vec * vec_length + wcol_offset) * cols：当前线程在block内的行偏移
        column_offset：block的全局列偏移
        */
        //转置矩阵当前的列偏移实际为结果矩阵的行偏移
      
            #pragma unroll
            for(int i=0;i<2;i++)
            {
                long row=((m_index_vec << 3)+ wrow_offset*2 + i) ;
                //column_offset为block的列偏移，warp_id*16为warp的列偏移， wcol_offset为warp内的列偏移
                long col=column_offset + warp_id*16 + wcol_offset;
                output_matrix_ = output_matrix +(row*cols)+col;
                //结果矩阵的块内列偏移为转置矩阵的行偏移
                if(row<rowEdge)
                {
                    if(col<colEdge)
                    *output_matrix_ = output_fragment_[i];
                    if((col+8)<colEdge)
                    *(output_matrix_+8) = output_fragment_[i+2];
                }
            }
        
        }
    };


    struct mmaOutputTile_tf32_map{
        //
        // Member variables
        //
        // int lane_id_;
        // int valid_thread;
        int warp_id;
        int warpin_id;
        int wrow_offset;
        int wcol_offset;
        // The register file fragment with the results to store
        const float* output_fragment_;
        float* output_matrix_;

        // Constructor
        __device__ __forceinline__ mmaOutputTile_tf32_map(
            int lane_id, 
            float* output_fragment)
        {
            output_fragment_ = output_fragment;
            //有数据的线程，因为当vec_len小于8时，一个warp内并不是所有线程都参与运算
            //由于C的转置问题，所以在C转置矩阵中找前vec_le列
            // lane_id_ = lane_id;
            warp_id = lane_id>>5;
            warpin_id=lane_id&31;
            // valid_thread = warpin_id &3;
            wrow_offset = (warpin_id>>2) * 2;
            wcol_offset = (warpin_id &3) <<1;
        }

        // Store
        __device__ __forceinline__ void Store(long m_index_vec, long column_offset,
            long cols, float* output_matrix,int rowEdge, int colEdge){
        /*将线程需要搬运的数据定位在全局的行位置,
        m_index_vec = blockIdx.x;
        column_offset=dimN_index= blockIdx.y * Tile_N;
        m_index_vec * vec_length：block所在全局的行偏移;
        (m_index_vec * vec_length + wcol_offset) * cols：当前线程在block内的行偏移
        column_offset：block的全局列偏移
        */
        //转置矩阵当前的列偏移实际为结果矩阵的行偏移
      
            #pragma unroll
            for(int i=0;i<2;i++)
            {
                long row=((m_index_vec << 3)+ wcol_offset + i) ;
                //column_offset为block的列偏移，warp_id*16为warp的列偏移， wcol_offset为warp内的列偏移
                long col=column_offset+wrow_offset+ (warp_id<<4);
                output_matrix_ = output_matrix +(row*cols)+col;
                //结果矩阵的块内列偏移为转置矩阵的行偏移
                if(row<rowEdge)
                {
                    if(col<colEdge)
                    *output_matrix_ = output_fragment_[i];
                    if((col+1)<colEdge)
                    *(output_matrix_+1) = output_fragment_[i+2];
                }
            }
        
        }
    };


    //tf32 - 16
    // 4 warps Tile_N = 128 8-bit v=2 4 8
    struct mmaOutputTile_tf32_16{
        //
        // Member variables
        //
        // int lane_id_;
        // int valid_thread;
        int warp_id;
        int warpin_id;
        // int wrow_offset;
        // int wcol_offset;
        // The register file fragment with the results to store
        const float* output_fragment_;
        float* output_matrix_;

        // Constructor
        __device__ __forceinline__ mmaOutputTile_tf32_16(
            int lane_id, 
            float* output_fragment)
        {
            output_fragment_ = output_fragment;
            //有数据的线程，因为当vec_len小于8时，一个warp内并不是所有线程都参与运算
            //由于C的转置问题，所以在C转置矩阵中找前vec_le列
            // lane_id_ = lane_id;
            warp_id = lane_id>>5;
            warpin_id=lane_id&31;
            // valid_thread = warpin_id &3;
            // wrow_offset = warpin_id%4;
            // wcol_offset = warpin_id/4;
        }

        // Store
        __device__ __forceinline__ void Store(long m_index_vec, long column_offset,
            long cols, float* output_matrix,int rowEdge, int colEdge){
        /*将线程需要搬运的数据定位在全局的行位置,
        m_index_vec = blockIdx.x;
        column_offset=dimN_index= blockIdx.y * Tile_N;
        m_index_vec * vec_length：block所在全局的行偏移;
        (m_index_vec * vec_length + wcol_offset) * cols：当前线程在block内的行偏移
        column_offset：block的全局列偏移
        */
        //转置矩阵当前的列偏移实际为结果矩阵的行偏移
      
            #pragma unroll
            for(int i=0;i<2;i++)
            {
                long row=((m_index_vec << 4)+ warpin_id/4 + i*8) ;
                //column_offset为block的列偏移，warp_id*16为warp的列偏移， wcol_offset为warp内的列偏移
                long col=column_offset + warp_id*8 + ((warpin_id%4)*2);
                output_matrix_ = output_matrix +(row*cols)+col;
                //结果矩阵的块内列偏移为转置矩阵的行偏移
                if(row<rowEdge)
                {
                    if(col<colEdge)
                    *output_matrix_ = output_fragment_[2*i];
                    if((col+1)<colEdge)
                    *(output_matrix_+1) = output_fragment_[1+2*i];
                }
            }
        
        }
    };

 // 4 warps Tile_N = 128 8-bit v=2 4 8
    struct mmaOutputTile_fp16_test{
        //
        // Member variables
        //
        // int lane_id_;
        // int valid_thread;
        int warp_id;
        int warpin_id;
        int wrow_offset;
        int wcol_offset;
        int group;
        int temp;
        // The register file fragment with the results to store
        half* output_fragment_;
        float2* output_matrix_;

        // Constructor
        __device__ __forceinline__ mmaOutputTile_fp16_test(
            int lane_id, 
            half* output_fragment)
        {
            output_fragment_ = output_fragment;
            //有数据的线程，因为当vec_len小于8时，一个warp内并不是所有线程都参与运算
            //由于C的转置问题，所以在C转置矩阵中找前vec_le列
            // lane_id_ = lane_id;
            warp_id = lane_id>>5;
            warpin_id=lane_id&31;
            // valid_thread = warpin_id &3;

            if(warpin_id < 8) group=0;
            else if(warpin_id < 16) group=2;
            else if(warpin_id < 24) group=1;
            else group=3;
            temp = 0;
            if((warpin_id%8)>=4) temp = 1;

            // wrow_offset = warpin_id>>2;
            wcol_offset = ((warpin_id &3) <<1 )+ temp;
        }

        // Store
        __device__ __forceinline__ void Store(long m_index_vec, long column_offset,
            long cols, float2* output_matrix,int rowEdge, int colEdge){
            //第一次shuffle
            //temp=0: 分别取第1,3个数； temp=1: 分别取第0,2个数；
            half2 ex;
            if(group < 2){
                ex.x =  output_fragment_[2];
                ex.y =  output_fragment_[3];
            }else{
                ex.x =  output_fragment_[0];
                ex.y =  output_fragment_[1];
            }
            //做shuffle
            ex = __shfl_xor_sync(0xffffffff, ex, 8, 32);
            //shuffle完，更新dense_tile_
            if(group < 2){
                output_fragment_[2] = ex.x;
                output_fragment_[3] = ex.y;
            }else{
                output_fragment_[0] = ex.x;
                output_fragment_[1] = ex.y;
            }

            //第二次shuffle
            if(temp == 0){
                ex.x =  output_fragment_[1];
                ex.y =  output_fragment_[3];
            }else{
                ex.x =  output_fragment_[0];
                ex.y =  output_fragment_[2];
            }
            //做shuffle
            ex = __shfl_xor_sync(0xffffffff, ex, 4, 32);
            //shuffle完，更新dense_tile_
            if(temp == 0){
                output_fragment_[1] = ex.x;
                output_fragment_[3] = ex.y;
            }else{
                output_fragment_[0] = ex.x;
                output_fragment_[2] = ex.y;
            }


            long row=(m_index_vec << 3)+ wcol_offset ;
            long col=column_offset/4 + (warp_id<<2) + group;
            output_matrix_ = output_matrix +(row*cols/4)+col;

            // if(blockIdx.x==0 and blockIdx.y ==0 and threadIdx.x==4){
            //     for(int i=0;i<4;i++)
            //     printf("%f\n", __half2float(output_fragment_[i]));
            //     printf("%d, %d, %ld, %ld, %d\n", wcol_offset, temp, row, col, group);
            // }

            //结果矩阵的块内列偏移为转置矩阵的行偏移
            //c(i), c(i+2)
            float2 * output_fragment_v2 = reinterpret_cast< float2 *>(output_fragment_);
            if(row<rowEdge)
            {
                *output_matrix_ = *(output_fragment_v2);
            }
   
        
        }
    };




//  //fp16 16
// struct mmaOutputTile_fp16_v2{
//         //
//         // Member variables
//         //
//         // int lane_id_;
//         // int valid_thread;
//         int warp_id;
//         int warpin_id;
//         int wrow_offset;
//         int wcol_offset;
//         // The register file fragment with the results to store
//         const half* output_fragment_;
//         half* output_matrix_;

//         // Constructor
//         __device__ __forceinline__ mmaOutputTile_fp16_v2(
//             int lane_id, 
//             half* output_fragment)
//         {
//             output_fragment_ = output_fragment;
//             //有数据的线程，因为当vec_len小于8时，一个warp内并不是所有线程都参与运算
//             //由于C的转置问题，所以在C转置矩阵中找前vec_le列
//             // lane_id_ = lane_id;
//             warp_id = lane_id>>5;
//             warpin_id=lane_id&31;
//             // valid_thread = warpin_id &3;
//             wrow_offset = warpin_id>>2;
//             wcol_offset = (warpin_id &3) <<1;
//         }

//         // Store
//         __device__ __forceinline__ void Store(long m_index_vec, long column_offset,
//             long cols, half* output_matrix,int rowEdge, int colEdge){
//         /*将线程需要搬运的数据定位在全局的行位置,
//         m_index_vec = blockIdx.x;
//         column_offset=dimN_index= blockIdx.y * Tile_N;
//         m_index_vec * vec_length：block所在全局的行偏移;
//         (m_index_vec * vec_length + wcol_offset) * cols：当前线程在block内的行偏移
//         column_offset：block的全局列偏移
//         */
//         //转置矩阵当前的列偏移实际为结果矩阵的行偏移
      
//             #pragma unroll
//             for(int i=0;i<2;i++)
//             {
//                 long row=((m_index_vec << 3)+ wcol_offset + i) ;
//                 long col=column_offset+wrow_offset+ (warp_id<<4);
//                 // const long output_offset = ((m_index_vec << 3)+ wcol_offset + i) * cols + column_offset;
//                 output_matrix_ = output_matrix +(row*cols)+col;
//                 //结果矩阵的块内列偏移为转置矩阵的行偏移
//                 //c(i), c(i+2)
//                 if(row<rowEdge)
//                 {
//                     if(col<colEdge)
//                     *output_matrix_ = output_fragment_[i];
//                     if((col+8)<colEdge)
//                     *(output_matrix_+8) = output_fragment_[i+2];
//                 }
//             }
        
//         }
//     };



    struct mmaOutputTile_fp16_map{
        //
        // Member variables
        //
        // int lane_id_;
        // int valid_thread;
        int warp_id;
        int warpin_id;
        int wrow_offset;
        int wcol_offset;
        // The register file fragment with the results to store
        const half* output_fragment_;
        half* output_matrix_;

        // Constructor
        __device__ __forceinline__ mmaOutputTile_fp16_map(
            int lane_id, 
            half* output_fragment)
        {
            output_fragment_ = output_fragment;
            //有数据的线程，因为当vec_len小于8时，一个warp内并不是所有线程都参与运算
            //由于C的转置问题，所以在C转置矩阵中找前vec_le列
            // lane_id_ = lane_id;
            warp_id = lane_id>>5;
            warpin_id=lane_id&31;
            // valid_thread = warpin_id &3;
            wrow_offset = (warpin_id>>2) * 2;
            wcol_offset = (warpin_id &3) <<1;
        }

        // Store
        __device__ __forceinline__ void Store(long m_index_vec, long column_offset,
            long cols, half* output_matrix,int rowEdge, int colEdge){
        /*将线程需要搬运的数据定位在全局的行位置,
        m_index_vec = blockIdx.x;
        column_offset=dimN_index= blockIdx.y * Tile_N;
        m_index_vec * vec_length：block所在全局的行偏移;
        (m_index_vec * vec_length + wcol_offset) * cols：当前线程在block内的行偏移
        column_offset：block的全局列偏移
        */
        //转置矩阵当前的列偏移实际为结果矩阵的行偏移
      
            #pragma unroll
            for(int i=0;i<2;i++)
            {
                long row=((m_index_vec << 3)+ wcol_offset + i) ;
                long col=column_offset+wrow_offset+ (warp_id<<4);
                // const long output_offset = ((m_index_vec << 3)+ wcol_offset + i) * cols + column_offset;
                output_matrix_ = output_matrix +(row*cols)+col;
                //结果矩阵的块内列偏移为转置矩阵的行偏移
                //c(i), c(i+2)
                if(row<rowEdge)
                {
                    if(col<colEdge)
                    *output_matrix_ = output_fragment_[i];
                    if((col+1)<colEdge)
                    *(output_matrix_+1) = output_fragment_[i+2];
                }
            }
        
        }
    };

    #endif