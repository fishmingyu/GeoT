
#include <mma.h>
#include <cstdint>
#include <stdio.h>
#include <cuda_fp16.h>
#include <torch/torch.h>
// using namespace nvcuda;


    //Tile_N = 128 threads_per_block = 128
    struct mmaDenseTile_fp16{
        const float *  values_;
        const int *  column_idxs_;
        const int rhs_cols_;
        const int lane_id_;
        const int warpin_id;
        const int warp_id;
        const half *matrix_base_;
        float *dense_tile_;
        //存放当前线程拿到的一个double值，并将该double值拆分成4个half分别进行转置放置
        float *sparse_fragment_;

        __device__ __forceinline__ mmaDenseTile_fp16(
        long row_offset_vec,
        const double * values,
        const int *  column_idxs,
	    int rhs_cols,
        int offset, 
        int lane_id, 
        const double*  matrix, 
        //row_offsets= column_indices_tile
        // const int *row_offsets,
        float * dense_tile,
        float *sparse_fragment):
            rhs_cols_(rhs_cols),
            lane_id_(lane_id),
            warpin_id(lane_id & 31),
            warp_id(lane_id>>5),
            //每行16个线程，每个线程搬运1个double,
            matrix_base_(reinterpret_cast<const half *>(matrix + offset)),
            // row_offsets_base_(row_offsets),
            values_(reinterpret_cast<const float *>(values + row_offset_vec*2) + (lane_id & 31)),
            column_idxs_(column_idxs + row_offset_vec + ((lane_id & 31)>>2)),
            dense_tile_(dense_tile),
            sparse_fragment_(sparse_fragment)
            {}
    
        __device__ __forceinline__ void Fetch(int colEdge, int dimN_index){

            sparse_fragment_[0]=__ldg(values_);
            const long row_offsets_ = __ldg(column_idxs_);
            values_ += 32;
            column_idxs_ += 8;
            const int global_offset = (warp_id<<4) + ((warpin_id&3)<<2);
            const long offset = (row_offsets_*rhs_cols_) + global_offset;
            at::Half dense_tile_half_temp[4]={0.0,0.0,0.0,0.0};
            half * dense_tile_half = reinterpret_cast<half *>(dense_tile_half_temp);
            for(int i=0;i<4;i++)
            {
                if((dimN_index + global_offset+i)<colEdge)
                dense_tile_half[i]=__ldg(matrix_base_ +offset+ i);
            }
        
            // const int global_offset = (warp_id<<2) + (warpin_id&3);
            // dense_tile_half =reinterpret_cast<const half *>(matrix_base_ + (row_offsets_)*rhs_cols_ + global_offset);
            //warp内部开始shuffle
            //shuffle需要先打包需要交换的值
            int xid=(warpin_id>>2)&1;  //列0，1，0，1
            //定义临时数组tmp
            float tmp[2];
            at::Half *p=reinterpret_cast<at::Half *>(tmp);
            if(xid==0)
            {
                p[0]=dense_tile_half_temp[2];
                p[1]=dense_tile_half_temp[3];
            }
            else
            {
                p[0]=dense_tile_half_temp[0];
                p[1]=dense_tile_half_temp[1];
            }
            //第一次shuflle
            
            tmp[0] = __shfl_xor_sync(0xffffffff, tmp[0], 4,32);
            //交换后赋值
            if(xid==0)
            {
            p[3]=p[1];
            p[1]=p[0];
            p[0]=dense_tile_half_temp[0];
            p[2]=dense_tile_half_temp[1];
            }else
            {
            p[0]=p[0];
            p[2]=p[1];
            p[1]=dense_tile_half_temp[2];
            p[3]=dense_tile_half_temp[3];
            }
            
            //tmp交替写入dense_tile_
            int warpin_offset=((warpin_id&3)<<2) + (xid<<1);
            int k=(warpin_id&3)>>1; //行 0，0，1，1
            if(k==0){
                *(dense_tile_+(warp_id<<6) + (warpin_offset<<2)+ (warpin_id>>3))=tmp[0];
                *(dense_tile_+(warp_id<<6) + ((warpin_offset+1)<<2)+(warpin_id>>3))=tmp[1];
            }
            else{
                *(dense_tile_+(warp_id<<6) + ((warpin_offset+1)<<2) +(warpin_id>>3))=tmp[1];
                *(dense_tile_+(warp_id<<6) + (warpin_offset<<2) + (warpin_id>>3))=tmp[0];
            }
        }

        // Load the residual and compute the matrix product
        __device__ __forceinline__ void ResidueLoad(int colEdge, int dimN_index){
            sparse_fragment_[0]=__ldg(values_);
            const long row_offsets_ = __ldg(column_idxs_);
            const int global_offset = (warp_id<<4) + ((warpin_id&3)<<2);
            at::Half dense_tile_half_temp[4]={0.0,0.0,0.0,0.0};
            half * dense_tile_half = reinterpret_cast<half *>(dense_tile_half_temp);
            if(row_offsets_ >= 0){
                 const long offset = (row_offsets_*rhs_cols_) + global_offset;
                // matrix_base_=matrix_base_ + (row_offsets_*rhs_cols_) + global_offset;
                for(int i=0;i<4;i++)
                {
                    if((dimN_index + global_offset+i)<colEdge)
                    dense_tile_half[i]=__ldg(matrix_base_ + offset+i);
                    // if(blockIdx.y==1 && threadIdx.x==32){
                    // printf("11111 ");
                    // printf("%.1f  ",__half2float(dense_tile_half[i]));
                    //  printf("%d  ", row_offsets_);
                    //  printf("%d  ", global_offset);
                    //  printf("\n");
                    //  }

                }
            }
            // if(blockIdx.y==1 && threadIdx.x==32){
            // printf("666666 ");
            // for(int i=0;i<4;i++)
            // printf("%.1f  ",__half2float(dense_tile_half[i]));
            // printf("\n");}
            //warp内部开始shuffle
            //shuffle需要先打包需要交换的值
            int xid=(warpin_id>>2)&1;  //列0，1，0，1
            //定义临时数组tmp
            float tmp[2];
            at::Half *p=reinterpret_cast<at::Half *>(tmp);
            if(xid==0)
            {
                p[0]=dense_tile_half_temp[2];
                p[1]=dense_tile_half_temp[3];
            }
            else
            {
                p[0]=dense_tile_half_temp[0];
                p[1]=dense_tile_half_temp[1];
            }
            //第一次shuflle
            tmp[0] = __shfl_xor_sync(0xffffffff, tmp[0], 4,32);
            //交换后赋值
            if(xid==0)
            {
            p[3]=p[1];
            p[1]=p[0];
            p[0]=dense_tile_half_temp[0];
            p[2]=dense_tile_half_temp[1];
            }else
            {
            p[0]=p[0];
            p[2]=p[1];
            p[1]=dense_tile_half_temp[2];
            p[3]=dense_tile_half_temp[3];
            }

            //tmp交替写入dense_tile_
            int warpin_offset=((warpin_id&3)<<2) + (xid<<1);
            int k=(warpin_id&3)>>1; //行 0，0，1，1
            if(k==0){
                *(dense_tile_+(warp_id<<6) + (warpin_offset<<2) + (warpin_id>>3))=tmp[0];
                *(dense_tile_+(warp_id<<6) + ((warpin_offset+1)<<2) +(warpin_id>>3))=tmp[1];
            }
            else{
                *(dense_tile_+(warp_id<<6) + ((warpin_offset+1)<<2) +(warpin_id>>3))=tmp[1];
                *(dense_tile_+(warp_id<<6) + (warpin_offset<<2) + (warpin_id>>3))=tmp[0];
            }
            
        }
    };

    //fp16 16
    struct mmaDenseTile_fp16_16{
        const float *  values_;
        const int *  column_idxs_;
        const int rhs_cols_;
        const int lane_id_;
        const int warpin_id;
        const int warp_id;
        const half *matrix_base_;
        half *dense_tile_;
        //存放当前线程拿到的一个double值，并将该double值拆分成4个half分别进行转置放置
        float *sparse_fragment_;

        __device__ __forceinline__ mmaDenseTile_fp16_16(
        long row_offset_vec,
        const double * values,
        const int *  column_idxs,
	    int rhs_cols,
        int offset, 
        int lane_id, 
        const double*  matrix, 
        //row_offsets= column_indices_tile
        // const int *row_offsets,
        float * dense_tile,
        float *sparse_fragment):
            rhs_cols_(rhs_cols),
            lane_id_(lane_id),
            warpin_id(lane_id & 31),
            warp_id(lane_id>>5),
            //每行16个线程，每个线程搬运1个double,
            matrix_base_(reinterpret_cast<const half *>(matrix + offset)),
            // row_offsets_base_(row_offsets),
            values_(reinterpret_cast<const float *>(values + row_offset_vec*4) + (lane_id & 31)),
            column_idxs_(column_idxs + row_offset_vec + (((lane_id & 31)%4)*2)),
            dense_tile_(reinterpret_cast< half *>(dense_tile)),
            sparse_fragment_(sparse_fragment)
            {}
    
        __device__ __forceinline__ void Fetch(int colEdge, int dimN_index){

            for(int i=0;i<2;i++){
                sparse_fragment_[i]=__ldg(values_);
                values_ += 32;
            }
                // half * sparse_fragment_fp16 = reinterpret_cast<half *>(sparse_fragment_);
                // if(blockIdx.x==0 and blockIdx.y==0 and threadIdx.x==5)
                // {
                //     printf("%f\n", __half2float(sparse_fragment_fp16[0]));
                //     printf("%f\n", __half2float(sparse_fragment_fp16[2]));
                //     printf("%f\n", __half2float(sparse_fragment_fp16[1]));
                //     printf("%f\n", __half2float(sparse_fragment_fp16[3]));
                // }

            at::Half dense_tile_half_temp[2]={0.0,0.0};
            half * dense_tile_half = reinterpret_cast<half *>(dense_tile_half_temp);
            const int global_offset = (warp_id<<3) + (warpin_id/4);

            if((global_offset+dimN_index)<colEdge)
            {
                for(int i=0;i<2;i++)
                {
                    const long row_offsets_ = __ldg(column_idxs_ + i);
                    const long offset = (row_offsets_*rhs_cols_) + global_offset;
                    dense_tile_half[i]=__ldg(matrix_base_ +offset);

                }
            }
            for(int i=0;i<2;i++)
            *(dense_tile_ + i)=dense_tile_half[i];

            column_idxs_ += 8;
                //     if( blockIdx.x==0 and blockIdx.y==1 and blockIdx.z==0 and threadIdx.x==0)
                // {
                //     printf("%f %f \n", __half2float( dense_tile_[0]), __half2float(dense_tile_[1]));
                // }
        
            // const int global_offset = (warp_id<<2) + (warpin_id&3);
            // dense_tile_half =reinterpret_cast<const half *>(matrix_base_ + (row_offsets_)*rhs_cols_ + global_offset);
            //warp内部开始shuffle
            //shuffle需要先打包需要交换的值
            
        }

        // Load the residual and compute the matrix product
        __device__ __forceinline__ void ResidueLoad(int colEdge, int dimN_index, int residue){
            int col_offset = (warpin_id%4)*2;
            half * sparse_fragment_fp16 = reinterpret_cast<half *>(sparse_fragment_);
            const half * values_fp16 = reinterpret_cast<const half *>(values_);
            values_fp16 -= (8-residue)*(warpin_id/4);
            for(int c=0;c<2;c++){
                if((col_offset + c )< residue){
                    sparse_fragment_fp16[c] = __ldg(values_fp16+c);
                    sparse_fragment_fp16[c+2] = __ldg(values_fp16+c+(8*residue));
                }
                else{
                    sparse_fragment_fp16[c] = 0.0;
                    sparse_fragment_fp16[c+2] = 0.0;
                }

                // if(blockIdx.x==0 and blockIdx.y==0 and threadIdx.x==5)
                // {
                //     printf("%f\n", __half2float(sparse_fragment_fp16[c]));
                //     printf("%f\n", __half2float(sparse_fragment_fp16[c+2]));
                // }
            }

            // for(int i=0;i<2;i++){
            //     sparse_fragment_[i]=__ldg(values_);
            //     values_ += 32;
            // }



            const int global_offset = (warp_id<<3) + ((warpin_id/4));
            at::Half dense_tile_half_temp[2]={0.0,0.0};
            half * dense_tile_half = reinterpret_cast<half *>(dense_tile_half_temp);
            if((global_offset+dimN_index)<colEdge)
            {
                for(int i=0;i<2;i++){
                    long row_offsets_ = -1;
                    if((col_offset + i )< residue)
                    row_offsets_ = __ldg(column_idxs_ + i);
                    // const long row_offsets_ = __ldg(column_idxs_ + i);
                    if(row_offsets_ >= 0)
                    {
                        const long offset = (row_offsets_*rhs_cols_) + global_offset;
                        dense_tile_half[i]=__ldg(matrix_base_ +offset);
                    }
                
                }
            }
            for(int i=0;i<2;i++)
            *(dense_tile_ + i)=dense_tile_half[i];
                //        if( blockIdx.x==0 and blockIdx.y==0 and blockIdx.z==0 and threadIdx.x==0)
                // {
                //     printf("%f %f \n", __half2float( dense_tile_[0]), __half2float(dense_tile_[1]));
                //     half * l =reinterpret_cast< half *>(sparse_fragment_);
                //     printf("%f %f \n", __half2float( dense_tile_[0]), __half2float(dense_tile_[1]));
                // }

            // if(blockIdx.y==1 && threadIdx.x==32){
            // printf("666666 ");
            // for(int i=0;i<4;i++)
            // printf("%.1f  ",__half2float(dense_tile_half[i]));
            // printf("\n");}
            //warp内部开始shuffle
            //shuffle需要先打包需要交换的值
           
            
        }
    };

    //fp16 ns
    struct mmaDenseTile_fp16_ns{
        const float *  values_;
        const int *  column_idxs_;
        const int rhs_cols_;
        const int lane_id_;
        const int warpin_id;
        const int warp_id;
        const half *matrix_base_;
        half *dense_tile_;
        //存放当前线程拿到的一个double值，并将该double值拆分成4个half分别进行转置放置
        float *sparse_fragment_;

        __device__ __forceinline__ mmaDenseTile_fp16_ns(
        long row_offset_vec,
        const double * values,
        const int *  column_idxs,
	    int rhs_cols,
        int offset, 
        int lane_id, 
        const double*  matrix, 
        //row_offsets= column_indices_tile
        // const int *row_offsets,
        float * dense_tile,
        float *sparse_fragment):
            rhs_cols_(rhs_cols),
            lane_id_(lane_id),
            warpin_id(lane_id & 31),
            warp_id(lane_id>>5),
            //每行16个线程，每个线程搬运1个double,
            matrix_base_(reinterpret_cast<const half *>(matrix + offset)),
            // row_offsets_base_(row_offsets),
            values_(reinterpret_cast<const float *>(values + row_offset_vec*2) + (lane_id & 31)),
            column_idxs_(column_idxs + row_offset_vec + ((lane_id & 31)>>2)),
            dense_tile_(reinterpret_cast< half *>(dense_tile)),
            sparse_fragment_(sparse_fragment)
            {}
    
        __device__ __forceinline__ void Fetch(int colEdge, int dimN_index){

            sparse_fragment_[0]=__ldg(values_);
            const long row_offsets_ = __ldg(column_idxs_);
            values_ += 32;
            column_idxs_ += 8;
            const int global_offset = (warp_id<<4) + ((warpin_id&3)<<2);
            const long offset = (row_offsets_*rhs_cols_) + global_offset;
            at::Half dense_tile_half_temp[4]={0.0,0.0,0.0,0.0};
            half * dense_tile_half = reinterpret_cast<half *>(dense_tile_half_temp);
            for(int i=0;i<4;i++)
            {
                if((dimN_index + global_offset+i)<colEdge)
                dense_tile_half[i]=*(matrix_base_ +offset+ i);
            }
        
            // const int global_offset = (warp_id<<2) + (warpin_id&3);
            // dense_tile_half =reinterpret_cast<const half *>(matrix_base_ + (row_offsets_)*rhs_cols_ + global_offset);
            //warp内部开始shuffle
            //shuffle需要先打包需要交换的值
            
            //tmp交替写入dense_tile_
            for(int i=0; i<4; i++)
                *(dense_tile_+(warp_id<<7) + (warpin_id/4) + ((warpin_id%4)*32) + i*8)=dense_tile_half[i];
 
        }

        // Load the residual and compute the matrix product
        __device__ __forceinline__ void ResidueLoad(int colEdge, int dimN_index){
            sparse_fragment_[0]=__ldg(values_);
            const long row_offsets_ = __ldg(column_idxs_);
            const int global_offset = (warp_id<<4) + ((warpin_id&3)<<2);
            at::Half dense_tile_half_temp[4]={0.0,0.0,0.0,0.0};
            half * dense_tile_half = reinterpret_cast<half *>(dense_tile_half_temp);
            if(row_offsets_ >= 0){
                 const long offset = (row_offsets_*rhs_cols_) + global_offset;
                // matrix_base_=matrix_base_ + (row_offsets_*rhs_cols_) + global_offset;
                for(int i=0;i<4;i++)
                {
                    if((dimN_index + global_offset+i)<colEdge)
                    dense_tile_half[i]=*(matrix_base_ + offset+i);
                    // if(blockIdx.y==1 && threadIdx.x==32){
                    // printf("11111 ");
                    // printf("%.1f  ",__half2float(dense_tile_half[i]));
                    //  printf("%d  ", row_offsets_);
                    //  printf("%d  ", global_offset);
                    //  printf("\n");
                    //  }

                }
            }
            // if(blockIdx.y==1 && threadIdx.x==32){
            // printf("666666 ");
            // for(int i=0;i<4;i++)
            // printf("%.1f  ",__half2float(dense_tile_half[i]));
            // printf("\n");}
            //warp内部开始shuffle
            //shuffle需要先打包需要交换的值
            for(int i=0; i<4; i++)
                *(dense_tile_+(warp_id<<7) + (warpin_id/4) + ((warpin_id%4)*32) + i*8)=dense_tile_half[i];
            
        }
    };    




        //Tile_N = 128 threads_per_block = 128
    struct mmaDenseTile_tf32{
        const float *  values_;
        const int *  column_idxs_;
        const int rhs_cols_;
        const int lane_id_;
        const int warpin_id;
        const int warp_id;
        const float *matrix_base_;
        float *dense_tile_;
        //存放当前线程拿到的一个double值，并将该double值拆分成4个half分别进行转置放置
        float *sparse_fragment_;

        __device__ __forceinline__ mmaDenseTile_tf32(
        long row_offset_vec,
        const float * values,
        const int *  column_idxs,
	    int rhs_cols,
        int offset, 
        int lane_id, 
        const float*  matrix, 
        //row_offsets= column_indices_tile
        // const int *row_offsets,
        float * dense_tile,
        float *sparse_fragment):
            rhs_cols_(rhs_cols),
            lane_id_(lane_id),
            warpin_id(lane_id & 31),
            warp_id(lane_id>>5),
            //当前block在全局的列偏移
            matrix_base_(matrix + offset),
            //8的意思是vector的长度
            values_((values + row_offset_vec*8) + (lane_id & 31)),
            //对4*16的RHS读取，每行连续读8个线程，共4行，所以需要>>3
            column_idxs_(column_idxs + row_offset_vec + ((lane_id & 31)>>3)),
            dense_tile_(dense_tile),
            sparse_fragment_(sparse_fragment)
            {}
    
        __device__ __forceinline__ void Fetch(int colEdge, int dimN_index){

            sparse_fragment_[0]= __ldg(values_);
            const long row_offsets_ = __ldg(column_idxs_);
            values_ += 32;
            column_idxs_ += 4;
            // (warp_id<<4) 每个warp有16列
            //行偏移,(warpin_id%8)*2),每行8个线程，每个线程读两个float数
            const int global_offset = (warp_id<<4) + ((warpin_id%8)*2);
            const long offset = (row_offsets_*rhs_cols_) + global_offset;
            float dense_tile_fp32[2]={0.0,0.0};
            for(int i=0;i<2;i++)
            {
                if((dimN_index+global_offset+i)<colEdge)
                dense_tile_fp32[i]=__ldg(matrix_base_ +offset+ i);
            }     
            //    if(threadIdx.x==35 & blockIdx.y==0)
            // {
            //     printf("%d\n",warp_id);
            //     printf("%d\n",((warpin_id%8)*2)*4);
            //     printf("%d\n",(((warpin_id%8)*2)+1)*4);
            //     printf("%d\n",warpin_id/8);
            //     printf("%d\n",dense_tile_fp32[0]);
            //     printf("%d\n",dense_tile_fp32[1]);
            //     printf("%d\n",row_offsets_*rhs_cols_);
            //     printf("%d\n",global_offset);
            // }
            int k=(warpin_id/4)%2; //T0,1,2,3,8,9,...
            if(k==0){
                *(dense_tile_ + warp_id*64 + ((warpin_id%8)*2)*4 + warpin_id/8) = dense_tile_fp32[0];
                *(dense_tile_ +  warp_id*64 + (((warpin_id%8)*2)+1)*4  + warpin_id/8) = dense_tile_fp32[1];
            }
            else{
                *(dense_tile_ +  warp_id*64 + (((warpin_id%8)*2)+1)*4 + warpin_id/8) = dense_tile_fp32[1];
                *(dense_tile_ +  warp_id*64 + ((warpin_id%8)*2)*4 + warpin_id/8) =dense_tile_fp32[0];
            }
        }

        // Load the residual and compute the matrix product
        __device__ __forceinline__ void ResidueLoad(int colEdge, int dimN_index){
            sparse_fragment_[0]=__ldg(values_);
            const long row_offsets_ = __ldg(column_idxs_);
            const int global_offset = (warp_id<<4) + ((warpin_id%8)*2);
            float dense_tile_fp32[2]={0.0,0.0};
            if(row_offsets_ >= 0){
                const long offset = (row_offsets_*rhs_cols_) + global_offset;
                // matrix_base_=matrix_base_ + (row_offsets_*rhs_cols_) + global_offset;
                 for(int i=0;i<2;i++)
                 {
                    if((dimN_index+global_offset+i)<colEdge)
                    dense_tile_fp32[i]=__ldg(matrix_base_ +offset+ i);
                 }   
            }
            // if(threadIdx.x==36 & blockIdx.y==0)
            // {
            //     printf("%d\n",dimN_index+global_offset);
            //     printf("%d\n",colEdge);
            // }
            int k=(warpin_id/4)%2; //T0,1,2,3,8,9,...
            if(k==0){
                *(dense_tile_ +  warp_id*64 + ((warpin_id%8)*2)*4 + warpin_id/8) = dense_tile_fp32[0];
                *(dense_tile_ +  warp_id*64 + (((warpin_id%8)*2)+1)*4 + warpin_id/8) = dense_tile_fp32[1];
            }
            else{
                *(dense_tile_ +  warp_id*64 + (((warpin_id%8)*2)+1)*4 + warpin_id/8) = dense_tile_fp32[1];
                *(dense_tile_ +  warp_id*64 + ((warpin_id%8)*2)*4 + warpin_id/8) = dense_tile_fp32[0];
            }
        }
    };
    

    //16
    struct mmaDenseTile_tf32_16{
        const float *  values_;
        const int *  column_idxs_;
        const int rhs_cols_;
        const int lane_id_;
        const int warpin_id;
        const int warp_id;
        const float *matrix_base_;
        float *dense_tile_;
        //存放当前线程拿到的一个double值，并将该double值拆分成4个half分别进行转置放置
        float *sparse_fragment_;

        __device__ __forceinline__ mmaDenseTile_tf32_16(
        long row_offset_vec,
        const float * values,
        const int *  column_idxs,
	    int rhs_cols,
        int offset, 
        int lane_id, 
        const float*  matrix, 
        //row_offsets= column_indices_tile
        // const int *row_offsets,
        float * dense_tile,
        float *sparse_fragment):
            rhs_cols_(rhs_cols),
            lane_id_(lane_id),
            warpin_id(lane_id & 31),
            warp_id(lane_id>>5),
            //当前block在全局的列偏移
            matrix_base_(matrix + offset),
            //8的意思是vector的长度
            values_((values + row_offset_vec*16) + (lane_id & 31)),
            //对4*16的RHS读取，每行连续读8个线程，共4行，所以需要>>3
            column_idxs_(column_idxs + row_offset_vec + ((lane_id & 31)%4)),
            dense_tile_(dense_tile),
            sparse_fragment_(sparse_fragment)
            {}
    
        __device__ __forceinline__ void Fetch(int colEdge, int dimN_index){

            for(int i =0; i<2; i++){
            sparse_fragment_[i]= __ldg(values_);
            values_ += 32;}
            const long row_offsets_ = __ldg(column_idxs_);
            // values_ += 32;
            column_idxs_ += 4;
            // (warp_id<<4) 每个warp有16列
            //行偏移,(warpin_id%8)*2),每行8个线程，每个线程读两个float数
            const int global_offset = (warp_id<<3) + ((warpin_id/4));
            const long offset = (row_offsets_*rhs_cols_) + global_offset;

            float dense_tile_fp32[1]={0.0};
            if((dimN_index+global_offset)<colEdge)
            dense_tile_fp32[0] =__ldg(matrix_base_ +offset);
             
             *(dense_tile_)=dense_tile_fp32[0];
        }

        // Load the residual and compute the matrix product
        __device__ __forceinline__ void ResidueLoad(int colEdge, int dimN_index, int residue){

            values_ -= (4-residue)*(warpin_id/4);
            for(int i =0; i<2; i++){
                if((warpin_id%4) < residue)
                {
                    sparse_fragment_[i]= __ldg(values_);
                }else{
                    sparse_fragment_[i]= 0.0;
                }
                values_ += (8*residue);
            }


            long row_offsets_ = -1;
            if((warpin_id%4) < residue)
                row_offsets_ = __ldg(column_idxs_);
            const int global_offset = (warp_id<<3) + ((warpin_id/4));
            // float dense_tile_fp32[2]={0.0,0.0};
            float dense_tile_fp32[1]={0.0};
            if(row_offsets_ >= 0){
                const long offset = (row_offsets_*rhs_cols_) + global_offset;
                if((dimN_index+global_offset)<colEdge)
               dense_tile_fp32[0] = __ldg(matrix_base_ +offset);
                
            }
            *(dense_tile_)=dense_tile_fp32[0];

        }
    };

// tf32 - ns
    struct mmaDenseTile_tf32_ns{
        const float *  values_;
        const int *  column_idxs_;
        const int rhs_cols_;
        const int lane_id_;
        const int warpin_id;
        const int warp_id;
        const float *matrix_base_;
        float *dense_tile_;
        //存放当前线程拿到的一个double值，并将该double值拆分成4个half分别进行转置放置
        float *sparse_fragment_;

        __device__ __forceinline__ mmaDenseTile_tf32_ns(
        long row_offset_vec,
        const float * values,
        const int *  column_idxs,
	    int rhs_cols,
        int offset, 
        int lane_id, 
        const float*  matrix, 
        //row_offsets= column_indices_tile
        // const int *row_offsets,
        float * dense_tile,
        float *sparse_fragment):
            rhs_cols_(rhs_cols),
            lane_id_(lane_id),
            warpin_id(lane_id & 31),
            warp_id(lane_id>>5),
            //当前block在全局的列偏移
            matrix_base_(matrix + offset),
            //8的意思是vector的长度
            values_((values + row_offset_vec*8) + (lane_id & 31)),
            //对4*16的RHS读取，每行连续读8个线程，共4行，所以需要>>3
            column_idxs_(column_idxs + row_offset_vec + ((lane_id & 31)>>3)),
            dense_tile_(dense_tile),
            sparse_fragment_(sparse_fragment)
            {}
    
        __device__ __forceinline__ void Fetch(int colEdge, int dimN_index){

            sparse_fragment_[0]= __ldg(values_);
            const long row_offsets_ = __ldg(column_idxs_);
            values_ += 32;
            column_idxs_ += 4;
            // (warp_id<<4) 每个warp有16列
            //行偏移,(warpin_id%8)*2),每行8个线程，每个线程读两个float数
            const int global_offset = (warp_id<<4) + ((warpin_id%8)*2);
            const long offset = (row_offsets_*rhs_cols_) + global_offset;
            float dense_tile_fp32[2]={0.0,0.0};
            for(int i=0;i<2;i++)
            {
                if((dimN_index+global_offset+i)<colEdge)
                dense_tile_fp32[i]=__ldg(matrix_base_ +offset+ i);
            }     
            //    if(threadIdx.x==35 & blockIdx.y==0)
            // {
            //     printf("%d\n",warp_id);
            //     printf("%d\n",((warpin_id%8)*2)*4);
            //     printf("%d\n",(((warpin_id%8)*2)+1)*4);
            //     printf("%d\n",warpin_id/8);
            //     printf("%d\n",dense_tile_fp32[0]);
            //     printf("%d\n",dense_tile_fp32[1]);
            //     printf("%d\n",row_offsets_*rhs_cols_);
            //     printf("%d\n",global_offset);
            // }
     
                *(dense_tile_ + warp_id*64 + ((warpin_id%8)*2)*4 + warpin_id/8) = dense_tile_fp32[0];
                *(dense_tile_ +  warp_id*64 + (((warpin_id%8)*2)+1)*4  + warpin_id/8) = dense_tile_fp32[1];


        }

        // Load the residual and compute the matrix product
        __device__ __forceinline__ void ResidueLoad(int colEdge, int dimN_index){
            sparse_fragment_[0]=__ldg(values_);
            const long row_offsets_ = __ldg(column_idxs_);
            const int global_offset = (warp_id<<4) + ((warpin_id%8)*2);
            float dense_tile_fp32[2]={0.0,0.0};
            if(row_offsets_ >= 0){
                const long offset = (row_offsets_*rhs_cols_) + global_offset;
                // matrix_base_=matrix_base_ + (row_offsets_*rhs_cols_) + global_offset;
                 for(int i=0;i<2;i++)
                 {
                    if((dimN_index+global_offset+i)<colEdge)
                    dense_tile_fp32[i]=__ldg(matrix_base_ +offset+ i);
                 }   
            }
            // if(threadIdx.x==36 & blockIdx.y==0)
            // {
            //     printf("%d\n",dimN_index+global_offset);
            //     printf("%d\n",colEdge);
            // }
            // int k=(warpin_id/4)%2; //T0,1,2,3,8,9,...
            // if(k==0){
                *(dense_tile_ +  warp_id*64 + ((warpin_id%8)*2)*4 + warpin_id/8) = dense_tile_fp32[0];
                *(dense_tile_ +  warp_id*64 + (((warpin_id%8)*2)+1)*4 + warpin_id/8) = dense_tile_fp32[1];
            // }
            // else{
            //     *(dense_tile_ +  warp_id*64 + (((warpin_id%8)*2)+1)*4 + warpin_id/8) = dense_tile_fp32[1];
            //     *(dense_tile_ +  warp_id*64 + ((warpin_id%8)*2)*4 + warpin_id/8) = dense_tile_fp32[0];
            // }
        }
    };


        //Tile_N = 128 threads_per_block = 128
    struct mmaDenseTile_fp16_v2{
        const float *  values_;
        const int *  column_idxs_;
        const int rhs_cols_;
        const int lane_id_;
        const int warpin_id;
        const int warp_id;
        const half2 *matrix_base_;
        half2 *dense_tile_;
        //存放当前线程拿到的一个double值，并将该double值拆分成4个half分别进行转置放置
        float *sparse_fragment_;

        __device__ __forceinline__ mmaDenseTile_fp16_v2(
        long row_offset_vec,
        const double * values,
        const int *  column_idxs,
	    int rhs_cols,
        int offset, 
        int lane_id, 
        const double*  matrix, 
        //row_offsets= column_indices_tile
        // const int *row_offsets,
        float * dense_tile,
        float *sparse_fragment):
            rhs_cols_(rhs_cols),
            lane_id_(lane_id),
            warpin_id(lane_id & 31),
            warp_id(lane_id>>5),
            //每行16个线程，每个线程搬运1个double,
            matrix_base_(reinterpret_cast<const half2 *>(matrix + offset)),
            // row_offsets_base_(row_offsets),
            values_(reinterpret_cast<const float *>(values + row_offset_vec*2) + (lane_id & 31)),
            column_idxs_(column_idxs + row_offset_vec + (((lane_id & 31)%4)*2)),
            dense_tile_(reinterpret_cast< half2 *>(dense_tile)),
            sparse_fragment_(sparse_fragment)
            {}
    
        __device__ __forceinline__ void Fetch(int colEdge, int dimN_index){

            sparse_fragment_[0]=__ldg(values_);
            values_ += 32;
                // half * sparse_fragment_fp16 = reinterpret_cast< half *>(sparse_fragment_);
                // if(blockIdx.x==0 and blockIdx.y==0 and threadIdx.x==4)
                // {
                //     printf("%f, %f\n", __half2float(sparse_fragment_fp16[0]), __half2float(sparse_fragment_fp16[1]));
                // }

            int temp = 0;
            if((warpin_id%8)>=4) temp = 1;
            long col_temp = __ldg(column_idxs_ + temp);
            for(int i=0;i<2;i++)
            {
                 int global_offset = (warp_id<<3) + (warpin_id/8) + (i*4);
                 long offset = (col_temp*rhs_cols_/2) + global_offset;
                dense_tile_[i]=__ldg(matrix_base_ +offset);
            }
            //temp=0: 分别取第1,3个数； temp=1: 分别取第0,2个数；
            half2 ex;
            if(temp == 0){
                ex.x =  dense_tile_[0].y;
                ex.y =  dense_tile_[1].y;
            }else{
                ex.x =  dense_tile_[0].x;
                ex.y =  dense_tile_[1].x;
            }
            //做shuffle
            ex = __shfl_xor_sync(0xffffffff, ex, 4,32);
            //shuffle完，更新dense_tile_
            if(temp == 0){
                dense_tile_[0].y = ex.x;
                dense_tile_[1].y = ex.y;
            }else{
                dense_tile_[0].x = ex.x;
                dense_tile_[1].x = ex.y;
            }
            column_idxs_ += 8;
        }

        // Load the residual and compute the matrix product
        __device__ __forceinline__ void ResidueLoad(int colEdge, int dimN_index, int residue){
            int col_offset = (warpin_id%4)*2;
            half * sparse_fragment_fp16 = reinterpret_cast<half *>(sparse_fragment_);
            const half * values_fp16 = reinterpret_cast<const half *>(values_);
            values_fp16 -= (8-residue)*(warpin_id/4);
            for(int c=0;c<2;c++){
                if((col_offset + c )< residue)
                sparse_fragment_fp16[c] = __ldg(values_fp16+c);
                else
                sparse_fragment_fp16[c] = 0.0;

                // if(blockIdx.x==0 and blockIdx.y==0 and threadIdx.x==4)
                // {
                //     printf("%f\n", __half2float(sparse_fragment_fp16[c]));
                // }
            }

            int temp = 0;
            if((warpin_id%8)>=4) temp = 1;
            long col_temp = -1;

            if((col_offset + temp )< residue)
            col_temp = __ldg(column_idxs_ + temp);
  
            
            if(col_temp >= 0)
            {
                for(int i=0;i<2;i++)
                {
                        int global_offset = (warp_id<<3) + (warpin_id/8) + (i*4);
                        long offset = (col_temp*rhs_cols_/2) + global_offset;
                        dense_tile_[i]=__ldg(matrix_base_ +offset);
                }
            }else{
                for(int i=0;i<2;i++)
                dense_tile_[i] = __floats2half2_rn(0.0f, 0.0f);
            }

            half2 ex;
            if(temp == 0){
                ex.x =  dense_tile_[0].y;
                ex.y =  dense_tile_[1].y;
            }else{
                ex.x =  dense_tile_[0].x;
                ex.y =  dense_tile_[1].x;
            }
            //做shuffle
            ex = __shfl_xor_sync(0xffffffff, ex, 4,32);
            //shuffle完，更新dense_tile_
            if(temp == 0){
                dense_tile_[0].y = ex.x;
                dense_tile_[1].y = ex.y;
            }else{
                dense_tile_[0].x = ex.x;
                dense_tile_[1].x = ex.y;
            }
            // if(blockIdx.x==0 and blockIdx.y==0 and threadIdx.x==4)
            // {
            //     printf("%f, %f, %f, %f\n",  __half2float(dense_tile_[0].x),  __half2float(dense_tile_[0].y),  __half2float(dense_tile_[1].x),  __half2float(dense_tile_[1].y));
            // }
        }
    };
struct mmaDenseTile_fp16_ori{
        const float *  values_;
        const int *  column_idxs_;
        const int rhs_cols_;
        const int lane_id_;
        const int warpin_id;
        const int warp_id;
        const half2 *matrix_base_;
        half2 *dense_tile_;
        //存放当前线程拿到的一个double值，并将该double值拆分成4个half分别进行转置放置
        float *sparse_fragment_;

        __device__ __forceinline__ mmaDenseTile_fp16_ori(
        long row_offset_vec,
        const double * values,
        const int *  column_idxs,
	    int rhs_cols,
        int offset, 
        int lane_id, 
        const double*  matrix, 
        //row_offsets= column_indices_tile
        // const int *row_offsets,
        float * dense_tile,
        float *sparse_fragment):
            rhs_cols_(rhs_cols),
            lane_id_(lane_id),
            warpin_id(lane_id & 31),
            warp_id(lane_id>>5),
            //每行16个线程，每个线程搬运1个double,
            matrix_base_(reinterpret_cast<const half2 *>(matrix + offset)),
            // row_offsets_base_(row_offsets),
            values_(reinterpret_cast<const float *>(values + row_offset_vec*2) + (lane_id & 31)),
            column_idxs_(column_idxs + row_offset_vec + (((lane_id & 31)%4)*2)),
            dense_tile_(reinterpret_cast< half2 *>(dense_tile)),
            sparse_fragment_(sparse_fragment)
            {}
    
        __device__ __forceinline__ void Fetch(int colEdge, int dimN_index){

            sparse_fragment_[0]=__ldg(values_);
            values_ += 32;
            int temp = 0;
            if((warpin_id%8)>=4) temp = 1;
            long col_temp = __ldg(column_idxs_ + temp);
            for(int i=0;i<2;i++)
            {
                 int global_offset = (warp_id<<3) + (warpin_id/8) + (i*4);
                 long offset = (col_temp*rhs_cols_/2) + global_offset;
                dense_tile_[i]=__ldg(matrix_base_ +offset);
            }
            //temp=0: 分别取第1,3个数； temp=1: 分别取第0,2个数；
            half2 ex;
            if(temp == 0){
                ex.x =  dense_tile_[0].y;
                ex.y =  dense_tile_[1].y;
            }else{
                ex.x =  dense_tile_[0].x;
                ex.y =  dense_tile_[1].x;
            }
            //做shuffle
            ex = __shfl_xor_sync(0xffffffff, ex, 4,32);
            //shuffle完，更新dense_tile_
            if(temp == 0){
                dense_tile_[0].y = ex.x;
                dense_tile_[1].y = ex.y;
            }else{
                dense_tile_[0].x = ex.x;
                dense_tile_[1].x = ex.y;
            }
            column_idxs_ += 8;

        }

        // Load the residual and compute the matrix product
        __device__ __forceinline__ void ResidueLoad(int colEdge, int dimN_index){


            sparse_fragment_[0]=__ldg(values_);

            int temp = 0;
            if((warpin_id%8)>=4) temp = 1;

            long col_temp = __ldg(column_idxs_ + temp);
            if(col_temp >= 0)
            {
                for(int i=0;i<2;i++)
                {
                        int global_offset = (warp_id<<3) + (warpin_id/8) + (i*4);
                        long offset = (col_temp*rhs_cols_/2) + global_offset;
                        dense_tile_[i]=__ldg(matrix_base_ +offset);
                }
            }else{
                for(int i=0;i<2;i++)
                dense_tile_[i] = __floats2half2_rn(0.0f, 0.0f);
            }

            half2 ex;
            if(temp == 0){
                ex.x =  dense_tile_[0].y;
                ex.y =  dense_tile_[1].y;
            }else{
                ex.x =  dense_tile_[0].x;
                ex.y =  dense_tile_[1].x;
            }
            //做shuffle
            ex = __shfl_xor_sync(0xffffffff, ex, 4,32);
            //shuffle完，更新dense_tile_
            if(temp == 0){
                dense_tile_[0].y = ex.x;
                dense_tile_[1].y = ex.y;
            }else{
                dense_tile_[0].x = ex.x;
                dense_tile_[1].x = ex.y;
            }
            // if(blockIdx.x==0 and blockIdx.y==0 and threadIdx.x==4)
            // {
            //     printf("%f, %f, %f, %f\n",  __half2float(dense_tile_[0].x),  __half2float(dense_tile_[0].y),  __half2float(dense_tile_[1].x),  __half2float(dense_tile_[1].y));
            // }
            //             if(blockIdx.x==0 and blockIdx.y==1 and threadIdx.x==0)
            // {
            //     half * temp = reinterpret_cast< half *>(sparse_fragment_);
            //     printf("%f, %f\n",  __half2float(temp[0]),  __half2float(temp[1]));
            //     printf("%d, %f, %f, %f, %f\n", col_temp,  __half2float(dense_tile_[0].x),  __half2float(dense_tile_[0].y),  __half2float(dense_tile_[1].x),  __half2float(dense_tile_[1].y));
            // }
        }
    };
        //Tile_N = 128 threads_per_block = 128
    struct mmaDenseTile_tf32_v2{
        const float *  values_;
        const int *  column_idxs_;
        const int rhs_cols_;
        const int lane_id_;
        const int warpin_id;
        const int warp_id;
        const float *matrix_base_;
        float *dense_tile_;
        //存放当前线程拿到的一个double值，并将该double值拆分成4个half分别进行转置放置
        float *sparse_fragment_;

        __device__ __forceinline__ mmaDenseTile_tf32_v2(
        long row_offset_vec,
        const float * values,
        const int *  column_idxs,
	    int rhs_cols,
        int offset, 
        int lane_id, 
        const float*  matrix, 
        //row_offsets= column_indices_tile
        // const int *row_offsets,
        float * dense_tile,
        float *sparse_fragment):
            rhs_cols_(rhs_cols),
            lane_id_(lane_id),
            warpin_id(lane_id & 31),
            warp_id(lane_id>>5),
            //当前block在全局的列偏移
            matrix_base_(matrix + offset),
            //8的意思是vector的长度
            values_((values + row_offset_vec*8) + (lane_id & 31)),
            //对4*16的RHS读取，每行连续读8个线程，共4行，所以需要>>3
            column_idxs_(column_idxs + row_offset_vec + ((lane_id & 31)%4)),
            dense_tile_(dense_tile),
            sparse_fragment_(sparse_fragment)
            {}
    
        __device__ __forceinline__ void Fetch(int colEdge, int dimN_index){

            sparse_fragment_[0]= __ldg(values_);
            const long row_offsets_ = __ldg(column_idxs_);
            values_ += 32;
            column_idxs_ += 4;
            // (warp_id<<4) 每个warp有16列
            //行偏移,(warpin_id%8)*2),每行8个线程，每个线程读两个float数
            const int global_offset = (warp_id<<4) + (warpin_id/4);
            const long offset = (row_offsets_*rhs_cols_) + global_offset;
            // float dense_tile_fp32[2]={0.0,0.0};
            for(int i=0;i<2;i++)
            {
                // if((dimN_index+global_offset+i)<colEdge)
                dense_tile_[i]=__ldg(matrix_base_ + offset + i*8);
            } 
            // for(int i=0;i<2;i++)
            //     dense_tile_[i]=dense_tile_fp32[i];    
        }

        // Load the residual and compute the matrix product
        __device__ __forceinline__ void ResidueLoad(int colEdge, int dimN_index, int residue){
            int col_offset = (warpin_id%4);
            long row_offsets_ = -1;
            values_ -= (4-residue)*(warpin_id/4);
            if(col_offset < residue){
                sparse_fragment_[0]=__ldg(values_);
                row_offsets_ = __ldg(column_idxs_);
            }
            const int global_offset = (warp_id<<4) + (warpin_id/4);
            // float dense_tile_fp32[2]={0.0,0.0};
            if(row_offsets_ >= 0){
                const long offset = (row_offsets_*rhs_cols_) + global_offset;
                // matrix_base_=matrix_base_ + (row_offsets_*rhs_cols_) + global_offset;
                 for(int i=0;i<2;i++)
                 {
                    // if((dimN_index+global_offset+i*8)<colEdge)
                    dense_tile_[i]=__ldg(matrix_base_ +offset+ i*8);
                 }   
            }else{
                dense_tile_[0] = 0.0;
                dense_tile_[1] = 0.0;
            }
            // for(int i=0;i<2;i++)
            //     dense_tile_[i]=dense_tile_fp32[i];
        }
    };


        //Tile_N = 128 threads_per_block = 128
    struct mmaDenseTile_tf32_v2_map{
        const float *  values_;
        const int *  column_idxs_;
        const int rhs_cols_;
        const int lane_id_;
        const int warpin_id;
        const int warp_id;
        const float2 *matrix_base_;
        float *dense_tile_;
        //存放当前线程拿到的一个double值，并将该double值拆分成4个half分别进行转置放置
        float *sparse_fragment_;

        __device__ __forceinline__ mmaDenseTile_tf32_v2_map(
        long row_offset_vec,
        const float * values,
        const int *  column_idxs,
	    int rhs_cols,
        int offset, 
        int lane_id, 
        const float*  matrix, 
        //row_offsets= column_indices_tile
        // const int *row_offsets,
        float * dense_tile,
        float *sparse_fragment):
            rhs_cols_(rhs_cols),
            lane_id_(lane_id),
            warpin_id(lane_id & 31),
            warp_id(lane_id>>5),
            //当前block在全局的列偏移
            matrix_base_(reinterpret_cast<const float2 *>(matrix + offset)),
            //8的意思是vector的长度
            values_((values + row_offset_vec*8) + (lane_id & 31)),
            //对4*16的RHS读取，每行连续读8个线程，共4行，所以需要>>3
            column_idxs_(column_idxs + row_offset_vec + ((lane_id & 31)%4)),
            dense_tile_(dense_tile),
            sparse_fragment_(sparse_fragment)
            {}
    
        __device__ __forceinline__ void Fetch(int colEdge, int dimN_index){

            sparse_fragment_[0]= __ldg(values_);
            const long row_offsets_ = __ldg(column_idxs_);
            values_ += 32;
            column_idxs_ += 4;
            // (warp_id<<4) 每个warp有16列
            //行偏移,(warpin_id%8)*2),每行8个线程，每个线程读两个float数
            const int global_offset = (warp_id<<3) + (warpin_id/4);
            const long offset = (row_offsets_*rhs_cols_/2) + global_offset;
            // float dense_tile_fp32[2]={0.0,0.0};
            // for(int i=0;i<2;i++)
            // {
            //     dense_tile_[i]=__ldg(matrix_base_ + offset + i*8);
            // } 
            float2 temp = __ldg(matrix_base_ + offset);
            dense_tile_[0] = temp.x;
            dense_tile_[1] = temp.y;
        }

        // Load the residual and compute the matrix product
        __device__ __forceinline__ void ResidueLoad(int colEdge, int dimN_index, int residue){
            int col_offset = (warpin_id%4);
            long row_offsets_ = -1;
            values_ -= (4-residue)*(warpin_id/4);
            if(col_offset < residue){
                sparse_fragment_[0]=__ldg(values_);
                row_offsets_ = __ldg(column_idxs_);
            }
            const int global_offset = (warp_id<<3) + (warpin_id/4);
            if(row_offsets_ >= 0){
                const long offset = (row_offsets_*rhs_cols_/2) + global_offset;
                float2 temp = __ldg(matrix_base_ + offset);
                dense_tile_[0] = temp.x;
                dense_tile_[1] = temp.y;
            }else{
                dense_tile_[0] = 0.0;
                dense_tile_[1] = 0.0;
            }

        }
    };

    struct mmaDenseTile_tf32_ones{
        const float *  values_;
        const int *  column_idxs_;
        const int rhs_cols_;
        const int lane_id_;
        const int warpin_id;
        const int warp_id;
        const float *matrix_base_;
        float *dense_tile_;
        //存放当前线程拿到的一个double值，并将该double值拆分成4个half分别进行转置放置
        float *sparse_fragment_;

        __device__ __forceinline__ mmaDenseTile_tf32_ones(
        long row_offset_vec,
        const float * values,
        const int *  column_idxs,
	    int rhs_cols,
        int offset, 
        int lane_id, 
        const float*  matrix, 
        //row_offsets= column_indices_tile
        // const int *row_offsets,
        float * dense_tile,
        float *sparse_fragment):
            rhs_cols_(rhs_cols),
            lane_id_(lane_id),
            warpin_id(lane_id & 31),
            warp_id(lane_id>>5),
            //当前block在全局的列偏移
            matrix_base_(matrix + offset),
            //8的意思是vector的长度
            values_((values + row_offset_vec*8) + (lane_id & 31)),
            //对4*16的RHS读取，每行连续读8个线程，共4行，所以需要>>3
            column_idxs_(column_idxs + row_offset_vec + ((lane_id & 31)%4)),
            dense_tile_(dense_tile),
            sparse_fragment_(sparse_fragment)
            {}
    
        __device__ __forceinline__ void Fetch(int colEdge, int dimN_index){

            sparse_fragment_[0]= __ldg(values_);
            const long row_offsets_ = __ldg(column_idxs_);
            values_ += 32;
            column_idxs_ += 4;
            // (warp_id<<4) 每个warp有16列
            //行偏移,(warpin_id%8)*2),每行8个线程，每个线程读两个float数
            const int global_offset = (warp_id<<4) + (warpin_id/4);
            const long offset = (row_offsets_*rhs_cols_) + global_offset;
            // float dense_tile_fp32[2]={0.0,0.0};
            for(int i=0;i<2;i++)
            {
                if((dimN_index+global_offset+i)<colEdge)
                dense_tile_[i]=__ldg(matrix_base_ + offset + i*8);
                else 
                dense_tile_[i] = 0.0;
            } 
            // for(int i=0;i<2;i++)
            //     dense_tile_[i]=dense_tile_fp32[i];    
        }

        // Load the residual and compute the matrix product
        __device__ __forceinline__ void ResidueLoad(int colEdge, int dimN_index, int residue){
            int col_offset = (warpin_id%4);
            long row_offsets_ = -1;
            values_ -= (4-residue)*(warpin_id/4);
            if(col_offset < residue){
                sparse_fragment_[0]=__ldg(values_);
                row_offsets_ = __ldg(column_idxs_);
            }
            const int global_offset = (warp_id<<4) + (warpin_id/4);
            // float dense_tile_fp32[2]={0.0,0.0};
            if(row_offsets_ >= 0){
                const long offset = (row_offsets_*rhs_cols_) + global_offset;
                // matrix_base_=matrix_base_ + (row_offsets_*rhs_cols_) + global_offset;
                 for(int i=0;i<2;i++)
                 {
                    if((dimN_index+global_offset+i*8)<colEdge)
                        dense_tile_[i]=__ldg(matrix_base_ +offset+ i*8);
                    else 
                    dense_tile_[i] = 0.0;
                 }   
            }
            // for(int i=0;i<2;i++)
            //     dense_tile_[i]=dense_tile_fp32[i];
        }
    };

    //Tile_N = 128 threads_per_block = 128
    struct mmaDenseTile_fp16_ori_v2{
        const float *  values_;
        const int *  column_idxs_;
        const int rhs_cols_;
        const int lane_id_;
        const int warpin_id;
        const int warp_id;
        const half *matrix_base_;
        half *dense_tile_;
        //存放当前线程拿到的一个double值，并将该double值拆分成4个half分别进行转置放置
        float *sparse_fragment_;

        __device__ __forceinline__ mmaDenseTile_fp16_ori_v2(
        long row_offset_vec,
        const double * values,
        const int *  column_idxs,
	    int rhs_cols,
        int offset, 
        int lane_id, 
        const double*  matrix, 
        //row_offsets= column_indices_tile
        // const int *row_offsets,
        float * dense_tile,
        float *sparse_fragment):
            rhs_cols_(rhs_cols),
            lane_id_(lane_id),
            warpin_id(lane_id & 31),
            warp_id(lane_id>>5),
            //每行16个线程，每个线程搬运1个double,
            matrix_base_(reinterpret_cast<const half *>(matrix + offset)),
            // row_offsets_base_(row_offsets),
            values_(reinterpret_cast<const float *>(values + row_offset_vec*2) + (lane_id & 31)),
            column_idxs_(column_idxs + row_offset_vec + (((lane_id & 31)%4)*2)),
            dense_tile_(reinterpret_cast< half *>(dense_tile)),
            sparse_fragment_(sparse_fragment)
            {}
    
        __device__ __forceinline__ void Fetch(int colEdge, int dimN_index){

            sparse_fragment_[0]=__ldg(values_);
            values_ += 32;
            long col_temp[2];
            for(int k=0; k<2; k++)
                col_temp[k] = __ldg(column_idxs_ + k);
            for(int i=0;i<2;i++)
            {
                const int global_offset = (warp_id<<4) + (warpin_id/4) + (i*8);
                for(int j=0; j<2; j++)
                {
                    const long offset = (col_temp[j]*rhs_cols_) + global_offset;
                    dense_tile_[i*2+j]=__ldg(matrix_base_ +offset);
                }
            }
            column_idxs_ += 8;
        }

        // Load the residual and compute the matrix product
        __device__ __forceinline__ void ResidueLoad(int colEdge, int dimN_index, int residue){
            int col_offset = (warpin_id%4)*2;
            half * sparse_fragment_fp16 = reinterpret_cast<half *>(sparse_fragment_);
            const half * values_fp16 = reinterpret_cast<const half *>(values_);
            values_fp16 -= (8-residue)*(warpin_id/4);
            for(int c=0;c<2;c++){
                if((col_offset + c )< residue)
                sparse_fragment_fp16[c] = __ldg(values_fp16+c);
                else
                sparse_fragment_fp16[c] = 0.0;

                // if(blockIdx.x==0 and blockIdx.y==0 and threadIdx.x==4)
                // {
                //     printf("%f\n", __half2float(sparse_fragment_fp16[c]));
                // }
            }
            
            long col_temp[2] = {-1, -1};
            for(int k=0; k<2; k++)
                if((col_offset + k )< residue)
                    col_temp[k] = __ldg(column_idxs_ + k);

            // at::Half dense_tile_half_temp[4]={0.0,0.0,0.0,0.0};
            // half * dense_tile_half = reinterpret_cast<half *>(dense_tile_half_temp);
            for(int i=0;i<2;i++)
            {
                const int global_offset = (warp_id<<4) + (warpin_id/4) + (i*8);
                    for(int j=0; j<2; j++)
                    {
                        if(col_temp[j] >= 0)
                        {
                            const long offset = (col_temp[j]*rhs_cols_) + global_offset;
                            dense_tile_[i*2+j]=__ldg(matrix_base_ +offset);
                        }else{
                            dense_tile_[i*2+j]=0.0;
                        }
                    }
            }

            // for(int i=0;i<4;i++)
            //     *(dense_tile_ + i)=dense_tile_half[i];
        }
    };

    //Tile_N = 128 threads_per_block = 128
    struct mmaDenseTile_fp16_map{
        const float *  values_;
        const int *  column_idxs_;
        const int rhs_cols_;
        const int lane_id_;
        const int warpin_id;
        const int warp_id;
        const half2 *matrix_base_;
        half *dense_tile_;
        //存放当前线程拿到的一个double值，并将该double值拆分成4个half分别进行转置放置
        float *sparse_fragment_;

        __device__ __forceinline__ mmaDenseTile_fp16_map(
        long row_offset_vec,
        const double * values,
        const int *  column_idxs,
	    int rhs_cols,
        int offset, 
        int lane_id, 
        const double*  matrix, 
        //row_offsets= column_indices_tile
        // const int *row_offsets,
        float * dense_tile,
        float *sparse_fragment):
            rhs_cols_(rhs_cols),
            lane_id_(lane_id),
            warpin_id(lane_id & 31),
            warp_id(lane_id>>5),
            //每行16个线程，每个线程搬运1个double,
            matrix_base_(reinterpret_cast<const half2 *>(matrix + offset)),
            // row_offsets_base_(row_offsets),
            values_(reinterpret_cast<const float *>(values + row_offset_vec*2) + (lane_id & 31)),
            column_idxs_(column_idxs + row_offset_vec + (((lane_id & 31)%4)*2)),
            dense_tile_(reinterpret_cast< half *>(dense_tile)),
            sparse_fragment_(sparse_fragment)
            {}
    
        __device__ __forceinline__ void Fetch(int colEdge, int dimN_index){

            sparse_fragment_[0]=__ldg(values_);
            values_ += 32;
            long col_temp[2];
            for(int k=0; k<2; k++)
                col_temp[k] = __ldg(column_idxs_ + k);
            for(int i=0;i<2;i++)
            {
                const int global_offset = (warp_id<<3) + (warpin_id/4);

                const long offset = (col_temp[i]*(rhs_cols_/2)) + global_offset;
                half2 temp = __ldg(matrix_base_ +offset);
                dense_tile_[i]= temp.x;
                dense_tile_[i + 2]= temp.y;
            
            }
            column_idxs_ += 8;
        }

        // Load the residual and compute the matrix product
        __device__ __forceinline__ void ResidueLoad(int colEdge, int dimN_index, int residue){
            int col_offset = (warpin_id%4)*2;
            half * sparse_fragment_fp16 = reinterpret_cast<half *>(sparse_fragment_);
            const half * values_fp16 = reinterpret_cast<const half *>(values_);
            values_fp16 -= (8-residue)*(warpin_id/4);
            for(int c=0;c<2;c++){
                if((col_offset + c )< residue)
                sparse_fragment_fp16[c] = __ldg(values_fp16+c);
                else
                sparse_fragment_fp16[c] = 0.0;

                // if(blockIdx.x==0 and blockIdx.y==0 and threadIdx.x==4)
                // {
                //     printf("%f\n", __half2float(sparse_fragment_fp16[c]));
                // }
            }
            
            long col_temp[2] = {-1, -1};
            for(int k=0; k<2; k++)
                if((col_offset + k )< residue)
                    col_temp[k] = __ldg(column_idxs_ + k);

            // at::Half dense_tile_half_temp[4]={0.0,0.0,0.0,0.0};
            // half * dense_tile_half = reinterpret_cast<half *>(dense_tile_half_temp);
            for(int i=0;i<2;i++)
            {
                const int global_offset = (warp_id<<3) + (warpin_id/4);

                if(col_temp[i] >= 0)
                {
                    const long offset = (col_temp[i]*(rhs_cols_/2)) + global_offset;
                    half2 temp = __ldg(matrix_base_ +offset);
                    dense_tile_[i]= temp.x;
                    dense_tile_[i + 2]= temp.y;
                }else{
                    dense_tile_[i]= 0.0;
                    dense_tile_[i + 2]= 0.0;
                }
                
            }

            // for(int i=0;i<4;i++)
            //     *(dense_tile_ + i)=dense_tile_half[i];
        }
    };

    // //Tile_N = 128 threads_per_block = 128
    // struct mmaDenseTile_fp16_test{
    //     const float *  values_;
    //     const int *  column_idxs_;
    //     const int rhs_cols_;
    //     const int lane_id_;
    //      int warpin_id;
    //      int warp_id;
    //     const float2 *matrix_base_;
    //     half *dense_tile_;
    //     float *dense_tile_share_;
    //     half *dense_tile_share_half;
    //     //存放当前线程拿到的一个double值，并将该double值拆分成4个half分别进行转置放置
    //     float *sparse_fragment_;
    //     int row1;
    //     int col1;
    //     int group;

    //     __device__ __forceinline__ mmaDenseTile_fp16_test(
    //     long row_offset_vec,
    //     const double * values,
    //     const int *  column_idxs,
	//     int rhs_cols,
    //     int offset, 
    //     int lane_id, 
    //     const float2*  matrix, 
    //     //row_offsets= column_indices_tile
    //     // const int *row_offsets,
    //     float * dense_tile,
    //     float * dense_tile_share,
    //     float *sparse_fragment):
    //         rhs_cols_(rhs_cols),
    //         lane_id_(lane_id),
    //         // warpin_id(lane_id & 31),
    //         // warp_id(lane_id>>5),
    //         //每行16个线程，每个线程搬运1个double,
    //         matrix_base_(matrix + offset),
    //         // row_offsets_base_(row_offsets),
    //         values_(reinterpret_cast<const float *>(values + row_offset_vec*2) + (lane_id & 31)),
    //         // column_idxs_(column_idxs + row_offset_vec),
    //         dense_tile_(reinterpret_cast< half *>(dense_tile)),
    //         sparse_fragment_(sparse_fragment)
    //         {
    //             warpin_id = lane_id & 31;
    //             warp_id = lane_id>>5;
    //             //计算row偏移
    //             row1 = warpin_id / 2;
    //             if(warpin_id>=16)  row1 -=8;
    //             column_idxs_ = column_idxs + row_offset_vec + row1;
    //             //计算col偏移
    //             col1 = (warpin_id%2)*2 + warpin_id/16;

    //             //计算第几组
    //             // group = warpin_id/4;
    //             group = (warpin_id/16)*4 + warpin_id%4;
    //             int group_id =  warpin_id / 4;
    //             if(warpin_id>=16)  group_id -=4;
    //             dense_tile_share_ = dense_tile_share + warp_id*64 + (warpin_id/4)*8;
    //             dense_tile_share_half = reinterpret_cast< half *>(dense_tile_share + warp_id*64 + group*8) + group_id;
    //             dense_tile_share_ += (warpin_id%4)*2;
    //         }
    
    //     __device__ __forceinline__ void Fetch(int colEdge, int dimN_index){

    //         // if( blockIdx.x==0 and  blockIdx.y==0 and blockIdx.z==0 and threadIdx.x==5){
    //         //     printf("%d,%d,%d\n", row1, col1, group);
    //         // }
    //         sparse_fragment_[0]=__ldg(values_);
    //         values_ += 32;
    //         long col_temp = __ldg(column_idxs_);

    //         const int global_offset = (warp_id<<4) + col1*4;

    //         const long offset = (col_temp*rhs_cols_) + global_offset;
    //         float2 temp = __ldg(matrix_base_ +offset/4);

    //         // dense_tile_share_ += (warpin_id%4)*2;
    //         // if(warpin_id>15){
    //             *(dense_tile_share_+1 )= temp.y;
    //             *(dense_tile_share_)= temp.x;
    //         // }else{
    //         //     *(dense_tile_share_)= temp.x;                    
    //         //     *(dense_tile_share_+1)= temp.y;
    //         // }
    //         __syncwarp();
    //         // if(warpin_id>15){
    //         dense_tile_[0] = *(dense_tile_share_half);
    //         dense_tile_[1] = *(dense_tile_share_half+4);
    //         dense_tile_[2] = *(dense_tile_share_half+8);
    //         dense_tile_[3] = *(dense_tile_share_half+12);
    //         // else{
    //         //     dense_tile_[3] = *(dense_tile_share_half+12);
    //         //     dense_tile_[2] = *(dense_tile_share_half+8);
    //         //     dense_tile_[1] = *(dense_tile_share_half+4);
    //         //     dense_tile_[0] = *(dense_tile_share_half);
    //         // }
    //         column_idxs_ += 8;
    //     }

    //     // Load the residual and compute the matrix product
    //     __device__ __forceinline__ void ResidueLoad(int colEdge, int dimN_index, int residue){
    //         int col_offset = (warpin_id%4)*2;
    //         half * sparse_fragment_fp16 = reinterpret_cast<half *>(sparse_fragment_);
    //         const half * values_fp16 = reinterpret_cast<const half *>(values_);
    //         values_fp16 -= (8-residue)*(warpin_id/4);
    //         for(int c=0;c<2;c++){
    //             if((col_offset + c )< residue)
    //             sparse_fragment_fp16[c] = __ldg(values_fp16+c);
    //             else
    //             sparse_fragment_fp16[c] = 0.0;
    //         }
            
    //         long col_temp = -1;
    //         if((warpin_id/4)< residue)
    //             col_temp = __ldg(column_idxs_);
    //             const int global_offset = (warp_id<<4) + col1*4;
    //             float2 temp = make_float2(0.0f, 0.0f);
    //             if(col_temp >= 0)
    //             {
    //                 const long offset = (col_temp*rhs_cols_) + global_offset;
    //                 temp = __ldg(matrix_base_ +offset/4);
    //             }

    //         // if(warpin_id>15){
    //             *(dense_tile_share_+1)= temp.y;
    //             *(dense_tile_share_)= temp.x;
    //         // }else{
    //         //     *(dense_tile_share_)= temp.x;                    
    //         //     *(dense_tile_share_+1)= temp.y;
    //         // }
    //         __syncwarp();
    //         // if(warpin_id>15){
    //         dense_tile_[0] = *(dense_tile_share_half);
    //         dense_tile_[1] = *(dense_tile_share_half+4);
    //         dense_tile_[2] = *(dense_tile_share_half+8);
    //         dense_tile_[3] = *(dense_tile_share_half+12);
    //         // else{
    //         //     dense_tile_[3] = *(dense_tile_share_half+12);
    //         //     dense_tile_[2] = *(dense_tile_share_half+8);
    //         //     dense_tile_[1] = *(dense_tile_share_half+4);
    //         //     dense_tile_[0] = *(dense_tile_share_half);
    //         // }
                
    //     }
    // };


      //Tile_N = 128 threads_per_block = 128
      // shuffle
    struct mmaDenseTile_fp16_test{
        const float *  values_;
        const int *  column_idxs_;
        const int rhs_cols_;
        const int lane_id_;
         int warpin_id;
         int warp_id;
         int group;
         int temp;
        const float2 *matrix_base_;
        float2 *dense_tile_;
        half *dense_tile_half;
        //存放当前线程拿到的一个double值，并将该double值拆分成4个half分别进行转置放置
        float *sparse_fragment_;

        __device__ __forceinline__ mmaDenseTile_fp16_test(
        long row_offset_vec,
        const double * values,
        const int *  column_idxs,
	    int rhs_cols,
        int offset, 
        int lane_id, 
        const float2*  matrix, 
        //row_offsets= column_indices_tile
        // const int *row_offsets,
        float * dense_tile,
        float *sparse_fragment):
            rhs_cols_(rhs_cols),
            lane_id_(lane_id),
            // warpin_id(lane_id & 31),
            // warp_id(lane_id>>5),
            //每行16个线程，每个线程搬运1个double,
            matrix_base_(reinterpret_cast<const float2 *>(matrix + offset)),
            // row_offsets_base_(row_offsets),
            values_(reinterpret_cast<const float *>(values + row_offset_vec*2) + (lane_id & 31)),
            column_idxs_(column_idxs + row_offset_vec + (((lane_id & 31)%4)*2)),
            dense_tile_(reinterpret_cast< float2 *>(dense_tile)),
            dense_tile_half(reinterpret_cast< half *>(dense_tile)),
            sparse_fragment_(sparse_fragment)
            {
                warpin_id = lane_id & 31;
                warp_id = lane_id>>5;
                if(warpin_id < 8) group=0;
                else if(warpin_id < 16) group=2;
                else if(warpin_id < 24) group=1;
                else group=3;
                temp = 0;
                if((warpin_id%8)>=4) temp = 1;
            }
    
        __device__ __forceinline__ void Fetch(int colEdge, int dimN_index){

            sparse_fragment_[0]=__ldg(values_);
            values_ += 32;

            // int temp = 0;
            // if((warpin_id%8)>=4) temp = 1;
            long col_temp = __ldg(column_idxs_ + temp);

            long offset = (col_temp*rhs_cols_/4) + (warp_id<<2) + group;
             *(dense_tile_) =__ldg(matrix_base_ +offset);
            
            //第一次shuffle
            //temp=0: 分别取第1,3个数； temp=1: 分别取第0,2个数；
            half2 ex;
            if(temp == 0){
                ex.x =  dense_tile_half[1];
                ex.y =  dense_tile_half[3];
            }else{
                ex.x =  dense_tile_half[0];
                ex.y =  dense_tile_half[2];
            }
            //做shuffle
            ex = __shfl_xor_sync(0xffffffff, ex, 4, 32);
            //shuffle完，更新dense_tile_
            if(temp == 0){
                dense_tile_half[1] = ex.x;
                dense_tile_half[3] = ex.y;
            }else{
                dense_tile_half[0] = ex.x;
                dense_tile_half[2] = ex.y;
            }

            //第二次shuffle
            //temp=0: 分别取第1,3个数； temp=1: 分别取第0,2个数；
            // half2 ex;
            if(group < 2){
                ex.x =  dense_tile_half[2];
                ex.y =  dense_tile_half[3];
            }else{
                ex.x =  dense_tile_half[0];
                ex.y =  dense_tile_half[1];
            }
            //做shuffle
            ex = __shfl_xor_sync(0xffffffff, ex, 8, 32);
            //shuffle完，更新dense_tile_
            if(group < 2){
                dense_tile_half[2] = ex.x;
                dense_tile_half[3] = ex.y;
            }else{
                dense_tile_half[0] = ex.x;
                dense_tile_half[1] = ex.y;
            }

            column_idxs_ += 8;
        }

        // Load the residual and compute the matrix product
        __device__ __forceinline__ void ResidueLoad(int colEdge, int dimN_index, int residue){
            int col_offset = (warpin_id%4)*2;
            half * sparse_fragment_fp16 = reinterpret_cast<half *>(sparse_fragment_);
            const half * values_fp16 = reinterpret_cast<const half *>(values_);
            values_fp16 -= (8-residue)*(warpin_id/4);
            for(int c=0;c<2;c++){
                if((col_offset + c )< residue)
                sparse_fragment_fp16[c] = __ldg(values_fp16+c);
                else
                sparse_fragment_fp16[c] = 0.0;

                // if(blockIdx.x==0 and blockIdx.y==0 and threadIdx.x==4)
                // {
                //     printf("%f\n", __half2float(sparse_fragment_fp16[c]));
                // }
            }

            // int temp = 0;
            // if((warpin_id%8)>=4) temp = 1;
            long col_temp = -1;

            if((col_offset + temp )< residue)
            col_temp = __ldg(column_idxs_ + temp);
  
            
            if(col_temp >= 0)
            {
                long offset = (col_temp*rhs_cols_/4) + (warp_id<<2) + group;
                *(dense_tile_) =__ldg(matrix_base_ +offset);
            }else{
                *(dense_tile_) = make_float2(0.0f, 0.0f);
            }

            half2 ex;
            if(temp == 0){
                ex.x =  dense_tile_half[1];
                ex.y =  dense_tile_half[3];
            }else{
                ex.x =  dense_tile_half[0];
                ex.y =  dense_tile_half[2];
            }
            //做shuffle
            ex = __shfl_xor_sync(0xffffffff, ex, 4, 32);
            //shuffle完，更新dense_tile_
            if(temp == 0){
                dense_tile_half[1] = ex.x;
                dense_tile_half[3] = ex.y;
            }else{
                dense_tile_half[0] = ex.x;
                dense_tile_half[2] = ex.y;
            }

            //第二次shuffle
            //temp=0: 分别取第1,3个数； temp=1: 分别取第0,2个数；
            // half2 ex;
            if(group < 2){
                ex.x =  dense_tile_half[2];
                ex.y =  dense_tile_half[3];
            }else{
                ex.x =  dense_tile_half[0];
                ex.y =  dense_tile_half[1];
            }
            //做shuffle
            ex = __shfl_xor_sync(0xffffffff, ex, 8, 32);
            //shuffle完，更新dense_tile_
            if(group < 2){
                dense_tile_half[2] = ex.x;
                dense_tile_half[3] = ex.y;
            }else{
                dense_tile_half[0] = ex.x;
                dense_tile_half[1] = ex.y;
            }
 
        }
    };



    // struct mmaDenseTile_fp16_test{
    //     const float *  values_;
    //     const int *  column_idxs_;
    //     const int rhs_cols_;
    //     const int lane_id_;
    //      int warpin_id;
    //      int warp_id;
    //     const float2 *matrix_base_;
    //     half *dense_tile_;
    //     float *dense_tile_share_;
    //     half *dense_tile_share_half;
    //     //存放当前线程拿到的一个double值，并将该double值拆分成4个half分别进行转置放置
    //     float *sparse_fragment_;
    //     int row1;
    //     int col1;
    //     int group;

    //     __device__ __forceinline__ mmaDenseTile_fp16_test(
    //     long row_offset_vec,
    //     const double * values,
    //     const int *  column_idxs,
	//     int rhs_cols,
    //     int offset, 
    //     int lane_id, 
    //     const float2*  matrix, 
    //     //row_offsets= column_indices_tile
    //     // const int *row_offsets,
    //     float * dense_tile,
    //     float * dense_tile_share,
    //     float *sparse_fragment):
    //         rhs_cols_(rhs_cols),
    //         lane_id_(lane_id),
    //         // warpin_id(lane_id & 31),
    //         // warp_id(lane_id>>5),
    //         //每行16个线程，每个线程搬运1个double,
    //         matrix_base_(matrix + offset),
    //         // row_offsets_base_(row_offsets),
    //         values_(reinterpret_cast<const float *>(values + row_offset_vec*2) + (lane_id & 31)),
    //         // column_idxs_(column_idxs + row_offset_vec),
    //         dense_tile_(reinterpret_cast< half *>(dense_tile)),
    //         sparse_fragment_(sparse_fragment)
    //         {
    //             warpin_id = lane_id & 31;
    //             warp_id = lane_id>>5;
    //             //计算row偏移
    //             row1 = warpin_id / 2;
    //             if(warpin_id>=16)  row1 -=8;
    //             column_idxs_ = column_idxs + row_offset_vec + row1;
    //             //计算col偏移
    //             col1 = (warpin_id%2)*2 + warpin_id/16;

    //             //计算第几组
    //             // group = warpin_id/4;
    //             group = (warpin_id/16)*4 + warpin_id%4;
    //             int groupin_id =  warpin_id / 4;

    //             int pad =0;
    //             if(warpin_id>=16) 
    //             {
    //                 groupin_id -=4;
    //                 pad = 1;
    //             }


    //             dense_tile_share_ = dense_tile_share + warp_id*65 + (warpin_id/4)*8 + pad;
    //             dense_tile_share_half = reinterpret_cast< half *>(dense_tile_share + warp_id*65 + group*8 + pad) + groupin_id;
    //             dense_tile_share_ += (warpin_id%4)*2;
    //         }
    
    //     __device__ __forceinline__ void Fetch(int colEdge, int dimN_index){

    //         // if( blockIdx.x==0 and  blockIdx.y==0 and blockIdx.z==0 and threadIdx.x==5){
    //         //     printf("%d,%d,%d\n", row1, col1, group);
    //         // }
    //         sparse_fragment_[0]=__ldg(values_);
    //         values_ += 32;
    //         long col_temp = __ldg(column_idxs_);

    //         const int global_offset = (warp_id<<4) + col1*4;

    //         const long offset = (col_temp*rhs_cols_) + global_offset;
    //         float2 temp = __ldg(matrix_base_ +offset/4);

    //         // dense_tile_share_ += (warpin_id%4)*2;
    //         // if(warpin_id>15){
    //             *(dense_tile_share_+1 )= temp.y;
    //             *(dense_tile_share_)= temp.x;
    //         // }else{
    //         //     *(dense_tile_share_)= temp.x;                    
    //         //     *(dense_tile_share_+1)= temp.y;
    //         // }
    //         __syncwarp();
    //         // if(warpin_id>15){
    //         dense_tile_[0] = *(dense_tile_share_half);
    //         dense_tile_[1] = *(dense_tile_share_half+4);
    //         dense_tile_[2] = *(dense_tile_share_half+8);
    //         dense_tile_[3] = *(dense_tile_share_half+12);
    //         // else{
    //         //     dense_tile_[3] = *(dense_tile_share_half+12);
    //         //     dense_tile_[2] = *(dense_tile_share_half+8);
    //         //     dense_tile_[1] = *(dense_tile_share_half+4);
    //         //     dense_tile_[0] = *(dense_tile_share_half);
    //         // }
    //         column_idxs_ += 8;
    //     }

    //     // Load the residual and compute the matrix product
    //     __device__ __forceinline__ void ResidueLoad(int colEdge, int dimN_index, int residue){
    //         int col_offset = (warpin_id%4)*2;
    //         half * sparse_fragment_fp16 = reinterpret_cast<half *>(sparse_fragment_);
    //         const half * values_fp16 = reinterpret_cast<const half *>(values_);
    //         values_fp16 -= (8-residue)*(warpin_id/4);
    //         for(int c=0;c<2;c++){
    //             if((col_offset + c )< residue)
    //             sparse_fragment_fp16[c] = __ldg(values_fp16+c);
    //             else
    //             sparse_fragment_fp16[c] = 0.0;
    //         }
            
    //         long col_temp = -1;
    //         if((warpin_id/4)< residue)
    //             col_temp = __ldg(column_idxs_);
    //             const int global_offset = (warp_id<<4) + col1*4;
    //             float2 temp = make_float2(0.0f, 0.0f);
    //             if(col_temp >= 0)
    //             {
    //                 const long offset = (col_temp*rhs_cols_) + global_offset;
    //                 temp = __ldg(matrix_base_ +offset/4);
    //             }

    //         // if(warpin_id>15){
    //             *(dense_tile_share_+1)= temp.y;
    //             *(dense_tile_share_)= temp.x;
    //         // }else{
    //         //     *(dense_tile_share_)= temp.x;                    
    //         //     *(dense_tile_share_+1)= temp.y;
    //         // }
    //         __syncwarp();
    //         // if(warpin_id>15){
    //         dense_tile_[0] = *(dense_tile_share_half);
    //         dense_tile_[1] = *(dense_tile_share_half+4);
    //         dense_tile_[2] = *(dense_tile_share_half+8);
    //         dense_tile_[3] = *(dense_tile_share_half+12);
    //         // else{
    //         //     dense_tile_[3] = *(dense_tile_share_half+12);
    //         //     dense_tile_[2] = *(dense_tile_share_half+8);
    //         //     dense_tile_[1] = *(dense_tile_share_half+4);
    //         //     dense_tile_[0] = *(dense_tile_share_half);
    //         // }
                
    //     }
    // };




        struct mmaDenseTile_fp16_ori_ones{
        const float *  values_;
        const int *  column_idxs_;
        const int rhs_cols_;
        const int lane_id_;
        const int warpin_id;
        const int warp_id;
        const half *matrix_base_;
        half *dense_tile_;
        //存放当前线程拿到的一个double值，并将该double值拆分成4个half分别进行转置放置
        float *sparse_fragment_;

        __device__ __forceinline__ mmaDenseTile_fp16_ori_ones(
        long row_offset_vec,
        const double * values,
        const int *  column_idxs,
	    int rhs_cols,
        int offset, 
        int lane_id, 
        const double*  matrix, 
        //row_offsets= column_indices_tile
        // const int *row_offsets,
        float * dense_tile,
        float *sparse_fragment):
            rhs_cols_(rhs_cols),
            lane_id_(lane_id),
            warpin_id(lane_id & 31),
            warp_id(lane_id>>5),
            //每行16个线程，每个线程搬运1个double,
            matrix_base_(reinterpret_cast<const half *>(matrix + offset)),
            // row_offsets_base_(row_offsets),
            values_(reinterpret_cast<const float *>(values + row_offset_vec*2) + (lane_id & 31)),
            column_idxs_(column_idxs + row_offset_vec + (((lane_id & 31)%4)*2)),
            dense_tile_(reinterpret_cast< half *>(dense_tile)),
            sparse_fragment_(sparse_fragment)
            {}
    
        __device__ __forceinline__ void Fetch(int colEdge, int dimN_index){

            sparse_fragment_[0]=__ldg(values_);
            values_ += 32;
            long col_temp[2];
            for(int k=0; k<2; k++)
                col_temp[k] = __ldg(column_idxs_ + k);
            for(int i=0;i<2;i++)
            {
                const int global_offset = (warp_id<<4) + (warpin_id/4) + (i*8);
                if(global_offset < colEdge){
                    for(int j=0; j<2; j++)
                    {
                        const long offset = (col_temp[j]*rhs_cols_) + global_offset;
                        dense_tile_[i*2+j]=__ldg(matrix_base_ +offset);
                    }
                }else{
                    for(int j=0; j<2; j++)
                    {
                        dense_tile_[i*2+j]=__float2half(0.0);
                    }
                }
            }
            column_idxs_ += 8;
        }

        // Load the residual and compute the matrix product
        __device__ __forceinline__ void ResidueLoad(int colEdge, int dimN_index, int residue){
            int col_offset = (warpin_id%4)*2;
            half * sparse_fragment_fp16 = reinterpret_cast<half *>(sparse_fragment_);
            const half * values_fp16 = reinterpret_cast<const half *>(values_);
            values_fp16 -= (8-residue)*(warpin_id/4);
            for(int c=0;c<2;c++){
                if((col_offset + c )< residue)
                sparse_fragment_fp16[c] = __ldg(values_fp16+c);
                else
                sparse_fragment_fp16[c] = 0.0;

                // if(blockIdx.x==0 and blockIdx.y==0 and threadIdx.x==4)
                // {
                //     printf("%f\n", __half2float(sparse_fragment_fp16[c]));
                // }
            }
            
            long col_temp[2] = {-1, -1};
            for(int k=0; k<2; k++)
                if((col_offset + k )< residue)
                    col_temp[k] = __ldg(column_idxs_ + k);

            // at::Half dense_tile_half_temp[4]={0.0,0.0,0.0,0.0};
            // half * dense_tile_half = reinterpret_cast<half *>(dense_tile_half_temp);
            for(int i=0;i<2;i++)
            {
                const int global_offset = (warp_id<<4) + (warpin_id/4) + (i*8);
                if(global_offset < colEdge){
                    for(int j=0; j<2; j++)
                    {
                        if(col_temp[j] >= 0)
                        {
                            const long offset = (col_temp[j]*rhs_cols_) + global_offset;
                            dense_tile_[i*2+j]=__ldg(matrix_base_ +offset);
                        }else{
                            dense_tile_[i*2+j]=0.0;
                        }
                    }
                }else{
                    for(int j=0; j<2; j++)
                    {
                        dense_tile_[i*2+j]=__float2half(0.0);
                    }
                }
            }

            // for(int i=0;i<4;i++)
            //     *(dense_tile_ + i)=dense_tile_half[i];
        }
    };






struct mmaDenseTile_tf32_sr{
        const float *  values_;
        const int *  column_idxs_;
        const int rhs_cols_;
        const int lane_id_;
        const int warpin_id;
        const int warp_id;
        const float *matrix_base_;
        float *dense_tile_;
        //存放当前线程拿到的一个double值，并将该double值拆分成4个half分别进行转置放置
        float *sparse_fragment_;

        __device__ __forceinline__ mmaDenseTile_tf32_sr(
        long row_offset_vec,
        const float * values,
        const int *  column_idxs,
	    int rhs_cols,
        int offset, 
        int lane_id, 
        const float*  matrix, 
        //row_offsets= column_indices_tile
        // const int *row_offsets,
        float * dense_tile,
        float *sparse_fragment):
            rhs_cols_(rhs_cols),
            lane_id_(lane_id),
            warpin_id(lane_id & 31),
            warp_id(lane_id>>5),
            //当前block在全局的列偏移
            matrix_base_(matrix + offset),
            //8的意思是vector的长度
            values_((values + row_offset_vec*8) + (lane_id & 31)),
            //对4*16的RHS读取，每行连续读8个线程，共4行，所以需要>>3
            column_idxs_(column_idxs + row_offset_vec + ((lane_id & 31)%4)),
            dense_tile_(dense_tile),
            sparse_fragment_(sparse_fragment)
            {}
    
        __device__ __forceinline__ void Fetch(int colEdge, int dimN_index){

            sparse_fragment_[0]= __ldg(values_);
            const long row_offsets_ = __ldg(column_idxs_);
            values_ += 32;
            column_idxs_ += 4;
            // (warp_id<<4) 每个warp有16列
            //行偏移,(warpin_id%8)*2),每行8个线程，每个线程读两个float数
            const int global_offset = (warp_id<<4) + (warpin_id/4);
            const long offset = (row_offsets_*rhs_cols_) + global_offset;
            // float dense_tile_fp32[2]={0.0,0.0};
            for(int i=0;i<2;i++)
            {
                // if((dimN_index+global_offset+i)<colEdge)
                dense_tile_[i]=__ldg(matrix_base_ + offset + i*8);
            } 
            // for(int i=0;i<2;i++)
            //     dense_tile_[i]=dense_tile_fp32[i];    
        }

        // Load the residual and compute the matrix product
        __device__ __forceinline__ void ResidueLoad(int colEdge, int dimN_index, int residue){

            sparse_fragment_[0]= __ldg(values_);
            const long row_offsets_ = __ldg(column_idxs_);
            
            const int global_offset = (warp_id<<4) + (warpin_id/4);
            // float dense_tile_fp32[2]={0.0,0.0};
            if(row_offsets_ >= 0){
                const long offset = (row_offsets_*rhs_cols_) + global_offset;
                // matrix_base_=matrix_base_ + (row_offsets_*rhs_cols_) + global_offset;
                 for(int i=0;i<2;i++)
                 {
                    // if((dimN_index+global_offset+i*8)<colEdge)
                    dense_tile_[i]=__ldg(matrix_base_ +offset+ i*8);
                 }   
            }else{
                dense_tile_[0] = 0.0;
                dense_tile_[1] = 0.0;
            }
            // for(int i=0;i<2;i++)
            //     dense_tile_[i]=dense_tile_fp32[i];
        }
    };


