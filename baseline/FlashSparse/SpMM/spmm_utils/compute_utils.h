#ifndef SPMM_COMPUTE_UTILS_H
#define SPMM_COMPUTE_UTILS_H
#include <mma.h>
#include <cstdint>



    struct mmaComputeUtils_fp16{
        // Shared memory buffers
        // const uint32_t* lhs_tile_;
        const uint32_t* dense_tile_;
        // Register file fragment to accumulate results into
        uint32_t * output_fragment_;
        int lane_id_;
        uint32_t *lhs_fragment;

        // Constructor
        __device__ __forceinline__ mmaComputeUtils_fp16(
           float * dense_tile,
            uint32_t* output_fragment,
            int lane_id,
            float *sparse_fragment):
            // lhs_tile_(reinterpret_cast<const  uint32_t*>(lhs_tile)),
            lane_id_(lane_id),
            dense_tile_(reinterpret_cast<const  uint32_t*>(dense_tile)),
            output_fragment_(output_fragment),
            lhs_fragment(reinterpret_cast<uint32_t*>(sparse_fragment)){}
        
        // Compute
        __device__ __forceinline__ void TileMAC(){

            uint32_t rhs_fragment[2];
            int warp_id=lane_id_>>5;
            // densetile + 所在的warp + 第1or2个32块。符合mma数据布局
            #pragma unroll
            for(int i=0;i<2;i++){
                rhs_fragment[i] = *(dense_tile_ + (warp_id<<6) + (i<<5)+ (lane_id_&31)); 
            }

            // if((lane_id_ % 32) < ValuesBlockWidth)
            //     lhs_fragment[0] = lhs_tile_[lane_id_ % ValuesBlockWidth];
            // else
            //     lhs_fragment[0] = 0;
            // lhs_fragment[0] = lhs_tile_[lane_id_&31];
            // __syncwarp();
    
            asm("mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 \t"
                "{%0,%1}, \t"
                "{%2,%3}, \t"
                "{%4}, \t"
                "{%0,%1}; ":
                "+r"(output_fragment_[0]), "+r"(output_fragment_[1]):
                "r"(rhs_fragment[0]),  "r"(rhs_fragment[1]),
                "r"(lhs_fragment[0])
            );
            // asm("mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 \t"
            //     "{%0,%1}, \t"
            //     "{%2,%3}, \t"
            //     "{%4}, \t"
            //     "{%5,%6}; ":
            //     "=f"(output_fragment_[0]), "=f"(output_fragment_[1]):
            //     "f"(rhs_fragment[0]),  "f"(rhs_fragment[1]),
            //     "f"(lhs_fragment[0]),
            //     "f"(output_fragment_[0]), "f"(output_fragment_[1])
            // );
            
        }

        __device__ __forceinline__ void TileMACResidue(){
        // uint32_t lhs_fragment[1];
        uint32_t rhs_fragment[2];
        int warp_id=lane_id_>>5;
        // densetile + 所在的warp + 第1or2个32块。符合mma数据布局
        #pragma unroll
        for(int i=0;i<2;i++){
            rhs_fragment[i] = *(dense_tile_ + (warp_id<<6) + (i<<5)+(lane_id_&31)); 
        }

        // if((lane_id_ &31) < ValuesBlockWidth)
        //     lhs_fragment[0] = lhs_tile_[(lane_id_ % ValuesBlockWidth)];
        // else
        //     lhs_fragment[0] = 0;
        // lhs_fragment[0] = lhs_tile_[lane_id_&31];
        // __syncwarp();
        asm("mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 \t"
                "{%0,%1}, \t"
                "{%2,%3}, \t"
                "{%4}, \t"
                "{%0,%1}; ":
                "+r"(output_fragment_[0]), "+r"(output_fragment_[1]):
                "r"(rhs_fragment[0]),  "r"(rhs_fragment[1]),
                "r"(lhs_fragment[0])
            );
        
        }
    };


    //fp16 16
        struct mmaComputeUtils_fp16_16{
        // Shared memory buffers
        // const uint32_t* lhs_tile_;
        uint32_t* rhs_fragment;
        // Register file fragment to accumulate results into
        uint32_t * output_fragment_;
        int lane_id_;
        uint32_t *lhs_fragment;

        // Constructor
        __device__ __forceinline__ mmaComputeUtils_fp16_16(
           float * dense_tile,
            uint32_t* output_fragment,
            int lane_id,
            float *sparse_fragment):
            // lhs_tile_(reinterpret_cast<const  uint32_t*>(lhs_tile)),
            lane_id_(lane_id),
            rhs_fragment(reinterpret_cast< uint32_t*>(dense_tile)),
            output_fragment_(output_fragment),
            lhs_fragment(reinterpret_cast<uint32_t*>(sparse_fragment)){}
        
        // Compute
        __device__ __forceinline__ void TileMAC(){

            // if((lane_id_ % 32) < ValuesBlockWidth)
            //     lhs_fragment[0] = lhs_tile_[lane_id_ % ValuesBlockWidth];
            // else
            //     lhs_fragment[0] = 0;
            // lhs_fragment[0] = lhs_tile_[lane_id_&31];
            // __syncwarp();
    
            asm("mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 \t"
                "{%0,%1}, \t"
                "{%2,%3}, \t"
                "{%4}, \t"
                "{%0,%1}; ":
                "+r"(output_fragment_[0]), "+r"(output_fragment_[1]):
                "r"(lhs_fragment[0]),  "r"(lhs_fragment[1]),
                "r"(rhs_fragment[0])
            );
            // asm("mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 \t"
            //     "{%0,%1}, \t"
            //     "{%2,%3}, \t"
            //     "{%4}, \t"
            //     "{%5,%6}; ":
            //     "=f"(output_fragment_[0]), "=f"(output_fragment_[1]):
            //     "f"(rhs_fragment[0]),  "f"(rhs_fragment[1]),
            //     "f"(lhs_fragment[0]),
            //     "f"(output_fragment_[0]), "f"(output_fragment_[1])
            // );
            
        }

        __device__ __forceinline__ void TileMACResidue(){


        // if((lane_id_ &31) < ValuesBlockWidth)
        //     lhs_fragment[0] = lhs_tile_[(lane_id_ % ValuesBlockWidth)];
        // else
        //     lhs_fragment[0] = 0;
        // lhs_fragment[0] = lhs_tile_[lane_id_&31];
        // __syncwarp();
        asm("mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 \t"
                "{%0,%1}, \t"
                "{%2,%3}, \t"
                "{%4}, \t"
                "{%0,%1}; ":
                "+r"(output_fragment_[0]), "+r"(output_fragment_[1]):
                "r"(lhs_fragment[0]),  "r"(lhs_fragment[1]),
                "r"(rhs_fragment[0])
            );
        
        }
    };



    struct mmaComputeUtils_tf32{
        // Shared memory buffers
        // const uint32_t* lhs_tile_;
        const float* dense_tile_;
        // Register file fragment to accumulate results into
        float * output_fragment_;
        int lane_id_;
        uint32_t *lhs_fragment;

        // Constructor
        __device__ __forceinline__ mmaComputeUtils_tf32(
            float* dense_tile,
            float* output_fragment,
            int lane_id,
            float *sparse_fragment):
            // lhs_tile_(reinterpret_cast<const  uint32_t*>(lhs_tile)),
            lane_id_(lane_id),
            dense_tile_(dense_tile),
            output_fragment_(output_fragment),
            lhs_fragment(reinterpret_cast<uint32_t*>(sparse_fragment)){}
        
        // Compute
        __device__ __forceinline__ void TileMAC(){

        float rhs_fragment1[2];
        // densetile + 所在的warp + 第1or2个32块。符合mma数据布局
        #pragma unroll
        for(int i=0;i<2;i++){
            rhs_fragment1[i] =  *(dense_tile_ + (lane_id_/32)*64 + lane_id_%32 + i*32); 
        }
        uint32_t* rhs_fragment = reinterpret_cast<uint32_t *>(rhs_fragment1);
    
        asm volatile(
        "mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};\n"
            : "=f"(output_fragment_[0]), "=f"(output_fragment_[1]), "=f"(output_fragment_[2]), "=f"(output_fragment_[3])
            : "r"(rhs_fragment[0]), "r"(rhs_fragment[1]), "r"(lhs_fragment[0]), "f"(output_fragment_[0]), "f"(output_fragment_[1]), "f"(output_fragment_[2]), "f"(output_fragment_[3]));
            
        }

        __device__ __forceinline__ void TileMACResidue(){
        // uint32_t lhs_fragment[1];
        float rhs_fragment1[2];
        // densetile + 所在的warp + 第1or2个32块。符合mma数据布局
        #pragma unroll
        for(int i=0;i<2;i++){
            rhs_fragment1[i] = *(dense_tile_ + (lane_id_/32)*64 + lane_id_%32 + i*32); 
        }
        uint32_t* rhs_fragment = reinterpret_cast<uint32_t *>(rhs_fragment1);
            

       asm volatile(
        "mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};\n"
            : "=f"(output_fragment_[0]), "=f"(output_fragment_[1]), "=f"(output_fragment_[2]), "=f"(output_fragment_[3])
            : "r"(rhs_fragment[0]), "r"(rhs_fragment[1]), "r"(lhs_fragment[0]), "f"(output_fragment_[0]), "f"(output_fragment_[1]), "f"(output_fragment_[2]), "f"(output_fragment_[3]));      
        }
        
        
    };

    //tf32-16

    struct mmaComputeUtils_tf32_16{
        // Shared memory buffers
        // const uint32_t* lhs_tile_;
        uint32_t* rhs_fragment;
        // Register file fragment to accumulate results into
        float * output_fragment_;
        int lane_id_;
        uint32_t *lhs_fragment;

        // Constructor
        __device__ __forceinline__ mmaComputeUtils_tf32_16(
            float* dense_tile,
            float* output_fragment,
            int lane_id,
            float *sparse_fragment):
            // lhs_tile_(reinterpret_cast<const  uint32_t*>(lhs_tile)),
            lane_id_(lane_id),
            rhs_fragment(reinterpret_cast<uint32_t*>(dense_tile)),
            output_fragment_(output_fragment),
            lhs_fragment(reinterpret_cast<uint32_t*>(sparse_fragment)){}
        
        // Compute
        __device__ __forceinline__ void TileMAC(){

        
    
        asm volatile(
        "mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};\n"
            : "=f"(output_fragment_[0]), "=f"(output_fragment_[1]), "=f"(output_fragment_[2]), "=f"(output_fragment_[3])
            : "r"(lhs_fragment[0]), "r"(lhs_fragment[1]), "r"(rhs_fragment[0]), "f"(output_fragment_[0]), "f"(output_fragment_[1]), "f"(output_fragment_[2]), "f"(output_fragment_[3]));
            
        }

        __device__ __forceinline__ void TileMACResidue(){

    
       asm volatile(
        "mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};\n"
            : "=f"(output_fragment_[0]), "=f"(output_fragment_[1]), "=f"(output_fragment_[2]), "=f"(output_fragment_[3])
            : "r"(lhs_fragment[0]), "r"(lhs_fragment[1]), "r"(rhs_fragment[0]), "f"(output_fragment_[0]), "f"(output_fragment_[1]), "f"(output_fragment_[2]), "f"(output_fragment_[3]));      
        }
        
        
    };



    struct mmaComputeUtils_fp16_v2{
        // Shared memory buffers
        // const uint32_t* lhs_tile_;
        uint32_t* rhs_fragment;
        // Register file fragment to accumulate results into
        uint32_t * output_fragment_;
        int lane_id_;
        uint32_t *lhs_fragment;

        // Constructor
        __device__ __forceinline__ mmaComputeUtils_fp16_v2(
           float * dense_tile,
            uint32_t* output_fragment,
            int lane_id,
            float *sparse_fragment):
            // lhs_tile_(reinterpret_cast<const  uint32_t*>(lhs_tile)),
            lane_id_(lane_id),
            rhs_fragment(reinterpret_cast<uint32_t*>(dense_tile)),
            output_fragment_(output_fragment),
            lhs_fragment(reinterpret_cast<uint32_t*>(sparse_fragment)){}
        
        // Compute
        __device__ __forceinline__ void TileMAC(){

            // uint32_t rhs_fragment[2];
            // int warp_id=lane_id_>>5;
            // // densetile + 所在的warp + 第1or2个32块。符合mma数据布局
            // #pragma unroll
            // for(int i=0;i<2;i++){
            //     rhs_fragment[i] = *(dense_tile_ + (warp_id<<6) + (i<<5)+ (lane_id_&31)); 
            // }

            // if((lane_id_ % 32) < ValuesBlockWidth)
            //     lhs_fragment[0] = lhs_tile_[lane_id_ % ValuesBlockWidth];
            // else
            //     lhs_fragment[0] = 0;
            // lhs_fragment[0] = lhs_tile_[lane_id_&31];
            // __syncwarp();
    
            asm("mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 \t"
                "{%0,%1}, \t"
                "{%2,%3}, \t"
                "{%4}, \t"
                "{%0,%1}; ":
                "+r"(output_fragment_[0]), "+r"(output_fragment_[1]):
                "r"(rhs_fragment[0]),  "r"(rhs_fragment[1]),
                "r"(lhs_fragment[0])
            );
            // asm("mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 \t"
            //     "{%0,%1}, \t"
            //     "{%2,%3}, \t"
            //     "{%4}, \t"
            //     "{%5,%6}; ":
            //     "=f"(output_fragment_[0]), "=f"(output_fragment_[1]):
            //     "f"(rhs_fragment[0]),  "f"(rhs_fragment[1]),
            //     "f"(lhs_fragment[0]),
            //     "f"(output_fragment_[0]), "f"(output_fragment_[1])
            // );
            
        }

        __device__ __forceinline__ void TileMACResidue(){
        // uint32_t lhs_fragment[1];
        // uint32_t rhs_fragment[2];
        // int warp_id=lane_id_>>5;
        // // densetile + 所在的warp + 第1or2个32块。符合mma数据布局
        // #pragma unroll
        // for(int i=0;i<2;i++){
        //     rhs_fragment[i] = *(dense_tile_ + (warp_id<<6) + (i<<5)+(lane_id_&31)); 
        // }

        // if((lane_id_ &31) < ValuesBlockWidth)
        //     lhs_fragment[0] = lhs_tile_[(lane_id_ % ValuesBlockWidth)];
        // else
        //     lhs_fragment[0] = 0;
        // lhs_fragment[0] = lhs_tile_[lane_id_&31];
        // __syncwarp();
        asm("mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 \t"
                "{%0,%1}, \t"
                "{%2,%3}, \t"
                "{%4}, \t"
                "{%0,%1}; ":
                "+r"(output_fragment_[0]), "+r"(output_fragment_[1]):
                "r"(rhs_fragment[0]),  "r"(rhs_fragment[1]),
                "r"(lhs_fragment[0])
            );
        
        }
    };



        struct mmaComputeUtils_tf32_v2{
        // Shared memory buffers
        // const uint32_t* lhs_tile_;
        uint32_t* rhs_fragment;
        // Register file fragment to accumulate results into
        float * output_fragment_;
        int lane_id_;
        uint32_t *lhs_fragment;

        // Constructor
        __device__ __forceinline__ mmaComputeUtils_tf32_v2(
            float* dense_tile,
            float* output_fragment,
            int lane_id,
            float *sparse_fragment):
            // lhs_tile_(reinterpret_cast<const  uint32_t*>(lhs_tile)),
            lane_id_(lane_id),
            rhs_fragment(reinterpret_cast<uint32_t*>(dense_tile)),
            output_fragment_(output_fragment),
            lhs_fragment(reinterpret_cast<uint32_t*>(sparse_fragment)){}
        
        // Compute
        __device__ __forceinline__ void TileMAC(){

        // float rhs_fragment1[2];
        // // densetile + 所在的warp + 第1or2个32块。符合mma数据布局
        // #pragma unroll
        // for(int i=0;i<2;i++){
        //     rhs_fragment1[i] =  *(dense_tile_ + (lane_id_/32)*64 + lane_id_%32 + i*32); 
        // }
        // uint32_t* rhs_fragment = reinterpret_cast<uint32_t *>(rhs_fragment1);
    
        asm volatile(
        "mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};\n"
            : "=f"(output_fragment_[0]), "=f"(output_fragment_[1]), "=f"(output_fragment_[2]), "=f"(output_fragment_[3])
            : "r"(rhs_fragment[0]), "r"(rhs_fragment[1]), "r"(lhs_fragment[0]), "f"(output_fragment_[0]), "f"(output_fragment_[1]), "f"(output_fragment_[2]), "f"(output_fragment_[3]));
            
        }

        __device__ __forceinline__ void TileMACResidue(){
        // // uint32_t lhs_fragment[1];
        // float rhs_fragment1[2];
        // // densetile + 所在的warp + 第1or2个32块。符合mma数据布局
        // #pragma unroll
        // for(int i=0;i<2;i++){
        //     rhs_fragment1[i] = *(dense_tile_ + (lane_id_/32)*64 + lane_id_%32 + i*32); 
        // }
        // uint32_t* rhs_fragment = reinterpret_cast<uint32_t *>(rhs_fragment1);
            

       asm volatile(
        "mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};\n"
            : "=f"(output_fragment_[0]), "=f"(output_fragment_[1]), "=f"(output_fragment_[2]), "=f"(output_fragment_[3])
            : "r"(rhs_fragment[0]), "r"(rhs_fragment[1]), "r"(lhs_fragment[0]), "f"(output_fragment_[0]), "f"(output_fragment_[1]), "f"(output_fragment_[2]), "f"(output_fragment_[3]));      
        }
        
        
    };
#endif
