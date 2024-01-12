#pragma once

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cusparse.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

typedef int64_t Index;
typedef float DType;
#define FULLMASK 0xffffffff
#define CEIL(x, y) (((x) + (y)-1) / (y))
#define MIN(x, y) ((x) < (y) ? (x) : (y))
#define MAX(x, y) ((x) > (y) ? (x) : (y))
