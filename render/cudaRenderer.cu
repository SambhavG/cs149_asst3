#include <string>
#include <algorithm>
#include <math.h>
#include <stdio.h>
#include <vector>
#include <iostream>
#include <stdexcept>
#include "cycleTimer.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>

#include "cudaRenderer.h"
#include "image.h"
#include "noise.h"
#include "sceneLoader.h"
#include "util.h"

#define BLOCKSIZE 256
#define SCAN_BLOCK_DIM 1024  // needed by sharedMemExclusiveScan implementation
#include "exclusiveScan.cu_inl"
#include "circleBoxTest.cu_inl"

#define DEBUG

#ifdef DEBUG
#define cudaCheckError(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr, "CUDA Error: %s at %s:%d\n",
        cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
#else
#define cudaCheckError(ans) ans
#endif

////////////////////////////////////////////////////////////////////////////////////////
// Putting all the cuda kernels here
///////////////////////////////////////////////////////////////////////////////////////

struct GlobalConstants {

    SceneName sceneName;

    int numCircles;
    float* position;
    float* velocity;
    float* color;
    float* radius;

    int imageWidth;
    int imageHeight;
    float* imageData;

    int numCirclesUp;
    int numGridCells;
    int numRegions;
    int maxRegionsPerSmall;
    float smallSize;
};

// Global variable that is in scope, but read-only, for all cuda
// kernels.  The __constant__ modifier designates this variable will
// be stored in special "constant" memory on the GPU. (we didn't talk
// about this type of memory in class, but constant memory is a fast
// place to put read-only variables).
__constant__ GlobalConstants cuConstRendererParams;

// read-only lookup tables used to quickly compute noise (needed by
// advanceAnimation for the snowflake scene)
__constant__ int    cuConstNoiseYPermutationTable[256];
__constant__ int    cuConstNoiseXPermutationTable[256];
__constant__ float  cuConstNoise1DValueTable[256];

// color ramp table needed for the color ramp lookup shader
#define COLOR_MAP_SIZE 5
__constant__ float  cuConstColorRamp[COLOR_MAP_SIZE][3];

// including parts of the CUDA code from external files to keep this
// file simpler and to seperate code that should not be modified
#include "noiseCuda.cu_inl"
#include "lookupColor.cu_inl"

// kernelClearImageSnowflake -- (CUDA device code)
//
// Clear the image, setting the image to the white-gray gradation that
// is used in the snowflake image
__global__ void kernelClearImageSnowflake() {

    int imageX = blockIdx.x * blockDim.x + threadIdx.x;
    int imageY = blockIdx.y * blockDim.y + threadIdx.y;

    int width = cuConstRendererParams.imageWidth;
    int height = cuConstRendererParams.imageHeight;

    if (imageX >= width || imageY >= height)
        return;

    int offset = 4 * (imageY * width + imageX);
    float shade = .4f + .45f * static_cast<float>(height-imageY) / height;
    float4 value = make_float4(shade, shade, shade, 1.f);

    // write to global memory: As an optimization, I use a float4
    // store, that results in more efficient code than if I coded this
    // up as four seperate fp32 stores.
    *(float4*)(&cuConstRendererParams.imageData[offset]) = value;
}

// kernelClearImage --  (CUDA device code)
//
// Clear the image, setting all pixels to the specified color rgba
__global__ void kernelClearImage(float r, float g, float b, float a) {

    int imageX = blockIdx.x * blockDim.x + threadIdx.x;
    int imageY = blockIdx.y * blockDim.y + threadIdx.y;

    int width = cuConstRendererParams.imageWidth;
    int height = cuConstRendererParams.imageHeight;

    if (imageX >= width || imageY >= height)
        return;

    int offset = 4 * (imageY * width + imageX);
    float4 value = make_float4(r, g, b, a);

    // write to global memory: As an optimization, I use a float4
    // store, that results in more efficient code than if I coded this
    // up as four seperate fp32 stores.
    *(float4*)(&cuConstRendererParams.imageData[offset]) = value;
}

// kernelAdvanceFireWorks
// 
// Update the position of the fireworks (if circle is firework)
__global__ void kernelAdvanceFireWorks() {
    const float dt = 1.f / 60.f;
    const float pi = 3.14159;
    const float maxDist = 0.25f;

    float* velocity = cuConstRendererParams.velocity;
    float* position = cuConstRendererParams.position;
    float* radius = cuConstRendererParams.radius;

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= cuConstRendererParams.numCircles)
        return;

    if (0 <= index && index < NUM_FIREWORKS) { // firework center; no update 
        return;
    }

    // determine the fire-work center/spark indices
    int fIdx = (index - NUM_FIREWORKS) / NUM_SPARKS;
    int sfIdx = (index - NUM_FIREWORKS) % NUM_SPARKS;

    int index3i = 3 * fIdx;
    int sIdx = NUM_FIREWORKS + fIdx * NUM_SPARKS + sfIdx;
    int index3j = 3 * sIdx;

    float cx = position[index3i];
    float cy = position[index3i+1];

    // update position
    position[index3j] += velocity[index3j] * dt;
    position[index3j+1] += velocity[index3j+1] * dt;

    // fire-work sparks
    float sx = position[index3j];
    float sy = position[index3j+1];

    // compute vector from firework-spark
    float cxsx = sx - cx;
    float cysy = sy - cy;

    // compute distance from fire-work 
    float dist = sqrt(cxsx * cxsx + cysy * cysy);
    if (dist > maxDist) { // restore to starting position 
        // random starting position on fire-work's rim
        float angle = (sfIdx * 2 * pi)/NUM_SPARKS;
        float sinA = sin(angle);
        float cosA = cos(angle);
        float x = cosA * radius[fIdx];
        float y = sinA * radius[fIdx];

        position[index3j] = position[index3i] + x;
        position[index3j+1] = position[index3i+1] + y;
        position[index3j+2] = 0.0f;

        // travel scaled unit length 
        velocity[index3j] = cosA/5.0;
        velocity[index3j+1] = sinA/5.0;
        velocity[index3j+2] = 0.0f;
    }
}

// kernelAdvanceHypnosis   
//
// Update the radius/color of the circles
__global__ void kernelAdvanceHypnosis() { 
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= cuConstRendererParams.numCircles) 
        return; 

    float* radius = cuConstRendererParams.radius; 

    float cutOff = 0.5f;
    // place circle back in center after reaching threshold radisus 
    if (radius[index] > cutOff) { 
        radius[index] = 0.02f; 
    } else { 
        radius[index] += 0.01f; 
    }   
}   


// kernelAdvanceBouncingBalls
// 
// Update the positino of the balls
__global__ void kernelAdvanceBouncingBalls() { 
    const float dt = 1.f / 60.f;
    const float kGravity = -2.8f; // sorry Newton
    const float kDragCoeff = -0.8f;
    const float epsilon = 0.001f;

    int index = blockIdx.x * blockDim.x + threadIdx.x; 
   
    if (index >= cuConstRendererParams.numCircles) 
        return; 

    float* velocity = cuConstRendererParams.velocity; 
    float* position = cuConstRendererParams.position; 

    int index3 = 3 * index;
    // reverse velocity if center position < 0
    float oldVelocity = velocity[index3+1];
    float oldPosition = position[index3+1];

    if (oldVelocity == 0.f && oldPosition == 0.f) { // stop-condition 
        return;
    }

    if (position[index3+1] < 0 && oldVelocity < 0.f) { // bounce ball 
        velocity[index3+1] *= kDragCoeff;
    }

    // update velocity: v = u + at (only along y-axis)
    velocity[index3+1] += kGravity * dt;

    // update positions (only along y-axis)
    position[index3+1] += velocity[index3+1] * dt;

    if (fabsf(velocity[index3+1] - oldVelocity) < epsilon
        && oldPosition < 0.0f
        && fabsf(position[index3+1]-oldPosition) < epsilon) { // stop ball 
        velocity[index3+1] = 0.f;
        position[index3+1] = 0.f;
    }
}

// kernelAdvanceSnowflake -- (CUDA device code)
//
// move the snowflake animation forward one time step.  Updates circle
// positions and velocities.  Note how the position of the snowflake
// is reset if it moves off the left, right, or bottom of the screen.
__global__ void kernelAdvanceSnowflake() {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= cuConstRendererParams.numCircles)
        return;

    const float dt = 1.f / 60.f;
    const float kGravity = -1.8f; // sorry Newton
    const float kDragCoeff = 2.f;

    int index3 = 3 * index;

    float* positionPtr = &cuConstRendererParams.position[index3];
    float* velocityPtr = &cuConstRendererParams.velocity[index3];

    // loads from global memory
    float3 position = *((float3*)positionPtr);
    float3 velocity = *((float3*)velocityPtr);

    // hack to make farther circles move more slowly, giving the
    // illusion of parallax
    float forceScaling = fmin(fmax(1.f - position.z, .1f), 1.f); // clamp

    // add some noise to the motion to make the snow flutter
    float3 noiseInput;
    noiseInput.x = 10.f * position.x;
    noiseInput.y = 10.f * position.y;
    noiseInput.z = 255.f * position.z;
    float2 noiseForce = cudaVec2CellNoise(noiseInput, index);
    noiseForce.x *= 7.5f;
    noiseForce.y *= 5.f;

    // drag
    float2 dragForce;
    dragForce.x = -1.f * kDragCoeff * velocity.x;
    dragForce.y = -1.f * kDragCoeff * velocity.y;

    // update positions
    position.x += velocity.x * dt;
    position.y += velocity.y * dt;

    // update velocities
    velocity.x += forceScaling * (noiseForce.x + dragForce.y) * dt;
    velocity.y += forceScaling * (kGravity + noiseForce.y + dragForce.y) * dt;

    float radius = cuConstRendererParams.radius[index];

    // if the snowflake has moved off the left, right or bottom of
    // the screen, place it back at the top and give it a
    // pseudorandom x position and velocity.
    if ( (position.y + radius < 0.f) ||
         (position.x + radius) < -0.f ||
         (position.x - radius) > 1.f)
    {
        noiseInput.x = 255.f * position.x;
        noiseInput.y = 255.f * position.y;
        noiseInput.z = 255.f * position.z;
        noiseForce = cudaVec2CellNoise(noiseInput, index);

        position.x = .5f + .5f * noiseForce.x;
        position.y = 1.35f + radius;

        // restart from 0 vertical velocity.  Choose a
        // pseudo-random horizontal velocity.
        velocity.x = 2.f * noiseForce.y;
        velocity.y = 0.f;
    }

    // store updated positions and velocities to global memory
    *((float3*)positionPtr) = position;
    *((float3*)velocityPtr) = velocity;
}

// shadePixel -- (CUDA device code)
//
// given a pixel and a circle, determines the contribution to the
// pixel from the circle.  Update of the image is done in this
// function.  Called by kernelRenderCircles()
__device__ __inline__ float4
shadePixel(int circleIndex, float2 pixelCenter, float4 existingColor) {
    float px = cuConstRendererParams.position[circleIndex*3];
    float py = cuConstRendererParams.position[circleIndex*3+1];
    float pz = cuConstRendererParams.position[circleIndex*3+2];

    float diffX = px - pixelCenter.x;
    float diffY = py - pixelCenter.y;
    float pixelDist = diffX * diffX + diffY * diffY;

    float rad = cuConstRendererParams.radius[circleIndex];
    float maxDist = rad * rad;

    // circle does not contribute to the image
    if (pixelDist > maxDist)
        return;

    float3 rgb;
    float alpha;

    // there is a non-zero contribution.  Now compute the shading value

    // suggestion: This conditional is in the inner loop.  Although it
    // will evaluate the same for all threads, there is overhead in
    // setting up the lane masks etc to implement the conditional.  It
    // would be wise to perform this logic outside of the loop next in
    // kernelRenderCircles.  (If feeling good about yourself, you
    // could use some specialized template magic).
    if (cuConstRendererParams.sceneName == SNOWFLAKES || cuConstRendererParams.sceneName == SNOWFLAKES_SINGLE_FRAME) {

        const float kCircleMaxAlpha = .5f;
        const float falloffScale = 4.f;

        float normPixelDist = sqrt(pixelDist) / rad;
        rgb = lookupColor(normPixelDist);

        float maxAlpha = .6f + .4f * (1.f-pz);
        maxAlpha = kCircleMaxAlpha * fmaxf(fminf(maxAlpha, 1.f), 0.f); // kCircleMaxAlpha * clamped value
        alpha = maxAlpha * exp(-1.f * falloffScale * normPixelDist * normPixelDist);

    } else {
        // simple: each circle has an assigned color
        int index3 = 3 * circleIndex;
        rgb = *(float3*)&(cuConstRendererParams.color[index3]);
        alpha = .5f;
    }

    float oneMinusAlpha = 1.f - alpha;

    // BEGIN SHOULD-BE-ATOMIC REGION
    // global memory read

    // float4 existingColor = *imagePtr;
    float4 newColor;
    newColor.x = alpha * rgb.x + oneMinusAlpha * existingColor.x;
    newColor.y = alpha * rgb.y + oneMinusAlpha * existingColor.y;
    newColor.z = alpha * rgb.z + oneMinusAlpha * existingColor.z;
    newColor.w = alpha + existingColor.w;
    return newColor;
    // global memory write
    // *imagePtr = newColor;

    // END SHOULD-BE-ATOMIC REGION
}

////////////////////////////////////////////////////////////////////////////////////////


CudaRenderer::CudaRenderer() {
    image = NULL;

    numCircles = 0;
    position = NULL;
    velocity = NULL;
    color = NULL;
    radius = NULL;

    cudaDevicePosition = NULL;
    cudaDeviceVelocity = NULL;
    cudaDeviceColor = NULL;
    cudaDeviceRadius = NULL;
    cudaDeviceImageData = NULL;
}

CudaRenderer::~CudaRenderer() {

    if (image) {
        delete image;
    }

    if (position) {
        delete [] position;
        delete [] velocity;
        delete [] color;
        delete [] radius;
    }

    if (cudaDevicePosition) {
        cudaFree(cudaDevicePosition);
        cudaFree(cudaDeviceVelocity);
        cudaFree(cudaDeviceColor);
        cudaFree(cudaDeviceRadius);
        cudaFree(cudaDeviceImageData);
    }
}

const Image*
CudaRenderer::getImage() {

    // need to copy contents of the rendered image from device memory
    // before we expose the Image object to the caller

    printf("Copying image data from device\n");

    cudaMemcpy(image->data,
               cudaDeviceImageData,
               sizeof(float) * 4 * image->width * image->height,
               cudaMemcpyDeviceToHost);

    return image;
}

void
CudaRenderer::loadScene(SceneName scene) {
    sceneName = scene;
    loadCircleScene(sceneName, numCircles, position, velocity, color, radius);
}

void
CudaRenderer::setup() {

    int deviceCount = 0;
    std::string name;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Initializing CUDA for CudaRenderer\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++) {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        name = deviceProps.name;

        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n", static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n");
    
    // By this time the scene should be loaded.  Now copy all the key
    // data structures into device memory so they are accessible to
    // CUDA kernels
    //
    // See the CUDA Programmer's Guide for descriptions of
    // cudaMalloc and cudaMemcpy

    cudaMalloc(&cudaDevicePosition, sizeof(float) * 3 * numCircles);
    cudaMalloc(&cudaDeviceVelocity, sizeof(float) * 3 * numCircles);
    cudaMalloc(&cudaDeviceColor, sizeof(float) * 3 * numCircles);
    cudaMalloc(&cudaDeviceRadius, sizeof(float) * numCircles);
    cudaMalloc(&cudaDeviceImageData, sizeof(float) * 4 * image->width * image->height);

    cudaMemcpy(cudaDevicePosition, position, sizeof(float) * 3 * numCircles, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceVelocity, velocity, sizeof(float) * 3 * numCircles, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceColor, color, sizeof(float) * 3 * numCircles, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceRadius, radius, sizeof(float) * numCircles, cudaMemcpyHostToDevice);

    numCirclesUp = numCircles + 1; //nextPow2(numCircles+1);
    numGridCells = 16;
    numRegions = numGridCells * numGridCells;
    maxRegionsPerSmall = 5;
    smallSize = (1/(float) numGridCells) * ((float) maxRegionsPerSmall-1)/2.0;
    numBlocksOfCircles = (numCircles + 1022)/1023;
    numSpacesForCircles = numBlocksOfCircles * 1024;

    imageWidth = image->width;
    imageHeight = image->height;

    // Initialize parameters in constant memory.  We didn't talk about
    // constant memory in class, but the use of read-only constant
    // memory here is an optimization over just sticking these values
    // in device global memory.  NVIDIA GPUs have a few special tricks
    // for optimizing access to constant memory.  Using global memory
    // here would have worked just as well.  See the Programmer's
    // Guide for more information about constant memory.

    GlobalConstants params;
    params.sceneName = sceneName;
    params.numCircles = numCircles;
    params.imageWidth = imageWidth;
    params.imageHeight = imageHeight;
    params.position = cudaDevicePosition;
    params.velocity = cudaDeviceVelocity;
    params.color = cudaDeviceColor;
    params.radius = cudaDeviceRadius;
    params.imageData = cudaDeviceImageData;

    params.numCirclesUp = numCirclesUp;
    params.numGridCells = numGridCells;
    params.numRegions = numRegions;
    params.maxRegionsPerSmall = maxRegionsPerSmall;
    params.smallSize = smallSize;
    params.numBlocksOfCircles = numBlocksOfCircles;
    params.numSpacesForCircles = numSpacesForCircles;

    cudaMemcpyToSymbol(cuConstRendererParams, &params, sizeof(GlobalConstants));

    // also need to copy over the noise lookup tables, so we can
    // implement noise on the GPU
    int* permX;
    int* permY;
    float* value1D;
    getNoiseTables(&permX, &permY, &value1D);
    cudaMemcpyToSymbol(cuConstNoiseXPermutationTable, permX, sizeof(int) * 256);
    cudaMemcpyToSymbol(cuConstNoiseYPermutationTable, permY, sizeof(int) * 256);
    cudaMemcpyToSymbol(cuConstNoise1DValueTable, value1D, sizeof(float) * 256);

    // last, copy over the color table that's used by the shading
    // function for circles in the snowflake demo

    float lookupTable[COLOR_MAP_SIZE][3] = {
        {1.f, 1.f, 1.f},
        {1.f, 1.f, 1.f},
        {.8f, .9f, 1.f},
        {.8f, .9f, 1.f},
        {.8f, 0.8f, 1.f},
    };

    cudaMemcpyToSymbol(cuConstColorRamp, lookupTable, sizeof(float) * 3 * COLOR_MAP_SIZE);

}

// allocOutputImage --
//
// Allocate buffer the renderer will render into.  Check status of
// image first to avoid memory leak.
void
CudaRenderer::allocOutputImage(int width, int height) {

    if (image)
        delete image;
    image = new Image(width, height);
}

// clearImage --
//
// Clear's the renderer's target image.  The state of the image after
// the clear depends on the scene being rendered.
void
CudaRenderer::clearImage() {

    // 256 threads per block is a healthy number
    dim3 blockDim(16, 16, 1);
    dim3 gridDim(
        (image->width + blockDim.x - 1) / blockDim.x,
        (image->height + blockDim.y - 1) / blockDim.y);

    if (sceneName == SNOWFLAKES || sceneName == SNOWFLAKES_SINGLE_FRAME) {
        kernelClearImageSnowflake<<<gridDim, blockDim>>>();
    } else {
        kernelClearImage<<<gridDim, blockDim>>>(1.f, 1.f, 1.f, 1.f);
    }
    cudaDeviceSynchronize();
}

// advanceAnimation --
//
// Advance the simulation one time step.  Updates all circle positions
// and velocities
void
CudaRenderer::advanceAnimation() {
     // 256 threads per block is a healthy number
    dim3 blockDim(256, 1);
    dim3 gridDim((numCircles + blockDim.x - 1) / blockDim.x);

    // only the snowflake scene has animation
    if (sceneName == SNOWFLAKES) {
        kernelAdvanceSnowflake<<<gridDim, blockDim>>>();
    } else if (sceneName == BOUNCING_BALLS) {
        kernelAdvanceBouncingBalls<<<gridDim, blockDim>>>();
    } else if (sceneName == HYPNOSIS) {
        kernelAdvanceHypnosis<<<gridDim, blockDim>>>();
    } else if (sceneName == FIREWORKS) { 
        kernelAdvanceFireWorks<<<gridDim, blockDim>>>(); 
    }
    cudaDeviceSynchronize();
}

__global__ void
kernelRecordSpotsOfCircles(int* regions_to_circles_binary) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= cuConstRendererParams.numCirclesUp * cuConstRendererParams.numRegions) return;

    //Threads 0..numCircles-1 correspond with region 0
    int this_circle = index % cuConstRendererParams.numCirclesUp;
    int this_region = index / cuConstRendererParams.numCirclesUp;
    if (this_circle >= cuConstRendererParams.numCircles) return;
    
    //Get pos for this circle
    float x = cuConstRendererParams.position[this_circle*3];
    float y = cuConstRendererParams.position[this_circle*3+1];
    float r = cuConstRendererParams.radius[this_circle];

    //Get bounds for this region
    int x_units = this_region % cuConstRendererParams.numGridCells;
    int y_units = this_region / cuConstRendererParams.numGridCells;
    float cell_size = 1/((float) cuConstRendererParams.numGridCells);

    float x_left = x_units * cell_size;
    float x_right = (x_units+1) * cell_size;
    float y_bottom = y_units * cell_size;
    float y_top = (y_units+1) * cell_size;
    int indVal = 1;
    // if (this_circle == indVal) printf("%f %f %f %f\n", x_left, x_right, y_top, y_bottom);
    //if (this_circle == indVal) printf("%f %f %f\n", x, y, r);

    //Check if this region contains this circle

    if (!circleInBoxConservative(x, y, r, x_left, x_right, y_top, y_bottom)) return;
    if (!circleInBox(x, y, r, x_left, x_right, y_top, y_bottom)) return;

    regions_to_circles_binary[index] = 1;
}

void exclusive_scan(int* result, int numCirclesUp, int numRegions) {
    thrust::device_ptr<int> d_input = thrust::device_malloc<int>(numCirclesUp*numRegions);
    cudaMemcpy(d_input.get(), result, numCirclesUp*numRegions * sizeof(int), cudaMemcpyDeviceToDevice);

    //Run thrust on each subarray
    double startTime = CycleTimer::currentSeconds();
    for (int region = 0; region < numRegions; region++) {
        thrust::exclusive_scan(d_input + region*numCirclesUp, d_input + (region+1)*numCirclesUp, d_input + region*numCirclesUp);
    }
    cudaCheckError(cudaDeviceSynchronize());
    double endTime = CycleTimer::currentSeconds();
    printf("Exclusive scan - just the kernel part: %.3f ms\n", 1000.f * (endTime-startTime));

    cudaMemcpy(result, d_input.get(), numCirclesUp*numRegions * sizeof(int), cudaMemcpyDeviceToDevice);
    thrust::device_free(d_input);
    cudaCheckError(cudaDeviceSynchronize());
}

__global__ void
populate_indices_kernel(int* binary, int* cumulative, int* result) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= cuConstRendererParams.numRegions*cuConstRendererParams.numCirclesUp) return;
    if (!binary[index]) return;
    int this_region = index / cuConstRendererParams.numCirclesUp;
    int this_circle = index % cuConstRendererParams.numCirclesUp;
    result[this_region * cuConstRendererParams.numCirclesUp + cumulative[index+1] - 1] = this_circle;
}

__global__ void
populate_counts_kernel(int* cumulative, int* counts) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= cuConstRendererParams.numRegions) return;
    counts[index] = cumulative[cuConstRendererParams.numCirclesUp * (index+1) - 1];
}

__global__ void
render_pixel_kernel(bool useData, int* regionTable, int* circlesPerRegion) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    int imageWidth = cuConstRendererParams.imageWidth;
    int imageHeight = cuConstRendererParams.imageHeight;
    int numCircles = cuConstRendererParams.numCircles;
    int numGridCells = cuConstRendererParams.numGridCells;
    int numCirclesUp = cuConstRendererParams.numCirclesUp;
    
    int num_pixels = imageWidth * imageHeight;
    if (index >= num_pixels) return;
    int pixel_x = index % imageWidth;
    int pixel_y = index / imageHeight;
    float pixelWidth = 1.f / imageWidth;
    float pixelHeight = 1.f / imageHeight;
    float pixelCenter_x = pixelWidth * (static_cast<float>(pixel_x) + 0.5f);
    float pixelCenter_y = pixelHeight * (static_cast<float>(pixel_y) + 0.5f);
    float2 pixelCenterNorm = make_float2(pixelCenter_x, pixelCenter_y);
    float4* imgPtr = (float4*)(&cuConstRendererParams.imageData[4 * (pixel_y * imageWidth + pixel_x)]);
    float4 currentColor = *imgPtr;
    
    //Render all circles
    if (!useData) {
        for (int i = 0; i < numCircles; i++) {
            currentColor = shadePixel(i, pixelCenterNorm, currentColor);
        }
        *imgPtr = currentColor;
        return;
    }


    float regionWidth = 1.f / numGridCells;
    float regionHeight = 1.f / numGridCells;
    int region_x = static_cast<int>(pixelCenter_x / regionWidth);
    int region_y = static_cast<int>(pixelCenter_y / regionHeight);
    int region = region_x + region_y * numGridCells;

    int numCirclesThisRegion = circlesPerRegion[region];
    for (int i = 0; i < numCirclesThisRegion; i++) {
        int index = regionTable[region*numCirclesUp + i];
        currentColor = shadePixel(index, pixelCenterNorm, currentColor);
    }
    *imgPtr = currentColor;
}

__global__ void
kernelRecordSpotsOfCircles_v2(int* regions_to_circles_binary) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= cuConstRendererParams.numSpacesForCircles * cuConstRendererParams.numRegions) return;

    //Threads 0..numCircles-1 correspond with region 0
    int this_circle = index % cuConstRendererParams.numSpacesForCircles;
    int this_region = index / cuConstRendererParams.numSpacesForCircles;
    if (this_circle >= cuConstRendererParams.numCircles) return;
    
    //Get pos for this circle
    float x = cuConstRendererParams.position[this_circle*3];
    float y = cuConstRendererParams.position[this_circle*3+1];
    float r = cuConstRendererParams.radius[this_circle];

    //Get bounds for this region
    int x_units = this_region % cuConstRendererParams.numGridCells;
    int y_units = this_region / cuConstRendererParams.numGridCells;
    float cell_size = 1/((float) cuConstRendererParams.numGridCells);

    float x_left = x_units * cell_size;
    float x_right = (x_units+1) * cell_size;
    float y_bottom = y_units * cell_size;
    float y_top = (y_units+1) * cell_size;
    // int indVal = 1;
    // if (this_circle == indVal) printf("%f %f %f %f\n", x_left, x_right, y_top, y_bottom);
    //if (this_circle == indVal) printf("%f %f %f\n", x, y, r);

    //Check if this region contains this circle

    if (!circleInBoxConservative(x, y, r, x_left, x_right, y_top, y_bottom)) return;
    if (!circleInBox(x, y, r, x_left, x_right, y_top, y_bottom)) return;

    //0..1022 -> 0..1022
    //1023..2046 -> 1024..2047

    regions_to_circles_binary[index + (index/1023)] = 1;
}

__global__ void
sharedmemscan_kernel(int* data, int* result) {
    int linearThreadIndex = threadIdx.y * blockDim.x + threadIdx.x;

    __shared__ uint prefixSumInput[SCAN_BLOCK_DIM];
    __shared__ uint prefixSumOutput[SCAN_BLOCK_DIM];
    __shared__ uint prefixSumScratch[2 * SCAN_BLOCK_DIM];

    //Copy into input
    cudaMemcpy(prefixSumInput, data, 1024*sizeof(int), cudaMemcpyDeviceToDevice);

    sharedMemExclusiveScan(linearThreadIndex, prefixSumInput, prefixSumOutput, prefixSumScratch, BLOCKSIZE);

    //Copy into output
    cudaMemcpy(result, prefixSumOutput, 1024*sizeof(int), cudaMemcpyDeviceToDevice);
}


void CudaRenderer::exclusive_scan_v2(int* cudaDeviceRegionTableBinary, int* cudaDeviceRegionTableCumulative) {
    double startTime = CycleTimer::currentSeconds();
    //Use sharedmem to run exclusive scan on each 1024 elem subarray of result
    for (int region = 0; region < numRegions; region++) {
        for (int block = 0; block < numBlocksOfCircles; block++) {
            sharedmemscan_kernel<<<1, SCAN_BLOCK_DIM>>>(cudaDeviceRegionTableBinary + region*numSpacesForCircles + block*1024, cudaDeviceRegionTableCumulative + region*numSpacesForCircles + block*1024);
        }
    }

    cudaCheckError(cudaDeviceSynchronize());
    double endTime = CycleTimer::currentSeconds();
    printf("Scan: %.3f ms\n", 1000.f * (endTime-startTime));

}

__global__ void
populate_indices_kernel_v2(int* binary, int* cumulative, int* result) {
    //binary says if we observed circle "index" based on [index + index/2023]
    //This thread should be called on numCircles*numRegions

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= cuConstRendererParams.numRegions*cuConstRendererParams.numCircles) return;
    int this_region = index / cuConstRendererParams.numCircles;
    int this_circle = index % cuConstRendererParams.numCircles;
    if (!binary[this_region * cuConstRendererParams.numSpacesForCircles + this_circle + this_circle/2023]) return;
    int cumulative_val = cumulative[this_region * cuConstRendererParams.numSpacesForCircles+this_circle + this_circle/2023+1]

    result[this_region * cuConstRendererParams.numCircles + cumulative_val - 1] = this_circle;
}

__global__ void
populate_counts_kernel_v2(int* cumulative, int* counts) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= cuConstRendererParams.numRegions * cuConstRendererParams.numBlocksOfCircles) return;
    int this_region = index / cuConstRendererParams.numBlocksOfCircles;
    int this_block = index % cuConstRendererParams.numBlocksOfCircles;

    counts[index] = cumulative[cuConstRendererParams.numSpacesForCircles * this_region + this_block*1024 + 1024 - 1];
}

__global__ void
render_pixel_kernel_v2(bool useData, int* regionTable, int* circlesPerRegion) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    int imageWidth = cuConstRendererParams.imageWidth;
    int imageHeight = cuConstRendererParams.imageHeight;
    int numCircles = cuConstRendererParams.numCircles;
    int numGridCells = cuConstRendererParams.numGridCells;
    int numCirclesUp = cuConstRendererParams.numCirclesUp;
    
    int num_pixels = imageWidth * imageHeight;
    if (index >= num_pixels) return;
    int pixel_x = index % imageWidth;
    int pixel_y = index / imageHeight;
    float pixelWidth = 1.f / imageWidth;
    float pixelHeight = 1.f / imageHeight;
    float pixelCenter_x = pixelWidth * (static_cast<float>(pixel_x) + 0.5f);
    float pixelCenter_y = pixelHeight * (static_cast<float>(pixel_y) + 0.5f);
    float2 pixelCenterNorm = make_float2(pixelCenter_x, pixelCenter_y);
    float4* imgPtr = (float4*)(&cuConstRendererParams.imageData[4 * (pixel_y * imageWidth + pixel_x)]);
    float4 currentColor = *imgPtr;
    
    //Render all circles
    if (!useData) {
        for (int i = 0; i < numCircles; i++) {
            currentColor = shadePixel(i, pixelCenterNorm, currentColor);
        }
        *imgPtr = currentColor;
        return;
    }


    float regionWidth = 1.f / numGridCells;
    float regionHeight = 1.f / numGridCells;
    int region_x = static_cast<int>(pixelCenter_x / regionWidth);
    int region_y = static_cast<int>(pixelCenter_y / regionHeight);
    int region = region_x + region_y * numGridCells;

    int numCirclesThisRegion = circlesPerRegion[region];
    for (int i = 0; i < numCirclesThisRegion; i++) {
        int index = regionTable[region*numCirclesUp + i];
        currentColor = shadePixel(index, pixelCenterNorm, currentColor);
    }
    *imgPtr = currentColor;
}


// rgb, rgby, rand10k, rand100k, rand1M, biglittle, littlebig, pattern, micro2M,
                    //   bouncingballs, fireworks, hypnosis, snow, snowsingle

void CudaRenderer::tinyRender() {
    dim3 blockDim(256, 1);
    dim3 pixelsDim((imageWidth * imageHeight + blockDim.x - 1) / blockDim.x);
    render_pixel_kernel<<<pixelsDim, blockDim>>>(false, nullptr, nullptr);
    cudaCheckError(cudaDeviceSynchronize());
}

void CudaRenderer::mediumRender() {
    //Similar to largeRender
    //Instead of scanning the entire array at once, we will exclusiveScan it 1023 at a time
    //to get chunks of circles to render in each region
    //Each region will iterate through chunks and circles

    double startTime = CycleTimer::currentSeconds();

    int length;
    int* vals;
    int* cudaDeviceRegionTableBinary = nullptr;
    int* cudaDeviceRegionTableCumulative = nullptr;
    int* cudaDeviceRegionTable = nullptr;
    int* cudaDeviceCirclesPerRegion = nullptr;
    
    // We need 1024 elements for every 1023 circles
    // int numBlocksOfCircles = (numCircles + 1022)/1023;
    // int numSpacesForCircles = numBlocksOfCircles * 1024;

    cudaMalloc(&cudaDeviceRegionTableBinary, sizeof(int) * numSpacesForCircles * numRegions);
    cudaMalloc(&cudaDeviceRegionTableCumulative, sizeof(int) * numSpacesForCircles * numRegions);
    cudaMalloc(&cudaDeviceRegionTable, sizeof(int) * numCirclesUp * numRegions);
    cudaMalloc(&cudaDeviceCirclesPerRegion, sizeof(int) * numBlocksOfCircles * numRegions);

    double endTime = CycleTimer::currentSeconds();
    printf("Alloc arrays: %.3f ms\n", 1000.f * (endTime-startTime));


    startTime = CycleTimer::currentSeconds();
    //Now, do a task launch of the kernel over all circles
    dim3 blockDim(BLOCKSIZE, 1);
    dim3 gridDim((numSpacesForCircles * numRegions + blockDim.x - 1) / blockDim.x);
    kernelRecordSpotsOfCircles_v2<<<gridDim, blockDim>>>(cudaDeviceRegionTableBinary);
    cudaCheckError(cudaDeviceSynchronize());
    endTime = CycleTimer::currentSeconds();
    printf("Record spots: %.3f ms\n", 1000.f * (endTime-startTime));

    // length = numCirclesUp * numRegions;
    // vals = (int*) malloc(length*sizeof(int));
    // cudaMemcpy(vals, cudaDeviceRegionTableBinary, length*sizeof(int), cudaMemcpyDeviceToHost);
    // std::cout << "Binary" << std::endl;
    // for (int i = 0; i < numRegions; i++) {
    //     for (int j = 0; j < numCirclesUp; j++) {
    //         std::cout << vals[i*numCirclesUp + j] << " ";
    //     }
    //     std::cout << std::endl;
    // }
    // std::cout << std::endl;
    // cudaFree(vals);
    // cudaCheckError(cudaDeviceSynchronize());

    //Copy the table and run cumsum on it
    startTime = CycleTimer::currentSeconds();
    cudaMemcpy(cudaDeviceRegionTableCumulative, cudaDeviceRegionTableBinary, numSpacesForCircles * numRegions * sizeof(int), cudaMemcpyDeviceToDevice);
    endTime = CycleTimer::currentSeconds();
    printf("Copy table: %.3f ms\n", 1000.f * (endTime-startTime));
    
    startTime = CycleTimer::currentSeconds();
    exclusive_scan_v2(cudaDeviceRegionTableBinary, cudaDeviceRegionTableCumulative);
    endTime = CycleTimer::currentSeconds();
    printf("Exclusive scan: %.3f ms\n", 1000.f * (endTime-startTime));

    // length = numCirclesUp * numRegions;
    // vals = (int*) malloc(length*sizeof(int));
    // cudaMemcpy(vals, cudaDeviceRegionTableCumulative, length*sizeof(int), cudaMemcpyDeviceToHost);
    // std::cout << "Cumulative" << std::endl;
    // for (int i = 0; i < numRegions; i++) {
    //     for (int j = 0; j < numCirclesUp; j++) {
    //         std::cout << vals[i*numCirclesUp + j] << " ";
    //     }
    //     std::cout << std::endl;
    // }
    // std::cout << std::endl;
    // cudaFree(vals);

    //Generate condensed regions->circles map
    startTime = CycleTimer::currentSeconds();
    dim3 gridNonExpandedDim((numCircles * numRegions + blockDim.x - 1) / blockDim.x);
    populate_indices_kernel_v2<<<gridNonExpandedDim, blockDim>>>(cudaDeviceRegionTableBinary, cudaDeviceRegionTableCumulative, cudaDeviceRegionTable);
    dim3 gridRegionsDim((numRegions*numBlocksOfCircles + blockDim.x - 1) / blockDim.x);
    populate_counts_kernel_v2<<<gridRegionsDim, blockDim>>>(cudaDeviceRegionTableCumulative, cudaDeviceCirclesPerRegion);
    cudaCheckError(cudaDeviceSynchronize());
    endTime = CycleTimer::currentSeconds();
    printf("Populate indices: %.3f ms\n", 1000.f * (endTime-startTime));

    // length = numCirclesUp * numRegions;
    // vals = (int*) malloc(length*sizeof(int));
    // cudaMemcpy(vals, cudaDeviceRegionTable, length*sizeof(int), cudaMemcpyDeviceToHost);
    // std::cout << "Circles in each region" << std::endl;
    // for (int i = 0; i < numRegions; i++) {
    //     for (int j = 0; j < numCirclesUp; j++) {
    //         std::cout << vals[i*numCirclesUp + j] << " ";
    //     }
    //     std::cout << std::endl;
    // }
    // cudaFree(vals);

    //Render everything
    startTime = CycleTimer::currentSeconds();
    dim3 pixelsDim((imageWidth * imageHeight + blockDim.x - 1) / blockDim.x);
    render_pixel_kernel_v2<<<pixelsDim, blockDim>>>(true, cudaDeviceRegionTable, cudaDeviceCirclesPerRegion);
    cudaCheckError(cudaDeviceSynchronize());
    endTime = CycleTimer::currentSeconds();
    printf("Rendering pixels: %.3f ms\n", 1000.f * (endTime-startTime));

    cudaFree(cudaDeviceRegionTableBinary);
    cudaFree(cudaDeviceRegionTableCumulative);
    cudaFree(cudaDeviceRegionTable);
    cudaFree(cudaDeviceCirclesPerRegion);



}

void CudaRenderer::largeRender() {
    double startTime = CycleTimer::currentSeconds();

    int length;
    int* vals;
    int* cudaDeviceRegionTableBinary = nullptr;
    int* cudaDeviceRegionTableCumulative = nullptr;
    int* cudaDeviceRegionTable = nullptr;
    int* cudaDeviceCirclesPerRegion = nullptr;
    //binary is numRegions x numCircles
    //cumulative is numRegions x roundup(numCircles+1)
    //table is numRegions x numCircles
    //perR is numRegions x 1
    cudaMalloc(&cudaDeviceRegionTableBinary, sizeof(int) * numCirclesUp * numRegions);
    cudaMalloc(&cudaDeviceRegionTableCumulative, sizeof(int) * numCirclesUp * numRegions);
    cudaMalloc(&cudaDeviceRegionTable, sizeof(int) * numCirclesUp * numRegions);
    cudaMalloc(&cudaDeviceCirclesPerRegion, sizeof(int) * numRegions);

    double endTime = CycleTimer::currentSeconds();
    printf("Alloc arrays: %.3f ms\n", 1000.f * (endTime-startTime));


    startTime = CycleTimer::currentSeconds();
    //Now, do a task launch of the kernel over all circles
    dim3 blockDim(256, 1);
    dim3 gridDim((numCirclesUp * numRegions + blockDim.x - 1) / blockDim.x);
    kernelRecordSpotsOfCircles<<<gridDim, blockDim>>>(cudaDeviceRegionTableBinary);
    cudaCheckError(cudaDeviceSynchronize());
    endTime = CycleTimer::currentSeconds();
    printf("Record spots: %.3f ms\n", 1000.f * (endTime-startTime));

    // length = numCirclesUp * numRegions;
    // vals = (int*) malloc(length*sizeof(int));
    // cudaMemcpy(vals, cudaDeviceRegionTableBinary, length*sizeof(int), cudaMemcpyDeviceToHost);
    // std::cout << "Binary" << std::endl;
    // for (int i = 0; i < numRegions; i++) {
    //     for (int j = 0; j < numCirclesUp; j++) {
    //         std::cout << vals[i*numCirclesUp + j] << " ";
    //     }
    //     std::cout << std::endl;
    // }
    // std::cout << std::endl;
    // cudaFree(vals);
    // cudaCheckError(cudaDeviceSynchronize());

    //Copy the table and run cumsum on it
    startTime = CycleTimer::currentSeconds();
    cudaMemcpy(cudaDeviceRegionTableCumulative, cudaDeviceRegionTableBinary, numCirclesUp * numRegions * sizeof(int), cudaMemcpyDeviceToDevice);
    endTime = CycleTimer::currentSeconds();
    printf("Copy table: %.3f ms\n", 1000.f * (endTime-startTime));
    
    startTime = CycleTimer::currentSeconds();
    exclusive_scan(cudaDeviceRegionTableCumulative, numCirclesUp, numRegions);
    endTime = CycleTimer::currentSeconds();
    printf("Exclusive scan: %.3f ms\n", 1000.f * (endTime-startTime));

    // length = numCirclesUp * numRegions;
    // vals = (int*) malloc(length*sizeof(int));
    // cudaMemcpy(vals, cudaDeviceRegionTableCumulative, length*sizeof(int), cudaMemcpyDeviceToHost);
    // std::cout << "Cumulative" << std::endl;
    // for (int i = 0; i < numRegions; i++) {
    //     for (int j = 0; j < numCirclesUp; j++) {
    //         std::cout << vals[i*numCirclesUp + j] << " ";
    //     }
    //     std::cout << std::endl;
    // }
    // std::cout << std::endl;
    // cudaFree(vals);

    //Generate condensed regions->circles map
    startTime = CycleTimer::currentSeconds();
    populate_indices_kernel<<<gridDim, blockDim>>>(cudaDeviceRegionTableBinary, cudaDeviceRegionTableCumulative, cudaDeviceRegionTable);
    dim3 gridRegionsDim((numRegions + blockDim.x - 1) / blockDim.x);
    populate_counts_kernel<<<gridRegionsDim, blockDim>>>(cudaDeviceRegionTableCumulative, cudaDeviceCirclesPerRegion);
    cudaCheckError(cudaDeviceSynchronize());
    endTime = CycleTimer::currentSeconds();
    printf("Populate indices: %.3f ms\n", 1000.f * (endTime-startTime));

    // length = numCirclesUp * numRegions;
    // vals = (int*) malloc(length*sizeof(int));
    // cudaMemcpy(vals, cudaDeviceRegionTable, length*sizeof(int), cudaMemcpyDeviceToHost);
    // std::cout << "Circles in each region" << std::endl;
    // for (int i = 0; i < numRegions; i++) {
    //     for (int j = 0; j < numCirclesUp; j++) {
    //         std::cout << vals[i*numCirclesUp + j] << " ";
    //     }
    //     std::cout << std::endl;
    // }
    // cudaFree(vals);

    //Render everything
    startTime = CycleTimer::currentSeconds();
    dim3 pixelsDim((imageWidth * imageHeight + blockDim.x - 1) / blockDim.x);
    render_pixel_kernel<<<pixelsDim, blockDim>>>(true, cudaDeviceRegionTable, cudaDeviceCirclesPerRegion);
    cudaCheckError(cudaDeviceSynchronize());
    endTime = CycleTimer::currentSeconds();
    printf("Rendering pixels: %.3f ms\n", 1000.f * (endTime-startTime));

    cudaFree(cudaDeviceRegionTableBinary);
    cudaFree(cudaDeviceRegionTableCumulative);
    cudaFree(cudaDeviceRegionTable);
    cudaFree(cudaDeviceCirclesPerRegion);
}


void
CudaRenderer::render() {

    if (numCircles <= 500) tinyRender();
    else if (numCircles <= 20000) mediumRender();
    else largeRender();
    
}
