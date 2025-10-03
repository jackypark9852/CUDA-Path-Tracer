
optixNeuralTexture demonstrates the use of the Cooperative Vectors API introduced in OptiX 9.0.
The Cooperative Vectors API provides access to the NVIDIA Tensor Cores inside ray tracing shader programs.
This sample is loading a neural compressed .ntc texture file into memory, 
and decompressing each texture sample at run time in the closest-hit shader program. 
The primary benefit of using Neural Texture Compression is a significant memory savings. 
In this case, optixNeuralTexture uses 80x less VRAM to store textures in memory
compared to the uncompressed textures used in the optixMeshViewer sample.
For more details, or to reproduce or customize the texture used in this sample, 
please see samples_exp\data\NeuralTexture\README.txt

A minimal copy of libntc source code has been included for the purpose of 
building this sample automatically without requiring any other external downloads. 
libntc was cloned at SHA aed2267688ffe1a03ede3bee8acedc223a4dea60.
In the future, this sample may link to and automatically download the public version of libntc.

To use libntc in your own projects, please find the latest version of the NTC SDK on Github.

The official NTC SDK includes command line and GUI tools for 
compressing textures and exploring the neural training parameters. 
The NTC SDK also includes example inference code for DirectX, Vulkan, CUDA, and Slang.
