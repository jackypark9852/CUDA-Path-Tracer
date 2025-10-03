
WaterBottle.ntc file has been created using the Neural Texture Compression (NTC) SDK, available on Github.

WaterBottle.ntc was created using the following command line with the ntc-cli executable from the NTC SDK:

ntc-cli.exe -c --loadManifest=WaterBottle-ntc-manifest.json -o WaterBottle.ntc --fp8weights --bitsPerPixel=1.5 --kPixelsPerBatch=128 --trainingSteps=300000

You can compare the visual quality and memory usage of the optixNeuralTexture sample to optixMeshViewer. The settings above produce good visual quality with an in-memory footprint considerably smaller than the PNG textures used in optixMeshViewer.

optixNeuralTexture texture memory: 816384 bytes (latents) + 26240 bytes (weights)
optixMeshViewer texture memory: 64 Mbytes (4 PNG textures, 2k x 2k, 32 bits per pixel)
Compression factor: 80x vs uncompressed  /  20x vs DDS BC7
