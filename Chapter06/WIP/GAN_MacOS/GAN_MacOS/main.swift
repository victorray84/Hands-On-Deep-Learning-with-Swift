//
//  main.swift
//  GAN_MacOS
//
//  Created by joshua.newnham on 19/02/2019.
//  Copyright © 2019 Joshua Newnham. All rights reserved.
//

import Foundation
import AppKit
import MetalKit
import MetalPerformanceShaders

let dataPath = FileManager.default.urls(for: .desktopDirectory, in: .userDomainMask)[0].appendingPathComponent("data")

let weightsPath = FileManager.default.urls(for: .desktopDirectory, in: .userDomainMask)[0].appendingPathComponent("weights")

let exportsPath = FileManager.default.urls(for: .desktopDirectory, in: .userDomainMask)[0].appendingPathComponent("exports")

// Create device
guard let device = MTLCreateSystemDefaultDevice() else{
    fatalError("Failed to reference GPU")
}

// Make sure the current device supports MetalPerformanceShaders
guard MPSSupportsMTLDevice(device) else{
    fatalError("Metal Performance Shaders not supported for current device")
}

// Create command queue
guard let commandQueue = device.makeCommandQueue() else{
    fatalError("Failed to create command queue")
}

//print(dataPath.appendingPathComponent("x_test").appendingPathExtension("data").absoluteString)

guard let dataLoader = DataLoader(
    device: device,
    imagesURL: dataPath.appendingPathComponent("x_test").appendingPathExtension("data")) else{
        fatalError("Failed to create an instance of a DataLoader")
}

//var count = 0
//while dataLoader.hasNext(){
//    if let batch = dataLoader.nextBatch(){
//        count += 1
//        print(count)
//
//        if let image = dataLoader.toNSImage(mpsImage: batch[0]){
//            print("hello world")
//        }
//    }
//}

print("Creating GAN")

let gan = GAN.createGAN(
    withCommandQueue: commandQueue,
    weightsPathURL: weightsPath,
    exportImagesURL: exportsPath,
    mode:.training)

print("Training starting")

gan.train(withDataLoader: dataLoader) {
    print("Training Finished")

    if let generatedImages = gan.generateSamples(10, syncronizeWithCPU: true){
        for generatedImage in generatedImages{
            if let nsImage = dataLoader.toNSImage(mpsImage: generatedImage){
                print("NSImage created")
            }
        }
    }
}

print("Finished")
