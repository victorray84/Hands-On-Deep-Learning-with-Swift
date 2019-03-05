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

var count = 0
while dataLoader.hasNext(){
    if let batch = dataLoader.nextBatch(){
        count += 1
        print(count)
        
        if let rawBytes = batch[0].toFloatArray()?.map({ (val) -> UInt8 in
            return UInt8((val * 127.5) + 127.5)
        }){
            if let cgImage = CGImage.fromByteArray(
                bytes: rawBytes,
                width: batch[0].width,
                height: batch[0].height,
                channels: batch[0].featureChannels){
                
                let image = NSImage(
                    cgImage: cgImage,
                    size: NSSize(width: cgImage.width, height: cgImage.height))
                
                print("hello world")
            }
        }
    }
}

//print("Creating GAN")
//
//let gan = GAN.createGAN(
//    withCommandQueue: commandQueue,
//    weightsPathURL: weightsPath,
//    exportImagesURL: exportsPath,
//    mode:.training)
//
//gan.train(withDataLoader: dataLoader) {
//    print("Finished training")
//
//    if let samples = gan.generateSamples(10){
//        print("sample created")
//    }
//}

print("Finished")
