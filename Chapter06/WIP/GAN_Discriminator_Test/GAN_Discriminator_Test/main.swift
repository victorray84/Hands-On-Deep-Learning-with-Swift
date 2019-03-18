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

guard let dataLoaderFor0 = DataLoader(
    device: device,
    imagesURL: dataPath.appendingPathComponent("x_train").appendingPathExtension("data"),
    labelsURL:dataPath.appendingPathComponent("y_train").appendingPathExtension("data"),
    label:0)
    else{
        fatalError("Failed to create an instance of a DataLoader")
    }

guard let dataLoaderFor1 = DataLoader(
    device: device,
    imagesURL: dataPath.appendingPathComponent("x_train").appendingPathExtension("data"),
    labelsURL:dataPath.appendingPathComponent("y_train").appendingPathExtension("data"),
    label:1)
    else{
        fatalError("Failed to create an instance of a DataLoader")
}

print("Creating Network")


let network = DiscriminatorNetwork(
    withCommandQueue: commandQueue,
    weightsPathURL: weightsPath,
    inputShape: Shape(width:28, height:28, channels:1),
    mode: .training, learningRate: 0.002,
    momentumScale: 0.2)

print("Training starting")

//network.train(
//    dataLoaderA: dataLoaderFor0,
//    dataLoaderB: dataLoaderFor1) {
//
//}

//print("Finished training")
//
//let inferenceNetwork = DiscriminatorNetwork(
//    withCommandQueue: commandQueue,
//    weightsPathURL: weightsPath,
//    inputShape: Shape(width:28, height:28, channels:1),
//    mode: .inference, learningRate: 0.005,
//    momentumScale: 0.5)
//
//print("perform inference on digit 0")
//dataLoaderFor0.reset()
//if let images = dataLoaderFor0.nextBatch(){
//    for image in images{
//        print(inferenceNetwork.predict(image))
//    }
//}
//
//print("perform inference on digit 1")
//dataLoaderFor1.reset()
//if let images = dataLoaderFor1.nextBatch(){
//    for image in images{
//        print(inferenceNetwork.predict(image))
//    }
//}


let sampleGenerator = PooledGANSampleGenerator(device, 100)
if let sample = sampleGenerator.generate(1){
    print(sample[0].toFloatArray())
}
