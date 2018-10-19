//: [Previous](@previous)

import Foundation
import AppKit
import AVFoundation
import CoreGraphics
import MetalKit
import MetalPerformanceShaders
import Accelerate
import GLKit
import PlaygroundSupport

let BASE_VALID_PATH = "Sketches/preprocessed/valid"

let validPath = PlaygroundSupport
    .playgroundSharedDataDirectory
    .appendingPathComponent(BASE_VALID_PATH)

// Create device
guard let device = MTLCreateSystemDefaultDevice() else{
    fatalError("Failed to reference GPU")
}

// Make sure the current device supports MetalPerformanceShaders
guard MPSSupportsMTLDevice(device) else{
    fatalError("Metal Performance Shaders not supported for current device")
}

/*
 The command queue (MTLCommandQueue) is the object that queues and submits commands to the
 device for execution.
 */
guard let commandQueue = device.makeCommandQueue() else{
    fatalError("Failed to create CommandQueue")
}

/*:
 ### Training in batches
 One of the hyperparameters you'll adjust when training a neural network is the batch size i.e.
 how much data you expose your network to a during each *step*. It, the batch size, offers practical and
 tuning capabilities. Sometimes it's not feasible to fit all your data into memory therefore its necessary
 to work in smaller batches, the other is that your network's loss function may be susceptible by the
 size due to adjusting to the mean, working in smaller batches provides some way of fine tuning that would
 be otherwise missed when working on the larger sample.
 
 Our dataloader will return a predefined batch and continue returning a batch until no data is available;
 afterwards we would reset it and start from the begining - below is an extract demonstrating this.
 */

// let dataLoader = DataLoader(device: device, sourcePathURL: trainPath)
//var batch = dataLoader.getNextBatch()
//
//while batch != nil && DataLoader.getBatchCount(batch: batch) > 0{
//    print(DataLoader.getBatchCount(batch: batch))
//    batch = dataLoader.getNextBatch()
//}
//print("finished")

/*:
 ### Training
 Training is a iterative process of having our network make predictions and then adjusting the node weights
 based on the loss (*typically the mean squared error between the **predicted value** and **actual value***).
 */

// create our data loader
let dataLoader = DataLoader(device: device, sourcePathURL: validPath)

let network = SketchCNN(
    withCommandQueue: commandQueue,
    inputShape: Shape(width:dataLoader.imageWidth,
                      height:dataLoader.imageHeight,
                      channels:dataLoader.featureChannels),
    numberOfClasses: dataLoader.numberOfClasses,
    mode:SketchCNN.NetworkMode.inference)

if let sample = dataLoader.getNextBatch(){
    network.predict(x: sample.images[0]) { (probs) in
        print("Actual value \(sample.labels[0].label ?? "")")
        if let probs = probs{
            print("Probabilities \(probs)")
        }
        print("Finished")
    }
}


//: [Next](@next)
