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

let BASE_WEIGHTS_PATH = "sketch_cnn_weights"

let BASE_VALID_PATH = "Sketches/preprocessed/valid"

let weightsPath = PlaygroundSupport
    .playgroundSharedDataDirectory
    .appendingPathComponent("\(BASE_WEIGHTS_PATH)")

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

/*:
 ### Training
 Training is a iterative process of having our network make predictions and then adjusting the node weights
 based on the loss (*typically the mean squared error between the **predicted value** and **actual value***).
 */

// Create our data loader
let dataLoader = DataLoader(device: device, sourcePathURL: validPath)

// We pass in the target shape which will be used to scale the inputs accordingly
let targetShape = Shape(
    width:dataLoader.imageWidth,
    height:dataLoader.imageHeight,
    channels:dataLoader.featureChannels)

// Create our training network
let network = SketchCNN(
    withCommandQueue: commandQueue,
    inputShape: targetShape,
    numberOfClasses: dataLoader.numberOfClasses,
    weightsPathURL: weightsPath,
    mode: .inference)

//autoreleasepool{
    guard let commandBuffer = commandQueue.makeCommandBuffer() else{
        fatalError()
    }
    
    if let batch = dataLoader.nextBatch(commandBuffer: commandBuffer){
        let img = batch.images[30]
        
//        network.predict(x: img) { (probs) in
//
//            print("Actual value \(batch.labels[30].label ?? "")")
//            if let probs = probs{
//                print("Probabilities \(probs)")
//                print("Predicted label \(dataLoader.labels[probs.argmax])")
//                print(dataLoader.labels)
//            }
//            print("Finished")
//        }
        
        network.predict(x: img) { (outputImage) in
            
            // convert output texture to image
            
            let img = outputImage 
            let tex = outputImage?.texture
            print("w \(outputImage?.width) h \(outputImage?.height) c \(outputImage?.featureChannels)")
            
            if let array = img?.toFloatArray(){
                var count = 0
                for i in 0..<array.count{
                    if array[i] > 0{
                        print("\(i) \(array[i])")
                        count += 1
                    }
                }
                
                print("\(Float(count)/Float(array.count))")
            }
        }
    }
//}


//: [Next](@next)
