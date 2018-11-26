/*:
 
 # [Hands-On Deep Learning with Swift]()
 ### Chapter 4 - Metal for Machine Learning
 *Writen by [Joshua Newnham](https://www.linkedin.com/in/joshuanewnham) and published by [Packt Publishing](https://www.packtpub.com/big-data-and-business-intelligence/machine-learning-core-ml)*
 
 **Playground Pages**
 - [Validation](Validation) page to test our model on our validation dataset
 - [Inference](Inference) page to use our model to perform inference on our own sketches
 */
import Foundation
import AppKit
import AVFoundation
import CoreGraphics
import MetalKit
import MetalPerformanceShaders
import Accelerate
import GLKit
import PlaygroundSupport


let BASE_TRAIN_PATH = "Sketches/preprocessed/train"

let trainPath = PlaygroundSupport
    .playgroundSharedDataDirectory
    .appendingPathComponent(BASE_TRAIN_PATH)

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


//let dataLoader = DataLoader(device: device, sourcePathURL: trainPath)
//var batch = dataLoader.getNextBatch()
//
//let network = SketchCNN(
//    withCommandQueue: commandQueue,
//    inputShape: Shape(width:dataLoader.imageWidth,
//                      height:dataLoader.imageHeight,
//                      channels:dataLoader.featureChannels),
//    numberOfClasses: dataLoader.numberOfClasses,
//    mode:SketchCNN.NetworkMode.inference)
//
//if batch != nil{
//    let x = batch!.images[0]
//    let y = batch!.labels[0]
//    network.predict(x: x) { (probs) in
//        print("predictions for \(y.label)")
//        print(probs)
//        print("finished")
//    }
//}

/*:
 ### Training
 Training is a iterative process of having our network make predictions and then adjusting the node weights
 based on the loss (*typically the mean squared error between the **predicted value** and **actual value***).
 */

// Create our data loader
let dataLoader = DataLoader(device: device, sourcePathURL: trainPath)

// We pass in the target shape which will be used to scale the inputs accordingly
let targetShape = Shape(
    width:dataLoader.imageWidth/2,
    height:dataLoader.imageHeight/2,
    channels:dataLoader.featureChannels)

// Create our training network
let network = SketchCNN(
    withCommandQueue: commandQueue,
    inputShape: targetShape,
    numberOfClasses: dataLoader.numberOfClasses,
    mode:SketchCNN.NetworkMode.training)

// Train
print("Training will begin")

network.train(withDataLoader: dataLoader) {
    print("Training did finish")
}

//let conv = MPSCNNConvolutionNode(source: x, weights: datasource)
//conv.paddingPolicy = MPSNNDefaultPadding(method: MPSNNPaddingMethod.sizeSame | MPSNNPaddingMethod.excludeEdges)

//: [Goto next Page](@next)