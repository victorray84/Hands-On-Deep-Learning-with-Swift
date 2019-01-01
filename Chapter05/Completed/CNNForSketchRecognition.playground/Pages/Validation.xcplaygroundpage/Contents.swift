/*:
 # [Hands-On Deep Learning with Swift]()
 ### Chapter 5 - Applying CNNs to recognise sketches (COMPLETE) 
 *Writen by [Joshua Newnham](https://www.linkedin.com/in/joshuanewnham) and published by [Packt Publishing](https://www.packtpub.com/big-data-and-business-intelligence/machine-learning-core-ml)*
 
 **Playground Pages**
 - [Training](Training) page to train our model
 - [Inference](Inference) page to use our model to perform inference on our own sketches
 */

/*:
 ## Validation
 In this page we use assess our model using the validation set; initially performed will give us a
 base score that we can use for reference to determine if our model is learning or not. After training
 we use this dataset to assess how well the model performs on data it **hasn't** seen during training.
 
 Typically (in production) the dataset is split into three groups;
 1. Training; data your model is exposed to
 2. Validation; data you assess how well your model is learning. This in turn is used to tweak the architecture and hyperparameters
 3. Test; data which is used to verify how well you model performs on unseen data (which has not been used to influence the model or hyperparameters)
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

let BASE_WEIGHTS_PATH = "sketch_cnn_weights"
let weightsPath = PlaygroundSupport
    .playgroundSharedDataDirectory
    .appendingPathComponent("\(BASE_WEIGHTS_PATH)")

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
 ### Validation
 We evaluate (and assess) our model on a dataset which was not used for training, the validation set.
 Typically you would also reserve another hold-out dataset which is omitted during training and used
 to evaluate how well your model generalises, this dataset is known as the test set - but not used here.
 */

// Create our data loader, passing in -1 to signal we want to use all samples for a single batch
let dataLoader = DataLoader(device: device, sourcePathURL: validPath, batchSize: -1)

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
    weightsPathURL:weightsPath,
    mode:SketchCNN.NetworkMode.inference)

// Perform inference for all the samples in our data loader, we'll keep track of the number of samples
// and how many times we predicted correctly to calculate the accuracy
var sampleCount : Float = 0.0
var predictionsCorrectCount : Float = 0.0

// Force our dataloader to the start
dataLoader.reset()

while dataLoader.hasNext(){
    autoreleasepool{
        guard let commandBuffer = commandQueue.makeCommandBuffer() else{
            fatalError()
        }
        
        if let batch = dataLoader.nextBatch(commandBuffer: commandBuffer){
            if let predictions = network.predict(X: batch.images){
                assert(predictions.count == batch.labels.count)
                
                for i in 0..<predictions.count{
                    sampleCount += 1.0
                    let predictedClass = dataLoader.labels[predictions[i].argmax]
                    let actualClass = batch.labels[i].label ?? ""
                    
                    predictionsCorrectCount += predictedClass == actualClass ? 1.0 : 0.0
                }
            }
        }
    }
}

let accuracy = predictionsCorrectCount/sampleCount

print("Validation accuracy \(accuracy)")

/*:
 [Goto the **Training** page](Training)
 
 [Goto the **Inference** page](Inference)
 */
