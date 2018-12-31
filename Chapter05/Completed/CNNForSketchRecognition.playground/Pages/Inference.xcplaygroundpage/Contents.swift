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
let weightsPath = PlaygroundSupport
    .playgroundSharedDataDirectory
    .appendingPathComponent("\(BASE_WEIGHTS_PATH)")

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
 ### Inference
 Once your model has satisfacorily been trained; it is now ready for infernce i.e. performing
 predictions on data in the wild. Here we'll perform classification on the input by the user,
 presenting the top labels that were predicted.
 */

// labels we'll use to make sense of the models output
let labels = [
    "airplane",
    "apple",
    "bear",
    "bee",
    "bicycle",
    "bufferfly",
    "car",
    "cat",
    "chair",
    "cup",
    "dog",
    "elephantt",
    "head",
    "helicoper",
    "monkey",
    "owl",
    "panda",
    "penguin",
    "pigeon",
    "rabbit",
    "sailboat",
    "teddy-bear"
]

// We pass in the target shape which will be used to scale the inputs accordingly
let targetShape = Shape(
    width:128,
    height:128,
    channels:1)

// Create our training network
let network = SketchCNN(
    withCommandQueue: commandQueue,
    inputShape: targetShape,
    numberOfClasses: 22,
    weightsPathURL:weightsPath,
    mode:SketchCNN.NetworkMode.inference)

// Create MPSImage
let placeholderImageDescriptor = MPSImageDescriptor(
    channelFormat: MPSImageFeatureChannelFormat.unorm8,
    width: targetShape.width,
    height: targetShape.height,
    featureChannels: targetShape.channels)

let placeholderImage = MPSImage(
    device: device,
    imageDescriptor: placeholderImageDescriptor)

PlaygroundPage.current.liveView = SketchCanvasView(
    frame: NSRect(x: 0,
                  y: 0,
                  width: 500,
                  height: 500),
    submitHandler:{(view, context) in
        guard let context = context else{
            return
        }
        
        let origin = MTLOrigin(
            x: 0, y: 0, z: 0)
        
        let size = MTLSize(
            width: targetShape.width,
            height: targetShape.height,
            depth: targetShape.channels)
        
        let region = MTLRegion(
            origin: origin,
            size: size)
        
        let bytesPerRow = targetShape.width * targetShape.channels
        
        placeholderImage.texture.replace(
            region: region,
            mipmapLevel: 0,
            withBytes: context.data!,
            bytesPerRow: bytesPerRow)
        
        let img = placeholderImage
        
        if let probabilities = network.predict(X: [placeholderImage]){
            // Expecting only a single imasge being passed to our network for inference
            assert(probabilities.count == 1)
            // Get an array of sorted indicies by probability, from high to low.
            let sortedIndies = probabilities[0].indices.sorted{
                probabilities[0][$0] > probabilities[0][$1]
            }
            
            let prob1 = String(format: "%.2f", probabilities[0][sortedIndies[0]])
            let label1 = labels[sortedIndies[0]]
            
            let prob2 = String(format: "%.2f", probabilities[0][sortedIndies[1]])
            let label2 = labels[sortedIndies[1]]
            
            
            let text = "I guess you are drawing either a \(label1) (\(prob1)) or \(label2) (\(prob2))"
            
            view.text = text
        }
})

//: [Next](@next)
