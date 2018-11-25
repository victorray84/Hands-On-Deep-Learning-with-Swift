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

guard let device = MTLCreateSystemDefaultDevice(),
    let commandQueue = device.makeCommandQueue() else{
    fatalError("Failed")
}

// Create our data loader
let dataLoader = DataLoader(commandQueue: commandQueue)

let trainingNetwork = Network2(
    withCommandQueue: commandQueue,
    inputShape: Shape(width:64, height:64, channels:1),
    numberOfClasses: dataLoader.numberOfClasses)

trainingNetwork.train(withDataLoader: dataLoader) {
    print("Finished")
}

//let inferenceNetwork = Network2(
//    withCommandQueue: commandQueue,
//    inputShape: Shape(width:64, height:64, channels:1),
//    numberOfClasses: dataLoader.numberOfClasses,
//    mode: Network2.NetworkMode.inference)
//
//inferenceNetwork.predict(x: dataLoader.getBatch(shuffle:false).images[4]) { (probs) in
//    print("Finished running inference on \(dataLoader.getBatch(shuffle:false).labels[4].label!)")
//    print(probs ?? "Undefined")
//}


