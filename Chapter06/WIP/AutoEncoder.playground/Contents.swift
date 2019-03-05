import Foundation
import AppKit
import MetalKit
import MetalPerformanceShaders
import Accelerate
import AVFoundation
import PlaygroundSupport
import CoreGraphics

let weightsPath = PlaygroundSupport
    .playgroundSharedDataDirectory
    .appendingPathComponent("\(AutoEncoderDataSource.FolderName)")

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

guard let dataLoader = DataLoader(
    device: device,
    imagesFile: "/data/x_test.data",
    labelsFile: "/data/y_test.data") else{
        fatalError("Failed to create an instance of a DataLoader")
}

//let batch = dataLoader.nextBatch()

let network = AutoEncoder(
    withCommandQueue: commandQueue,
    inputShape: Shape(width:28, height:28, channels:1),
    numberOfClasses: 10,
    weightsPathURL: weightsPath,
    mode: AutoEncoder.NetworkMode.training)

network.train(withDataLoader: dataLoader) {
    print("Finished training")
}
