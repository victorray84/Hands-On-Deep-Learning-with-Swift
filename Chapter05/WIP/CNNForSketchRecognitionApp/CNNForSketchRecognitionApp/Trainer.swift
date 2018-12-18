//
//  Trainer.swift
//  CNNForSketchRecognitionApp
//
//  Created by joshua.newnham on 12/11/2018.
//  Copyright © 2018 Joshua Newnham. All rights reserved.
//

import Foundation
import AppKit
import AVFoundation
import CoreGraphics
import MetalKit
import MetalPerformanceShaders
import Accelerate
import GLKit

class Trainer{
    
    public static let targetShape = Shape(
        width:128,
        height:128,
        channels:1)
    
    public static func validate(){
        let VALID_PATH = "/Users/joshua.newnham/Documents/Shared Playground Data/Sketches/preprocessed/valid/"
        
        //let TRAIN_PATH = "/Users/joshua.newnham/Documents/Shared Playground Data/Sketches/org/"
        
        let WEIGHTS_PATH = "/Users/joshua.newnham/Documents/Shared Playground Data/sketch_cnn_weights/"
        
        let validPath = URL(fileURLWithPath: VALID_PATH)
        let weightsPath = URL(fileURLWithPath: WEIGHTS_PATH)
        
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
        
        // Create our data loader
        let dataLoader = DataLoader(device: device, sourcePathURL: validPath, batchSize:1)
        
        // We pass in the target shape which will be used to scale the inputs accordingly
        //        let targetShape = Shape(
        //            width:dataLoader.imageWidth/2,
        //            height:dataLoader.imageHeight/2,
        //            channels:dataLoader.featureChannels)
        
        // Create our training network
        let network = SketchCNN(
            withCommandQueue: commandQueue,
            inputShape: Trainer.targetShape,
            numberOfClasses: dataLoader.numberOfClasses,
            weightsPathURL:weightsPath,
            mode:SketchCNN.NetworkMode.inference)
        
        var correctCount = 0.0
        var count = 0.0
        
        for _ in 0..<10{
            autoreleasepool{
                guard let commandBuffer = commandQueue.makeCommandBuffer() else{
                    fatalError()
                }
                
                if let batch = dataLoader.nextBatch(commandBuffer: commandBuffer){
                    for i in 0..<batch.images.count{
                        let img = batch.images[i]
                        let label = batch.labels[i]
                        let actualClass = label.label ?? ""
                        
                        network.predict(x: img) { (probs) in
                            
                            if let probs = probs{
                                //                            print("Probabilities \(probs)")
                                let predictedClass = dataLoader.labels[probs.argmax]
                                
                                count += 1.0
                                correctCount += predictedClass == actualClass ? 1.0 : 0.0
                                
                                print("\tPrediction \(predictedClass); Actual \(actualClass); Accuracy \(correctCount/count)")
                                
                                count += 1.0
                                correctCount += predictedClass == actualClass ? 1.0 : 0.0
                            }
                        }
                    }
                }
            }
        }
    }
    
    public static func train(){
        let TRAIN_PATH = "/Users/joshua.newnham/Documents/Shared Playground Data/Sketches/preprocessed/train/"
        
        let WEIGHTS_PATH = "/Users/joshua.newnham/Documents/Shared Playground Data/sketch_cnn_weights/"
        
        let trainPath = URL(fileURLWithPath: TRAIN_PATH)
        let weightsPath = URL(fileURLWithPath: WEIGHTS_PATH)        
        
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
        
        // Create our data loader
        let dataLoader = DataLoader(device: device, sourcePathURL: trainPath)
        
        // We pass in the target shape which will be used to scale the inputs accordingly
//        let targetShape = Shape(
//            width:dataLoader.imageWidth/2,
//            height:dataLoader.imageHeight/2,
//            channels:dataLoader.featureChannels)
        
        // Create our training network
        let network = SketchCNN(
            withCommandQueue: commandQueue,
            inputShape: Trainer.targetShape,
            numberOfClasses: dataLoader.numberOfClasses,
            weightsPathURL:weightsPath, 
            mode:SketchCNN.NetworkMode.training)
        
        // Train
        print("Training will begin")
        
        network.train(withDataLoader: dataLoader) {
            print("Training did finish")
        }
    }
}
