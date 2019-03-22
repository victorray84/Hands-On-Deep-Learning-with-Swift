//
//  GAN+Activations.swift
//  GAN_MacOS
//
//  Created by joshua.newnham on 22/03/2019.
//  Copyright © 2019 Joshua Newnham. All rights reserved.
//

import Foundation
import AppKit
import MetalKit
import MetalPerformanceShaders
import CoreGraphics

extension GAN{
    
    static func createRelu(_ x:MPSNNImageNode, _ name:String) -> MPSCNNNeuronNode{
        let activation = MPSCNNNeuronReLUNode(source: x)
        activation.resultImage.format = .float32
        activation.label = name
        return activation
    }
    
    /*:
     f(x) = alpha * x for x < 0, f(x) = x for x >= 0
     */
    static func createLeakyRelu(_ x:MPSNNImageNode, _ name:String) -> MPSCNNNeuronNode{
        let activation = MPSCNNNeuronReLUNode(source: x, a:0.2)
        activation.resultImage.format = .float32
        activation.label = name
        return activation
    }
    
    static func createSigmoid(_ x:MPSNNImageNode, _ name:String) -> MPSCNNNeuronNode{
        let activation = MPSCNNNeuronSigmoidNode(source:x)
        //let activation = MPSCNNNeuronHardSigmoidNode(source: x, a:1.0, b:Float.leastNonzeroMagnitude)
        activation.resultImage.format = .float32
        activation.label = name
        return activation
    }
    
    static func createTanH(_ x:MPSNNImageNode, _ name:String) -> MPSCNNNeuronNode{
        let activation = MPSCNNNeuronTanHNode(source: x)
        activation.resultImage.format = .float32
        activation.label = name
        return activation
    }
    
}
