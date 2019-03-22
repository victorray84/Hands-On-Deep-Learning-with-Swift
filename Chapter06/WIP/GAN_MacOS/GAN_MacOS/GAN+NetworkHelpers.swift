//
//  GAN+NetworkHelpers.swift
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
    
    func makeOptimizer(learningRate:Float, momentumScale:Float) -> MPSNNOptimizerAdam?{
        //    private func makeOptimizer(learningRate:Float, momentumScale:Float) -> MPSNNOptimizerStochasticGradientDescent?{
        
        let optimizerDescriptor = MPSNNOptimizerDescriptor(
            learningRate: learningRate,
            gradientRescale: 1.0,
            regularizationType: .None,
            regularizationScale: 1.0)
        
        let optimizer = MPSNNOptimizerAdam(
            device: self.device,
            beta1: Double(momentumScale),
            beta2: 0.999,
            epsilon: 1e-8,
            timeStep: 0,
            optimizerDescriptor: optimizerDescriptor)
        
        //        let optimizer = MPSNNOptimizerStochasticGradientDescent(
        //            device: self.device,
        //            momentumScale: momentumScale,
        //            useNestrovMomentum: true,
        //            optimizerDescriptor: optimizerDescriptor)
        
        return optimizer
    }
    
    private func makeMPSVector(count:Int, repeating:Float=0.0) -> MPSVector?{
        // Create a Metal buffer
        guard let buffer = self.device.makeBuffer(
            bytes: Array<Float32>(repeating: repeating, count: count),
            length: count * MemoryLayout<Float32>.size,
            options: [.storageModeShared]) else{
                return nil
        }
        
        // Create a vector descriptor
        let desc = MPSVectorDescriptor(
            length: count, dataType: MPSDataType.float32)
        
        // Create a vector with descriptor
        let vector = MPSVector(
            buffer: buffer, descriptor: desc)
        
        assert(vector.dataType == MPSDataType.float32)
        
        return vector
    }
    
    func createConvLayer(
        name:String,
        x:MPSNNImageNode,
        mode:NetworkMode,
        kernelSize:KernelSize = KernelSize(width:5, height:5),
        strideSize:KernelSize = KernelSize(width:2, height:2),
        inputFeatureChannels:Int,
        outputFeatureChannels:Int,
        datasources:inout [ConvnetDataSource],
        activationFunc:((MPSNNImageNode, String) -> MPSCNNNeuronNode)? = nil) -> [MPSNNFilterNode]{
        
        var layers = [MPSNNFilterNode]()
        
        // Create an optimizer iff we are training
//        var optimizer : MPSNNOptimizerStochasticGradientDescent? = nil
        var optimizer : MPSNNOptimizerAdam? = nil
        
        if mode == NetworkMode.training{
            optimizer = self.makeOptimizer(
                learningRate: self.learningRate,
                momentumScale: self.momentumScale)
        }
        
        let datasource = ConvnetDataSource(
            name: name,
            weightsPathURL:self.weightsPathURL,
            kernelSize: kernelSize,
            strideSize: strideSize,
            inputFeatureChannels: inputFeatureChannels,
            outputFeatureChannels: outputFeatureChannels,
            optimizer: optimizer)
        
        if mode == .training{
            datasource.weightsAndBiasesState = MPSCNNConvolutionWeightsAndBiasesState(
                device: self.device,
                cnnConvolutionDescriptor: datasource.descriptor())
            
            if let weightsMomentum = self.makeMPSVector(count: datasource.weightsLength),
                let biasMomentum = self.makeMPSVector(count: datasource.biasTermsLength){
                
                datasource.momentumVectors = [weightsMomentum, biasMomentum]
            }
            
            if let weightsVelocity = self.makeMPSVector(count: datasource.weightsLength),
                let biasVelocity = self.makeMPSVector(count: datasource.biasTermsLength){
                
                datasource.velocityVectors = [weightsVelocity, biasVelocity]
            }
        }
        
        datasources.append(datasource)
        
        let conv = MPSCNNConvolutionNode(source: x, weights: datasource)
        conv.resultImage.format = .float32
        conv.paddingPolicy = MPSNNDefaultPadding(method: MPSNNPaddingMethod.sizeSame)
        conv.label = "\(name)_conv"
        
        layers.append(conv)
        
        if let activationFunc = activationFunc{
            let activationNode = activationFunc(layers.last!.resultImage, "\(name)_activation")
            layers.append(activationNode)
        }
        
        return layers
    }
    
    func createTransposeConvLayer(
        name:String,
        x:MPSNNImageNode,
        mode:NetworkMode,
        kernelSize:KernelSize = KernelSize(width:5, height:5),
        strideSize:KernelSize = KernelSize(width:1, height:1),
        inputFeatureChannels:Int,
        outputFeatureChannels:Int,
        datasources:inout [ConvnetDataSource],
        upscale:Int=2,
        activationFunc:((MPSNNImageNode, String) -> MPSCNNNeuronNode)? = nil) -> [MPSNNFilterNode]{
        
        var layers = [MPSNNFilterNode]()
        
        // upscale image
        let upscale = MPSCNNUpsamplingNearestNode(
            source: x,
            integerScaleFactorX: upscale,
            integerScaleFactorY: upscale)
        upscale.label = "\(name)_upscale"
        
        layers.append(upscale)
        
        // Create an optimizer iff we are training;
//        var optimizer : MPSNNOptimizerStochasticGradientDescent? = nil
        var optimizer : MPSNNOptimizerAdam? = nil
        
        if mode == NetworkMode.training{
            optimizer = self.makeOptimizer(
                learningRate:self.learningRate,
                momentumScale: self.momentumScale)
        }
        
        let datasource = ConvnetDataSource(
            name: name,
            weightsPathURL:self.weightsPathURL,
            kernelSize: kernelSize,
            strideSize: strideSize,
            inputFeatureChannels: inputFeatureChannels,
            outputFeatureChannels: outputFeatureChannels,
            optimizer: optimizer)
        
        if mode == .training{
            datasource.weightsAndBiasesState = MPSCNNConvolutionWeightsAndBiasesState(
                device: self.device,
                cnnConvolutionDescriptor: datasource.descriptor())
            
            if let weightsMomentum = self.makeMPSVector(count: datasource.weightsLength),
                let biasMomentum = self.makeMPSVector(count: datasource.biasTermsLength){
                
                datasource.momentumVectors = [weightsMomentum, biasMomentum]
            }
            
            if let weightsVelocity = self.makeMPSVector(count: datasource.weightsLength),
                let biasVelocity = self.makeMPSVector(count: datasource.biasTermsLength){
                
                datasource.velocityVectors = [weightsVelocity, biasVelocity]
            }
        }
        
        datasources.append(datasource)
        
        let conv = MPSCNNConvolutionNode(source: upscale.resultImage, weights: datasource)
        conv.resultImage.format = .float32
        conv.paddingPolicy = MPSNNDefaultPadding(method: MPSNNPaddingMethod.sizeSame)
        conv.label = "\(name)_conv"
        
        layers.append(conv)
        
        if let activationFunc = activationFunc{
            let activationNode = activationFunc(layers.last!.resultImage, "\(name)_activation")
            layers.append(activationNode)
        }
        
        return layers
    }
    
    func createDenseLayer(
        name:String,
        input:MPSNNImageNode,
        mode:NetworkMode,
        kernelSize:KernelSize,
        inputFeatureChannels:Int,
        outputFeatureChannels:Int,
        datasources:inout [ConvnetDataSource],
        activationFunc:((MPSNNImageNode, String) -> MPSCNNNeuronNode)? = nil) -> [MPSNNFilterNode]{
        
        var layers = [MPSNNFilterNode]()
        
        var optimizer : MPSNNOptimizerAdam? = nil
        
        if mode == NetworkMode.training{
            optimizer = self.makeOptimizer(
                learningRate: self.learningRate,
                momentumScale: self.momentumScale)
        }
        
        let datasource = ConvnetDataSource(
            name: name,
            weightsPathURL: self.weightsPathURL,
            kernelSize: kernelSize,
            inputFeatureChannels: inputFeatureChannels,
            outputFeatureChannels: outputFeatureChannels,
            optimizer:optimizer)
        
        if mode == .training{
            datasource.weightsAndBiasesState = MPSCNNConvolutionWeightsAndBiasesState(
                device: self.device,
                cnnConvolutionDescriptor: datasource.descriptor())
            
            if let weightsMomentum = self.makeMPSVector(count: datasource.weightsLength),
                let biasMomentum = self.makeMPSVector(count: datasource.biasTermsLength){
                
                datasource.momentumVectors = [weightsMomentum, biasMomentum]
            }
            
            if let weightsVelocity = self.makeMPSVector(count: datasource.weightsLength),
                let biasVelocity = self.makeMPSVector(count: datasource.biasTermsLength){
                
                datasource.velocityVectors = [weightsVelocity, biasVelocity]
            }
        }
        
        datasources.append(datasource)
        
        let fc = MPSCNNFullyConnectedNode(
            source: input,
            weights: datasource)
        
        fc.resultImage.format = .float32
        fc.paddingPolicy = MPSNNDefaultPadding(method: MPSNNPaddingMethod.validOnly)
        fc.label = "\(name)_fc"
        
        layers.append(fc)
        
        if let activationFunc = activationFunc{
            let activationNode = activationFunc(layers.last!.resultImage, "\(name)_activation")
            layers.append(activationNode)
        }
        
        return layers
    }
}
