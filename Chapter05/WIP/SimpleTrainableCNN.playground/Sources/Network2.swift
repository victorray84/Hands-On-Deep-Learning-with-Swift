import Foundation
import AppKit
import AVFoundation
import CoreGraphics
import MetalKit
import MetalPerformanceShaders
import Accelerate
import PlaygroundSupport

public class Network2{
    public enum NetworkMode{
        case training
        case inference
    }
    
    let device : MTLDevice
    let commandQueue : MTLCommandQueue
    
    let mode : NetworkMode
    
    let inputShape : Shape
    let numberOfClasses : Int
    
    var optimizer : MPSNNOptimizerStochasticGradientDescent?
    var graph : MPSNNGraph?
    
    var datasources = [DataSource]()
    
    public init(withCommandQueue commandQueue:MTLCommandQueue,
                inputShape:Shape,
                numberOfClasses:Int,
                mode:NetworkMode=NetworkMode.training){
        
        self.device = commandQueue.device
        self.inputShape = inputShape
        self.numberOfClasses = numberOfClasses
        self.mode = mode
        self.commandQueue = commandQueue
        
        if mode == .training{
            self.optimizer = MPSNNOptimizerStochasticGradientDescent(
                device: self.device,
                learningRate: 0.001)
            
            self.graph = self.createTrainingGraph()
        } else{
            self.graph = self.createInferenceGraph()
        }
        
        print(graph!.debugDescription)
    }
    
    private func createInferenceGraph() -> MPSNNGraph?{
        let input = MPSNNImageNode(handle: nil)
        
        // Scale
        let scale = MPSNNLanczosScaleNode(
            source: input,
            outputSize: MTLSize(
                width: 64,
                height: 64,
                depth: 1))
        
        // OUTPUT = 128x128x1
        
        // -------------------------- //
        
        let conv1Datasource = DataSource(
            name: "conv1",
            kernelSize: KernelSize(width:3, height:3),
            strideSize: StrideSize(width:1, height:1),
            inputFeatureChannels: 1,
            outputFeatureChannels: 32,
            optimizer: self.optimizer)
        
        conv1Datasource.weightsAndBiasesState = MPSCNNConvolutionWeightsAndBiasesState(
            device: self.device,
            cnnConvolutionDescriptor: conv1Datasource.descriptor())
        
        self.datasources.append(conv1Datasource)
        
        let conv1 = MPSCNNConvolutionNode(source: scale.resultImage, weights: conv1Datasource)
        conv1.paddingPolicy = MPSNNDefaultPadding(method: MPSNNPaddingMethod.sizeSame)
        
        let conv1Activation = MPSCNNNeuronReLUNode(source: conv1.resultImage)
        
        let maxPool1 = MPSCNNPoolingMaxNode(source: conv1Activation.resultImage, filterSize: 2, stride:2)
        maxPool1.paddingPolicy = MPSNNDefaultPadding(method: MPSNNPaddingMethod.sizeSame)
        
        // OUTPUT = 64x64x32
        
        // -------------------------- //
        
        let conv2Datasource = DataSource(
            name: "conv2",
            kernelSize: KernelSize(width:3, height:3),
            strideSize: StrideSize(width:1, height:1),
            inputFeatureChannels: 32,
            outputFeatureChannels: 64,
            optimizer: self.optimizer)
        
        conv2Datasource.weightsAndBiasesState = MPSCNNConvolutionWeightsAndBiasesState(
            device: self.device,
            cnnConvolutionDescriptor: conv2Datasource.descriptor())
        
        self.datasources.append(conv2Datasource)
        
        let conv2 = MPSCNNConvolutionNode(source: maxPool1.resultImage, weights: conv2Datasource)
        conv2.paddingPolicy = MPSNNDefaultPadding(method: MPSNNPaddingMethod.sizeSame)
        
        let conv2Activation = MPSCNNNeuronReLUNode(source: conv2.resultImage)
        
        let maxPool2 = MPSCNNPoolingMaxNode(source: conv2Activation.resultImage, filterSize: 2, stride:2)
        maxPool2.paddingPolicy = MPSNNDefaultPadding(method: MPSNNPaddingMethod.sizeSame)
        
        // OUTPUT = 32x32x64
        
        // -------------------------- //
        
        let conv3Datasource = DataSource(
            name: "conv3",
            kernelSize: KernelSize(width:3, height:3),
            strideSize: StrideSize(width:1, height:1),
            inputFeatureChannels: 64,
            outputFeatureChannels: 128,
            optimizer: self.optimizer)
        
        conv3Datasource.weightsAndBiasesState = MPSCNNConvolutionWeightsAndBiasesState(
            device: self.device,
            cnnConvolutionDescriptor: conv3Datasource.descriptor())
        
        self.datasources.append(conv3Datasource)
        
        let conv3 = MPSCNNConvolutionNode(source: maxPool2.resultImage, weights: conv3Datasource)
        conv3.paddingPolicy = MPSNNDefaultPadding(method: MPSNNPaddingMethod.sizeSame)
        
        let conv3Activation = MPSCNNNeuronReLUNode(source: conv3.resultImage)
        
        let maxPool3 = MPSCNNPoolingMaxNode(source: conv3Activation.resultImage, filterSize: 2, stride:2)
        maxPool3.paddingPolicy = MPSNNDefaultPadding(method: MPSNNPaddingMethod.sizeSame)
        
        // OUTPUT = 16x16x128
        
        // -------------------------- //
        
        let fc1Datasource = DataSource(
            name: "fc1",
            kernelSize: KernelSize(width:16, height:16),
            strideSize: StrideSize(width:1, height:1),
            inputFeatureChannels: 128,
            outputFeatureChannels: 256,
            optimizer: self.optimizer)
        
        fc1Datasource.weightsAndBiasesState = MPSCNNConvolutionWeightsAndBiasesState(
            device: self.device,
            cnnConvolutionDescriptor: fc1Datasource.descriptor())
        
        self.datasources.append(fc1Datasource)
        
        let fc1 = MPSCNNFullyConnectedNode(source: maxPool3.resultImage, weights: fc1Datasource)
        
        let fc1Activation = MPSCNNNeuronReLUNode(source: fc1.resultImage)
        
        // OUTPUT = 256
        
        // -------------------------- //
        
        let fc2Datasource = DataSource(
            name: "fc2",
            kernelSize: KernelSize(width:1, height:1),
            strideSize: StrideSize(width:1, height:1),
            inputFeatureChannels: 256,
            outputFeatureChannels: self.numberOfClasses,
            optimizer: self.optimizer)
        
        fc2Datasource.weightsAndBiasesState = MPSCNNConvolutionWeightsAndBiasesState(
            device: self.device,
            cnnConvolutionDescriptor: fc2Datasource.descriptor())
        
        self.datasources.append(fc2Datasource)
        
        let fc2 = MPSCNNFullyConnectedNode(source: fc1Activation.resultImage, weights: fc2Datasource)
        
        // OUTPUT = 5
        
        // softmax
        let softmax = MPSCNNSoftMaxNode(source: fc2.resultImage)
        
        guard let graph = MPSNNGraph(
            device: self.device,
            resultImage: softmax.resultImage,
            resultImageIsNeeded: true) else{
                return nil
        }
        
        return graph
    }
    
    private func createTrainingGraph() -> MPSNNGraph?{
        let input = MPSNNImageNode(handle: nil)

        // Scale
        let scale = MPSNNLanczosScaleNode(
            source: input,
            outputSize: MTLSize(
                width: 32,
                height: 32,
                depth: 1))

        // OUTPUT = 32x32x1

        // -------------------------- //

        let conv1Datasource = DataSource(
            name: "conv1",
            kernelSize: KernelSize(width:3, height:3),
            strideSize: StrideSize(width:1, height:1),
            inputFeatureChannels: 1,
            outputFeatureChannels: 16,
            optimizer: self.optimizer)

        conv1Datasource.weightsAndBiasesState = MPSCNNConvolutionWeightsAndBiasesState(
            device: self.device,
            cnnConvolutionDescriptor: conv1Datasource.descriptor())

        self.datasources.append(conv1Datasource)

        let conv1 = MPSCNNConvolutionNode(source: scale.resultImage, weights: conv1Datasource)
        conv1.paddingPolicy = MPSNNDefaultPadding(method: MPSNNPaddingMethod.sizeSame)

        let conv1Activation = MPSCNNNeuronReLUNode(source: conv1.resultImage)

        let maxPool1 = MPSCNNPoolingMaxNode(source: conv1Activation.resultImage, filterSize: 2, stride:2)
        //maxPool1.paddingPolicy = MPSNNDefaultPadding(method: MPSNNPaddingMethod.sizeSame)

        // OUTPUT = 64x64x16

        // -------------------------- //

        let conv2Datasource = DataSource(
            name: "conv2",
            kernelSize: KernelSize(width:3, height:3),
            strideSize: StrideSize(width:1, height:1),
            inputFeatureChannels: 16,
            outputFeatureChannels: 64,
            optimizer: self.optimizer)

        conv2Datasource.weightsAndBiasesState = MPSCNNConvolutionWeightsAndBiasesState(
            device: self.device,
            cnnConvolutionDescriptor: conv2Datasource.descriptor())

        self.datasources.append(conv2Datasource)

        let conv2 = MPSCNNConvolutionNode(source: maxPool1.resultImage, weights: conv2Datasource)
        conv2.paddingPolicy = MPSNNDefaultPadding(method: MPSNNPaddingMethod.sizeSame)

        let conv2Activation = MPSCNNNeuronReLUNode(source: conv2.resultImage)

        let maxPool2 = MPSCNNPoolingMaxNode(source: conv2Activation.resultImage, filterSize: 2, stride:2)
        //maxPool2.paddingPolicy = MPSNNDefaultPadding(method: MPSNNPaddingMethod.sizeSame)

        // OUTPUT = 32x32x64

        //        // -------------------------- //
        //
        //        let conv3Datasource = DataSource(
        //            name: "conv3",
        //            kernelSize: KernelSize(width:3, height:3),
        //            strideSize: StrideSize(width:1, height:1),
        //            inputFeatureChannels: 64,
        //            outputFeatureChannels: 128,
        //            optimizer: self.optimizer)
        //
        //        conv3Datasource.weightsAndBiasesState = MPSCNNConvolutionWeightsAndBiasesState(
        //            device: self.device,
        //            cnnConvolutionDescriptor: conv3Datasource.descriptor())
        //
        //        self.datasources.append(conv3Datasource)
        //
        //        let conv3 = MPSCNNConvolutionNode(source: maxPool2.resultImage, weights: conv3Datasource)
        //        conv3.paddingPolicy = MPSNNDefaultPadding(method: MPSNNPaddingMethod.sizeSame)
        //
        //        let conv3Activation = MPSCNNNeuronReLUNode(source: conv3.resultImage)
        //
        //        let maxPool3 = MPSCNNPoolingMaxNode(source: conv3Activation.resultImage, filterSize: 2, stride:2)
        //        maxPool3.paddingPolicy = MPSNNDefaultPadding(method: MPSNNPaddingMethod.sizeSame)

        // OUTPUT = 16x16x128

        // -------------------------- //

        let fc1Datasource = DataSource(
            name: "fc1",
            kernelSize: KernelSize(width:32, height:32),
            strideSize: StrideSize(width:1, height:1),
            inputFeatureChannels: 64,
            outputFeatureChannels: 128,
            optimizer: self.optimizer)

        fc1Datasource.weightsAndBiasesState = MPSCNNConvolutionWeightsAndBiasesState(
            device: self.device,
            cnnConvolutionDescriptor: fc1Datasource.descriptor())

        self.datasources.append(fc1Datasource)

        let fc1 = MPSCNNFullyConnectedNode(source: maxPool2.resultImage, weights: fc1Datasource)

        let fc1Activation = MPSCNNNeuronReLUNode(source: fc1.resultImage)

        // OUTPUT = 256

        // -------------------------- //

        let fc2Datasource = DataSource(
            name: "fc2",
            kernelSize: KernelSize(width:1, height:1),
            strideSize: StrideSize(width:1, height:1),
            inputFeatureChannels: 128,
            outputFeatureChannels: self.numberOfClasses,
            optimizer: self.optimizer)

        fc2Datasource.weightsAndBiasesState = MPSCNNConvolutionWeightsAndBiasesState(
            device: self.device,
            cnnConvolutionDescriptor: fc2Datasource.descriptor())

        self.datasources.append(fc2Datasource)

        let fc2 = MPSCNNFullyConnectedNode(source: fc1Activation.resultImage, weights: fc2Datasource)

        // OUTPUT = 5

        // softmax
        let softmax = MPSCNNSoftMaxNode(source: fc2.resultImage)

        // === Define the loss function === //

        let lossDesc = MPSCNNLossDescriptor(
            type: MPSCNNLossType.softMaxCrossEntropy,
            reductionType: MPSCNNReductionType.mean)
        lossDesc.numberOfClasses = self.numberOfClasses

        let loss = MPSCNNLossNode(
            source: softmax.resultImage,
            lossDescriptor: lossDesc)

        // === Backwards pass ===

        let softmaxG = softmax.gradientFilter(withSource: loss.resultImage)

        let fc2Gradient = fc2.gradientFilter(withSource: softmaxG.resultImage)

        let fc1ActivationGradient = fc1Activation.gradientFilter(withSource: fc2Gradient.resultImage)

        let fc1Gradient = fc1.gradientFilter(withSource: fc1ActivationGradient.resultImage)

        let maxPool2Gradient = maxPool1.gradientFilter(withSource: fc1Gradient.resultImage)

        let covn2ActivationGradient = conv2Activation.gradientFilter(withSource:maxPool2Gradient.resultImage)

        let conv2Gradient = conv2.gradientFilter(withSource: covn2ActivationGradient.resultImage)

        let maxPool1Gradient = maxPool2.gradientFilter(withSource: conv2Gradient.resultImage)

        let covn1ActivationGradient = conv1Activation.gradientFilter(withSource:maxPool1Gradient.resultImage)

        let conv1Gradient = conv1.gradientFilter(withSource: covn1ActivationGradient.resultImage)

        //let scaleGradient = scale.gradientFilter(withSource: conv1Gradient.resultImage)

        guard let graph = MPSNNGraph(
            device: device,
            resultImage: conv1Gradient.resultImage,
            resultImageIsNeeded: false) else{
                return nil
        }

        return graph
    }
    
    public func predict(
        x:MPSImage,
        completationHandler handler: @escaping ([Float]?) -> Void) -> Void{
        
        guard let graph = self.graph else{
            return
        }
        
        graph.executeAsync(withSourceImages: [x]) { (outputImage, error) in
            DispatchQueue.main.async {
                if error != nil{
                    print(error!)
                    handler(nil)
                    return
                }
                
                if outputImage != nil{
                    guard let probs = outputImage!.toFloatArray() else{
                        handler(nil)
                        return
                    }
                    
                    handler(Array<Float>(probs))
                    return
                }
                
                handler(nil)
            }
        }
    }
    
    public func train(
        withDataLoader dataLoader:DataLoader,
        epochs : Int = 1000,
        completionHandler handler: @escaping () -> Void){
        
        autoreleasepool{
            let trainingSemaphore = DispatchSemaphore(value:1)
            
            var latestCommandBuffer : MTLCommandBuffer?
            
            for epoch in 1...epochs{
                let batch = dataLoader.getBatch()
                
                latestCommandBuffer = self.trainStep(
                    x:batch.images,
                    y:batch.labels,
                    semaphore:trainingSemaphore)
                
                if epoch % 50 == 0{
                    print("Epoch \(epoch)")
                }
            }
            
            latestCommandBuffer?.waitUntilCompleted()
            
            self.updateDatasources()
        }
        
        handler()
    }
    
    private func trainStep(
        x:[MPSImage],
        y:[MPSCNNLossLabels],
        semaphore:DispatchSemaphore) -> MTLCommandBuffer?{
        
        let _ = semaphore.wait(timeout: .distantFuture)
        
        guard let graph = self.graph,
            let commandBuffer = self.commandQueue.makeCommandBuffer() else{
                semaphore.signal()
                return nil
        }
        
        graph.encodeBatch(
            to: commandBuffer,
            sourceImages: [x],
            sourceStates: [y])
        
        commandBuffer.addCompletedHandler({ (commandBuffer) in
            semaphore.signal()
        })
        
        commandBuffer.commit()
        
        return commandBuffer
        
    }
    
    private func updateDatasources(){
        // Get command buffer to use in MetalPerformanceShaders.
        guard let commandBuffer = self.commandQueue.makeCommandBuffer() else{
            return
        }
        
        for datasource in self.datasources{
            datasource.synchronizeParameters(on: commandBuffer)
        }
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        for datasource in self.datasources{
            datasource.updateAndSaveParametersToDisk()
        }
    }
}
