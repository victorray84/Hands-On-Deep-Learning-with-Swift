import Foundation
import AppKit
import AVFoundation
import CoreGraphics
import MetalKit
import MetalPerformanceShaders
import Accelerate
import PlaygroundSupport

public class Network{
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
                width: self.inputShape.width,
                height: self.inputShape.height,
                depth: 1))
        
        // OUTPUT = 64x64x1
        
        // layer 1
        let conv1Datasource = DataSource(
            name: "conv1",
            kernelSize: KernelSize(width:3, height:3),
            strideSize: StrideSize(width:1, height:1),
            inputFeatureChannels: 1,
            outputFeatureChannels: 4,
            optimizer: nil)
        
        self.datasources.append(conv1Datasource)
        
        let conv1 = MPSCNNConvolutionNode(source: scale.resultImage, weights: conv1Datasource)
        conv1.paddingPolicy = MPSNNDefaultPadding(method: MPSNNPaddingMethod.sizeSame)
        
        let conv1Activation = MPSCNNNeuronReLUNode(source: conv1.resultImage)
        
        // OUTPUT = 64x64x4
        
        let maxPool1 = MPSCNNPoolingMaxNode(source: conv1Activation.resultImage, filterSize: 2, stride:2)
        maxPool1.paddingPolicy = MPSNNDefaultPadding(method: MPSNNPaddingMethod.sizeSame)
        
        // OUTPUT = 32x32x4
        let fc1Datasource = DataSource(
            name: "fc1",
            kernelSize: KernelSize(width:32, height:32),
            strideSize: StrideSize(width:1, height:1),
            inputFeatureChannels: 4,
            outputFeatureChannels: 128,
            optimizer: nil)
        
        self.datasources.append(fc1Datasource)
        
        let fc1 = MPSCNNFullyConnectedNode(source: maxPool1.resultImage, weights: fc1Datasource)
        
        let fc1Activation = MPSCNNNeuronReLUNode(source: fc1.resultImage)
        
        // OUTPUT = 1x1x64
        let fc2Datasource = DataSource(
            name: "fc2",
            kernelSize: KernelSize(width:1, height:1),
            strideSize: StrideSize(width:1, height:1),
            inputFeatureChannels: 128,
            outputFeatureChannels: self.numberOfClasses,
            optimizer: nil)
        
        self.datasources.append(fc2Datasource)
        
        let fc2 = MPSCNNFullyConnectedNode(source: fc1Activation.resultImage, weights: fc2Datasource)
        
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
                width: self.inputShape.width,
                height: self.inputShape.height,
                depth: 1))
        
        // OUTPUT = 64x64x1
        
        // layer 1
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
        
        // OUTPUT = 64x64x4
        
        let maxPool1 = MPSCNNPoolingMaxNode(source: conv1Activation.resultImage, filterSize: 2, stride:2)
        maxPool1.paddingPolicy = MPSNNDefaultPadding(method: MPSNNPaddingMethod.sizeSame)
        
        // OUTPUT = 32x32x4
        let fc1Datasource = DataSource(
            name: "fc1",
            kernelSize: KernelSize(width:32, height:32),
            strideSize: StrideSize(width:1, height:1),
            inputFeatureChannels: 32,
            outputFeatureChannels: 128,
            optimizer: self.optimizer)
        
        fc1Datasource.weightsAndBiasesState = MPSCNNConvolutionWeightsAndBiasesState(
            device: self.device,
            cnnConvolutionDescriptor: fc1Datasource.descriptor())
        
        self.datasources.append(fc1Datasource)
        
        let fc1 = MPSCNNFullyConnectedNode(source: maxPool1.resultImage, weights: fc1Datasource)
        
        let fc1Activation = MPSCNNNeuronReLUNode(source: fc1.resultImage)
        
        // OUTPUT = 1x1x64
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
        
        let maxPoolGradient = maxPool1.gradientFilter(withSource: fc1Gradient.resultImage)
        
        let covn1ActivationGradient = conv1Activation.gradientFilter(withSource:maxPoolGradient.resultImage)
        
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
