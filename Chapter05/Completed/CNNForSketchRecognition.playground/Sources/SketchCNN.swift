import Foundation
import AppKit
import MetalKit
import MetalPerformanceShaders
import PlaygroundSupport

class SketchCNN{
    
    public enum NetworkMode{
        case training
        case inference
    }
    
    let device : MTLDevice
    let commandQueue : MTLCommandQueue
    
    let mode : NetworkMode
    let weightsPathURL : URL
    
    let inputShape : Shape
    let numberOfClasses : Int
    
    var optimizer : MPSNNOptimizerStochasticGradientDescent?
    var graph : MPSNNGraph?
    
    var datasources = [SketchCNNDatasource]()
    
    public init(withCommandQueue commandQueue:MTLCommandQueue,
                inputShape:Shape,
                numberOfClasses:Int,
                weightsPathURL : URL,
                mode:NetworkMode=NetworkMode.training){
        
        self.device = commandQueue.device
        self.inputShape = inputShape
        self.numberOfClasses = numberOfClasses
        self.weightsPathURL = weightsPathURL
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
}

// MARK: - Util

extension SketchCNN{
    
    func createConvLayer(name:String,
                         x:MPSNNImageNode,
                         kernelSize:KernelSize,
                         strideSize:KernelSize = KernelSize(width:1, height:1),
                         inputFeatureChannels:Int,
                         outputFeatureChannels:Int,
                         includeMaxPooling:Bool=true) -> [MPSNNFilterNode]{
        
        let datasource = SketchCNNDatasource(
            name: name,
            weightsPathURL:self.weightsPathURL,
            kernelSize: kernelSize,
            strideSize: strideSize,
            inputFeatureChannels: inputFeatureChannels,
            outputFeatureChannels: outputFeatureChannels,
            optimizer: self.optimizer)
        
        if self.mode == .training{
            datasource.weightsAndBiasesState = MPSCNNConvolutionWeightsAndBiasesState(
                device: self.device,
                cnnConvolutionDescriptor: datasource.descriptor())
        }
        
        self.datasources.append(datasource)
        
        let conv = MPSCNNConvolutionNode(source: x, weights: datasource)
        conv.paddingPolicy = MPSNNDefaultPadding(method: MPSNNPaddingMethod.sizeSame)
        
        let relu = MPSCNNNeuronReLUNode(source: conv.resultImage)
        
        if !includeMaxPooling{
            return [conv, relu]
        }
        
        //let pooling = MPSCNNPoolingMaxNode(source: relu.resultImage, filterSize: 2)
        //pooling.paddingPolicy = MPSNNDefaultPadding(method: MPSNNPaddingMethod.excludeEdges)
        let pooling = MPSCNNPoolingAverageNode(source: relu.resultImage, filterSize: 2)
        
        if self.mode == .inference{
            return [conv, relu, pooling]
        }
        
        let dropout = MPSCNNDropoutNode(source: pooling.resultImage, keepProbability: 0.4)
        return [conv, relu, pooling, dropout]
    }
    
    func createDenseLayer(
        name:String,
        x:MPSNNImageNode,
        kernelSize:KernelSize,
        inputFeatureChannels:Int,
        outputFeatureChannels:Int,
        includeActivation:Bool=true) -> [MPSNNFilterNode]{
        
        let datasource = SketchCNNDatasource(
            name: name,
            weightsPathURL: self.weightsPathURL,
            kernelSize: kernelSize,
            inputFeatureChannels: inputFeatureChannels,
            outputFeatureChannels: outputFeatureChannels,
            optimizer: self.optimizer)
        
        if self.mode == .training{
            datasource.weightsAndBiasesState = MPSCNNConvolutionWeightsAndBiasesState(
                device: self.device,
                cnnConvolutionDescriptor: datasource.descriptor())
        }
        
        self.datasources.append(datasource)
        
        let fc = MPSCNNFullyConnectedNode(
            source: x,
            weights: datasource)
        
        fc.paddingPolicy = MPSNNDefaultPadding(method: MPSNNPaddingMethod.sizeSame)
        
        if !includeActivation{
            return [fc]
        }
        
        let relu = MPSCNNNeuronReLUNode(source: fc.resultImage)
        
        if self.mode == .inference{
            return [fc, relu]
        }
        
        let dropout = MPSCNNDropoutNode(source: relu.resultImage, keepProbability: 0.4)
        return [fc, relu, dropout]
    }
    
}

// MARK: - Inference

extension SketchCNN{
    
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
                    
                    handler(Array<Float>(probs[0..<self.numberOfClasses]))
                    return
                }
                
                handler(nil)
            }
        }
    }
    
    private func createInferenceGraph() -> MPSNNGraph?{
        // === Forward pass === //
        
        // placeholder node
        let input = MPSNNImageNode(handle: nil)
        
        // INPUT = 256x256x1
        
        // Scale
        let scale = MPSNNLanczosScaleNode(
            source: input,
            outputSize: MTLSize(
                width: self.inputShape.width,
                height: self.inputShape.height,
                depth: 1))
        
        // OUTPUT = 128x128x1
        
        // layer 1
        let layer1Nodes = self.createConvLayer(
            name: "l1",
            x: scale.resultImage,
            kernelSize: KernelSize(width:7, height:7),
            strideSize: KernelSize(width:1, height:1),
            inputFeatureChannels: 1,
            outputFeatureChannels: 32,
            includeMaxPooling: true)
        
        // OUTPUT = 64x64x32
        
        // layer 3
        let layer2Nodes = self.createConvLayer(
            name: "l2",
            x: layer1Nodes.last!.resultImage,
            kernelSize: KernelSize(width:5, height:5),
            strideSize: KernelSize(width:1, height:1),
            inputFeatureChannels: 32,
            outputFeatureChannels: 32,
            includeMaxPooling: true)
        
        // OUTPUT = 32x32x32
        
        // layer 3
        let layer3Nodes = self.createConvLayer(
            name: "l3",
            x: layer2Nodes.last!.resultImage,
            kernelSize: KernelSize(width:5, height:5),
            strideSize: KernelSize(width:1, height:1),
            inputFeatureChannels: 32,
            outputFeatureChannels: 64,
            includeMaxPooling: true)
        
        // OUTPUT = 16x16x64
        
        // layer 4
        let layer4Nodes = self.createConvLayer(
            name: "l4",
            x: layer3Nodes.last!.resultImage,
            kernelSize: KernelSize(width:3, height:3),
            strideSize: KernelSize(width:1, height:1),
            inputFeatureChannels: 64,
            outputFeatureChannels: 128)
        
        // OUTPUT = 8x8x128
        
        // fully connected layer
        let layer5Nodes = createDenseLayer(
            name: "l5",
            x: layer4Nodes.last!.resultImage,
            kernelSize: KernelSize(width:8, height:8),
            inputFeatureChannels: 128,
            outputFeatureChannels: 256)
        
        // OUTPUT = 256
        
        let layer6Nodes = createDenseLayer(
            name: "l6",
            x: layer5Nodes.last!.resultImage,
            kernelSize: KernelSize(width:1, height:1),
            inputFeatureChannels: 256,
            outputFeatureChannels: self.numberOfClasses,
            includeActivation: false)
        
        // OUTPUT = numberOfClasses
        
        let softmax = MPSCNNSoftMaxNode(source: layer6Nodes.last!.resultImage)
        
        guard let graph = MPSNNGraph(device: device,
                                     resultImage: softmax.resultImage,
                                     resultImageIsNeeded: true) else{
                                        return nil
        }
        
        return graph
    }
}

extension SketchCNN{
    
    public func train(
        withDataLoader dataLoader:DataLoader,
        epochs : Int = 100,
        completionHandler handler: @escaping () -> Void) -> Bool{
        
        autoreleasepool{
            let trainingSemaphore = DispatchSemaphore(value:1)
            
            var latestCommandBuffer : MTLCommandBuffer?
            
            for epoch in 1...epochs{
                var batch = dataLoader.getNextBatch()
                
                while batch != nil && DataLoader.getBatchCount(batch: batch) == dataLoader.batchSize{
                    latestCommandBuffer = self.trainStep(
                        x: batch!.images,
                        y: batch!.labels,
                        semaphore:trainingSemaphore)
                    
                    // update batch
                    batch = dataLoader.getNextBatch()
                }
                
                // wait for training to complete
                if latestCommandBuffer?.status != .completed{
                    latestCommandBuffer?.waitUntilCompleted()
                }
                
                // progressively save weights (always good practice)
                if epoch % 50 == 0{
                    print("Finished epoch \(epoch)")
                    updateDatasources()
                }
                
                // reset the current index of the data loader
                dataLoader.reset()
                
                // DEBUG - exit early 
                break
            }
            
            updateDatasources()
        }
        
        handler()
        
        return true
    }
    
    private func trainStep(x:[MPSImage], y:[MPSCNNLossLabels], semaphore:DispatchSemaphore) -> MTLCommandBuffer?{
        let _ = semaphore.wait(timeout: .distantFuture)
        
        guard let graph = self.graph else{
            semaphore.signal()
            return nil
        }
        
        // Get command buffer to use in MetalPerformanceShaders.
        guard let commandBuffer = self.commandQueue.makeCommandBuffer() else{
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
    
    private func createTrainingGraph() -> MPSNNGraph?{
        // === Forward pass === //
        
        // placeholder node
        let input = MPSNNImageNode(handle: nil)
        
        // INPUT = 256x256x1
        
        // Scale
        let scale = MPSNNLanczosScaleNode(
            source: input,
            outputSize: MTLSize(
                width: self.inputShape.width,
                height: self.inputShape.height,
                depth: 1))
        
        // OUTPUT = 128x128x1
        
        // layer 1
        let layer1Nodes = self.createConvLayer(
            name: "l1",
            x: scale.resultImage,
            kernelSize: KernelSize(width:7, height:7),
            strideSize: KernelSize(width:1, height:1),
            inputFeatureChannels: 1,
            outputFeatureChannels: 32,
            includeMaxPooling: true)
        
        // OUTPUT = 64x64x32
        
        // layer 2
        let layer2Nodes = self.createConvLayer(
            name: "l2",
            x: layer1Nodes.last!.resultImage,
            kernelSize: KernelSize(width:5, height:5),
            strideSize: KernelSize(width:1, height:1),
            inputFeatureChannels: 32,
            outputFeatureChannels: 64,
            includeMaxPooling: true)
        
        // OUTPUT = 32x32x64
        
        // layer 3
        let layer3Nodes = self.createConvLayer(
            name: "l3",
            x: layer2Nodes.last!.resultImage,
            kernelSize: KernelSize(width:3, height:3),
            strideSize: KernelSize(width:1, height:1),
            inputFeatureChannels: 64,
            outputFeatureChannels: 64)
        
        // OUTPUT = 16x16x64
        
        // fully connected layer
        let layer4Nodes = createDenseLayer(
            name: "l4",
            x: layer3Nodes.last!.resultImage,
            kernelSize: KernelSize(width:16, height:16),
            inputFeatureChannels: 64,
            outputFeatureChannels: 256)
        
        // OUTPUT = 256
        
        let layer5Nodes = createDenseLayer(
            name: "l5",
            x: layer4Nodes.last!.resultImage,
            kernelSize: KernelSize(width:1, height:1),
            inputFeatureChannels: 256,
            outputFeatureChannels: self.numberOfClasses,
            includeActivation: false)
        
        // OUTPUT = numberOfClasses
        
        let softmax = MPSCNNSoftMaxNode(source: layer5Nodes.last!.resultImage)
        
        // === Define the loss function ===
        
        // Loss function
        let lossDesc = MPSCNNLossDescriptor(
            type: MPSCNNLossType.softMaxCrossEntropy,
            reductionType: MPSCNNReductionType.mean)
        lossDesc.numberOfClasses = self.numberOfClasses
        
        let loss = MPSCNNLossNode(
            source: softmax.resultImage,
            lossDescriptor: lossDesc)
        
        // === Backwards pass ===
        
        let softmaxG = softmax.gradientFilter(withSource: loss.resultImage)
        
        // Keep track of the last nodes result image to pass it to the next
        var lastResultImage = softmaxG.resultImage
        
        let _ = layer5Nodes.reversed().map { (node) -> MPSNNGradientFilterNode in
            let gradientNode = node.gradientFilter(withSource: lastResultImage)
            lastResultImage = gradientNode.resultImage
            return gradientNode
        }

        let _ = layer4Nodes.reversed().map { (node) -> MPSNNGradientFilterNode in
            let gradientNode = node.gradientFilter(withSource: lastResultImage)
            lastResultImage = gradientNode.resultImage
            return gradientNode
        }
        
        let _ = layer3Nodes.reversed().map { (node) -> MPSNNGradientFilterNode in
            let gradientNode = node.gradientFilter(withSource: lastResultImage)
            lastResultImage = gradientNode.resultImage
            return gradientNode
        }
        
        let _ = layer2Nodes.reversed().map { (node) -> MPSNNGradientFilterNode in
            let gradientNode = node.gradientFilter(withSource: lastResultImage)
            lastResultImage = gradientNode.resultImage
            return gradientNode
        }
        
        let _ = layer1Nodes.reversed().map { (node) -> MPSNNGradientFilterNode in
            let gradientNode = node.gradientFilter(withSource: lastResultImage)
            lastResultImage = gradientNode.resultImage
            return gradientNode
        }
        
        return MPSNNGraph(device: self.device,
                          resultImage: lastResultImage,
                          resultImageIsNeeded: false)
    }
}