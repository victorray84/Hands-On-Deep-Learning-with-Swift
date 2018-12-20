import Foundation
import AppKit
import MetalKit
import MetalPerformanceShaders

public class SketchCNN{
    
    public enum NetworkMode{
        case training
        case inference
    }
    
    let device : MTLDevice
    let commandQueue : MTLCommandQueue
    
    let weightsPathURL : URL
    
    let mode : NetworkMode
    
    let inputShape : Shape
    let numberOfClasses : Int
    
    var optimizer : MPSNNOptimizerStochasticGradientDescent?
    var graph : MPSNNGraph?
    
    var datasources = [SketchCNNDatasource]()
    
    public init(withCommandQueue commandQueue:MTLCommandQueue,
                inputShape:Shape,
                numberOfClasses:Int,
                weightsPathURL:URL,
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
                learningRate: 0.01)
            
            self.optimizer!.options = MPSKernelOptions(arrayLiteral: MPSKernelOptions.verbose)
            
            self.graph = self.createTrainingGraph()
            
            //print(self.graph!.debugDescription)
        } else{
            self.graph = self.createInferenceGraph()
        }
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
                         includeMaxPooling:Bool=true,
                         poolSize : Int = 2,
                         dropoutProbability:Float=0.0) -> [MPSNNFilterNode]{
        
        let datasource = SketchCNNDatasource(
            name: name,
            weightsPathURL:self.weightsPathURL,
            kernelSize: kernelSize,
            strideSize: strideSize,
            inputFeatureChannels: inputFeatureChannels,
            outputFeatureChannels: outputFeatureChannels,
            optimizer: self.optimizer)
        
//        if self.mode == .training{
//            if datasource.load(){
//                let weights = self.device.makeBuffer(
//                    bytes: datasource.weights(),
//                    length: datasource.size, options: <#T##MTLResourceOptions#>)
//                datasource.weightsAndBiasesState = MPSCNNConvolutionWeightsAndBiasesState(
//                    device: self.device,
//                    cnnConvolutionDescriptor: datasource.descriptor())
//            }
//        }
        
        self.datasources.append(datasource)
        
        let conv = MPSCNNConvolutionNode(source: x, weights: datasource)
        conv.paddingPolicy = MPSNNDefaultPadding(method: MPSNNPaddingMethod.sizeSame)
        
        let relu = MPSCNNNeuronReLUNode(source: conv.resultImage)
        
        if !includeMaxPooling{
            return [conv, relu]
        }
        
        let pooling = MPSCNNPoolingMaxNode(source: relu.resultImage, filterSize: poolSize)
        pooling.paddingPolicy = MPSNNDefaultPadding(method: MPSNNPaddingMethod.validOnly)
        
        if self.mode == .inference || dropoutProbability == 0.0{
            return [conv, relu, pooling]
        }
        
        let dropout = MPSCNNDropoutNode(
            source: pooling.resultImage,
            keepProbability: (1.0 - dropoutProbability))
        
        return [conv, relu, pooling, dropout]
    }
    
    func createDenseLayer(
        name:String,
        x:MPSNNImageNode,
        kernelSize:KernelSize,
        inputFeatureChannels:Int,
        outputFeatureChannels:Int,
        includeActivation:Bool=true,
        dropoutProbability:Float=0.0) -> [MPSNNFilterNode]{
        
        let datasource = SketchCNNDatasource(
            name: name,
            weightsPathURL:self.weightsPathURL, 
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
        
        fc.paddingPolicy = MPSNNDefaultPadding(method: MPSNNPaddingMethod.validOnly)
        
        if !includeActivation{
            return [fc]
        }
        
        let relu = MPSCNNNeuronReLUNode(source: fc.resultImage)
        
        if self.mode == .inference || dropoutProbability == 0.0{
            return [fc, relu]
        }
        
        let dropout = MPSCNNDropoutNode(
            source: relu.resultImage,
            keepProbability: (1.0 - dropoutProbability))
        
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
                    
                    let oi = outputImage!
                    print("Output image w \(oi.width), h \(oi.height), c \(oi.featureChannels), no. \(oi.numberOfImages)")                    
                    
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
        
        // INPUT = 128x128x1
        
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
            strideSize: KernelSize(width:2, height:2),
            inputFeatureChannels: 1,
            outputFeatureChannels: 32,
            includeMaxPooling: false,
            poolSize: 0,
            dropoutProbability: 0.3)
        
        // OUTPUT = 64x64x32
        
        // layer 2
        let layer2Nodes = self.createConvLayer(
            name: "l2",
            x: layer1Nodes.last!.resultImage,
            kernelSize: KernelSize(width:5, height:5),
            strideSize: KernelSize(width:1, height:1),
            inputFeatureChannels: 32,
            outputFeatureChannels: 32,
            includeMaxPooling: true,
            poolSize: 2)        
        
        // OUTPUT = 32x32x32
        
        // layer 3
        let layer3Nodes = self.createConvLayer(
            name: "l3",
            x: layer2Nodes.last!.resultImage,
            kernelSize: KernelSize(width:5, height:5),
            strideSize: KernelSize(width:1, height:1),
            inputFeatureChannels: 32,
            outputFeatureChannels: 32,
            includeMaxPooling: true,
            poolSize: 2,
            dropoutProbability: 0.3)
        
        // OUTPUT = 16x16x32
        
        let layer4Nodes = self.createConvLayer(
            name: "l4",
            x: layer3Nodes.last!.resultImage,
            kernelSize: KernelSize(width:5, height:5),
            strideSize: KernelSize(width:1, height:1),
            inputFeatureChannels: 32,
            outputFeatureChannels: 32,
            includeMaxPooling: true,
            poolSize: 2,
            dropoutProbability: 0.3)
        
        // OUTPUT = 8x8x32
        
        // fully connected layer
        let layer5Nodes = createDenseLayer(
            name: "l5",
            x: layer4Nodes.last!.resultImage,
            kernelSize: KernelSize(width:8, height:8),
            inputFeatureChannels: 32,
            outputFeatureChannels: 64,
            includeActivation: true,
            dropoutProbability: 0.3)
        
        // OUTPUT = 64
        
        let layer6Nodes = createDenseLayer(
            name: "l6",
            x: layer5Nodes.last!.resultImage,
            kernelSize: KernelSize(width:1, height:1),
            inputFeatureChannels: 64,
            outputFeatureChannels: self.numberOfClasses,
            includeActivation: false)
        
        // OUTPUT = numberOfClasses
        
        let softmax = MPSCNNSoftMaxNode(source: layer6Nodes.last!.resultImage)
        
        guard let graph = MPSNNGraph(
            device: device,
            resultImage: softmax.resultImage,
            resultImageIsNeeded: true) else{
                return nil
        }
        
        return graph
    }
}

extension SketchCNN{
    
    @discardableResult
    public func train(
        withDataLoaderForTraining trainingDataLoader:DataLoader,
        dataLoaderForValidation:DataLoader? = nil,
        epochs : Int = 1,
        completionHandler handler: @escaping () -> Void) -> Bool{
        
        let trainingSemaphore = DispatchSemaphore(value:2)
        
        autoreleasepool{
            var latestCommandBuffer : MTLCommandBuffer?
            
            for epoch in 1...epochs{
                trainingDataLoader.reset()
                
                while trainingDataLoader.hasNext(){
                    latestCommandBuffer = self.trainStep(
                        withDataLoader: trainingDataLoader,
                        semaphore: trainingSemaphore)
                }
                
                print("Finished epoch \(epoch)")
            }
        }
        
        print("Finished training")
        updateDatasources()
        
//            for epoch in 1...epochs{
//                trainingDataLoader.reset()
//
//                    while trainingDataLoader.hasNext(){
//                            latestCommandBuffer = self.trainStep(
//                                withDataLoader:trainingDataLoader,
//                                semaphore:trainingSemaphore)
//                    }
//
//                    // wait for training to complete
//                    if latestCommandBuffer?.status != .completed{
//                        latestCommandBuffer?.waitUntilCompleted()
//                    }
//
//                print("Finished epoch \(epoch)")
//                updateDatasources()
        
                // progressively save weights (always good practice)
//                if epoch % 5 == 0{
//                    print("Finished epoch \(epoch)")
//                    updateDatasources()
//                }
                
                /// DEV
//                print("Finished epoch \(epoch)")
//                updateDatasources()
                ///
                // reset the current index of the data loader
//                trainingDataLoader.reset()
//                }
//        }
//
//        updateDatasources()
        
        handler()
        
        return true
    }
    
    private func trainStep(withDataLoader dataLoader:DataLoader, semaphore:DispatchSemaphore) -> MTLCommandBuffer?{
        //let _ = semaphore.wait(timeout: .distantFuture)
        
        guard let graph = self.graph else{
            semaphore.signal()
            return nil
        }
        
        // Get command buffer to use in MetalPerformanceShaders.
        guard let commandBuffer = self.commandQueue.makeCommandBuffer() else{
            semaphore.signal()
            return nil
        }
        
        // Get next batch
        guard let batch = dataLoader.nextBatch(commandBuffer:commandBuffer) else{
            semaphore.signal()
            return nil
        }
        
        graph.encodeBatch(to: commandBuffer,
                          sourceImages: [batch.images],
                          sourceStates: [batch.labels],
                          intermediateImages: nil,
                          destinationStates: nil)
        
        
        commandBuffer.addCompletedHandler({ (commandBuffer) in
            semaphore.signal()
        })
        
        commandBuffer.commit()
        
        commandBuffer.waitUntilCompleted()
        
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
        
        /// DEV
//        for datasource in self.datasources{
////            guard let cb = self.commandQueue.makeCommandBuffer() else{
////                continue
////            }
////
////            let wb = datasource.cnnConvolution!.exportWeightsAndBiases(
////                with: cb,
////                resultStateCanBeTemporary: false)
////
////            wb.synchronize(on: cb)
////
////            cb.commit()
////            cb.waitUntilCompleted()
//
//            guard let wb = datasource.weightsAndBiasesState else{
//                continue
//            }
//
//            let wtData = wb.weights.toArray(type: Float.self)
//            let biasData = wb.biases!.toArray(type: Float.self)
//            print("")
//        }
        ///
        
        /// DEV
//        guard let commandBuffer2 = self.commandQueue.makeCommandBuffer() else{
//            return
//        }
//
//        let wb = datasources[0].cnnConvolution!.exportWeightsAndBiases(
//            with: commandBuffer2,
//            resultStateCanBeTemporary: false)
//
//        wb.synchronize(on: commandBuffer2)
//
//        commandBuffer2.commit()
//        commandBuffer2.waitUntilCompleted()
//
//        let data = wb.weights.toArray(type: Float.self)
//        if data[0] == Float.nan{
//            print("Data is nan")
//        }
        
//        if let wb = self.datasources[0].weightsAndBiasesState{
//            let data = wb.weights.toArray(type: Float.self)
//            print(data[0..<5])
//        }
        ///
        
        for datasource in self.datasources{
            datasource.saveToDisk()
        }
    }
    
    private func createTrainingGraph() -> MPSNNGraph?{
        // === Forward pass === //
        
        // placeholder node
        let input = MPSNNImageNode(handle: nil)
        
        // INPUT = 128x128x1
        
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
            strideSize: KernelSize(width:2, height:2),
            inputFeatureChannels: 1,
            outputFeatureChannels: 32,
            includeMaxPooling: false,
            poolSize: 0,
            dropoutProbability: 0.3)
        
        // OUTPUT = 64x64x32
        
        // layer 2
        let layer2Nodes = self.createConvLayer(
            name: "l2",
            x: layer1Nodes.last!.resultImage,
            kernelSize: KernelSize(width:5, height:5),
            strideSize: KernelSize(width:1, height:1),
            inputFeatureChannels: 32,
            outputFeatureChannels: 32,
            includeMaxPooling: true,
            poolSize: 2)
        
        // OUTPUT = 32x32x32
        
        // layer 3
        let layer3Nodes = self.createConvLayer(
            name: "l3",
            x: layer2Nodes.last!.resultImage,
            kernelSize: KernelSize(width:5, height:5),
            strideSize: KernelSize(width:1, height:1),
            inputFeatureChannels: 32,
            outputFeatureChannels: 32,
            includeMaxPooling: true,
            poolSize: 2,
            dropoutProbability: 0.3)
        
        // OUTPUT = 16x16x32
        
        let layer4Nodes = self.createConvLayer(
            name: "l4",
            x: layer3Nodes.last!.resultImage,
            kernelSize: KernelSize(width:5, height:5),
            strideSize: KernelSize(width:1, height:1),
            inputFeatureChannels: 32,
            outputFeatureChannels: 32,
            includeMaxPooling: true,
            poolSize: 2,
            dropoutProbability: 0.3)
        
        // OUTPUT = 8x8x32
        
        // fully connected layer
        let layer5Nodes = createDenseLayer(
            name: "l5",
            x: layer4Nodes.last!.resultImage,
            kernelSize: KernelSize(width:8, height:8),
            inputFeatureChannels: 32,
            outputFeatureChannels: 64,
            includeActivation: true,
            dropoutProbability: 0.3)
        
        // OUTPUT = 64
        
        let layer6Nodes = createDenseLayer(
            name: "l6",
            x: layer5Nodes.last!.resultImage,
            kernelSize: KernelSize(width:1, height:1),
            inputFeatureChannels: 64,
            outputFeatureChannels: self.numberOfClasses,
            includeActivation: false)
        
        // OUTPUT = numberOfClasses
        
        let softmax = MPSCNNSoftMaxNode(source: layer6Nodes.last!.resultImage)
        
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
        loss.resultImage.format = .float32
        let softmaxG = softmax.gradientFilter(withSource: loss.resultImage)
        
        // Keep track of the last nodes result image to pass it to the next
        var lastResultImage = softmaxG.resultImage
        lastResultImage.format = .float32
        
        // Set training style; this can be sett via the trainingStyle property of a MPSCNNConvolutionGradientNode object.
        // Valid value are:
        //  MPSCNNConvolutionGradientNode.MPSNNTrainingStyle.updateDeviceCPU (to update on tthe CPU)
        //  MPSCNNConvolutionGradientNode.MPSNNTrainingStyle.updateDeviceGPU (to update on tthe GPU
        //  MPSCNNConvolutionGradientNode.MPSNNTrainingStyle.updateDeviceNone (don't update this layer)
        let trainingStyle = MPSNNTrainingStyle.updateDeviceGPU
        
        let _ = layer6Nodes.reversed().map { (node) -> MPSNNGradientFilterNode in
            let gradientNode = node.gradientFilter(withSource: lastResultImage)
            lastResultImage = gradientNode.resultImage
            
            lastResultImage.format = .float32
            
            if let cnnGradientNode = gradientNode as? MPSCNNConvolutionGradientNode{
                cnnGradientNode.trainingStyle = trainingStyle
            }
            
            return gradientNode
        }
        
        let _ = layer5Nodes.reversed().map { (node) -> MPSNNGradientFilterNode in
            let gradientNode = node.gradientFilter(withSource: lastResultImage)
            lastResultImage = gradientNode.resultImage
            
            lastResultImage.format = .float32
            
            if let cnnGradientNode = gradientNode as? MPSCNNConvolutionGradientNode{
                cnnGradientNode.trainingStyle = trainingStyle
            }
            
            return gradientNode
        }
        
        let _ = layer4Nodes.reversed().map { (node) -> MPSNNGradientFilterNode in
            let gradientNode = node.gradientFilter(withSource: lastResultImage)
            lastResultImage = gradientNode.resultImage
            
            lastResultImage.format = .float32
            
            if let cnnGradientNode = gradientNode as? MPSCNNConvolutionGradientNode{
                cnnGradientNode.trainingStyle = trainingStyle
            }
            
            return gradientNode
        }
        
        let _ = layer3Nodes.reversed().map { (node) -> MPSNNGradientFilterNode in
            let gradientNode = node.gradientFilter(withSource: lastResultImage)
            lastResultImage = gradientNode.resultImage
            
            lastResultImage.format = .float32
            
            if let cnnGradientNode = gradientNode as? MPSCNNConvolutionGradientNode{
                cnnGradientNode.trainingStyle = trainingStyle
            }
            
            return gradientNode
        }
        
        let _ = layer2Nodes.reversed().map { (node) -> MPSNNGradientFilterNode in
            let gradientNode = node.gradientFilter(withSource: lastResultImage)
            lastResultImage = gradientNode.resultImage
            
            lastResultImage.format = .float32
            
            if let cnnGradientNode = gradientNode as? MPSCNNConvolutionGradientNode{
                cnnGradientNode.trainingStyle = trainingStyle
            }
            
            return gradientNode
        }
        
        let _ = layer1Nodes.reversed().map { (node) -> MPSNNGradientFilterNode in
            let gradientNode = node.gradientFilter(withSource: lastResultImage)
            lastResultImage = gradientNode.resultImage
            
            lastResultImage.format = .float32
            
            if let cnnGradientNode = gradientNode as? MPSCNNConvolutionGradientNode{
                cnnGradientNode.trainingStyle = trainingStyle
            }
            
            return gradientNode
        }
        
        if let graph = MPSNNGraph(
            device: self.device,
            resultImage: lastResultImage,
            resultImageIsNeeded: false){
            
            graph.format = .float32
            
            return graph
        }
        
        return nil
    }
}
