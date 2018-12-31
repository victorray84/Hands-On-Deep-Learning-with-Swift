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
    
    public private(set) var learningRate : Float = 0.001
    
    public private(set) var momentumScale : Float = 0.2
    
    var graph : MPSNNGraph?
    
    var datasources = [SketchCNNDatasource]()
    
    public init(withCommandQueue commandQueue:MTLCommandQueue,
                inputShape:Shape,
                numberOfClasses:Int,
                weightsPathURL:URL,
                mode:NetworkMode=NetworkMode.training,
                learningRate:Float=0.001,
                momentumScale:Float=0.2){
        
        self.commandQueue = commandQueue
        self.device = commandQueue.device
        self.inputShape = inputShape
        self.numberOfClasses = numberOfClasses
        self.weightsPathURL = weightsPathURL
        self.mode = mode
        self.learningRate = learningRate
        self.momentumScale = momentumScale
        
        if mode == .training{
            self.graph = self.createTrainingGraph()
        } else{
            self.graph = self.createInferenceGraph()
        }
        
        //print(self.graph!.debugDescription)
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
            optimizer: self.makeOptimizer())
        
        if self.mode == .training{
            datasource.weightsAndBiasesState = MPSCNNConvolutionWeightsAndBiasesState(
                device: self.device,
                cnnConvolutionDescriptor: datasource.descriptor())
            
            if let weightsMomentum = self.makeMPSVector(count: datasource.weightsLength),
                let biasMomentum = self.makeMPSVector(count: datasource.biasTermsLength){
                
                datasource.momentumVectors = [weightsMomentum, biasMomentum]
            }
        }
        
        self.datasources.append(datasource)
        
        let conv = MPSCNNConvolutionNode(source: x, weights: datasource)
        conv.resultImage.format = .float32
        conv.paddingPolicy = MPSNNDefaultPadding(method: MPSNNPaddingMethod.sizeSame)
        conv.label = "\(name)_conv"
        
        let relu = MPSCNNNeuronReLUNode(source: conv.resultImage)
        relu.resultImage.format = .float32
        relu.label = "\(name)_relu"
        
        if !includeMaxPooling{
            return [conv, relu]
        }
        
        let pooling = MPSCNNPoolingMaxNode(source: relu.resultImage, filterSize: poolSize)
        pooling.resultImage.format = .float32
        pooling.paddingPolicy = MPSNNDefaultPadding(method: MPSNNPaddingMethod.validOnly)
        pooling.label = "\(name)_maxpooling"
        
        if self.mode == .inference || dropoutProbability == 0.0{
            return [conv, relu, pooling]
        }
        
        let dropout = MPSCNNDropoutNode(
            source: pooling.resultImage,
            keepProbability: (1.0 - dropoutProbability))
        dropout.resultImage.format = .float32
        dropout.label = "\(name)_dropout"
        
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
            optimizer: self.makeOptimizer())
        
        if self.mode == .training{
            datasource.weightsAndBiasesState = MPSCNNConvolutionWeightsAndBiasesState(
                device: self.device,
                cnnConvolutionDescriptor: datasource.descriptor())
        
            if let weightsMomentum = self.makeMPSVector(count: datasource.weightsLength),
                let biasMomentum = self.makeMPSVector(count: datasource.biasTermsLength){
                
                datasource.momentumVectors = [weightsMomentum, biasMomentum]
            }

        }
        
        self.datasources.append(datasource)
        
        let fc = MPSCNNFullyConnectedNode(
            source: x,
            weights: datasource)
        
        fc.resultImage.format = .float32
        fc.paddingPolicy = MPSNNDefaultPadding(method: MPSNNPaddingMethod.validOnly)
        fc.label = "\(name)_fc"
        
        if !includeActivation{
            return [fc]
        }
        
        let relu = MPSCNNNeuronReLUNode(source: fc.resultImage)
        relu.resultImage.format = .float32
        relu.label = "\(name)_relu"
        
        if self.mode == .inference || dropoutProbability == 0.0{
            return [fc, relu]
        }
        
        let dropout = MPSCNNDropoutNode(
            source: relu.resultImage,
            keepProbability: (1.0 - dropoutProbability))
        dropout.resultImage.format = .float32
        dropout.label = "\(name)_dropout"
        
        return [fc, relu, dropout]
    }
    
    private func makeMPSVector(count:Int, repeating:Float=0.0) -> MPSVector?{
        // Create a Metal buffer
        guard let buffer = self.device.makeBuffer(
            bytes: Array<Float32>(repeating: repeating, count: count),
            length: count * MemoryLayout<Float32>.size, options: [.storageModeShared]) else{
                return nil
        }
        
        // Create a vector descriptor
        let desc = MPSVectorDescriptor(
            length: count, dataType: MPSDataType.float32)
        
        // Create a vector with descriptor
        let vector = MPSVector(
            buffer: buffer, descriptor: desc)
        
        return vector
    }
    
    private func makeOptimizer() -> MPSNNOptimizerStochasticGradientDescent?{
        guard self.mode == .training else{
            return nil
        }
        
        let optimizerDescriptor = MPSNNOptimizerDescriptor(
            learningRate: self.learningRate,
            gradientRescale: 1.0,
            regularizationType: MPSNNRegularizationType.None,
            regularizationScale: 1.0)
        
        let optimizer = MPSNNOptimizerStochasticGradientDescent(
            device: self.device,
            momentumScale: self.momentumScale,
            useNestrovMomentum: true,
            optimizerDescriptor: optimizerDescriptor)
        
        optimizer.options = MPSKernelOptions(arrayLiteral: MPSKernelOptions.verbose)
        
        //print(optimizer.debugDescription)
        
        return optimizer
    }
}

// MARK: - Inference

extension SketchCNN{
    
    public func predict(X:[MPSImage]) -> [[Float]]?{
        
        guard let graph = self.graph else{
            return nil
        }
        
        if let commandBuffer = self.commandQueue.makeCommandBuffer(){
            let outputs = graph.encodeBatch(
                to: commandBuffer,
                sourceImages: [X],
                sourceStates: nil)
            
            // Syncronize the outputs after the prediction has been made
            // so we can get access to the values on the CPU
            outputs?.forEach({ (output) in
                output.synchronize(on: commandBuffer)
            })
            
            // Commit the command to the GPU
            commandBuffer.commit()
            
            // Wait for it to finish
            commandBuffer.waitUntilCompleted()
            
            // Process outputs
            if let outputs = outputs{
                let predictions = outputs.map({ (output) -> [Float] in
                    if let probs = output.toFloatArray(){
                        return Array<Float>(probs[0..<self.numberOfClasses])
                    }
                    return [Float]()
                })
                
                return predictions
            }
        }
        return nil
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
        softmax.label = "output"
        
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
        withDataLoaderForTraining trainDataLoader:DataLoader,
        dataLoaderForValidation validDataLoader:DataLoader? = nil,
        epochs : Int = 500,
        completionHandler handler: @escaping () -> Void) -> [(epoch:Int, accuracy:Float)]{
        
        var validationAccuracy = [(epoch:Int, accuracy:Float)]()
        
        let trainingSemaphore = DispatchSemaphore(value:2)
        
        var latestCommandBuffer : MTLCommandBuffer?
        
        // Check initial validation score
        if let validDataLoader = validDataLoader{
            let accuracy = self.validate(withDataLoader: validDataLoader)
            print("Initial model accuracy is \(accuracy)")
            
            validationAccuracy.append((epoch: 0, accuracy:accuracy))
        }
        
        for epoch in 1...epochs{
            autoreleasepool{
                trainDataLoader.reset()
                
                while trainDataLoader.hasNext(){
                    latestCommandBuffer = self.trainStep(
                        withDataLoader: trainDataLoader,
                        semaphore: trainingSemaphore)
                }
                
                // wait for training to complete
                if latestCommandBuffer?.status != .completed{
                    latestCommandBuffer?.waitUntilCompleted()
                }
                
                // reset Data loader
                trainDataLoader.reset()
                
                // Update and validate model every 5 epochs or on the last epoch
                if epoch % 5 == 0 || epoch == epochs{
                    print("Finished epoch \(epoch)")
                    updateDatasources()
                    
                    if let validDataLoader = validDataLoader{
                        let accuracy = self.validate(withDataLoader: validDataLoader)
                        print("Model Accuracy after \(epoch) epoch(s) is \(accuracy)")
                        
                        validationAccuracy.append((epoch: epoch, accuracy:accuracy))
                    }
                }
            }
        }
        
        print("Finished training")
        
        handler()
        
        return validationAccuracy
    }
    
    private func trainStep(withDataLoader dataLoader:DataLoader, semaphore:DispatchSemaphore) -> MTLCommandBuffer?{
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
        
        //        commandBuffer.waitUntilCompleted()
        
        return commandBuffer
    }
    
    func validate(withDataLoader dataLoader:DataLoader) -> Float{
        // Create inference network
        let inferenceNetwork = SketchCNN(
            withCommandQueue: self.commandQueue,
            inputShape: self.inputShape,
            numberOfClasses: self.numberOfClasses,
            weightsPathURL: self.weightsPathURL,
            mode: .inference)
        
        var sampleCount : Float = 0.0
        var predictionsCorrectCount : Float = 0.0
        
        dataLoader.reset()
        
        while dataLoader.hasNext(){
            autoreleasepool{
                guard let commandBuffer = commandQueue.makeCommandBuffer() else{
                    fatalError()
                }
                
                if let batch = dataLoader.nextBatch(commandBuffer: commandBuffer){
                    if let predictions = inferenceNetwork.predict(X: batch.images){
                        assert(predictions.count == batch.labels.count)
                        
                        for i in 0..<predictions.count{
                            sampleCount += 1.0
                            let predictedClass = dataLoader.labels[predictions[i].argmax]
                            let actualClass = batch.labels[i].label ?? ""
                            
                            predictionsCorrectCount += predictedClass == actualClass ? 1.0 : 0.0
                        }
                    }
                }
            }
        }
        
        return predictionsCorrectCount/sampleCount
    }
    
    private func updateDatasources(){
        // Syncronize weights and biases from GPU to CPU
        guard let commandBuffer = self.commandQueue.makeCommandBuffer() else{
            return
        }
        
        for datasource in self.datasources{
            datasource.synchronizeParameters(on: commandBuffer)
        }
        
        //        /// DEV
        //        for datasource in self.datasources{
        //            datasource.momentumVectors?[0].synchronize(on: commandBuffer)
        //            datasource.velocityVectors?[0].synchronize(on: commandBuffer)
        //        }
        //        ///
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        /// DEV
        if let wb = self.datasources[0].weightsAndBiasesState{
            let wtData = wb.weights.toArray(type: Float.self)
            let bData = wb.biases!.toArray(type: Float.self)
            print("\(wtData[0..<10])")
            print("\(bData[0..<10])")
        }
        ///
        
        // Persist the weightds and bias terms to disk
        for datasource in self.datasources{
            datasource.saveParametersToDisk()
        }
        
        //        self.graph?.reloadFromDataSources()
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
            
            graph.outputStateIsTemporary = true
            
            graph.format = .float32
            
            return graph
        }
        
        return nil
    }
}
