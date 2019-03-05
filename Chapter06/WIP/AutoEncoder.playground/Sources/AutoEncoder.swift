import Foundation
import AppKit
import MetalKit
import MetalPerformanceShaders

public class AutoEncoder{

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
    
    var datasources = [AutoEncoderDataSource]()
    
    public init(withCommandQueue commandQueue:MTLCommandQueue,
                inputShape:Shape,
                numberOfClasses:Int,
                weightsPathURL:URL,
                mode:NetworkMode = NetworkMode.training,
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
    }
    
}

// MARK: Creating the MPSNNGraph

extension AutoEncoder{
    
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
        
        return vector
    }
    
    private func createDenseLayer(name:String,
                          input:MPSNNImageNode,
                          kernelSize:KernelSize,
                          inputFeatureChannels:Int,
                          outputFeatureChannels:Int,
                          includeActivation:Bool=true) -> [MPSNNFilterNode]{
        
        let datasource = AutoEncoderDataSource(
            name: name,
            weightsPathURL: self.weightsPathURL,
            kernelSize: kernelSize,
            inputFeatureChannels: inputFeatureChannels,
            outputFeatureChannels: outputFeatureChannels,
            optimizer:makeOptimizer())
        
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
            source: input,
            weights: datasource)
        
        fc.resultImage.format = .float32
        fc.paddingPolicy = MPSNNDefaultPadding(method: MPSNNPaddingMethod.validOnly)
        fc.label = "\(name)_fc"
        
        var layers = [MPSNNFilterNode](arrayLiteral: fc)
        
        if includeActivation{
            let relu = MPSCNNNeuronReLUNode(source: fc.resultImage)
            relu.resultImage.format = .float32
            relu.label = "\(name)_relu"
            
            layers.append(relu)
        }
        
        return layers
    }
    
    func createTrainingGraph() -> MPSNNGraph?{
        guard #available(macOS 10.14.1, *) else{
            return nil
        }
        
        // Input placeholder
        let input = MPSNNImageNode(handle: nil)
        
        // == Forward-pass == //
        
        // Input -> hidden
        let zLayers = self.createDenseLayer(
            name: "l1",
            input: input,
            kernelSize: KernelSize(width:28, height:28), // size of our image
            inputFeatureChannels: 1,
            outputFeatureChannels: 300, // hidden layer size
            includeActivation: true)
        
        // hidden -> Output
        let logitLayers = self.createDenseLayer(
            name: "l1",
            input: zLayers.last!.resultImage,
            kernelSize: KernelSize(width:1, height:1),
            inputFeatureChannels: 300,
            outputFeatureChannels: 28 * 28 * 1, // output image size (width = 28, height = 28, channels = 1)
            includeActivation: false)
        
        let sigmoid = MPSCNNNeuronSigmoidNode(source: logitLayers.last!.resultImage)
        
        // reshape for the final output
        let output = MPSNNReshapeNode(
            source:sigmoid.resultImage,
            resultWidth: 28, resultHeight: 28, resultFeatureChannels:1)
        
        // == Loss ==
        let lossDesc = MPSCNNLossDescriptor(
            type: MPSCNNLossType.sigmoidCrossEntropy,
            reductionType: MPSCNNReductionType.mean)
        
        let loss = MPSCNNLossNode(
            source: output.resultImage,
            lossDescriptor: lossDesc)
        
        loss.resultImage.format = .float32
        
        // === Backwards pass ===
        let outputG = output.gradientFilter(withSource: loss.resultImage)
        
        // sigmoid
        let sigmoidG = sigmoid.gradientFilter(withSource: outputG.resultImage)
        
        // Keep track of the last nodes result image to pass it to the next
        var lastResultImage = sigmoidG.resultImage
        lastResultImage.format = .float32
        
        // propagate backwards through the other layers (logitLayers and zLayers)
        let trainingStyle = MPSNNTrainingStyle.updateDeviceGPU
        
        let _ = logitLayers.reversed().map { (node) -> MPSNNGradientFilterNode in
            let gradientNode = node.gradientFilter(withSource: lastResultImage)
            
            lastResultImage = gradientNode.resultImage
            lastResultImage.format = .float32
            
            if let cnnGradientNode = gradientNode as? MPSCNNConvolutionGradientNode{
                cnnGradientNode.trainingStyle = trainingStyle
            }
            
            return gradientNode
        }
        
        let _ = zLayers.reversed().map { (node) -> MPSNNGradientFilterNode in
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
    
    func createInferenceGraph() -> MPSNNGraph?{
        guard #available(macOS 10.14.1, *) else{
            return nil
        }
        
        // Input placeholder
        let input = MPSNNImageNode(handle: nil)
        
        // Input -> hidden
        let zLayers = self.createDenseLayer(
            name: "l1",
            input: input,
            kernelSize: KernelSize(width:28, height:28), // size of our image
            inputFeatureChannels: 1,
            outputFeatureChannels: 300, // hidden layer size
            includeActivation: true)
        
        // hidden -> Output
        let logitLayers = self.createDenseLayer(
            name: "l1",
            input: zLayers.last!.resultImage,
            kernelSize: KernelSize(width:1, height:1),
            inputFeatureChannels: 300,
            outputFeatureChannels: 28 * 28 * 1, // output image size (width = 28, height = 28, channels = 1)
            includeActivation: false)
        
        let sigmoid = MPSCNNNeuronSigmoidNode(source: logitLayers.last!.resultImage)
        
        // reshape
        let output = MPSNNReshapeNode(
            source:sigmoid.resultImage,
            resultWidth: 28, resultHeight: 28, resultFeatureChannels:1)
        
        if let graph = MPSNNGraph(
            device: self.device,
            resultImage: output.resultImage,
            resultImageIsNeeded: false){
            
            graph.outputStateIsTemporary = true
            
            graph.format = .float32
            
            return graph
        }
        
        return nil
    }
}

// MARK: Training

extension AutoEncoder{
    
    public func train(
        withDataLoader dataLoader:DataLoader,
        epochs:Int = 500,
        completionHandler handler: @escaping () -> Void){
        
        // Semaphore used to sync between CPU (loading data) and GPU (performing a training step)
        let trainingSemaphore = DispatchSemaphore(value:2)
        
        // Reference to the latest command buffer so we don't pre-maturely
        // try to proceed before finishing training
        var latestCommandBuffer : MTLCommandBuffer?
        
        for _ in 1...epochs{
            autoreleasepool{
                dataLoader.reset()
                
                while dataLoader.hasNext(){
                    latestCommandBuffer = self.trainStep(
                        withDataLoader:dataLoader,
                        semaphore:trainingSemaphore)
                }
                
                if latestCommandBuffer?.status == .completed{
                    latestCommandBuffer?.waitUntilCompleted()
                }
            }
        }
        
        updateDatasources()
        
        // Notify compeition handler that we have finished
        handler()
    }
    
    func trainStep(
        withDataLoader dataLoader:DataLoader,
        semaphore:DispatchSemaphore) -> MTLCommandBuffer?{
        
        let _ = semaphore.wait(timeout: .distantFuture)
        
        // Unwrap the graph
        guard let graph = self.graph else{
            semaphore.signal()
            return nil
        }
        
        // Get the command buffer to use in the MPS
        guard let commandBuffer = self.commandQueue.makeCommandBuffer() else{
            semaphore.signal()
            return nil
        }
        
        // Get next batch
        guard let batch = dataLoader.nextBatch() else{
            semaphore.signal()
            return nil
        }
        
        graph.encodeBatch(
            to: commandBuffer,
            sourceImages: [batch.images],
            sourceStates: nil,
            intermediateImages: nil,
            destinationStates: nil)
        
        commandBuffer.addCompletedHandler { (commandBuffer) in
            semaphore.signal()
        }
        
        commandBuffer.commit()
        
        return commandBuffer
    }
    
    func updateDatasources(){
        // Syncronize weights and bias terms from GPU to CPU
        guard let commandBuffer = self.commandQueue.makeCommandBuffer() else{
            return
        }
        
        for datasource in self.datasources{
            datasource.synchronizeParameters(on: commandBuffer)
        }
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        // Persist the weights and the bias terms to disk
        for datasource in datasources{
            datasource.saveParametersToDisk()
        }
    }
    
}
