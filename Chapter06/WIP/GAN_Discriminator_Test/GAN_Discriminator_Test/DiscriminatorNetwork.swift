import Foundation
import AppKit
import MetalKit
import MetalPerformanceShaders
import CoreGraphics

public class DiscriminatorNetwork{

    public enum NetworkMode{
        case training
        case inference
    }
    
    let device : MTLDevice
    let commandQueue : MTLCommandQueue
    
    let weightsPathURL : URL
    
    let mode : NetworkMode
    
    public private(set) var inputShape : Shape = (width:28, height:28, channels:1)
    
    public private(set) var learningRate : Float = 0.0002
    
    public private(set) var momentumScale : Float = 0.2 // beta1
    
    var sharedDatasources = [ConvnetDataSource]()
    
    private var graph : MPSNNGraph?
    
    public init(withCommandQueue commandQueue:MTLCommandQueue,
                weightsPathURL:URL,
                inputShape:Shape=(width:28, height:28, channels:1),
                mode:NetworkMode = NetworkMode.training,
                learningRate:Float=0.001,
                momentumScale:Float=0.5){
        
        self.commandQueue = commandQueue
        self.device = commandQueue.device
        self.weightsPathURL = weightsPathURL
        self.mode = mode
        self.learningRate = learningRate
        self.momentumScale = momentumScale
        
        self.graph = self.createDiscriminator(self.mode)
        //self.graph?.options = .verbose
    }
    
}

// MARK: Inference

extension DiscriminatorNetwork{
    
    public func predict(_ image:MPSImage) -> Float{
        guard let commandBuffer = self.commandQueue.makeCommandBuffer(),
            let graph = self.graph else{
            return -1
        }
        
        
        let output = graph.encode(to: commandBuffer, sourceImages: [image])
        
        // Syncronize the outputs after the prediction has been made
        // so we can get access to the values on the CPU
        output?.synchronize(on: commandBuffer)
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        guard let array = output?.toFloatArray() else{
            return -1
        }
        
        //let originalArray = image.toArray(padding: UInt8(0))!
        let originalArray = image.toArray(padding: Float(0))!
        
        let tex = image.texture
        let texb = output!.texture
        
        for i in 0..<originalArray.count{
            if Int.random(in: 0...10) > 8{
                continue
            }
            let a = originalArray[i]
            
            print("i \(i) ... a=\(a)")
                
        }
            
        for i in 0..<array.count{
            if Int.random(in: 0...10) > 8{
                continue
            }
            let b = array[i]
            print("i \(i) ...b=\(b)")
        }
        
        return array[0]
    }
}

// MARK: Training

extension DiscriminatorNetwork{
    
    public func train(
        dataLoaderA:DataLoader,
        dataLoaderB:DataLoader,
        epochs:Int = 10,
        completionHandler handler: @escaping () -> Void){
        
        for epoch in 1...epochs{
            
            var trainingSteps : Float = 0.0
            var loss : Float = 0.0
            
            dataLoaderA.reset()
            dataLoaderB.reset()
            
            while dataLoaderA.hasNext() && dataLoaderB.hasNext(){
                autoreleasepool{
                    if let stepLoss = self.trainStep(
                        dataLoaderA:dataLoaderA,
                        dataLoaderB:dataLoaderB){
                        trainingSteps += 1
                        loss += stepLoss
                        
                        if trainingSteps == 0 || Int(trainingSteps) % 20 == 0{
                            print("... loss \(loss/trainingSteps)")
                        }
                    }
                }
            }
            
            // Calculate the mean for both losses
            loss /= trainingSteps
            
            // Generate sample images every n epochs
            print("Finished epoch \(epoch); loss \(loss)")
            
            if epoch == 1 || epoch == epochs || epoch % 5 == 0{
                // Make the associated datasources to the discriminator trainable
                // so they are persisted
                self.sharedDatasources.forEach { (ds) in
                    ds.trainable = true
                }
                
                updateDatasources(self.sharedDatasources)
            }
        }
        
        // Notify compeition handler that we have finished
        DispatchQueue.main.async {
            handler()
        }
    }
    
    @discardableResult
    func trainStep(dataLoaderA:DataLoader, dataLoaderB:DataLoader) -> Float?{
        var loss : Float = 0.0
        
        guard let graph = self.graph else{
                return nil
        }
        
        // Train the Discriminator
        
        // 1. get the inputs (x and y)
        if let trueImages = dataLoaderA.nextBatch(),
            let trueLabels = dataLoaderA.createLabels(withValue: 0.9),
            let falseImages = dataLoaderB.nextBatch(),
            let falseLabels = dataLoaderB.createLabels(withValue: 0.0),
            let commandBuffer = self.commandQueue.makeCommandBuffer() {
            
            // Train the discriminator
            sharedDatasources.forEach { (ds) in
                ds.trainable = true
            }
            
//            for (image, label) in zip([trueImages + falseImages], [trueLabels + falseLabels]){
//                graph.encode(
//                    to: commandBuffer,
//                    sourceImages: image,
//                    sourceStates: label,
//                    intermediateImages: nil,
//                    destinationStates: nil)
//            }

            graph.encodeBatch(
                to: commandBuffer,
                sourceImages: [trueImages + falseImages],
                sourceStates: [trueLabels + falseLabels],
                intermediateImages: nil,
                destinationStates: nil)
            
            // Syncronise the loss labels so we can get access to them
            // to return to the caller
            for label in trueLabels + falseLabels{
                label.synchronize(on: commandBuffer)
            }
            
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
            
            // Calculate and store the loss
            for label in trueLabels + falseLabels{
                if let labelLosses = label.lossImage().toFloatArray(){
                    loss += labelLosses[0]
                }
            }
            
            loss /= Float(trueLabels.count + falseLabels.count)
        }
        
        return loss
    }
    
    func updateDatasources(_ datasources:[DataSource]){
        // Syncronize weights and bias terms from GPU to CPU
        guard let commandBuffer = self.commandQueue.makeCommandBuffer() else{
            return
        }
        
        for datasource in datasources{
            if datasource.trainable{
                datasource.synchronizeParameters(on: commandBuffer)
            }
        }

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        // Persist the weights and the bias terms to disk
        for datasource in datasources{
            if datasource.trainable{
                datasource.saveParametersToDisk()
            }
        }
    }
}

// MARK: Activation factory

extension DiscriminatorNetwork{
    
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
        let activation = MPSCNNNeuronReLUNode(source: x, a:0.1)
        activation.resultImage.format = .float32
        activation.label = name
        return activation
    }
    
    static func createSigmoid(_ x:MPSNNImageNode, _ name:String) -> MPSCNNNeuronNode{
        //let activation = MPSCNNNeuronSigmoidNode(source:x)
        let activation = MPSCNNNeuronHardSigmoidNode(source: x, a:1.0, b:Float.leastNonzeroMagnitude)
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
    
    static func createLinear(_ x:MPSNNImageNode, _ name:String) -> MPSCNNNeuronNode{
        let activation = MPSCNNNeuronLinearNode(source: x)
        activation.resultImage.format = .float32
        activation.label = name
        return activation
    }
    
}

// MARK: Builder functions

extension DiscriminatorNetwork{
    
    private func makeOptimizer() -> MPSNNOptimizerStochasticGradientDescent?{
        guard self.mode == .training else{
            return nil
        }
        
        let optimizerDescriptor = MPSNNOptimizerDescriptor(
            learningRate: self.learningRate,
            gradientRescale: 1.0,
            regularizationType: .None,
            regularizationScale: 0.0)
        
        let optimizer = MPSNNOptimizerStochasticGradientDescent(
            device: self.device,
            momentumScale: self.momentumScale,
            useNestrovMomentum: true,
            optimizerDescriptor: optimizerDescriptor)
        
        optimizer.options = MPSKernelOptions(arrayLiteral: MPSKernelOptions.verbose)
        
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
    
    func createConvLayer(name:String,
                         x:MPSNNImageNode,
                         kernelSize:KernelSize = KernelSize(width:5, height:5),
                         strideSize:KernelSize = KernelSize(width:2, height:2),
                         inputFeatureChannels:Int,
                         outputFeatureChannels:Int,
                         datasources:inout [ConvnetDataSource],
                         activationFunc:((MPSNNImageNode, String) -> MPSCNNNeuronNode)? = nil) -> [MPSNNFilterNode]{
        
        var layers = [MPSNNFilterNode]()
        
        // We are sharing datasources between our networks (specifically the Discriminator + GAN,
        // Generator + GAN - it's for this reason we cache them
        let datasource = ConvnetDataSource(
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
    
    private func createDenseLayer(name:String,
                                  input:MPSNNImageNode,
                                  kernelSize:KernelSize,
                                  inputFeatureChannels:Int,
                                  outputFeatureChannels:Int,
                                  datasources:inout [ConvnetDataSource],
                                  activationFunc:((MPSNNImageNode, String) -> MPSCNNNeuronNode)? = nil) -> [MPSNNFilterNode]{
        
        var layers = [MPSNNFilterNode]()
        
        // We are sharing datasources between our networks (specifically the Discriminator + GAN,
        // Generator + GAN - it's for this reason we cache them
        let datasource = ConvnetDataSource(
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

// MARK: Discriminator

extension DiscriminatorNetwork{
    
    func createDiscriminatorForwardPassNodes(
        _ x:MPSNNImageNode? = nil,
        _ inputShape:Shape?=nil) -> (nodes:[MPSNNFilterNode], datasources:[ConvnetDataSource]){
        
        var nodes = [MPSNNFilterNode]()
        
        let inputShape = inputShape ?? self.inputShape
        
        // keep track of the last input
        var lastOutput : MPSNNImageNode = x ?? MPSNNImageNode(handle: nil)
        //lastOutput.format = .float32
        
        // === Forward pass ===
        let layer1 = self.createConvLayer(
            name: "d_conv_1",
            x: lastOutput, // 28x28x1
            kernelSize: KernelSize(width:5, height:5),
            strideSize: StrideSize(width:2, height:2),
            inputFeatureChannels: inputShape.channels,
            outputFeatureChannels: 64,
            datasources: &self.sharedDatasources,
            activationFunc: DiscriminatorNetwork.createLeakyRelu)
        
        lastOutput = layer1.last!.resultImage // 14x14x64
        nodes += layer1
        
        let layer2 = self.createConvLayer(
            name: "d_conv_2",
            x: lastOutput, // 14x14x64
            kernelSize: KernelSize(width:5, height:5),
            strideSize: StrideSize(width:2, height:2),
            inputFeatureChannels: 64,
            outputFeatureChannels: 128,
            datasources: &self.sharedDatasources,
            activationFunc: DiscriminatorNetwork.createLeakyRelu)

        lastOutput = layer2.last!.resultImage // 7x7x128
        nodes += layer2

        let layer3 = self.createDenseLayer(
            name: "d_dense_1",
            input: lastOutput, // 7x7x128
            kernelSize: KernelSize(width:7, height:7),
            inputFeatureChannels: 128,
            outputFeatureChannels: 256,
            datasources: &self.sharedDatasources,
            activationFunc: DiscriminatorNetwork.createLeakyRelu)

        lastOutput = layer3.last!.resultImage // 1x1x256
        nodes += layer3

        let layer4 = self.createDenseLayer(
            name: "d_dense_2",
            input: lastOutput,// 1x1x256
            kernelSize: KernelSize(width:1, height:1),
            inputFeatureChannels: 256,
            outputFeatureChannels: 1,
            datasources: &self.sharedDatasources,
            activationFunc: DiscriminatorNetwork.createSigmoid)

        lastOutput = layer4.last!.resultImage
        nodes += layer4

        return (nodes:nodes, datasources:self.sharedDatasources)
    }
    
    func createDiscriminator(_ mode:NetworkMode) -> MPSNNGraph?{
        // Create placeholder for our input into the network
        let input = MPSNNImageNode(handle: nil)
        
        // Scale
        let scale = MPSNNLanczosScaleNode(
            source: input,
            outputSize: MTLSize(
                width: self.inputShape.width,
                height: self.inputShape.height,
                depth: 1))
        
        //input.format = .float32
        
        // Create the forward pass
        let (nodes, _) = createDiscriminatorForwardPassNodes(
            scale.resultImage,
            self.inputShape)
        
        // Obtain reference to the last node 
        var lastOutput = nodes.last!.resultImage
        
        if mode == .inference{
            guard let mpsGraph = MPSNNGraph(
                device: self.device,
                resultImage: lastOutput,
                resultImageIsNeeded: true) else{
                    return nil
            }
            
            return mpsGraph
        }
        
        // === Loss function ===
        let lossDesc = MPSCNNLossDescriptor(
            type: MPSCNNLossType.sigmoidCrossEntropy,
            //type: MPSCNNLossType.categoricalCrossEntropy,
            reductionType: MPSCNNReductionType.mean)
        
        let loss = MPSCNNLossNode(
            source: lastOutput,
            lossDescriptor: lossDesc)
        
        loss.resultImage.format = .float32
        lastOutput = loss.resultImage
        
        let trainingStyle = MPSNNTrainingStyle.updateDeviceGPU
        
        // === Backwards pass ===
        let _ = nodes.reversed().map { (node) -> MPSNNGradientFilterNode in
            let gradientNode = node.gradientFilter(withSource: lastOutput)
            
            lastOutput = gradientNode.resultImage
            lastOutput.format = .float32
            
            if let cnnGradientNode = gradientNode as? MPSCNNConvolutionGradientNode{
                cnnGradientNode.trainingStyle = trainingStyle
            }
            
            return gradientNode
        }
        
        guard let mpsGraph = MPSNNGraph(
            device: self.device,
            resultImage: lastOutput,
            resultImageIsNeeded: false) else{
                return nil
        }
        mpsGraph.outputStateIsTemporary = true
        mpsGraph.format = .float32
        
        return mpsGraph
    }
}
