import Foundation
import AppKit
import MetalKit
import MetalPerformanceShaders
import CoreGraphics

public class GAN{
    
    /*:
     Data object to encapsulate MPSNNGraph and associated datasources
     */
    class Graph{
        
        let mode : NetworkMode
        let graph : MPSNNGraph
        let datasources : [ConvnetDataSource]
        
        init(_ graph:MPSNNGraph, _ datasources:[ConvnetDataSource], _ mode:NetworkMode){
            self.mode = mode
            self.graph = graph
            self.datasources = datasources
        }
    }
    

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
    
    public private(set) var momentumScale : Float = 0.5 // beta1
    
    public private(set) var latentSize: Int = 100
    
    var sharedDatasources = [String:ConvnetDataSource]()
    
    private var discriminatorGraph : Graph?
    
    private var generatorGraph : Graph?
    private var adversarialGraph : Graph?
    
    private var sampleGenerator : GANSampleGenerator
    
    private var exportImagesURL : URL?
    
    // featureMapSize:Int, useBatchNorm:Bool
    
    static func createGAN(withCommandQueue commandQueue:MTLCommandQueue,
                       weightsPathURL:URL,
                       exportImagesURL:URL?=nil,
                       inputShape:Shape=(width:28, height:28, channels:1),
                       latentSize: Int = 100,
                       mode:NetworkMode = NetworkMode.training) -> GAN{
        
        return GAN(withCommandQueue: commandQueue,
                   weightsPathURL: weightsPathURL,
                   exportImagesURL:exportImagesURL,
                   inputShape:inputShape,
                   latentSize:latentSize,
                   mode:mode)
    }
    
    public init(withCommandQueue commandQueue:MTLCommandQueue,
                weightsPathURL:URL,
                exportImagesURL:URL?=nil,
                inputShape:Shape=(width:28, height:28, channels:1),
                latentSize: Int = 100,
                mode:NetworkMode = NetworkMode.training,
                learningRate:Float=0.005,
                momentumScale:Float=0.5){
        
        self.commandQueue = commandQueue
        self.device = commandQueue.device
        self.weightsPathURL = weightsPathURL
        self.exportImagesURL = exportImagesURL
        self.mode = mode
        self.learningRate = learningRate
        self.momentumScale = momentumScale
        
        self.sampleGenerator = PooledGANSampleGenerator(self.device, self.latentSize)
        
        // Create generator (for inference)
        self.generatorGraph = self.createGenerator(.inference)
        
        //print(self.generatorForInference.debugDescription)
        
        if self.mode == .training{
            // if training then ...
            // create a generator network for training (including the backprop)
            self.adversarialGraph = self.createGenerator(.training)
            // create discriminator for training
            self.discriminatorGraph = self.createDiscriminator(.training)
        }
    }
    
}

// MARK: Inference

extension GAN{
    
    public func generateSamples(_ batchCount:Int) -> [MPSImage]?{
        guard let commandBuffer = self.commandQueue.makeCommandBuffer(),
            let network = self.generatorGraph else{
            return nil
        }
        
        guard let x = self.sampleGenerator.generate(batchCount) else{
            return nil
        }
        
        let samples = network.graph.encodeBatch(
            to: commandBuffer,
            sourceImages: [x], sourceStates: nil)
        
        commandBuffer.commit()
        
        commandBuffer.waitUntilCompleted()
        
        return samples
    }
}

// MARK: Training

extension GAN{
    
    public func asyncTrain(withDataLoader dataLoader:DataLoader,
                      epochs:Int = 5,
                      completionHandler handler: @escaping () -> Void){
        
        DispatchQueue.global(qos: .userInitiated).async {
            self.train(withDataLoader: dataLoader,
                           epochs:epochs,
                           completionHandler: handler)
        }
    }
    
    public func train(
        withDataLoader dataLoader:DataLoader,
        epochs:Int = 1000,
        completionHandler handler: @escaping () -> Void){
        
        print("Starting training")
        
        for epoch in 1...epochs{
            dataLoader.reset()
            
            while dataLoader.hasNext(){
                autoreleasepool{
                    self.trainStep(withDataLoader:dataLoader)
                }
            }
            
            // Make the associated datasources to the discriminator trainable
            // so they are persisted
            self.discriminatorGraph!.datasources.forEach { (ds) in
                ds.trainable = true
            }
            
            updateDatasources(self.discriminatorGraph!.datasources)
            
            // Make the associated datasources to the discriminator NOT trainable
            // so they are NOT persisted
            self.discriminatorGraph!.datasources.forEach { (ds) in
                ds.trainable = false
            }
            
            updateDatasources(self.adversarialGraph!.datasources)
            
            // Generate sample images every n epochs
            if epoch == 1 || epoch % 5 == 0 {
                print("Finished epoch \(epoch)")
                
                self.generateImages(dataLoader.batchSize, forEpoch:epoch)
            }
        }
        
        // Notify compeition handler that we have finished
        DispatchQueue.main.async {
            handler()
        }
    }
    
    func trainStep(withDataLoader dataLoader:DataLoader){
        
        guard let discriminator = self.discriminatorGraph,
            let adversarial = self.adversarialGraph else{
                return
        }
        
        // Train the Discriminator
        
        // 1. get the inputs (x and y)
        if let trueImages = dataLoader.nextBatch(),
            let trueLabels = dataLoader.createLabels(withValue: 1.0),
            let falseImages = self.generateSamples(dataLoader.batchSize),
            let falseLabels = dataLoader.createLabels(withValue: 0.0),
            let commandBuffer = self.commandQueue.makeCommandBuffer() {
            
            // Train the discriminator
            discriminator.datasources.forEach { (ds) in
                ds.trainable = true
            }
            
            discriminator.graph.encodeBatch(to: commandBuffer,
                                            sourceImages: [trueImages + falseImages],
                                            sourceStates: [trueLabels + falseLabels])
            
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
        }
        
        // Train the generative adversarial network (aka adversarial) x2
        
        // disable 'learning' for the discriminator nodes
        adversarial.datasources.filter { $0.name.starts(with: "d_") }.forEach { (ds) in
            ds.trainable = false
        }
        
        for _ in 0..<2{
            if let commandBuffer = self.commandQueue.makeCommandBuffer(),
                let x = self.sampleGenerator.generate(dataLoader.batchSize),
                let y = dataLoader.createLabels(withValue: Float.random(in: 0.8...1.0)){
                
                
                adversarial.graph.encodeBatch(to: commandBuffer,
                                              sourceImages: [x],
                                              sourceStates: [y])
                
                commandBuffer.commit()
                commandBuffer.waitUntilCompleted()
            }
        }
    }
    
    func generateImages(_ sampleCount:Int, forEpoch epoch:Int){
        guard let generatedImages = self.generateSamples(sampleCount),
            let exportURL = self.exportImagesURL else{
                return
        }
        
        print("TODO")
        
        for (idx, image) in generatedImages.enumerated(){
            if let floatArray = image.toFloatArray(){
                print("hello")
                //                let byteArray = floatArray.map { UInt8($0 * 255.0) }
                //                let bytes = Data(fromArray: byteArray)
                //                if let bitmapImageRep = NSBitmapImageRep(data: bytes){
                //                    let pngData = bitmapImageRep.representation(using: .png, properties: [:])
                //                    do {
                //                        try pngData?.write(
                //                            to: exportURL.appendingPathComponent("generatedimage_(\(epoch)_\(idx).png"),
                //                            options: .atomic)
                //                    } catch { }
                //                }
            }
        }
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
            datasource.saveParametersToDisk()
        }
    }
}

// MARK: Activation factory

extension GAN{
    
    static func createRelu(_ x:MPSNNImageNode, _ name:String) -> MPSCNNNeuronNode{
        let activation = MPSCNNNeuronReLUNode(source: x)
        activation.resultImage.format = .float16
        activation.label = name
        return activation
    }
    
    static func createLeakyRelu(_ x:MPSNNImageNode, _ name:String) -> MPSCNNNeuronNode{
        let activation = MPSCNNNeuronReLUNode(source: x, a:0.2)
        activation.resultImage.format = .float16
        activation.label = name
        return activation
    }
    
    static func createSigmoid(_ x:MPSNNImageNode, _ name:String) -> MPSCNNNeuronNode{
        let activation = MPSCNNNeuronSigmoidNode(source:x)
        activation.resultImage.format = .float16
        activation.label = name
        return activation
    }
    
    static func createTanH(_ x:MPSNNImageNode, _ name:String) -> MPSCNNNeuronNode{
        let activation = MPSCNNNeuronTanHNode(source: x)
        activation.resultImage.format = .float16
        activation.label = name
        return activation
    }
    
}

// MARK: Builder functions

extension GAN{
    
    private func makeOptimizer() -> MPSNNOptimizerAdam?{
        guard self.mode == .training else{
            return nil
        }
        
        let optimizerDescriptor = MPSNNOptimizerDescriptor(
            learningRate: self.learningRate,
            gradientRescale: 1.0,
            regularizationType: .None,
            regularizationScale: 1.0)
        
        let optimizer = MPSNNOptimizerAdam(
            device: self.device,
            beta1: Double(self.momentumScale),
            beta2: 0.999,
            epsilon: 1e-8,
            timeStep: 0,
            optimizerDescriptor: optimizerDescriptor)
        
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
        if self.sharedDatasources[name] == nil{
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
                
                if let weightsVelocity = self.makeMPSVector(count: datasource.weightsLength),
                    let biasVelocity = self.makeMPSVector(count: datasource.biasTermsLength){
                    
                    datasource.velocityVectors = [weightsVelocity, biasVelocity]
                }
            }
            
            self.sharedDatasources[name] = datasource
        }
    
        let datasource = self.sharedDatasources[name]!
        
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
    
    func createTransposeConvLayer(name:String,
                         x:MPSNNImageNode,
                         kernelSize:KernelSize = KernelSize(width:5, height:5),
                         strideSize:KernelSize = KernelSize(width:1, height:1),
                         inputFeatureChannels:Int,
                         outputFeatureChannels:Int,
                         datasources:inout [ConvnetDataSource],
                         upscale:Int=2,
                         activationFunc:((MPSNNImageNode, String) -> MPSCNNNeuronNode)? = nil) -> [MPSNNFilterNode]{
        
        var layers = [MPSNNFilterNode]()
        
        // upscale image
        let upscale = MPSCNNUpsamplingBilinearNode(
            source: x,
            integerScaleFactorX: upscale,
            integerScaleFactorY: upscale)
        upscale.label = "\(name)_upscale"
        
        layers.append(upscale)
        
        // We are sharing datasources between our networks (specifically the Discriminator + GAN,
        // Generator + GAN - it's for this reason we cache them
        if self.sharedDatasources[name] == nil{
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
            
            self.sharedDatasources[name] = datasource
        }
    
        let datasource = self.sharedDatasources[name]!
        
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
        if self.sharedDatasources[name] == nil{
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
                
                if let weightsVelocity = self.makeMPSVector(count: datasource.weightsLength),
                    let biasVelocity = self.makeMPSVector(count: datasource.biasTermsLength){
                    
                    datasource.velocityVectors = [weightsVelocity, biasVelocity]
                }
                
            }
            
            self.sharedDatasources[name] = datasource
        }
        
        let datasource = self.sharedDatasources[name]!
        
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

extension GAN{
    
    func createDiscriminatorForwardPassNodes(
        _ x:MPSNNImageNode? = nil,
        _ inputShape:Shape?=nil) -> (nodes:[MPSNNFilterNode], datasources:[ConvnetDataSource]){
        
        var nodes = [MPSNNFilterNode]()
        var datasources = [ConvnetDataSource]()
        
        let inputShape = inputShape ?? self.inputShape
        
        // keep track of the last input
        var lastOutput : MPSNNImageNode = x ?? MPSNNImageNode(handle: nil)
        
        // === Forward pass ===
        let layer1 = self.createConvLayer(
            name: "d_conv_1",
            x: lastOutput, // 28x28x1
            kernelSize: KernelSize(width:5, height:4),
            strideSize: StrideSize(width:2, height:2),
            inputFeatureChannels: inputShape.channels,
            outputFeatureChannels: 64,
            datasources: &datasources,
            activationFunc: GAN.createLeakyRelu)
        
        lastOutput = layer1.last!.resultImage
        nodes += layer1
        
        let layer2 = self.createConvLayer(
            name: "d_conv_2",
            x: lastOutput, // 14x14x64
            kernelSize: KernelSize(width:5, height:4),
            strideSize: StrideSize(width:2, height:2),
            inputFeatureChannels: 64,
            outputFeatureChannels: 128,
            datasources: &datasources,
            activationFunc: GAN.createLeakyRelu)
        
        lastOutput = layer2.last!.resultImage
        nodes += layer2
        
        let layer3 = self.createDenseLayer(
            name: "d_dense_1",
            input: lastOutput,
            kernelSize: KernelSize(width:7, height:7),
            inputFeatureChannels: 128,
            outputFeatureChannels: 256,
            datasources: &datasources,
            activationFunc: GAN.createLeakyRelu)
        
        lastOutput = layer3.last!.resultImage
        nodes += layer3
        
        let layer4 = self.createDenseLayer(
            name: "d_dense_2",
            input: lastOutput,
            kernelSize: KernelSize(width:1, height:1),
            inputFeatureChannels: 256,
            outputFeatureChannels: 1,
            datasources: &datasources,
            activationFunc: GAN.createSigmoid)
        
        lastOutput = layer4.last!.resultImage
        nodes += layer4
        
        return (nodes:nodes, datasources:datasources)
    }
    
    func createDiscriminator(_ mode:NetworkMode) -> Graph?{
        // Create placeholder for our input into the network
        let input = MPSNNImageNode(handle: nil)
        
        // Create the forward pass
        let (nodes, datasources) = createDiscriminatorForwardPassNodes(input, self.inputShape)
        
        // Obtain reference to the last node 
        var lastOutput = nodes.last!.resultImage
        
        if mode == .inference{
            guard let mpsGraph = MPSNNGraph(
                device: self.device,
                resultImage: lastOutput,
                resultImageIsNeeded: true) else{
                    return nil
            }
            
            return Graph(mpsGraph, datasources, mode)
        }
        
        // === Loss function ===
        let lossDesc = MPSCNNLossDescriptor(
            type: MPSCNNLossType.sigmoidCrossEntropy,
            reductionType: MPSCNNReductionType.mean)
        
        let loss = MPSCNNLossNode(
            source: lastOutput,
            lossDescriptor: lossDesc)
        
        loss.resultImage.format = .float32
        
        lastOutput = loss.resultImage
        
        // === Backwards pass ===
        let _ = nodes.reversed().map { (node) -> MPSNNGradientFilterNode in
            let gradientNode = node.gradientFilter(withSource: lastOutput)
            
            lastOutput = gradientNode.resultImage
            lastOutput.format = .float32
            
            return gradientNode
        }
        
        guard let mpsGraph = MPSNNGraph(
            device: self.device,
            resultImage: lastOutput,
            resultImageIsNeeded: false) else{
                return nil
        }
        
        return Graph(mpsGraph, datasources, mode)
    }
}

// MARK: Generator

extension GAN{
    
    func createGenerator(_ mode:NetworkMode) -> Graph?{
        guard #available(OSX 10.14.1, *) else {
            return nil
        }
        
        var datasources = [ConvnetDataSource]()
        
        var nodes = [MPSNNFilterNode]()
        
        // Input placeholder
        let input = MPSNNImageNode(handle: nil)
        
        // keep track of the last input
        var lastOutput : MPSNNImageNode = input
        
        // Dense layers
        let layer1 = self.createDenseLayer(
            name: "g_dense_1",
            input: lastOutput,
            kernelSize: KernelSize(width:self.latentSize, height:1),
            inputFeatureChannels: 1,
            outputFeatureChannels: 7 * 7 * 128,
            datasources: &datasources,
            activationFunc: GAN.createLeakyRelu)
        
        lastOutput = layer1.last!.resultImage
        nodes += layer1
        
        // Reshape
        let reshapeNode = MPSNNReshapeNode(
            source: lastOutput,
            resultWidth: 7,
            resultHeight: 7,
            resultFeatureChannels: 128)
        
        nodes += [reshapeNode]        
        lastOutput = reshapeNode.resultImage
        
        // Conv layerss
        let layer2 = self.createTransposeConvLayer(
            name: "g_conv_1",
            x: lastOutput,
            kernelSize:KernelSize(width:5, height:5),
            strideSize:StrideSize(width:1, height:1),
            inputFeatureChannels: 128,
            outputFeatureChannels: 64,
            datasources: &datasources,
            upscale: 2,
            activationFunc:GAN.createLeakyRelu)
        
        lastOutput = layer2.last!.resultImage
        nodes += layer2
        
        let layer3 = self.createTransposeConvLayer(
            name: "g_conv_2",
            x: lastOutput,
            kernelSize:KernelSize(width:5, height:5),
            strideSize:StrideSize(width:1, height:1),
            inputFeatureChannels: 64,
            outputFeatureChannels: 1,
            datasources: &datasources,
            upscale: 2,
            activationFunc:GAN.createSigmoid)
        
        lastOutput = layer3.last!.resultImage
        nodes += layer3
        
        if mode == .inference{
            guard let mpsGraph = MPSNNGraph(
                device: self.device,
                resultImage: lastOutput,
                resultImageIsNeeded: true) else{
                    return nil
            }
            
            return Graph(mpsGraph, datasources, mode)
        }
        
        // Let's now attach the discriminator to our generator network
        let (discriminatorNodes, discriminatorDatasources) = self.createDiscriminatorForwardPassNodes(
            lastOutput, self.inputShape)
        
        discriminatorDatasources.forEach { (ds) in
            ds.trainable = false
            datasources.append(ds)
        }
        
        nodes += discriminatorNodes
        
        lastOutput = discriminatorNodes.last!.resultImage
        
        // === Loss function ===
        let lossDesc = MPSCNNLossDescriptor(
            type: MPSCNNLossType.sigmoidCrossEntropy,
            reductionType: MPSCNNReductionType.mean)
        
        let loss = MPSCNNLossNode(
            source: lastOutput,
            lossDescriptor: lossDesc)
        
        loss.resultImage.format = .float32
        
        lastOutput = loss.resultImage
        
        // === Backwards pass ===
        let _ = nodes.reversed().map { (node) -> MPSNNGradientFilterNode in
            let gradientNode = node.gradientFilter(withSource: lastOutput)
            
            lastOutput = gradientNode.resultImage
            lastOutput.format = .float32
            
            return gradientNode
        }
        
        guard let mpsGraph = MPSNNGraph(
            device: self.device,
            resultImage: lastOutput,
            resultImageIsNeeded: false) else{
                return nil
        }
        
        return Graph(mpsGraph, datasources, mode)
    }
}
