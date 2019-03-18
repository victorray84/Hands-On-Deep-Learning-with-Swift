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
                learningRate:Float=0.0, //0.0002,
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
            //print(self.adversarialGraph!.graph.debugDescription)
            
            // create discriminator for training
            self.discriminatorGraph = self.createDiscriminator(.training)
        }
    }
    
}

// MARK: Inference

extension GAN{
    
    public func generateSamples(_ batchCount:Int, syncronizeWithCPU:Bool=false) -> [MPSImage]?{
        guard let commandBuffer = self.commandQueue.makeCommandBuffer(),
            let generator = self.generatorGraph else{
            return nil
        }
        
        guard let x = self.sampleGenerator.generate(batchCount) else{
            return nil
        }
        
        let samples = generator.graph.encodeBatch(
            to: commandBuffer,
            sourceImages: [x], sourceStates: nil)
        
        if syncronizeWithCPU,
            let samples = samples{
            for sample in samples{
                sample.synchronize(on: commandBuffer)
            }
        }
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        return samples
    }
}

// MARK: Training

extension GAN{
    
    public func asyncTrain(withDataLoader dataLoader:DataLoader,
                      epochs:Int = 100,
                      completionHandler handler: @escaping () -> Void){
        
        DispatchQueue.global(qos: .userInitiated).async {
            self.train(withDataLoader: dataLoader,
                           epochs:epochs,
                           completionHandler: handler)
        }
    }
    
    public func train(
        withDataLoader dataLoader:DataLoader,
        epochs:Int = 3,
        completionHandler handler: @escaping () -> Void){
        
        for epoch in 1...epochs{
            
            var trainingSteps : Float = 0.0
            var discriminatorLoss : Float = 0.0
            var adversarialLoss : Float = 0.0
            
            dataLoader.reset()
            
            while dataLoader.hasNext(){
                autoreleasepool{
                    if let loss = self.trainStep(withDataLoader:dataLoader){
                        trainingSteps += 1
                        discriminatorLoss += loss.discriminatorLoss
                        adversarialLoss += loss.adversarialLoss
                        
                        if Int(trainingSteps) % 5 == 0{
                            print("... discriminatorLoss \(discriminatorLoss/trainingSteps), adversarialLoss \(adversarialLoss/trainingSteps)")
                        }
                    }
                }
            }
            
            // Calculate the mean for both losses
            discriminatorLoss /= trainingSteps
            adversarialLoss /= trainingSteps
            
            // DEV
//            inspectDatasources(d:self.discriminatorGraph!.datasources,
//                               g:self.generatorGraph!.datasources,
//                               a:self.adversarialGraph!.datasources)
            
            // Generate sample images every n epochs
            print("Finished epoch \(epoch); discriminator loss \(discriminatorLoss), adversarial loss \(adversarialLoss)")
            
            if epoch == 1 || epoch == epochs || epoch % 5 == 0{
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
                
                // Reload weights
                self.discriminatorGraph?.graph.reloadFromDataSources()
                self.generatorGraph?.graph.reloadFromDataSources()
                self.adversarialGraph?.graph.reloadFromDataSources()
                
                self.generateImages(dataLoader.batchSize, forEpoch:epoch, withDataLoader: dataLoader)
                
                // DEVELOPMENT
                autoreleasepool { () -> Void in
                    testDiscriminator(dataLoader:dataLoader)
                }
            }
        }
        
        // Notify compeition handler that we have finished
        DispatchQueue.main.async {
            handler()
        }
    }
    
    @discardableResult
    func trainStep(withDataLoader dataLoader:DataLoader) -> (discriminatorLoss:Float, adversarialLoss:Float)?{
        var discriminatorLoss : Float = 0.0
        var adversarialLoss : Float = 0.0
        
        guard let discriminator = self.discriminatorGraph,
            let adversarial = self.adversarialGraph else{
                return nil
        }
        
        // Train the Discriminator
        
        // 1. get the inputs (x and y)
        if let trueImages = dataLoader.nextBatch(),
            let trueLabels = dataLoader.createLabels(withValue: 0.9),
            let falseImages = self.generateSamples(dataLoader.batchSize, syncronizeWithCPU: true),
            let falseLabels = dataLoader.createLabels(withValue: 0.0),
            let commandBuffer = self.commandQueue.makeCommandBuffer() {
            
            // Train the discriminator
            discriminator.datasources.forEach { (ds) in
                ds.trainable = true
            }
            
            discriminator.graph.encodeBatch(
                to: commandBuffer,
                sourceImages: [trueImages + falseImages],
                sourceStates: [trueLabels + falseLabels],
                intermediateImages: nil,
                destinationStates: nil)
            
            // Syncronise the parameters so they are available for the adversarial network
            // TODO: Check this is needed
            for ds in discriminator.datasources{
                ds.synchronizeParameters(on: commandBuffer)
            }
            
            // Syncronise the loss labels so we can get access to them
            // to return to the caller
            for label in trueLabels + falseLabels{
                label.synchronize(on: commandBuffer)
            }
            
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
            
            // Calculate and store the loss
            for label in trueLabels + falseLabels{
                if let loss = label.lossImage().toFloatArray(){
                    discriminatorLoss += loss[0]
                }
            }
            
            discriminatorLoss /= Float(trueLabels.count + falseLabels.count)
        }
        
        // Train the generative adversarial network (aka adversarial)
        for _ in 0..<2{
            if let commandBuffer = self.commandQueue.makeCommandBuffer(){
                // Disable 'learning' for the discriminator nodes
                discriminator.datasources.forEach { (ds) in
                    ds.trainable = false
                }
//                adversarial.datasources.filter { $0.name.starts(with: "d_") }.forEach { (ds) in
//                    ds.trainable = false
//                }
                
                if let x = self.sampleGenerator.generate(dataLoader.batchSize),
                    let y = dataLoader.createLabels(withValue: 1.0){
                    
                    adversarial.graph.encodeBatch(
                        to: commandBuffer,
                        sourceImages: [x],
                        sourceStates: [y],
                        intermediateImages: nil,
                        destinationStates: nil)
                    
                    // Syncronoise the weights so they are available to the generator network
                    // TODO: Check this is needed
                    for ds in adversarial.datasources.filter( { $0.trainable } ){
                        ds.synchronizeParameters(on: commandBuffer)
                    }
                    
                    // Syncronise the loss labels so we can get access to them
                    // to return to the caller
                    for label in y{
                        label.synchronize(on: commandBuffer)
                    }
                    
                    commandBuffer.commit()
                    commandBuffer.waitUntilCompleted()
                    
                    // Calculate and store the loss
                    for label in y{
                        if let loss = label.lossImage().toFloatArray(){
                            adversarialLoss += loss[0]
                        }
                    }
                    
                    adversarialLoss /= Float(y.count)
                }
            }
        }
        
        return (discriminatorLoss:discriminatorLoss, adversarialLoss:adversarialLoss)
    }
    
    func generateImages(_ sampleCount:Int, forEpoch epoch:Int, withDataLoader dataLoader:DataLoader){
        guard let generatedImages = self.generateSamples(sampleCount, syncronizeWithCPU: true),
            let exportURL = self.exportImagesURL else{
                return
        }
        
        for (idx, image) in generatedImages.enumerated(){
            if let nsImage = dataLoader.toNSImage(mpsImage: image){
                var url = exportURL
                url.appendPathComponent("generatedimage_\(epoch)-\(idx).png")
                
                nsImage.pngWrite(
                    to: url,
                    options: Data.WritingOptions.atomic)
                
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
            if datasource.trainable{
                datasource.saveParametersToDisk()
            }
        }
    }
}

// MARK: Development

extension GAN{
    
    @discardableResult
    func testDiscriminator(dataLoader:DataLoader) -> (Float, Float, Float) {
        guard let discriminator = createDiscriminator(.inference) else{
            return (-1, -1, -1)
        }
        
        var count : Float = 0
        var trueCorrectCount : Float = 0
        var falseCorrectCount : Float = 0
        
        dataLoader.reset()
        
        if let trueImages = dataLoader.nextBatch(),
            let falseImages = self.generateSamples(dataLoader.batchSize, syncronizeWithCPU: true){
            //let falseImages = dataLoader.createDummyInput(withValue: 0.0, count: dataLoader.batchSize){
            
            for image in trueImages{
                let array = image.toFloatArray()
                let tex = image.texture
                
                if let nsImage = dataLoader.toNSImage(mpsImage: image){
                    let _ = nsImage
                }
            }
            
            for image in falseImages{
                let array = image.toFloatArray()
                let tex = image.texture

                if let nsImage = dataLoader.toNSImage(mpsImage: image){
                    let _ = nsImage
                }
            }
            
            if let commandBuffer = self.commandQueue.makeCommandBuffer(){
                let outputs = discriminator.graph.encodeBatch(
                    to: commandBuffer,
                    sourceImages: [trueImages],
                    sourceStates: nil)
                
                outputs?.forEach({ (output) in
                    output.synchronize(on: commandBuffer)
                })
                commandBuffer.commit()
                commandBuffer.waitUntilCompleted()
                
                // Process outputs
                if let outputs = outputs{
//                    let array = outputs[0].toFloatArray()
                    
                    let predictions = outputs.map({ (output) -> Float in
                        if let probs = output.toFloatArray(){
                            return probs[0]
                        }
                        return -1.0
                    })
                    
                    count += Float(predictions.count)
                    trueCorrectCount = predictions.reduce(0, { (res, prob) -> Float in
                        return res + (prob > 0.5 ? 1 : 0)
                    })
                }
            }
            
            if let commandBuffer = self.commandQueue.makeCommandBuffer(){
                let outputs = discriminator.graph.encodeBatch(
                    to: commandBuffer,
                    sourceImages: [falseImages],
                    sourceStates: nil)
                
                outputs?.forEach({ (output) in
                    output.synchronize(on: commandBuffer)
                })
                commandBuffer.commit()
                commandBuffer.waitUntilCompleted()
                
                // Process outputs
                if let outputs = outputs{
                    let predictions = outputs.map({ (output) -> Float in
                        if let probs = output.toFloatArray(){
                            return probs[0]
                        }
                        return -1.0
                    })
                    
                    count += Float(predictions.count)
                    falseCorrectCount = predictions.reduce(0, { (res, prob) -> Float in
                        return res + (prob < 0.5 ? 1 : 0)
                    })
                }
            }
        }
        
        let accuracy = (trueCorrectCount + falseCorrectCount)/count
        let trueAccuracy = trueCorrectCount/(count/2)
        let falseAccuracy = falseCorrectCount/(count/2)
        
        print("total accuracy \(accuracy), true accuracy \(trueAccuracy), false accuracy \(falseAccuracy)")
        
        return (accuracy, trueAccuracy, falseAccuracy)
        
    }
    
    func inspectDatasources(d:[DataSource], g:[DataSource], a:[DataSource]){
        guard let commandBuffer = self.commandQueue.makeCommandBuffer() else{
            return
        }
        
        for datasource in g{
            //datasource.synchronizeParameters(on: commandBuffer)
        }
        
        let tmp3 = g.first { (ds) -> Bool in
            return ds.name == "g_conv_2"
        }
        
        for datasource in a{
            //datasource.synchronizeParameters(on: commandBuffer)
        }
        
        let tmp4 = a.first { (ds) -> Bool in
            return ds.name == "g_conv_2"
        }
        
        let tmp2 = a.first { (ds) -> Bool in
            return ds.name == "d_dense_1"
        }
        
        for datasource in d{
            datasource.synchronizeParameters(on: commandBuffer)
        }
        
        // compare d_dense_2 from d and a
        let tmp1 = d.first { (ds) -> Bool in
            return ds.name == "d_dense_1"
        }
        
        let tmp0 = d.first { (ds) -> Bool in
            return ds.name == "d_conv_2"
        }
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
//        if tmp0 != nil{
//            let weightsDataA = tmp0!.weightsAndBiasesState!.weights.toArray(type: Float.self)
//            
//            print("tmp0")
//            print(weightsDataA[20...30])
//        }
//        
//        if tmp1 != nil{
//            let weightsDataA = tmp1!.weightsAndBiasesState!.weights.toArray(type: Float.self)
//            
//            print("tmp1")
//            print(weightsDataA[100...110])
//        }
        
//        if tmp1 != nil && tmp2 != nil{
//            let weightsDataA = tmp1!.weightsAndBiasesState!.weights.toArray(type: Float.self)
//            let weightsDataB = tmp2!.weightsAndBiasesState!.weights.toArray(type: Float.self)
//
//            print("tmp1")
//            print(weightsDataA[0...20])
//
//            print("tmp2")
//            print(weightsDataB[0...20])
//        }
//
        if tmp3 != nil && tmp4 != nil{
            let weightsDataA = tmp3!.weightsAndBiasesState!.weights.toArray(type: Float.self)
            let weightsDataB = tmp4!.weightsAndBiasesState!.weights.toArray(type: Float.self)

            print("tmp3")
            print(weightsDataA[20...40])

            print("tmp4")
            print(weightsDataB[20...40])
        }
        
        print("Hello world!")
        
    }
    
}

// MARK: Activation factory

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

// MARK: Builder functions

extension GAN{
    
    //private func makeOptimizer() -> MPSNNOptimizerAdam?{
    private func makeOptimizer() -> MPSNNOptimizerAdam?{
        guard self.mode == .training else{
            return nil
        }
        
        let optimizerDescriptor = MPSNNOptimizerDescriptor(
            learningRate: self.learningRate,
            gradientRescale: 1.0,
            regularizationType: .None,
            regularizationScale: 0.0)
        
        let optimizer = MPSNNOptimizerAdam(
            device: self.device,
            beta1: Double(self.momentumScale),
            beta2: 0.999,
            epsilon: 1e-8,
            timeStep: 0,
            optimizerDescriptor: optimizerDescriptor)
        
        optimizer.options = MPSKernelOptions(arrayLiteral: MPSKernelOptions.verbose)
        
//        let optimizer = MPSNNOptimizerStochasticGradientDescent(
//            device: self.device,
//            momentumScale: self.momentumScale,
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
        let upscale = MPSCNNUpsamplingNearestNode(
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
                
                if let weightsVelocity = self.makeMPSVector(count: datasource.weightsLength),
                    let biasVelocity = self.makeMPSVector(count: datasource.biasTermsLength){

                    datasource.velocityVectors = [weightsVelocity, biasVelocity]
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
            kernelSize: KernelSize(width:5, height:5),
            strideSize: StrideSize(width:2, height:2),
            inputFeatureChannels: inputShape.channels,
            outputFeatureChannels: 64,
            datasources: &datasources,
            activationFunc: GAN.createRelu)
        
        lastOutput = layer1.last!.resultImage // 14x14x64
        nodes += layer1
        
        let layer2 = self.createConvLayer(
            name: "d_conv_2",
            x: lastOutput, // 14x14x64
            kernelSize: KernelSize(width:5, height:5),
            strideSize: StrideSize(width:2, height:2),
            inputFeatureChannels: 64,
            outputFeatureChannels: 128,
            datasources: &datasources,
            activationFunc: GAN.createRelu)
        
        lastOutput = layer2.last!.resultImage // 7x7x128
        nodes += layer2
        
        let layer3 = self.createDenseLayer(
            name: "d_dense_1",
            input: lastOutput, // 7x7x128
            kernelSize: KernelSize(width:7, height:7),
            inputFeatureChannels: 128,
            outputFeatureChannels: 256,
            datasources: &datasources,
            activationFunc: GAN.createRelu)
        
        lastOutput = layer3.last!.resultImage // 1x1x256
        nodes += layer3
        
        let layer4 = self.createDenseLayer(
            name: "d_dense_2",
            input: lastOutput,// 1x1x256
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
        
        // Scale
        let scale = MPSNNLanczosScaleNode(
            source: input,
            outputSize: MTLSize(
                width: self.inputShape.width,
                height: self.inputShape.height,
                depth: 1))
        
        // Create the forward pass
        let (nodes, datasources) = createDiscriminatorForwardPassNodes(
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
        
        //let trainingStyle = MPSNNTrainingStyle.updateDeviceGPU
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
            activationFunc: GAN.createRelu)
        
        lastOutput = layer1.last!.resultImage
        nodes += layer1
        
        // Reshape
        let reshapeNode = MPSNNReshapeNode(
            source: lastOutput,
            resultWidth: 7,
            resultHeight: 7,
            resultFeatureChannels: 128)
        
        lastOutput = reshapeNode.resultImage
        nodes += [reshapeNode]
        
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
            activationFunc:GAN.createRelu)
        
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
            activationFunc:GAN.createTanH)
        
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
        
        mpsGraph.format = .float32
        
        return Graph(mpsGraph, datasources, mode)
    }
}
