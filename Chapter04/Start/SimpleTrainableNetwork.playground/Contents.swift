/*:
 
 # [Hands-On Deep Learning with Swift]()
 ### Chapter 4 - Metal for Machine Learning
 *Writen by [Joshua Newnham](https://www.linkedin.com/in/joshuanewnham) and published by [Packt Publishing](https://www.packtpub.com/big-data-and-business-intelligence/machine-learning-core-ml)*
 */

import Foundation
import AppKit
import MetalKit
import MetalPerformanceShaders
import Accelerate
import AVFoundation
import PlaygroundSupport
import CoreGraphics

// Required to run tasks in the background
PlaygroundPage.current.needsIndefiniteExecution = true

typealias KernelSize = (width:Int, height:Int)

typealias Sample = (image:MPSImage, label:MPSCNNLossLabels)

typealias Batch = (images:[MPSImage], labels:[MPSCNNLossLabels])

// MARK: - DataLoader

/*
 Class responsible for providing our model with data to sample from
 */
class DataLoader{
    
    public let channelFormat = MPSImageFeatureChannelFormat.unorm8
    public let imageWidth  = 2
    public let imageHeight = 2
    public let featureChannels = 1
    public let imagePixelsCount = 4
    public let numberOfClasses = 4
    
    /*
     A MPSImageDescriptor object describes a attributes of MPSImage and is used to
     create one (see MPSImage discussion below)
     */
    lazy var imageDescriptor : MPSImageDescriptor = {
        var imageDescriptor = MPSImageDescriptor(
            channelFormat:self.channelFormat,
            width: self.imageWidth,
            height: self.imageHeight,
            featureChannels:self.featureChannels)
        return imageDescriptor
    }()
    
    public var images = [UInt8]()
    public var labels = [UInt8]()
    
    public var count : Int{
        get{
            return self.images.count
        }
    }
    
    init?() {
        for i in 0..<4{
            for j in 0..<4{
                images.append((i == j) ? UInt8(255) : UInt8(0))
            }
        }
        
        for i in 0..<4{
            labels.append(UInt8(i))
        }
        
        print("Initilised \(self.images.count) images and \(self.labels.count) labels")
    }
    
    public func getImage(withDevice device:MTLDevice, atIndex index:Int) -> MPSImage?{
        // Create image
        let image = MPSImage(device:device, imageDescriptor: self.imageDescriptor)
        
        return setImageData(image, withDataFromIndex: index)
    }
    
    private func setImageData(_ image:MPSImage, withDataFromIndex index:Int) -> MPSImage?{
        // Calculate start and end index
        let sIdx = index * self.imagePixelsCount
        let eIdx = sIdx + self.imagePixelsCount
        
        // Check that the indecies are within the bounds of our array
        guard sIdx >= 0, eIdx <= self.images.count else {
            return nil
        }
        
        // Get image data
        var imageData = [UInt8]()
        imageData += self.images[(sIdx..<eIdx)]
        
        // Copy the image data into the image
        image.texture.replace(
            region: MTLRegion(origin: MTLOrigin(x: 0,
                                                y: 0,
                                                z: 0),
                              size: MTLSize(width: self.imageWidth,
                                            height: self.imageHeight,
                                            depth: 1)),
            mipmapLevel: 0,
            slice: 0,
            withBytes: imageData,
            bytesPerRow: self.imageWidth * self.featureChannels,
            bytesPerImage: 0)
        
        return image
    }
    
    public func getLabel(withDevice device:MTLDevice,
                         atIndex index:Int) -> MPSCNNLossLabels?{
        guard index >= 0, index < self.labels.count else {
            return nil
        }
        
        let labelIndex = Int(self.labels[index])
        
        // TODO create and return label
        return nil
    }
    
    public func getSample(withDevice device:MTLDevice, atIndex index:Int) -> Sample?{
        guard let image = self.getImage(withDevice: device, atIndex: index),
            let label = self.getLabel(withDevice: device, atIndex: index) else{
                return nil
        }
        
        return Sample(image:image, label:label)
    }
    
    public func getSamples(withDevice device:MTLDevice) -> Batch?{
        var images = [MPSImage]()
        var labels = [MPSCNNLossLabels]()
        
        for index in 0..<self.labels.count{
            if Int(index) < 0 || Int(index) >= self.count{
                break
            }
            
            if let sample = self.getSample(withDevice: device, atIndex: index){
                images.append(sample.image)
                labels.append(sample.label)
            }
        }
        
        return (images:images, labels:labels)
    }
}

// MARK: - Datasource

class Datasource : NSObject, MPSCNNConvolutionDataSource{
    
    static let FolderName = "SimpleWeights"
    
    let name : String
    let kernelSize : KernelSize
    let inputFeatureChannels : Int
    let outputFeatureChannels : Int
    
    var optimizer : MPSNNOptimizerStochasticGradientDescent?
    var weightsAndBiasesState : MPSCNNConvolutionWeightsAndBiasesState?
    
    var weightsData : Data?
    
    lazy var cnnDescriptor : MPSCNNConvolutionDescriptor = {
        let descriptor = MPSCNNConvolutionDescriptor(
            kernelWidth: self.kernelSize.width,
            kernelHeight: self.kernelSize.height,
            inputFeatureChannels: self.inputFeatureChannels,
            outputFeatureChannels: self.outputFeatureChannels)
        
        return descriptor
    }()
    
    init(name:String, kernelSize:KernelSize,
         inputFeatureChannels:Int, outputFeatureChannels:Int,
         optimizer:MPSNNOptimizerStochasticGradientDescent? = nil) {
        self.name = name
        self.kernelSize = kernelSize
        self.inputFeatureChannels = inputFeatureChannels
        self.outputFeatureChannels = outputFeatureChannels
        self.optimizer = optimizer
    }
    
    func label() -> String? {
        return self.name
    }
    
    func dataType() -> MPSDataType {
        return MPSDataType.float32
    }
    
    func descriptor() -> MPSCNNConvolutionDescriptor {
        return self.cnnDescriptor
    }
    
    func purge() {
        self.weightsData = nil
    }
    
    public func weights() -> UnsafeMutableRawPointer{
        return UnsafeMutableRawPointer(mutating: (self.weightsData! as NSData).bytes)
    }
    
    public func biasTerms() -> UnsafeMutablePointer<Float>?{
        return nil
    }
    
    func copy(with zone: NSZone? = nil) -> Any {
        let copy = Datasource(
            name: self.name,
            kernelSize: self.kernelSize,
            inputFeatureChannels: self.inputFeatureChannels,
            outputFeatureChannels: self.outputFeatureChannels,
            optimizer: self.optimizer)
        
        copy.weightsAndBiasesState = self.weightsAndBiasesState
        return copy as Any
    }
}

// MARK: Datasource load methods

extension Datasource{
    
    func load() -> Bool {
        print("load")
        self.weightsData = self.loadWeights()
        return self.weightsData != nil
    }

    private func loadWeights() -> Data?{
        let url = playgroundSharedDataDirectory
            .appendingPathComponent("\(Datasource.FolderName)/\(self.name)_conv.data")
        
        do{
            return try Data(contentsOf:url)
        } catch{
            // TODO initilize weights with random values
            return nil
        }
    }
}

// MARK: Datasource update methods

extension Datasource{
    
    /*
     Update method called on the CPU (set via the Nodes trainingStyle property)
    */
    func update(with gradientState: MPSCNNConvolutionGradientState,
                sourceState: MPSCNNConvolutionWeightsAndBiasesState) -> Bool {
        return false
    }
    
    /*
     Update method called on the CPU (set via the Nodes trainingStyle property)
    */
    func update(with commandBuffer: MTLCommandBuffer,
                gradientState: MPSCNNConvolutionGradientState,
                sourceState: MPSCNNConvolutionWeightsAndBiasesState) -> MPSCNNConvolutionWeightsAndBiasesState? {
        
        // TODO Handle update on the GPU
        
        return nil
    }
    
    func synchronizeParameters(on commandBuffer:MTLCommandBuffer){
        guard let weightsAndBiasesState = self.weightsAndBiasesState else{
            return
        }
        
        // TODO syncronize weights
    }
}

// MARK: Datasource save

extension Datasource{
    
    private func checkFolderExists(atPath:URL) -> Bool{
        var isDirectory = ObjCBool(true)
        if !FileManager.default.fileExists(atPath: atPath.path, isDirectory: &isDirectory){
            
            do {
                try FileManager.default.createDirectory(at: atPath, withIntermediateDirectories: false, attributes: nil)
            } catch let error as NSError {
                print(error.localizedDescription);
                return false
            }
        }
        
        return true
    }
    
    func updateAndSaveParametersToDisk(){
        guard let weightsAndBiasesState = self.weightsAndBiasesState else{
            return
        }
        
        // TODO Update local weights with adjusted weights saved in weightsAndBiasesState
        
        self.saveToDisk()
    }
    
    func saveToDisk() -> Bool{
        print("saveToDisk")
        return self.saveWeightsToDisk()
    }
    
    func saveWeightsToDisk() -> Bool{
        print("saveWeightsToDisk")
        guard let data = self.weightsData else{
            return false
        }
        
        // check the folder exists
        self.checkFolderExists(atPath: playgroundSharedDataDirectory.appendingPathComponent("\(Datasource.FolderName)"))
        
        let url = playgroundSharedDataDirectory.appendingPathComponent("\(Datasource.FolderName)/\(self.name)_conv.data")
        
        do{
            try data.write(to: url, options: NSData.WritingOptions.atomicWrite)
            print("Saved weights to \(url.absoluteString)")
            return true
        } catch{
            print("Failed to save weights to disk \(error)")
            return false
        }
    }
}

// MARK: - Network facade

class Network{
    
    enum NetworkMode{
        case training
        case inference
    }
    
    let device : MTLDevice
    let commandQueue : MTLCommandQueue
    
    var mode : NetworkMode = NetworkMode.training
    
    var graph : MPSNNGraph?
    
    var datasources = [Datasource]()
    
    init(withCommandQueue commandQueue:MTLCommandQueue, mode:NetworkMode=NetworkMode.training){
        self.device = commandQueue.device
        self.mode = mode
        self.commandQueue = commandQueue
        
        self.graph = mode == .training ? self.createTrainingGraph() : self.createInferenceGraph()
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
                    handler(outputImage!.toFloatArray())
                    return
                }
                
                handler(nil)
            }
        }
    }
    
    public func train(
        withDataLoader dataLoader:DataLoader,
        epochs : Int = 500,
        completionHandler handler: @escaping () -> Void) -> Bool{
        
        // TODO implement training set
        
        return true
    }
    
    
    private func createTrainingGraph() -> MPSNNGraph?{
        // TODO Create training graph
        return nil
    }
    
    private func createInferenceGraph() -> MPSNNGraph?{
        // TODO Create inference graph
        return nil
    }
}

// MARK: - Performing inference

// Create device
guard let device = MTLCreateSystemDefaultDevice() else{
    fatalError("Failed to reference GPU")
}

// Make sure the current device supports MetalPerformanceShaders
guard MPSSupportsMTLDevice(device) else{
    fatalError("Metal Performance Shaders not supported for current device")
}

// Create command queue
guard let commandQueue = device.makeCommandQueue() else{
    fatalError("Failed to create command queue")
}

guard let dataLoader = DataLoader() else{
    fatalError("Failed to create an instance of a DataLoader")
}

// Get a couple of samples that we will use for inference
let sample = dataLoader.getSample(withDevice: device, atIndex: 0)
let sample2 = dataLoader.getSample(withDevice: device, atIndex: 3)

// Perform inference using our network with a untrained model
let untrainedNetworkForInference = Network(withCommandQueue: commandQueue, mode: .inference)
untrainedNetworkForInference.predict(x: sample!.image, completationHandler: { (result) in
    print("Results from **untrained** network for label \(sample!.label.label!)")
    print(result!)
})

untrainedNetworkForInference.predict(x: sample2!.image, completationHandler: { (result) in
    print("Results from **untrained** network for label \(sample2!.label.label!)")
    print(result!)
})

// Train our model 
let trainingNetwork = Network(withCommandQueue: commandQueue)
trainingNetwork.train(withDataLoader: dataLoader) {
    print("Finished training!")
}

// Perform inference using our network with a trained model
let trainedNetworkForInference = Network(withCommandQueue: commandQueue, mode: .inference)
trainedNetworkForInference.predict(x: sample!.image, completationHandler: { (result) in
    print("Results from **trained** network for label \(sample!.label.label!)")
    print(result!)
})

trainedNetworkForInference.predict(x: sample2!.image, completationHandler: { (result) in
    print("Results from **trained** network for label \(sample2!.label.label!)")
    print(result!)
})

