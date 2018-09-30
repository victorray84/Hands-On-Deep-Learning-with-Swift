/*:
 
 # [Hands-On Deep Learning with Swift]()
 ### Chapter 3 - Metal for Machine Learning
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

let MNIST_IMAGE_WIDTH = 28
let MNIST_IMAGE_HEIGHT = 28
let MNIST_FEATURE_CHANNELS = 1 // grayscale
let MNIST_NUMBER_OF_CLASSES = 10 // 0 - 9

typealias KernelSize = (width:Int, height:Int)

public class NodeDataSource : NSObject, MPSCNNConvolutionDataSource{
    
    let name : String
    let kernelSize : KernelSize
    let inputFeatureChannels : Int
    let outputFeatureChannels : Int
    
    var weightsData : Data?
    var biasTermsData : Data?

    internal lazy var nodeDescriptor : MPSCNNConvolutionDescriptor = {
        let desc = MPSCNNConvolutionDescriptor(
            kernelWidth: self.kernelSize.width,
            kernelHeight: self.kernelSize.height,
            inputFeatureChannels: self.inputFeatureChannels,
            outputFeatureChannels: self.outputFeatureChannels)
        
        return desc
    }()
    
    init(name:String,
         kernelSize:KernelSize,
         inputFeatureChannels:Int,
         outputFeatureChannels:Int) {
        self.name = name
        self.kernelSize = kernelSize
        self.inputFeatureChannels = inputFeatureChannels
        self.outputFeatureChannels = outputFeatureChannels
    }
    
    public func label() -> String?{
        return self.name
    }

    public func dataType() -> MPSDataType{
        return MPSDataType.float32
    }
    
    public func descriptor() -> MPSCNNConvolutionDescriptor{
        return self.nodeDescriptor
    }
    
    public func load() -> Bool{
        self.weightsData = self.loadWeights()
        self.biasTermsData = self.loadBiasTerms()
        
        return self.weightsData != nil
    }

    private func loadWeights() -> Data?{
        return self.loadFile("/weights/\(self.name)_wts.data")
        
    }

    private func loadBiasTerms() -> Data?{
        return self.loadFile("/weights/\(self.name)_bias_terms.data")
    }

    private func loadFile(_ path:String) -> Data?{
        let pathComponentns = path.components(separatedBy: ".")
        
        guard let url = Bundle.main.url(
            forResource: pathComponentns[0],
            withExtension: pathComponentns[1]) else{
                return nil
        }
        
        return try? Data(contentsOf: url)
    }

    public func purge(){
        self.weightsData = nil
        self.biasTermsData = nil
    }
    
    public func weights() -> UnsafeMutableRawPointer{
        return UnsafeMutableRawPointer(mutating: (self.weightsData! as NSData).bytes)
    }

    public func biasTerms() -> UnsafeMutablePointer<Float>?{
        guard let biasTermsData = self.biasTermsData else{
            return nil
        }
        
        return UnsafeMutableRawPointer(
            mutating: (biasTermsData as NSData).bytes).bindMemory(
                to: Float.self,
                capacity: self.outputFeatureChannels * MemoryLayout<Float>.size)
    }
    
    public func copy(with zone:NSZone? = nil) -> Any{
        return NodeDataSource(
            name: self.name,
            kernelSize: self.kernelSize,
            inputFeatureChannels: self.inputFeatureChannels,
            outputFeatureChannels: self.outputFeatureChannels) as Any
    }
}

class Network{
    
    let device : MTLDevice
    let commandQueue : MTLCommandQueue
    
    var graph : MPSNNGraph?
    
    init(withCommandQueue commandQueue:MTLCommandQueue,
         graphBuilder:(MTLDevice) -> MPSNNGraph?) {
        self.commandQueue = commandQueue
        self.device = self.commandQueue.device
        
        self.graph = graphBuilder(self.device)
    }
    
    public func forward(
        inputImage:MPSImage,
        completationHandler handler: @escaping ([Float]?) -> Void) -> Void{
        
        guard let graph = self.graph else{
            return
        }

        graph.executeAsync(withSourceImages: [inputImage]) { (outputImage, error) in
            DispatchQueue.main.async {
                if error != nil{
                    print(error!)
                    handler(nil)
                    return
                }
                
                if outputImage != nil{
                    handler(self.getProbabilities(fromImage:outputImage!))
                    return
                }
                
                handler(nil)
            }
        }
    }
    
    private func getProbabilities(fromImage image:MPSImage) -> [Float]?{
        /*
         An MPSImage object can contain multiple CNN images for batch processing. In order
         to create an MPSImage object that contains N images, create an MPSImageDescriptor object
         with the numberOfImages property set to N. The length of the 2D texture array (i.e.
         the number of slices) will be equal to ((featureChannels+3)/4)*numberOfImages,
         where consecutive (featureChannels+3)/4 slices of this array represent one image.
        */
        let numberOfSlices = ((image.featureChannels + 3)/4) * image.numberOfImages
        
        /*
         If featureChannels<=4 and numberOfImages=1 (i.e. only one slice is needed to represent the image),
         the underlying metal texture type is chosen to be MTLTextureType.type2D rather than
         MTLTextureType.type2DArray as explained above.
         */
        let totalChannels = image.featureChannels <= 2 ?
            image.featureChannels : numberOfSlices * 4
        
        /*
         If featureChannels<=4 and numberOfImages=1 (i.e. only one slice is needed to represent
         the image), the underlying metal texture type is chosen to be MTLTextureType.type2D
         rather than MTLTextureType.type2DArray
         */
        let paddedFeatureChannels = image.featureChannels <= 2 ? image.featureChannels : 4
        
        let stride = image.width * image.height * paddedFeatureChannels
        
        let count =  image.width * image.height * totalChannels * image.numberOfImages

        var outputUInt16 = [UInt16](repeating: 0, count: count)

        let bytesPerRow = image.width * paddedFeatureChannels * MemoryLayout<UInt16>.size

        let region = MTLRegion(
            origin: MTLOrigin(x: 0, y: 0, z: 0),
            size: MTLSize(width: image.width, height: image.height, depth: 1))

        for sliceIndex in 0..<numberOfSlices{
            image.texture.getBytes(&(outputUInt16[stride * sliceIndex]),
                                   bytesPerRow:bytesPerRow,
                                   bytesPerImage:0,
                                   from: region,
                                   mipmapLevel:0,
                                   slice:sliceIndex)
        }
        
        // Convert UInt16 array into Float32 (Float in Swift)
        var output = [Float](repeating: 0, count: outputUInt16.count)

        var bufferUInt16 = vImage_Buffer(data: &outputUInt16,
                                         height: 1,
                                         width: UInt(outputUInt16.count),
                                         rowBytes: outputUInt16.count * 2)

        var bufferFloat32 = vImage_Buffer(data: &output,
                                          height: 1,
                                          width: UInt(outputUInt16.count),
                                          rowBytes: outputUInt16.count * 4)
        
        if vImageConvert_Planar16FtoPlanarF(&bufferUInt16, &bufferFloat32, 0) != kvImageNoError {
            print("Failed to convert UInt16 array to Float32 array")
            return nil
        }

        return output
    }
}

/*
 Shallow Neural Network
 */

func createShallowNetwork(
    withDevice device:MTLDevice) -> MPSNNGraph?{

    // placeholder node
    let input = MPSNNImageNode(handle: nil)
    
    // Scale
    let scale = MPSNNLanczosScaleNode(
        source: input,
        outputSize: MTLSize(
            width: MNIST_IMAGE_WIDTH,
            height: MNIST_IMAGE_HEIGHT,
            depth: 1))
    
    let fc = MPSCNNFullyConnectedNode(
        source: scale.resultImage,
        weights: NodeDataSource(
            name: "shallow_fc",
            kernelSize: KernelSize(
                width:MNIST_IMAGE_WIDTH,
                height:MNIST_IMAGE_HEIGHT),
            inputFeatureChannels: MNIST_FEATURE_CHANNELS,
            outputFeatureChannels: MNIST_NUMBER_OF_CLASSES))
    
    let softmax = MPSCNNSoftMaxNode(source: fc.resultImage)
    
    guard let graph = MPSNNGraph(
        device: device,
        resultImage: softmax.resultImage,
        resultImageIsNeeded: true) else{
            return nil
    }

    print(graph.debugDescription)

    return graph
}

func create1HiddenLayerNetwork(
    withDevice device:MTLDevice) -> MPSNNGraph?{
    
    let hiddenUnits = 32
    
    // placeholder node
    let input = MPSNNImageNode(handle: nil)
    
    // Scale
    let scale = MPSNNLanczosScaleNode(
        source: input,
        outputSize: MTLSize(
            width: MNIST_IMAGE_WIDTH,
            height: MNIST_IMAGE_HEIGHT,
            depth: 1))
    
    let fc1 = MPSCNNFullyConnectedNode(
        source: scale.resultImage,
        weights: NodeDataSource(
            name: "h1_fc_1",
            kernelSize: KernelSize(
                width:MNIST_IMAGE_HEIGHT,
                height:MNIST_IMAGE_HEIGHT),
            inputFeatureChannels: MNIST_FEATURE_CHANNELS,
            outputFeatureChannels: hiddenUnits)) // 32 hidden units at layer 1
    
    let relu1 = MPSCNNNeuronReLUNode(source: fc1.resultImage)
    
    let fc2 = MPSCNNFullyConnectedNode(
        source: relu1.resultImage,
        weights: NodeDataSource(
            name: "h1_fc_2",
            kernelSize: KernelSize(width:1, height:1),
            inputFeatureChannels: hiddenUnits,
            outputFeatureChannels: MNIST_NUMBER_OF_CLASSES))
    
    let softmax = MPSCNNSoftMaxNode(source: fc2.resultImage)
    
    guard let graph = MPSNNGraph(
        device: device,
        resultImage: softmax.resultImage,
        resultImageIsNeeded: true) else{
            return nil
    }
    
    print(graph.debugDescription)
    
    return graph
}

// Create device
guard let device = MTLCreateSystemDefaultDevice() else{
    fatalError("Failed to reference GPU")
}

// Create command queue
guard let commandQueue = device.makeCommandQueue() else{
    fatalError("Failed to create command queue")
}

let network = Network(
    withCommandQueue: commandQueue,
    graphBuilder: create1HiddenLayerNetwork)

// Create MPSImage
let placeholderImageDescriptor = MPSImageDescriptor(
    channelFormat: MPSImageFeatureChannelFormat.unorm8,
    width: MNIST_NUMBER_OF_CLASSES,
    height: MNIST_IMAGE_HEIGHT,
    featureChannels: MNIST_FEATURE_CHANNELS)

let placeholderImage = MPSImage(
    device: device,
    imageDescriptor: placeholderImageDescriptor)

PlaygroundPage.current.liveView = DigitCanvasView(
    frame: NSRect(x: 0,
                  y: 0,
                  width: 300,
                  height: 300),
    submitHandler:{(view, context) in
        guard let context = context else{
            return
        }

        let origin = MTLOrigin(
            x: 0, y: 0, z: 0)

        let size = MTLSize(
            width: MNIST_IMAGE_WIDTH,
            height: MNIST_IMAGE_HEIGHT,
            depth: 1)

        let region = MTLRegion(
            origin: origin,
            size: size)

        let bytesPerRow = MNIST_IMAGE_WIDTH * MNIST_FEATURE_CHANNELS

        placeholderImage.texture.replace(
            region: region,
            mipmapLevel: 0,
            withBytes: context.data!,
            bytesPerRow: bytesPerRow)
        
        let img = placeholderImage
        

        network.forward(
            inputImage: placeholderImage,
            completationHandler: { (probs) in
                if let probs = probs{
                    view.text = "Prediction \(probs.argmax)"
                }
        })
})
