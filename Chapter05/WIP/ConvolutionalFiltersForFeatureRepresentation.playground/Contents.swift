/*:
 
 # [Hands-On Deep Learning with Swift]()
 ### Chapter 5 - Applying CNNs to recognise sketches
 *Writen by [Joshua Newnham](https://www.linkedin.com/in/joshuanewnham) and published by [Packt Publishing](https://www.packtpub.com/big-data-and-business-intelligence/machine-learning-core-ml)*
 */
import Foundation
import AppKit
import AVFoundation
import CoreGraphics
import MetalKit
import MetalPerformanceShaders
import Accelerate
import GLKit
import PlaygroundSupport

// Required to run tasks in the background
PlaygroundPage.current.needsIndefiniteExecution = true

//: #### Datasource
//: Create a Datasource that will be used by our network; here we are manually setting our filters to extract specific features from an input

class CNNDatasource : NSObject, MPSCNNConvolutionDataSource{
    
    let name : String
    let kernelSize : (width:Int, height:Int) = (width:3, height:3)
    let inputFeatureChannels : Int = 1
    let outputFeatureChannels : Int = 0 // TODO add number of output feature channels
    
    var weightsData : Data?
    
    init(name:String) {
        
        self.name = name
    }
    
    /*
     Alerts MPS what sort of weights are provided by the object
     For MPSCNNConvolution, MPSDataTypeUInt8, MPSDataTypeFloat16 and MPSDataTypeFloat32 are supported for normal convolutions using MPSCNNConvolution. MPSCNNBinaryConvolution assumes weights to be of type MPSDataTypeUInt32 always.
    */
    public func dataType() -> MPSDataType{
        return MPSDataType.float32
    }
    
    /*
     Return a MPSCNNConvolutionDescriptor
    */
    @available(OSX 10.13, *)
    public func descriptor() -> MPSCNNConvolutionDescriptor{
        let desc = MPSCNNConvolutionDescriptor(
            kernelWidth: self.kernelSize.width,
            kernelHeight: self.kernelSize.height,
            inputFeatureChannels: self.inputFeatureChannels,
            outputFeatureChannels: self.outputFeatureChannels)
        
        return desc
    }
    
    /*
     Returns a pointer to the weights for the convolution.
     The type of each entry in array is given by -dataType. The number of entries is equal to:
     @code
     inputFeatureChannels * outputFeatureChannels * kernelHeight * kernelWidth
     @endcode
    */
    public func weights() -> UnsafeMutableRawPointer{
        guard let weightsData = self.weightsData else{
            fatalError("weightsData is null")
        }
        
        return UnsafeMutableRawPointer(mutating: (weightsData as NSData).bytes)
    }
    
    /*
     Returns a pointer to the bias terms for the convolution.
     Note: bias terms are always float (a single precision IEEE-754 float and represents one bias) , even when the weights are not.
    */
    public func biasTerms() -> UnsafeMutablePointer<Float>?{
        return nil
    }
    
    
    /*
     Each load alert will be balanced by a purge later, when MPS no longer needs the data from this object. Load will always be called atleast once after initial construction or each purge of the object before anything else is called.
     Note: load may be called to merely inspect the descriptor. In some circumstances, it may be worthwhile to postpone weight and bias construction until they are actually needed to save touching memory and keep the working set small. The load function is intended to be an opportunity to open files or mark memory no longer purgeable.
     */
    public func load() -> Bool{
        // TODO Create filters to extract features
        
        let kernelMatricies = [[[Float]]]()
        
        // Initilize weights
        let weightsCount = self.outputFeatureChannels
            * self.kernelSize.height
            * self.kernelSize.width
            * self.inputFeatureChannels
        
        var weightsArray = Array<Float>(repeating: 0, count: weightsCount)
        
        // Expected format :: weights[ outputChannel ][ kernelY ][ kernelX ][ inputChannel ]
        for o in 0..<self.outputFeatureChannels {
            for ky in 0..<self.kernelSize.height {
                for kx in 0..<self.kernelSize.width {
                    for i in 0..<self.inputFeatureChannels {
                        let weightsArrayIndex = ((o * self.kernelSize.height + ky)
                            * self.kernelSize.width + kx)
                            * self.inputFeatureChannels + i
                        weightsArray[weightsArrayIndex] = kernelMatricies[o][ky][kx]
                    }
                }
            }
        }
        
        self.weightsData = Data(buffer: UnsafeBufferPointer(start: &weightsArray, count: weightsArray.count))
        
        return true
    }
    
    /* Alerts the data source that the data is no longer needed */
    public func purge(){
        self.weightsData = nil
    }
    
    
    /* A label that is transferred to the convolution at init time */
    public func label() -> String?{
        return self.name
    }
    
    public func copy(with zone: NSZone? = nil) -> Any{
        return CNNDatasource(name: self.name) as Any
    }
}

//: #### Network

class Network{
    
    let commandQueue : MTLCommandQueue
    let device : MTLDevice
    
    var graph : MPSNNGraph?
    
    init(withCommandQueue commandQueue:MTLCommandQueue) {
        self.commandQueue = commandQueue
        self.device = self.commandQueue.device
        
        self.graph = createInferenceGraph()
    }
    
    internal func createInferenceGraph() -> MPSNNGraph?{
        // Define networks input (placeholder)
        let input = MPSNNImageNode(handle: nil)
        
        // Scale input
        let scale = MPSNNLanczosScaleNode(
            source:input,
            outputSize: MTLSize(
                width: 128,
                height: 128,
                depth: 1))
        
        // Our convolution layer
        let conv = MPSCNNConvolutionNode(
            source: scale.resultImage,
            weights: CNNDatasource(name: "conv"))
        
        conv.paddingPolicy = MPSNNDefaultPadding(method: .validOnly)
        
        return MPSNNGraph(
            device: self.device,
            resultImage: conv.resultImage,
            resultImageIsNeeded: true)
    }
    
    func forward(image:MPSImage, completionHandler handler: @escaping (MPSImage?) -> Void){
        guard let graph = self.graph else{
            handler(nil)
            return
        }
        
        // To deliver optimal performance we leave some resources used in MPSCNN
        // to be released at next call of autoreleasepool, so the user can
        // decide the appropriate time to release this
        autoreleasepool{
            /*
             This function will synchronously encode the graph on a private command buffer, commit it to a MPS internal command queue and return. The GPU will start working.
             When the GPU is done, the completion handler will be called.  You should use the intervening time to encode other work for execution on the GPU, so that the GPU stays busy and doesn't clock down.
             */
            graph.executeAsync(withSourceImages: [image], completionHandler:{
                (graphOutput:MPSImage?, graphError:Error?) in
                // Handler is of type MPSNNGraphCompletionHandler
                
                if let error = graphError{
                    print(error)
                    DispatchQueue.main.async { handler(nil) }
                } else if let output = graphOutput{
                    DispatchQueue.main.async { handler(output) }
                } else{
                    DispatchQueue.main.async { handler(nil) }
                }
            })
        }
    }
}

//: #### Metal setup

/*
 A Metal device ([MTLDevice]()(https://developer.apple.com/documentation/metal/mtldevice))) is the interface to the GPU. It supports methods for creating objects such as function libraries and textures.
 */
guard let device = MTLCreateSystemDefaultDevice() else{
    fatalError("Failed to get reference to GPU")
}

// Make sure the current device supports MetalPerformanceShaders.
guard MPSSupportsMTLDevice(device) else {
    fatalError("Metal Performance Shaders not Supported on current Device")
}

/*:
 The command queue (MTLCommandQueue) is the object that queues and submits commands to the device for execution.
 */
guard let commandQueue = device.makeCommandQueue() else{
    fatalError("Failed to make CommandQueue")
}

//: ---
//: **Perform inference**

let dataLoader = DataLoader(commandQueue:commandQueue)

guard let inputImage = dataLoader.loadImage(
    filename: "VerticalRectangle",
    fileExtension: "png") else{
        fatalError("Failed to load image")
}

// TODO Create function to extract a single channel from a models output (extractChannelAsCGImage)


let network = Network(withCommandQueue: commandQueue)
network.forward(image: inputImage) { (image) in
    if let outputImage = image{
        print("Output image :: height:\(outputImage.height), width:\(outputImage.width), feature channels:\(outputImage.featureChannels), Number of images:\(outputImage.numberOfImages)")
        
        
        guard let buffer = outputImage.toFloatArray() else{
            fatalError("Failed to export Float array")
        }
        
        // TODO: Create individual images for each of the channels
        // (where each channel is the convolution for each of our specified filters)
        var extractedChannels = [CGImage]()
        
        
        let verticalFilterChannel = extractedChannels[0]
        let horizontalFilterChannel = extractedChannels[1]
        let diagonal45FilterChannel = extractedChannels[2]
        let diagonal135FilterChannel = extractedChannels[3]
        
        // Create a view to present the pixel intensity for each of our images
        let barPlotViewFrame = NSRect(
            x: 0, y: 0,
            width: 250, height: 250)
        
        let barPlotView = BarPlotView(
            frame: barPlotViewFrame)
        
        // TODO Add bars to our bar plot using the extracted features
        
        // Set our view to the PLaygrounds liveView
        PlaygroundPage.current.liveView = barPlotView
    }
}

