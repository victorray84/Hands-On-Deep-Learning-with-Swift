import AppKit
import MetalKit
import MetalPerformanceShaders

public typealias Sample = (image:MPSImage, label:MPSCNNLossLabels)

public typealias Batch = (images:[MPSImage], labels:[MPSCNNLossLabels])

public class DataLoader{
    
    public let channelFormat = MPSImageFeatureChannelFormat.unorm8
    public let imageWidth  = 28
    public let imageHeight = 28
    public let featureChannels = 1
    public let numberOfClasses = 10
    
    let commandQueue : MTLCommandQueue
    let device : MTLDevice
    let textureLoader : MTKTextureLoader
    
    public init(commandQueue:MTLCommandQueue, sourceDirectory:String, validationSplit:Float=0.2){
        self.commandQueue = commandQueue
        self.device = self.commandQueue.device
        
        /*
         The MTKTextureLoader class simplifies the effort required to load your texture data
         into a Metal app. This class can load images from common file formats such as
         PNG, JPEG, and TIFF.
         */
        self.textureLoader = MTKTextureLoader(device:device)
        

//        let documentsURL = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
//        do {
//            let fileURLs = try fileManager.contentsOfDirectory(at: documentsURL, includingPropertiesForKeys: nil)
//            // process files
//        } catch {
//            print("Error while enumerating files \(documentsURL.path): \(error.localizedDescription)")
//        }
    }
    
    
}
