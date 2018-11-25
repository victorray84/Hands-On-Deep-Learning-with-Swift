import Foundation
import CoreGraphics
import MetalKit
import MetalPerformanceShaders

public class DataLoader{
    
    let device : MTLDevice
    let commandQueue : MTLCommandQueue
    let textureLoader : MTKTextureLoader
    
    public init(commandQueue:MTLCommandQueue) {
        self.device = commandQueue.device
        self.commandQueue = commandQueue
        
        /*
         The MTKTextureLoader class simplifies the effort required to load your texture data
         into a Metal app. This class can load images from common file formats such as
         PNG, JPEG, and TIFF.
         */
        self.textureLoader = MTKTextureLoader(device:device)
    }
    
    public func loadImage(filename:String, fileExtension:String, grayscale:Bool=true) -> MPSImage?{
        guard let inputImage = MPSImage.loadFrom(
            url: Bundle.main.url(forResource: filename, withExtension: fileExtension)!,
            usingTextureLoader: self.textureLoader) else{
                print("Unable to load image")
                return nil
        }
        
        print("Input image :: height:\(inputImage.height), width:\(inputImage.width), feature channels:\(inputImage.featureChannels)")
        
        if !grayscale{
            return inputImage
        }
        
        guard let commandBuffer = self.commandQueue.makeCommandBuffer() else{
            print("Unable to create command buffer")
            return nil
        }
        
        guard let greyScaleInputImage = inputImage.convertToGrayscale(usingCommandBuffer: commandBuffer) else{
            print("Unable to run grayscale filter on input image")
            return nil
        }
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        return greyScaleInputImage
    }
    
}
