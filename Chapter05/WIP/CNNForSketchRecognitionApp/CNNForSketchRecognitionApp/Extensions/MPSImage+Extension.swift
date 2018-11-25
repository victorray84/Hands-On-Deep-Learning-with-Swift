import MetalKit
import MetalPerformanceShaders
import Accelerate

public extension MPSImage {
    
    public static func loadFrom(url:URL, usingTextureLoader textureLoader:MTKTextureLoader) -> MPSImage?{
        guard let texture = try? textureLoader.newTexture(
            URL: url,
            options: [MTKTextureLoader.Option.SRGB : NSNumber(value: false)]) else{
                return nil
        }
        
        return MPSImage(texture: texture, featureChannels: 3)
    }
}

public extension MPSImage{
    
    /*
     https://developer.apple.com/documentation/metalperformanceshaders/mpsimage
     Some image types, such as those found in convolutional neural networks (CNN), differ from a standard texture in that
     they may have more than 4 channels per pixel. While the channels could hold RGBA data, they will more commonly hold a number
     of structural permutations upon an RGBA image as the neural network progresses. It is not uncommon for each pixel to have
     32 or 64 channels in it.
     
     Since a standard MTLTexture object cannot have more than 4 channels, the additional channels are stored in
     slices of a 2D texture array (i.e. a texture of type MTLTextureType.type2DArray) such that 4 consecutive
     channels are stored in each slice of this array. If the number of feature channels is N, the number of
     array slices needed is (N+3)/4. For example, a 9-channel CNN image with a width of 3 and a height of 2
     will be stored as follows:
     */
    @nonobjc public func toFloatArray() -> [Float]?{
        /*
         An MPSImage object can contain multiple CNN images for batch processing. In order
         to create an MPSImage object that contains N images, create an MPSImageDescriptor object
         with the numberOfImages property set to N. The length of the 2D texture array (i.e.
         the number of slices) will be equal to ((featureChannels+3)/4)*numberOfImages,
         where consecutive (featureChannels+3)/4 slices of this array represent one image.
         */
        let numberOfSlices = ((self.featureChannels + 3)/4) * self.numberOfImages
        
        /*
         If featureChannels<=4 and numberOfImages=1 (i.e. only one slice is needed to represent the image),
         the underlying metal texture type is chosen to be MTLTextureType.type2D rather than
         MTLTextureType.type2DArray as explained above.
         */
        let totalChannels = self.featureChannels <= 2 ?
            self.featureChannels : numberOfSlices * 4
        
        /*
         If featureChannels<=4 and numberOfImages=1 (i.e. only one slice is needed to represent
         the image), the underlying metal texture type is chosen to be MTLTextureType.type2D
         rather than MTLTextureType.type2DArray
         */
        let paddedFeatureChannels = self.featureChannels <= 2 ? self.featureChannels : 4
        
        let stride = self.width * self.height * paddedFeatureChannels
        
        let count =  self.width * self.height * totalChannels * self.numberOfImages
        
        var outputUInt16 = [UInt16](repeating: 0, count: count)
        
        let bytesPerRow = self.width * paddedFeatureChannels * MemoryLayout<UInt16>.size
        
        let region = MTLRegion(
            origin: MTLOrigin(x: 0, y: 0, z: 0),
            size: MTLSize(width: self.width, height: self.height, depth: 1))
        
        for sliceIndex in 0..<numberOfSlices{
            self.texture.getBytes(&(outputUInt16[stride * sliceIndex]),
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

// MARK: - Conversion functions

public extension MPSImage{
    
    public func convertToGrayscale(usingCommandBuffer commandBuffer:MTLCommandBuffer) -> MPSImage?{
        
        guard let srcColorSpace = CGColorSpace(name: CGColorSpace.sRGB),
            let dstColorSpace = CGColorSpace(name: CGColorSpace.linearGray) else {
                return nil
        }
        
        let conversionInfo = CGColorConversionInfo(
            src: srcColorSpace,dst: dstColorSpace)
        
        let conversion = MPSImageConversion(
            device: commandBuffer.device,
            srcAlpha: .alphaIsOne,
            destAlpha: .alphaIsOne,
            backgroundColor: nil,
            conversionInfo: conversionInfo)
        
        // Create destination texture
        let dstDescriptor = MPSImageDescriptor(
            channelFormat: MPSImageFeatureChannelFormat.unorm8,
            width: self.texture.width,
            height: self.texture.height,
            featureChannels: 1)
        
        // we use preImage to hold preprocesing intermediate results
        //let dstImage = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: dstDescriptor)
        let dstImage = MPSImage(device: device, imageDescriptor: dstDescriptor)
        
        conversion.encode(
            commandBuffer: commandBuffer,
            sourceImage: self,
            destinationImage: dstImage)
        
        return dstImage
    }
    
}
