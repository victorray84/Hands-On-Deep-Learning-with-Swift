import Foundation
import MetalKit
import MetalPerformanceShaders
import Accelerate

public extension MPSImage{
    
    public func toFloatArray() -> [Float]?{
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
