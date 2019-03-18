import AppKit
import MetalKit
import MetalPerformanceShaders

public class DataLoader{
    
    /// Property of the MPSImageDescriptor to describe the channel format
    public let channelFormat = MPSImageFeatureChannelFormat.float32
    // Property of the MPSImageDescriptor to describe the image width
    public let imageWidth  = 28
    /// Property of the MPSImageDescriptor to describe the image height
    public let imageHeight = 28
    /// Property of the MPSImageDescriptor to describe the number of channels (1 being grayscale)
    public let featureChannels = 1
    
    /// Enable to have each batch first shuffled before retrieving
    public var shuffle : Bool = true
    MPSCNNConvolution
    public private(set) var label : UInt8 = 0
    
    /// Total number of files
    var count : Int{
        get{
            return (self.imagesData.count / self.imageStride)
        }
    }
    
    /// Number of bytes per image
    var imageStride : Int{
        get{
            return (self.imageWidth * self.imageHeight * self.featureChannels)
        }
    }
    
    private let device : MTLDevice
    
    /// Size of our mini-batches
    public private(set) var batchSize : Int = 0
    
    public var imagesData = [Float32]()
    
    /*
     The size of our MPSImage pool that is used by the DataLoader (this to
     avoid repeatitive allocation
     */
    private var poolSize : Int{
        get{
            return self.batchSize * 4
        }
    }
    
    /// The pool of MPSImages
    var mpsImagePool = [MPSImage]()
    
    /// Pointer into our pool of MPSImage objects
    private var mpsImagePoolIndex = 0
    
    /// Current sample index
    private var currentIndex = 0
    
    private var nextImageIndex = [Int]()
    
    /*:
     A MPSImageDescriptor object describes a attributes of MPSImage and is used to
     create one (see MPSImage discussion below)
     */
    lazy var imageDescriptor : MPSImageDescriptor = {
        var imageDescriptor = MPSImageDescriptor(
            channelFormat:self.channelFormat,
            width: self.imageWidth,
            height: self.imageHeight,
            featureChannels:self.featureChannels)
        imageDescriptor.numberOfImages = 1
        
        return imageDescriptor
    }()
    
    public init?(device:MTLDevice,
                 imagesURL:URL,
                 labelsURL:URL,
                 label:UInt8,
                 batchSize:Int=4){
        
        self.device = device
        self.label = label
        self.batchSize = batchSize
        
        guard let labelsData = try? Data(contentsOf: labelsURL) else{
            return nil
        }
        
        let labels = labelsData.withUnsafeBytes {
            (bytes: UnsafePointer<UInt8>) -> [UInt8] in
            return Array(UnsafeBufferPointer(start: bytes, count: labelsData.count / MemoryLayout<UInt>.stride))
        }
        
        guard let imagesData = try? Data(contentsOf: imagesURL) else{
                return nil
        }
        
        let imageByteData = imagesData.withUnsafeBytes { (bytes: UnsafePointer<UInt8>) -> [UInt8] in
            return Array(UnsafeBufferPointer(start: bytes, count: imagesData.count / MemoryLayout<UInt>.stride))
        }

        // Normalise the data
        let imageFloatData = imageByteData.map({ (byte) -> Float32 in
            return (Float32(byte) - 127.5) / 127.5
        })
        
        self.imagesData = [Float32]()
        
        for (i, l) in labels.enumerated(){
            if l != self.label{
                continue
            }
            
            let sIdx = Int(i * self.imageWidth * self.imageHeight * self.featureChannels)
            let eIdx = Int(sIdx + self.imageWidth * self.imageHeight * self.featureChannels)
            
            let data = imageFloatData[sIdx..<eIdx]
            self.imagesData = self.imagesData + data
        }
        
        self.reset()
    }
}

// MARK: - Utility functions

extension DataLoader{
    
    func toNSImage(mpsImage image:MPSImage) -> NSImage?{
        guard let cgImage = self.toCGImage(mpsImage: image) else{
            return nil
        }
        
        return NSImage(
            cgImage: cgImage,
            size: NSSize(width: cgImage.width, height: cgImage.height))
    }
    
    func toCGImage(mpsImage image:MPSImage) -> CGImage?{
        if let rawBytes = image.toFloatArray()?.map({ (val) -> UInt8 in
            return UInt8((val * 127.5) + 127.5)
        }){
            return CGImage.fromByteArray(
                bytes: rawBytes,
                width: self.imageWidth,
                height: self.imageHeight,
                channels: self.featureChannels)
        }
        
        return nil
    }
    
}

// MARK: - Sample methods

extension DataLoader{
    
    /** Create the pool og MPSImages which will be used and reused by the dataloader */
    private func initMpsImagePool(){
        self.mpsImagePool.removeAll()
        
        let descriptor = self.imageDescriptor
        for _ in 0..<self.poolSize{
            self.mpsImagePool.append(MPSImage(device: self.device, imageDescriptor: descriptor))
        }
    }
    
    /// Call this before begining a batch; this resets the current index and pooling index
    public func reset(){
        self.currentIndex = 0
        self.mpsImagePoolIndex = 0
        
        self.nextImageIndex.removeAll()
        
        for i in 0..<self.count{
            self.nextImageIndex.append(i)
        }
        
        if self.shuffle{
            self.nextImageIndex.shuffle()
        }
    }
    
    /// Return true if there is another batch available to consume
    public func hasNext() -> Bool{
        return (self.currentIndex + self.batchSize) < self.count
    }
    
    /**
     Return the next available batch; its the developers responsibility to ensure that another batch is available
     before calling this method
     */
    public func nextBatch() -> [MPSImage]?{
        if self.mpsImagePool.count < self.poolSize{
            //MPSImage(device: self.device, imageDescriptor: descriptor)self.initMpsImagePool()
        }
        
        var batchImages = [MPSImage]()
        
        // Get current batch range
        let range = self.currentIndex..<(self.currentIndex + self.batchSize)
        let indicies = self.nextImageIndex[range]
        
        // Advance index
        self.currentIndex += self.batchSize
        
        // Populate batch
        for index in indicies{
            // vectorise label
            guard let image = self.getInputForImage(atIndex: index) else{
                fatalError("Failed to create image or label for index \(index)")
            }
            
            // add images to our batch
            batchImages.append(image)
        }
        
        return batchImages
    }
}

// MARK: - Image loading

extension DataLoader{
    
    /** Updates a MPSImage from our pool and returns it */
    public func getInputForImage(atIndex index:Int) -> MPSImage?{
        // Calculate start and end index
        let sIdx = index * self.imageStride
        let eIdx = sIdx + self.imageStride
        
        // Check that the indecies are within the bounds of our array
        guard sIdx >= 0, eIdx <= self.imagesData.count else {
            return nil
        }
        
        // Get image data
        var imageData = [Float32]()
        imageData += self.imagesData[(sIdx..<eIdx)]
        
        // get a unsafe pointer to our image data
        let dataPointer = UnsafeMutableRawPointer(mutating: imageData)
        
        // update the data of the associated MPSImage object (with the image data)
        self.mpsImagePool[self.mpsImagePoolIndex].writeBytes(
            dataPointer,
            dataLayout: MPSDataLayout.HeightxWidthxFeatureChannels,
            imageIndex: 0)
        
        let image = self.mpsImagePool[mpsImagePoolIndex]
        
        // increase pointer to our pool
        self.mpsImagePoolIndex = (self.mpsImagePoolIndex + 1) % self.poolSize
        
        return image
    }
}

// MARK: Label creation

extension DataLoader{
    
    func createLabels(withValue value:Float, variance:Float=0.1) -> [MPSCNNLossLabels]?{
        var labels = [MPSCNNLossLabels]()
        
        for _ in 0..<self.batchSize{
            var labelVec = [Float32](repeating: 0, count: 1)
            labelVec[0] = max(min(1.0, value + Float.random(in: -variance...variance)), 0.0) // Add some variance to the labels
            let labelData = Data(fromArray: labelVec)
            
            guard let labelDesc = MPSCNNLossDataDescriptor(
                data: labelData,
                layout: MPSDataLayout.HeightxWidthxFeatureChannels,
                size: MTLSize(width: 1, height: 1, depth: 1)) else{
                    return nil
            }
            
            let label = MPSCNNLossLabels(
                device: device,
                labelsDescriptor: labelDesc)
            
            labels.append(label)
        }
        
        return labels
    }
    
}
