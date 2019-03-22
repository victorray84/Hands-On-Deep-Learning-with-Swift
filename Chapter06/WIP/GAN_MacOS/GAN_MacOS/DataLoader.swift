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
    
    public var imagesData = [Float]()
    
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
                 batchSize:Int=8){
        
        self.device = device
        self.batchSize = batchSize               
        
        guard let imagesData = try? Data(contentsOf: imagesURL) else{
                return nil
        }
        
        let imageByteData = imagesData.withUnsafeBytes { (bytes: UnsafePointer<UInt8>) -> [UInt8] in
            return Array(UnsafeBufferPointer(start: bytes, count: imagesData.count / MemoryLayout<UInt8>.stride))
        }

        // Normalise the data
        self.imagesData = imageByteData.map({ (byte) -> Float in
            return (Float32(byte) - 127.5) / 127.5
        })
        
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
            self.initMpsImagePool()
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

// MARK: - DEVELOPMENT

extension DataLoader{
    
    public func createDummyInput(withValue value:Float=0.0, count:Int=1) -> [MPSImage]?{
        var dummyImages = [MPSImage]()
        
        for _ in 0..<count{
            let imageData = [Float](repeating: value, count: self.imageWidth * self.imageHeight * self.featureChannels)
            
            // get a unsafe pointer to our image data
            let dataPointer = UnsafeMutableRawPointer(mutating: imageData)
            
            let image = MPSImage(
                device: self.device,
                imageDescriptor: self.imageDescriptor)
            
            // update the data of the associated MPSImage object (with the image data)
            image.writeBytes(
                dataPointer,
                dataLayout: MPSDataLayout.HeightxWidthxFeatureChannels,
                imageIndex: 0)
            
            dummyImages.append(image)
        }
        
        return dummyImages
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
        var imageData = [Float]()
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
    
    func createLabels(withValue value:Float, variance:Float=0.01, batchSize:Int?=nil) -> [MPSCNNLossLabels]?{
        var labels = [MPSCNNLossLabels]()
        
        let batchSize = batchSize ?? self.batchSize
        
        for _ in 0..<batchSize{
            var labelVec = [Float](repeating: 0, count: 1)
            labelVec[0] = max(min(1.0, value + Float.random(in: -variance...variance)), 0.0) // Add some variance to the labels
            let labelData = Data(fromArray: labelVec)
            
            guard let labelDesc = MPSCNNLossDataDescriptor(
                data: labelData,
                layout: MPSDataLayout.featureChannelsxHeightxWidth,
//                layout: MPSDataLayout.HeightxWidthxFeatureChannels,
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
