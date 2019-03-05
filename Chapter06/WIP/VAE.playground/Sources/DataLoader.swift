import AppKit
import MetalKit
import MetalPerformanceShaders

public class DataLoader{
    
    /// Property of the MPSImageDescriptor to describe the channel format
    public let channelFormat = MPSImageFeatureChannelFormat.unorm8
    /// Property of the MPSImageDescriptor to describe the image width
    public let imageWidth  = 28
    /// Property of the MPSImageDescriptor to describe the image height
    public let imageHeight = 28
    /// Property of the MPSImageDescriptor to describe the number of channels (1 being grayscale)
    public let featureChannels = 1
    /// Number of classes in our dataset
    public var numberOfClasses : Int = 10
    
    /// Enable to have each batch first shuffled before retrieving
    public var shuffle : Bool = true
    
    /// Total number of files
    var count : Int{
        get{
            return self.labels.count
        }
    }
    
    private let device : MTLDevice
    
    /// Size of our mini-batches
    public private(set) var batchSize : Int = 0
    
    public var images = [UInt8]()
    public var labels = [UInt8]()
    
    /*
     The size of our MPSImage pool that is used by the DataLoader (this to
     avoid repeatitive allocation
     */
    private var poolSize : Int{
        get{
            return self.batchSize * 3
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
        
        // alterantive would be setting numberOfImages = batchSize and using
        // mpsImage.batchrepresentation to retreive the batch
        // https://developer.apple.com/documentation/metalperformanceshaders/mpsimage/2942495-batchrepresentation
        
        return imageDescriptor
    }()
    
    public init?(device:MTLDevice,
                imagesFile:String,
                labelsFile:String,
                batchSize:Int=4){
        
        self.device = device
        self.batchSize = batchSize
        
        let imagesPathComponents = imagesFile.components(separatedBy: ".")
        let labelsPathComponents = labelsFile.components(separatedBy: ".")
        
        guard let imagesURL = Bundle.main.url(
            forResource: imagesPathComponents[0],
            withExtension: imagesPathComponents[1]),
            let labelsURL = Bundle.main.url(
                forResource: labelsPathComponents[0],
                withExtension: labelsPathComponents[1])
            else{
                return nil
        }
        
        guard let imagesData = try? Data(contentsOf: imagesURL),
            let labelsData = try? Data(contentsOf: labelsURL) else{
                return nil
        }
        
        self.images = imagesData.withUnsafeBytes { (bytes: UnsafePointer<UInt8>) -> [UInt8] in
            return Array(UnsafeBufferPointer(start: bytes, count: imagesData.count / MemoryLayout<UInt>.stride))
        }
        
        self.labels = labelsData.withUnsafeBytes { (bytes: UnsafePointer<UInt8>) -> [UInt8] in
            return Array(UnsafeBufferPointer(start: bytes, count: labelsData.count / MemoryLayout<UInt>.stride))
        }
        
        self.reset()
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
        return (self.currentIndex + self.batchSize) <= self.count
    }
    
    /**
     Return the next available batch; its the developers responsibility to ensure that another batch is available
     before calling this method
     */
    public func nextBatch() -> Batch?{
        if self.mpsImagePool.count < self.poolSize{
            self.initMpsImagePool()
        }
        
        var batchImages = [MPSImage]()
        var batchLabels = [MPSCNNLossLabels]()
        
        // Get current batch range
        let range = self.currentIndex..<(self.currentIndex + self.batchSize)
        let indicies = self.nextImageIndex[range]
        
        // Advance index
        self.currentIndex += self.batchSize
        
        // Populate batch
        for index in indicies{            
            // vectorise label
            guard let vecLabel = self.getLabel(atIndex:index),
                let image = self.getImage(atIndex:index) else{
                fatalError("Failed to create image or label for index \(index)")
            }
            
            // add label and image to our batch
            batchLabels.append(vecLabel)
            batchImages.append(image)
        }
        
        return Batch(images:batchImages, labels:batchLabels)
    }
}

// MARK: - Image loading

extension DataLoader{
    
    /** Updates a MPSImage from our pool and returns it */
    public func getImage(atIndex index:Int) -> MPSImage?{
        // Calculate start and end index
        let sIdx = index * self.imageWidth * self.imageHeight * self.featureChannels
        let eIdx = sIdx + self.imageWidth * self.imageHeight * self.featureChannels
        
        // Check that the indecies are within the bounds of our array
        guard sIdx >= 0, eIdx < self.images.count else {
            return nil
        }
        
        // Get image data
        var imageData = [UInt8]()
        imageData += self.images[(sIdx..<eIdx)]
        
        // get a unsafe pointer to our image data
        let dataPointer = UnsafeMutableRawPointer(mutating: imageData)
        
        // update the data of the associated MPSImage object (with the image data)
        self.mpsImagePool[self.mpsImagePoolIndex].writeBytes(
            dataPointer,
            dataLayout: MPSDataLayout.HeightxWidthxFeatureChannels,
            imageIndex: 0)
        
        let mpsImage = self.mpsImagePool[mpsImagePoolIndex]
        
        // increase pointer to our pool
        self.mpsImagePoolIndex += 1
        self.mpsImagePoolIndex = (self.mpsImagePoolIndex + 1) % self.poolSize
        
        return mpsImage
    }
    
}

// MARK: - Label encoding

extension DataLoader{
    
    /** Helper function that vertocizes; returning its MPSCNNLossLabels representation */
    public func getLabel(atIndex index:Int) -> MPSCNNLossLabels?{
        guard index >= 0, index < self.labels.count else {
            return nil
        }
        
        let labelIndex = Int(self.labels[index])
        
        let numOfLabels = 10
        var labelVec = [Float](repeating: 0, count: numOfLabels)
        labelVec[labelIndex] = 1
        
        let labelData = Data(fromArray: labelVec)
        
        guard let labelDesc = MPSCNNLossDataDescriptor(
            data: labelData,
            layout: MPSDataLayout.HeightxWidthxFeatureChannels,
            size: MTLSize(width: 1, height: 1, depth: self.numberOfClasses)) else{
                return nil
        }
        
        let label = MPSCNNLossLabels(
            device: device,
            labelsDescriptor: labelDesc)
        
        return label
    }
}
