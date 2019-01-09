import AppKit
import MetalKit
import MetalPerformanceShaders

public class DataLoader{
    
    /// Property of the MPSImageDescriptor to describe the channel format
    public let channelFormat = MPSImageFeatureChannelFormat.unorm8
    /// Property of the MPSImageDescriptor to describe the image width
    public let imageWidth  = 128
    /// Property of the MPSImageDescriptor to describe the image height
    public let imageHeight = 128
    /// Property of the MPSImageDescriptor to describe the number of channels (1 being grayscale)
    public let featureChannels = 1
    /// Number of classes in our dataset
    public var numberOfClasses : Int = 22
    
    // Directory of sketch images we will be feeding our network
    var sketchFileUrls = [String:[URL]]()
    // List of labesl (where their index corresponds to the index returned by the network)
    public var labels = [String]()
    
    /// Total number of files
    var count : Int{
        return self.sketchFileUrls.reduce(0, { total, kvp in
            return total + kvp.value.count
        })
    }
    
    private let device : MTLDevice
    
    // Where the images reside
    private let sourcePathURL : URL
    
    // TODO Declare properties to support batching
    // shuffle, batchSize, poolSize, mpsImagePool, mpsImagePoolIndex, currentIndex
    
    /// Enable to have each batch first shuffled before retrieving
    public var shuffle : Bool = true

    /// Size of our mini-batches
    public private(set) var batchSize : Int = 0

    // Used to create the batches; populated each time the batch is reset
    private var batchPool = [SampleLookup]()

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
    
    // TODO Modify signature to include batchSize
    public init(device:MTLDevice,
                sourcePathURL:URL,
                batchSize:Int=4){
        
        self.device = device
        self.sourcePathURL = sourcePathURL
        self.batchSize = batchSize
        
        fetchSketchUrls()
        
        setLabels()
        
        self.numberOfClasses = self.sketchFileUrls.count
        
        self.reset()
    }
    
    private func fetchSketchUrls(){
        do {
            let dirUrls = try FileManager.default
                .contentsOfDirectory(at: self.sourcePathURL, includingPropertiesForKeys:nil)
            for dirUrl in dirUrls{
                guard let fileUrls = try? FileManager.default
                    .contentsOfDirectory(at: dirUrl, includingPropertiesForKeys:nil) else{
                        continue
                }
                let label = dirUrl.pathComponents.last!
                self.sketchFileUrls[label] = fileUrls
            }
        } catch {
            print("Error while enumerating files \(self.sourcePathURL.absoluteString): \(error.localizedDescription)")
        }
    }
    
    private func setLabels(){
        self.labels.removeAll()
        self.labels.append(contentsOf: Array(self.sketchFileUrls.keys).sorted())
    }
}

// MARK: - Sample methods

extension DataLoader{
    
    // TODO Implement a function (initMpsImagePool) to initilize the MPSImage pool (reusable array of MPSImage objects)
    private func initMpsImagePool(){
        self.mpsImagePool.removeAll()
        
        let descriptor = self.imageDescriptor
        for _ in 0..<self.poolSize{
            self.mpsImagePool.append(
                MPSImage(device: self.device, imageDescriptor: descriptor))
        }
    }
    
    // TODO Implement a function (populateBatchPool) to create an array of lookups for the whole dataset
    private func populateBatchPool(){
        batchPool.removeAll()
        
        // Keep track of how many we have added per label
        var labelCounts = [String:Int]()
        
        // batchPool with all tuples of image labels and indicies pairs
        var hasChanged = true
        while hasChanged{
            hasChanged = false
            for label in self.labels{
                let currentCount = labelCounts[label] ?? 0
                if currentCount >= self.sketchFileUrls[label]!.count{
                    continue // ignore
                }
                hasChanged = true // flag that we have changed
                batchPool.append(SampleLookup(label:label, index:currentCount))
                labelCounts[label] = currentCount + 1
            }
        }
        
        if self.shuffle{
            batchPool.shuffle()
        }
    }
    
    /// Call this before begining a batch; this resets the current index and pooling index
    public func reset(){
        self.currentIndex = 0
        self.mpsImagePoolIndex = 0
        
        self.populateBatchPool()
    }
    
    /// Return true if there is another batch available to consume
    public func hasNext() -> Bool{
        return (self.currentIndex + self.batchSize) <= self.count
    }
    
    /*:
     Return the next available batch; its the developers responsibility to ensure that another batch is available
     before calling this method
     */
    public func nextBatch(commandBuffer:MTLCommandBuffer) -> Batch?{
        if self.mpsImagePool.count < self.poolSize{
            self.initMpsImagePool()
        }

        var batchImages = [MPSImage]()
        var batchLabels = [MPSCNNLossLabels]()

        // Get current batch range
        let range = self.currentIndex..<(self.currentIndex + self.batchSize)
        // Get slice
        let batchLookups = self.batchPool[range]
        // Advance index
        self.currentIndex += self.batchSize
        
        // Populate batch
        for batchLookup in batchLookups{
            // vectorise label
            guard let vecLabel = self.vectorizeLabel(label: batchLookup.label) else{
                fatalError("No image found for label \(batchLookup.label)")
            }
            
            // get the image for a specific label and index
            guard let imageData = self.loadImageData(
                forLabel: batchLookup.label,
                atIndex: batchLookup.index) else{
                    fatalError("No image found for label \(batchLookup.label) at index \(batchLookup.index)")
            }

            // get a unsafe pointer to our image data
            let dataPointer = UnsafeMutableRawPointer(mutating: imageData)

            // update the data of the associated MPSImage object (with the image data)
            self.mpsImagePool[self.mpsImagePoolIndex].writeBytes(
                dataPointer,
                dataLayout: MPSDataLayout.HeightxWidthxFeatureChannels,
                imageIndex: 0)
            
            // add label and image to our batch
            batchLabels.append(vecLabel)
            batchImages.append(self.mpsImagePool[mpsImagePoolIndex])

            // increase pointer to our pool
            self.mpsImagePoolIndex += 1
            self.mpsImagePoolIndex = (self.mpsImagePoolIndex + 1) % self.poolSize
        }
        
        return Batch(images:batchImages, labels:batchLabels)
    }
}

// MARK: - Image loading

extension DataLoader{
    
    /** Helper function that load a given image and returns its byte representation */
    public func loadImageData(forLabel label:String, atIndex index:Int) -> [UInt8]?{
        if self.sketchFileUrls[label] == nil || index < 0 || index >= self.sketchFileUrls[label]!.count{
            return nil
        }
        
        let fileUrl = self.sketchFileUrls[label]![index]
        
        let img = NSImage(contentsOf: fileUrl)
        
        guard let cgImage = img?.cgImage else{
            return nil
        }
        
        return cgImage.toByteArray()
    }
    
}

// MARK: - Label encoding

extension DataLoader{
    
    /** Helper function that vertocizes; returning its MPSCNNLossLabels representation */
    public func vectorizeLabel(label:String) -> MPSCNNLossLabels?{
        if self.sketchFileUrls[label] == nil{
            return nil
        }
        
        guard let labelIndex = self.labels.firstIndex(of: label) else{
            return nil
        }
        
        var labelVec = [Float32](repeating: 0, count: self.numberOfClasses)
        labelVec[labelIndex] = 1
        
        let labelData = Data(fromArray: labelVec)
        
        guard let labelDesc = MPSCNNLossDataDescriptor(
            data: labelData,
            layout: MPSDataLayout.HeightxWidthxFeatureChannels,
            size: MTLSize(width: 1, height: 1, depth: self.numberOfClasses)) else{
                return nil
        }
        
        let lossLabel = MPSCNNLossLabels(
            device: device,
            labelsDescriptor: labelDesc)
        
        lossLabel.label = label
        
        return lossLabel
    }
}
