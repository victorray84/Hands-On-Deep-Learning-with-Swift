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
    
    // Used to create the batches; populated each time the batch is reset
    private var batchPool = [SampleLookup]()
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
                sourcePathURL:URL){
        
        self.device = device
        self.sourcePathURL = sourcePathURL
        
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
    
    // TODO Implement a functino (initMpsImagePool) to initilize the MPSImage pool (reusable array of MPSImage objects)
    
    // TODO Implement a functino (populateBatchPool) to create an array of lookups for the whole dataset -
    
    /// Call this before begining a batch; this resets the current index and pooling index
    public func reset(){
        // TODO Implement body to reset the data loader; called before each epoch
    }
    
    /// Return true if there is another batch available to consume
    public func hasNext() -> Bool{
        // TODO Implement body to return true if another batch is available otherwise false
        return false
    }
    
    /*:
    Return the next available batch; its the developers responsibility to ensure that another batch is available
    before calling this method
     */
    public func nextBatch(commandBuffer:MTLCommandBuffer) -> Batch?{
        // TODO Implement body to return the next batch (using the batchPool to lookup the associated images and
        // labels - we'll reuse the MPSImage's available in our pool to avoid the cost of object creation
        return nil
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
