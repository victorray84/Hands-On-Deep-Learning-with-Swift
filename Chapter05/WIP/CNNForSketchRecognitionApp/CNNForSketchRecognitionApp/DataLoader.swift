import AppKit
import MetalKit
import MetalPerformanceShaders

public class DataLoader{
    
    public let channelFormat = MPSImageFeatureChannelFormat.unorm8
    public let imageWidth  = 128
    public let imageHeight = 128
    public let featureChannels = 1
    public var numberOfClasses : Int = 0
    
    var sketchFileUrls = [String:[URL]]()
    
    public var labels = [String]()
    
    /*
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
    
    var count : Int{
        return self.sketchFileUrls.reduce(0, { total, kvp in
            return total + kvp.value.count
        })
    }
    
    let device : MTLDevice
    public let batchSize : Int
    let sourcePathURL : URL
    
    private var poolSize : Int{
        get{
            return self.batchSize * 3
        }
    }
    
    var mpsImagePool = [MPSImage]()
    
    // pointer into our pool of MPSImage objects
    private var mpsImagePoolIndex = 0
    
    // current sample index
    private var currentIndex = 0
    
    public init(device:MTLDevice,
                sourcePathURL:URL,
                batchSize:Int=66){
        
        self.device = device
        self.sourcePathURL = sourcePathURL
        self.batchSize = batchSize
        
        fetchSketchUrls()
        
        setLabels()
        
        self.numberOfClasses = self.sketchFileUrls.count
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
    
    public static func getBatchCount(batch:Batch?) -> Int{
        guard let batch = batch else{
            return 0
        }
        
        assert(batch.images.count == batch.labels.count)
        
        return batch.images.count
    }
}

// MARK: - Image loading

extension DataLoader{
    
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

// MARK: - Sample methods

extension DataLoader{
    
    private func initMpsImagePool(){
        self.mpsImagePool.removeAll()
        
        let descriptor = self.imageDescriptor
        for _ in 0..<self.poolSize{
            self.mpsImagePool.append(MPSImage(device: self.device, imageDescriptor: descriptor))
        }
    }
    
    public func reset(){
        self.currentIndex = 0
        self.mpsImagePoolIndex = 0
    }
    
    public func hasNext() -> Bool{
        let range = self.currentIndex..<(self.count - self.currentIndex)
        return range.count >= self.batchSize
    }
    
    public func nextBatch(commandBuffer:MTLCommandBuffer) -> Batch?{
        if self.mpsImagePool.count < self.poolSize{
            self.initMpsImagePool()
        }
        
        var batchImages = [MPSImage]()
        var batchLabels = [MPSCNNLossLabels]()
        
        var sampleAdded = true // flag to indicate if sample has been added or not (for early stopping)
        
        outerLoop: while batchImages.count < self.batchSize && sampleAdded{
            sampleAdded = false
            
            for label in self.labels{
                // vectorise label
                guard let vecLabel = self.vectorizeLabel(label: label) else{
                    fatalError("No image found for label \(label) found")
                }
                
                // get the image for a specific label and index
                guard let imageData = self.loadImageData(forLabel: label, atIndex: self.currentIndex) else{
                    //print("No image found for label \(label) at index \(self.currentIndex)")
                    continue
                }
                
                // flag that we have added a sample
                sampleAdded = true
                
                // get a unsafe pointer to our image data
                let dataPointer = UnsafeMutableRawPointer(mutating: imageData)
                
                //let mpsImage = MPSImage(device: self.device, imageDescriptor: self.imageDescriptor)
//                let mpsImage = MPSTemporaryImage(
//                    commandBuffer: commandBuffer,
//                    imageDescriptor: self.imageDescriptor)
                
                // update the data of the associated MPSImage object (with the image data)
                self.mpsImagePool[self.mpsImagePoolIndex].writeBytes(
                    dataPointer,
                    dataLayout: MPSDataLayout.HeightxWidthxFeatureChannels,
                    imageIndex: 0)
                
//                mpsImage.writeBytes(
//                    dataPointer,
//                    dataLayout: MPSDataLayout.HeightxWidthxFeatureChannels,
//                    imageIndex: 0)
                
                // add label and image to our batch
                batchLabels.append(vecLabel)
                batchImages.append(self.mpsImagePool[mpsImagePoolIndex])
                
                //batchImages.append(mpsImage)
                
                // increase pointer to our pool
                self.mpsImagePoolIndex += 1
                self.mpsImagePoolIndex = (self.mpsImagePoolIndex + 1) % self.poolSize
                
                // check if we need to stop
                if batchImages.count >= self.batchSize{
                    break outerLoop
                }
            }
            
            self.currentIndex += 1
        }
        
        if batchImages.count == 0 || batchImages.count != batchLabels.count{
            return nil
        }
        
        return Batch(images:batchImages, labels:batchLabels)
    }
    
    public func vectorizeLabel(label:String) -> MPSCNNLossLabels?{
        if self.sketchFileUrls[label] == nil{
            return nil
        }
        
        guard let labelIndex = self.labels.firstIndex(of: label) else{
            return nil
        }
        
        var labelVec = [Float](repeating: 0, count: self.numberOfClasses)
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
