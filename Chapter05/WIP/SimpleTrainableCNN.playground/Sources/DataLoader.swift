import Foundation
import AppKit
import AVFoundation
import CoreGraphics
import MetalKit
import MetalPerformanceShaders
import Accelerate
import GLKit

public class DataLoader{
    
    let width : Int = 64
    let height : Int = 64
    let channels : Int = 64
    
    let device : MTLDevice
    let commandQueue : MTLCommandQueue
    let textureLoader : MTKTextureLoader
    
    let trainingData = [
        (filename:"Circle", fileExtension:"png", label:0, labelName:"Circle"),
        (filename:"Square", fileExtension:"png", label:1, labelName:"Square"),
        (filename:"Triangle", fileExtension:"png", label:2, labelName:"Triangle"),
        (filename:"HorzRectangle", fileExtension:"png", label:3, labelName:"HorzRectangle"),
        (filename:"VertRectangle", fileExtension:"png", label:4, labelName:"VertRectangle")
    ]
    
    public var numberOfClasses : Int{
        get{
            return trainingData.count
        }
    }
    
    public var batchSize : Int{
        get{
            return trainingData.count
        }
    }
    
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
    
    private func loadImage(filename:String, fileExtension:String, grayscale:Bool=true) -> MPSImage?{
        guard let inputImage = MPSImage.loadFrom(
            url: Bundle.main.url(forResource: filename, withExtension: fileExtension)!,
            usingTextureLoader: self.textureLoader) else{
                print("Unable to load image")
                return nil
        }
        
        //print("Input image :: height:\(inputImage.height), width:\(inputImage.width), feature channels:\(inputImage.featureChannels)")
        
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
    
    private func loadLabel(label:Int, name:String) -> MPSCNNLossLabels?{
        var labelVec = [Float](repeating: 0, count: self.numberOfClasses)
        labelVec[label] = 1
        
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
        
        lossLabel.label = name
        
        return lossLabel
    }
    
    public func getBatch(shuffle:Bool=true) -> Batch{
        var images = [MPSImage]()
        var labels = [MPSCNNLossLabels]()
        
        var indicies = Array(0..<self.trainingData.count)
        if shuffle{
            indicies = indicies.shuffled()
        }
        
        for idx in indicies{
            let item = self.trainingData[idx]
            
            guard let image = self.loadImage(
                filename: item.filename,
                fileExtension: item.fileExtension,
                grayscale: true) else{
                    continue
            }
            
            guard let label = self.loadLabel(
                label:item.label,
                name:item.labelName) else{
                    continue
            }
            
            images.append(image)
            labels.append(label)
        }
        
        return (images:images, labels:labels)
    }
    
    public static func getBatchCount(batch:Batch?) -> Int{
        guard let batch = batch else{
            return 0
        }
        
        assert(batch.images.count == batch.labels.count)
        
        return batch.images.count
    }
}
