//
//  GANSampleGenerator.swift
//  GAN_MacOS
//
//  Created by joshua.newnham on 24/02/2019.
//  Copyright © 2019 Joshua Newnham. All rights reserved.
//

import Foundation
import AppKit
import MetalKit
import MetalPerformanceShaders
import GameKit

public class GANSampleGenerator : NSObject{
    
    /*
     A MPSImageDescriptor object describes a attributes of MPSImage and is used to
     create one (see MPSImage discussion below)
     */
    lazy var imageDescriptor : MPSImageDescriptor = {
        var imageDescriptor = MPSImageDescriptor(
            channelFormat:MPSImageFeatureChannelFormat.float32,
            width: self.latentSize,
            height: 1,
            featureChannels:1)
        return imageDescriptor
    }()
    
    let device : MTLDevice
    let latentSize : Int
    let randomSource : GKRandomSource
    
    lazy var imageData : [Float] = {
        return [Float](repeating: 0.0, count: self.latentSize) // resuze array for each image
    }()
    
    public init(_ device:MTLDevice, _ latentSize:Int) {
        self.device = device
        self.latentSize = latentSize
        self.randomSource = GKRandomSource.sharedRandom()
    }
    
    public func generate(_ batchSize:Int, _ imageData:[Float]?=nil) -> [MPSImage]?{
        var images = [MPSImage]()
        
        var imageData = imageData ?? self.imageData
        
        for _ in 0..<batchSize{
            let image = MPSImage(device: self.device, imageDescriptor: self.imageDescriptor)
            
            for i in 0..<imageData.count{
                imageData[i] = Float.randomNormal(
                    mean:  0.0001,
                    deviation: 0.5,
                    randomSource: self.randomSource)
            }
            
            // get a unsafe pointer to our image data
            let dataPointer = UnsafeMutableRawPointer(mutating: imageData)
            // Update data
            image.writeBytes(dataPointer, dataLayout: .HeightxWidthxFeatureChannels, imageIndex: 0)
            images.append(image)
            
        }
        
        return images
    }
    
}

class PooledGANSampleGenerator : GANSampleGenerator{
    
    private(set) var poolSize : Int
    
    private var index : Int = 0
    private var pool : [MPSImage] = [MPSImage]()
    
    override init(_ device: MTLDevice, _ latentSize: Int) {
        self.poolSize = 0
        super.init(device, latentSize)
        
        self.initPool()
    }
    
    init(_ device: MTLDevice, _ latentSize: Int, _ poolSize:Int) {
        self.poolSize = poolSize
        super.init(device, latentSize)
        
        self.initPool()
    }
    
    private func initPool(){
        guard self.poolSize > 0 else{
            return
        }
        
        for _ in 0..<self.poolSize{
            self.pool.append(MPSImage(device: self.device, imageDescriptor: self.imageDescriptor))
        }
    }
    
    public override func generate(_ batchSize:Int, _ imageData:[Float]?=nil) -> [MPSImage]?{
        if self.poolSize == 0{
            self.poolSize = batchSize * 3
            self.initPool()
        }
        
        guard batchSize < self.poolSize else{
            fatalError("Batch size is larger than the available pool")
        }
        
        var images = [MPSImage]()
        
        //var imageData = imageData ?? self.imageData
        var imageData = imageData ?? [Float](repeating: 0.0, count: self.latentSize)
        
        for _ in 0..<batchSize{
            let image = self.pool[self.index]
            self.index += 1
            self.index %= self.poolSize
            
            for i in 0..<imageData.count{
                imageData[i] = Float.randomNormal(
                    mean: 0.0001,
                    deviation: 0.5,
                    randomSource: self.randomSource)
            }
            
            // get a unsafe pointer to our image data
            let dataPointer = UnsafeMutableRawPointer(mutating: imageData)
            // Update data
            image.writeBytes(dataPointer, dataLayout: .HeightxWidthxFeatureChannels, imageIndex: 0)
            
            images.append(image)
            
        }
        
        return images
    }
}
