//
//  DataSource.swift
//  Hands-On Deep Learning with Swift - GAN
//
//  Created by joshua.newnham on 19/02/2019.
//  Copyright © 2019 Joshua Newnham. All rights reserved.
//

import Foundation
import AppKit
import MetalKit
import MetalPerformanceShaders

protocol DataSource {
    
    var name : String{ get }
    
    var trainable : Bool{ set get }
    
    var weightsAndBiasesState : MPSCNNConvolutionWeightsAndBiasesState?{ get }
    
    @discardableResult
    func saveParametersToDisk() -> Bool
    
    func saveWeightsToDisk() -> Bool
    
    func saveBiasTermsToDisk() -> Bool
    
    func synchronizeParameters(on commandBuffer:MTLCommandBuffer)
    
}

// MARK:

extension DataSource{
    
    @discardableResult
    func checkFolderExists(atPath:URL) -> Bool{
        var isDirectory = ObjCBool(true)
        if !FileManager.default.fileExists(atPath: atPath.path, isDirectory: &isDirectory){
            
            do {
                try FileManager.default.createDirectory(at: atPath, withIntermediateDirectories: false, attributes: nil)
            } catch let error as NSError {
                print(error.localizedDescription);
                return false
            }
        }
        
        return true
    }
    
}
