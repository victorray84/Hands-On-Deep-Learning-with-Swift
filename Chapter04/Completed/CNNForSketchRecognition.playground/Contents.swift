/*:
 
 # [Hands-On Deep Learning with Swift]()
 ### Chapter 4 - Metal for Machine Learning
 *Writen by [Joshua Newnham](https://www.linkedin.com/in/joshuanewnham) and published by [Packt Publishing](https://www.packtpub.com/big-data-and-business-intelligence/machine-learning-core-ml)*
 */
import Foundation
import AppKit
import AVFoundation
import CoreGraphics
import MetalKit
import MetalPerformanceShaders
import Accelerate
import GLKit
import PlaygroundSupport


let BASE_SKETCHES_PATH = "Sketches/preprocessed"
let SKETCHES_URL = PlaygroundSupport
    .playgroundSharedDataDirectory
    .appendingPathComponent(BASE_SKETCHES_PATH)

var sketchUrls = [String:[URL]]()

do {
    let dirUrls = try FileManager.default
        .contentsOfDirectory(at: SKETCHES_URL, includingPropertiesForKeys:nil)
    for dirUrl in dirUrls{
        guard let fileUrls = try? FileManager.default
            .contentsOfDirectory(at: dirUrl, includingPropertiesForKeys:nil) else{
                continue
        }
        let label = dirUrl.pathComponents.last!
        sketchUrls[label] = fileUrls
    }
} catch {
    print("Error while enumerating files \(SKETCHES_URL.absoluteString): \(error.localizedDescription)")
}

let fileUrl = sketchUrls["monkey"]![0]
print(fileUrl)

let data = Data(contentsOf: fileUrl)

let img = NSImage(contentsOf: fileUrl)

if let cgImage = img?.cgImage{
    let data = cgImage.toByteArray()!
    let dataPointer = UnsafeMutableRawPointer(mutating: data)
    
    guard let device = MTLCreateSystemDefaultDevice() else{
        fatalError("Failed to get reference to GPU")
    }
    
    var descriptor = MPSImageDescriptor(
        channelFormat: MPSImageFeatureChannelFormat.unorm8,
        width: cgImage.width,
        height: cgImage.height,
        featureChannels: cgImage.width/cgImage.bytesPerRow)
    descriptor.numberOfImages = 1
    let mpsImage = MPSImage(device: device, imageDescriptor: descriptor)
    mpsImage.writeBytes(dataPointer,
                        dataLayout: MPSDataLayout.HeightxWidthxFeatureChannels, imageIndex: 0)
    
    
}

//let url = URL(fileURLWithPath: "images", isDirectory: true)
//
//let fm = FileManager.default
//let fileManager = FileManager.default
//let documentsURL = fileManager.urls(for: .documentDirectory, in: .userDomainMask)[0]
//
//let temp = fileManager.contents(atPath: "images/airplane/1.png")
//
//do {
//    let fileURLs = try fileManager.contentsOfDirectory(at: url, includingPropertiesForKeys: nil)
//    for fileURL in fileURLs{
//        print(fileURL)
//    }
//} catch {
//    print("Error while enumerating files \(url.path): \(error.localizedDescription)")
//}
