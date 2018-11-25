/*:
 
 # [Hands-On Deep Learning with Swift]()
 ### Chapter 5 - Applying CNNs to recognise sketches
 *Writen by [Joshua Newnham](https://www.linkedin.com/in/joshuanewnham) and published by [Packt Publishing](https://www.packtpub.com/big-data-and-business-intelligence/machine-learning-core-ml)*
 */
import Foundation
import AppKit
import AVFoundation
import PlaygroundSupport
import CoreGraphics

// Required to run tasks in the background
PlaygroundPage.current.needsIndefiniteExecution = true

let MNIST_IMAGE_WIDTH = 28
let MNIST_IMAGE_HEIGHT = 28
let MNIST_FEATURE_CHANNELS = 1 // grayscale

func loadFile(_ path:String) -> Data?{
    let pathComponentns = path.components(separatedBy: ".")
    
    guard let url = Bundle.main.url(
        forResource: pathComponentns[0],
        withExtension: pathComponentns[1]) else{
            return nil
    }
    
    return try? Data(contentsOf: url)
}

func loadBiasTerms() -> Data?{
    return loadFile("/weights/h1_fc_1_bias_terms.data")
}

func loadWeights() -> Data?{
    return loadFile("/weights/h1_fc_1_wts.data")
}

func loadWeightsForIndex(index:Int) -> [Float]?{
    guard let data = loadWeights(),
        let biasTermsData = loadBiasTerms() else{
        return nil
    }
    
    let biasTerms = biasTermsData.toArray(type: Float.self)
    
    let count = MNIST_IMAGE_WIDTH * MNIST_IMAGE_HEIGHT * MNIST_FEATURE_CHANNELS
    let sIdx = count * index
    let eIdx = (sIdx + count)-1
    
    let dataArray = data.toArray(type: Float.self)
    var weights = Array(dataArray[sIdx...eIdx])
    
    return weights.map({ (w) -> Float in
        return w + biasTerms[index]
    })
}

func standardize(array:[Float]) -> [Float]{
    let min = array.min()!
    let max = array.max()!
    let r = max - min
    
    let z = array.map({ (w) -> Float in
        return 2 * ((w - min) / r) - 1
    })
    
    return z
}

guard let weights = loadWeightsForIndex(index: 7) else{
    fatalError("Unable to load weights file")
}

var stdWeights = standardize(array:weights)

let view = WeightsView(
    frame: NSRect(x: 20,
                  y: 20,
                  width: 300,
                  height: 300))

view.weights = stdWeights
PlaygroundPage.current.liveView = view
