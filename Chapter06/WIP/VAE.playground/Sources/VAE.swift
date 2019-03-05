
import Foundation
import AppKit
import MetalKit
import MetalPerformanceShaders

public class VAE{
    
    public enum NetworkMode{
        case training
        case inference
    }
    
    let device : MTLDevice
    let commandQueue : MTLCommandQueue
    
    let weightsPathURL : URL
    
    let mode : NetworkMode
    
    let inputShape : Shape
    let numberOfClasses : Int
    
    public private(set) var learningRate : Float = 0.001
    
    public private(set) var momentumScale : Float = 0.2
    
    var graph : MPSNNGraph?
    
    var datasources = [VAEDatasource]()
    
    public init(withCommandQueue commandQueue:MTLCommandQueue,
                inputShape:Shape,
                numberOfClasses:Int,
                weightsPathURL:URL,
                mode:NetworkMode=NetworkMode.training,
                learningRate:Float=0.001,
                momentumScale:Float=0.2){
        
        self.commandQueue = commandQueue
        self.device = commandQueue.device
        self.inputShape = inputShape
        self.numberOfClasses = numberOfClasses
        self.weightsPathURL = weightsPathURL
        self.mode = mode
        self.learningRate = learningRate
        self.momentumScale = momentumScale
        
        if mode == .training{
            self.graph = self.createTrainingGraph()
        } else{
            self.graph = self.createInferenceGraph()
        }
    }
}

extension VAE{
    
    func createTrainingGraph() -> MPSNNGraph?{
        return nil
    }
    
    func createInferenceGraph() -> MPSNNGraph?{
        return nil
    }
    
    private func createConvolutionalBlock(
        name:String,
        inputChannels:Int, outputChannels:Int,
        kernelSize:KernelSize, strideSize:StrideSize)
    
    private func createEncoder(){
        let input = MPSNNImageNode(handle: nil)
        
    }
}
