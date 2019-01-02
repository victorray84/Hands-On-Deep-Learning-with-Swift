import Foundation
import AppKit
import MetalKit
import MetalPerformanceShaders

public class SketchCNN{
    
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
    
    var datasources = [SketchCNNDatasource]()
    
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
        
        //print(self.graph!.debugDescription)
    }
}

// MARK: - Util

extension SketchCNN{
    
    func createConvLayer(name:String,
                         x:MPSNNImageNode,
                         kernelSize:KernelSize,
                         strideSize:KernelSize = KernelSize(width:1, height:1),
                         inputFeatureChannels:Int,
                         outputFeatureChannels:Int,
                         includeMaxPooling:Bool=true,
                         poolSize : Int = 2,
                         dropoutProbability:Float=0.0) -> [MPSNNFilterNode]{
        
        // TODO Implement convolutional block
        
        return []
    }
    
    func createDenseLayer(
        name:String,
        x:MPSNNImageNode,
        kernelSize:KernelSize,
        inputFeatureChannels:Int,
        outputFeatureChannels:Int,
        includeActivation:Bool=true,
        dropoutProbability:Float=0.0) -> [MPSNNFilterNode]{
        
        // TODO Implement dense layer block
        
        return []
    }
    
    private func makeMPSVector(count:Int, repeating:Float=0.0) -> MPSVector?{
        // TODO Create Vector
        return nil
    }
    
    private func makeOptimizer() -> MPSNNOptimizerStochasticGradientDescent?{
        guard self.mode == .training else{
            return nil
        }
        
        // TODO Create an instance of an MPSNNOptimizerStochasticGradientDescent with momentum
        return nil
    }
}

// MARK: - Inference

extension SketchCNN{
    
    public func predict(X:[MPSImage]) -> [[Float]]?{
        
        guard let graph = self.graph else{
            return nil
        }
        
        if let commandBuffer = self.commandQueue.makeCommandBuffer(){
            let outputs = graph.encodeBatch(
                to: commandBuffer,
                sourceImages: [X],
                sourceStates: nil)
            
            // Syncronize the outputs after the prediction has been made
            // so we can get access to the values on the CPU
            outputs?.forEach({ (output) in
                output.synchronize(on: commandBuffer)
            })
            
            // Commit the command to the GPU
            commandBuffer.commit()
            
            // Wait for it to finish
            commandBuffer.waitUntilCompleted()
            
            // Process outputs
            if let outputs = outputs{
                let predictions = outputs.map({ (output) -> [Float] in
                    if let probs = output.toFloatArray(){
                        return Array<Float>(probs[0..<self.numberOfClasses])
                    }
                    return [Float]()
                })
                
                return predictions
            }
        }
        return nil
    }
    
    private func createInferenceGraph() -> MPSNNGraph?{
        // TODO Create inference graph
        
        // 1. Create Input node
        
        // 2. Create convolutional layers
        
        // 3. Create dense layers
        
        // 4. Create SoftMax layer
        
        // 5. Create and return Graph
        
        return nil
    }
}

extension SketchCNN{
    
    @discardableResult
    public func train(
        withDataLoaderForTraining trainDataLoader:DataLoader,
        dataLoaderForValidation validDataLoader:DataLoader? = nil,
        epochs : Int = 500,
        completionHandler handler: @escaping () -> Void) -> [(epoch:Int, accuracy:Float)]{
        
        // Store validation accuracy as we train
        var validationAccuracy = [(epoch:Int, accuracy:Float)]()
        
        // TODO Create semaphore to coordinate (and work effectively with) the CPU and GPU activity
        
        // TODO Create variable to hold reference of the last command buffer
        
        // TODO Check initial validation score
        
        for epoch in 1...epochs{
            autoreleasepool{
                // TODO Reset the dataloader
                
                // TODO iterate through all the batches; performing a single training set on each
                
                // TODO wait for the final training step to complete
                
                // TODO Update and validate model every 5 epochs or on the last epoch
                // ... and output accuracy against the validation set
            }
        }
        
        print("Finished training")
        
        handler()
        
        return validationAccuracy
    }
    
    private func trainStep(withDataLoader dataLoader:DataLoader, semaphore:DispatchSemaphore) -> MTLCommandBuffer?{
        let _ = semaphore.wait(timeout: .distantFuture)
        
        guard let graph = self.graph else{
            semaphore.signal()
            return nil
        }
        
        // TODO Make a command buffer to execute the passing step on
        
        // TODO Get next batch
        
        // TODO Encode (passing in input and labels)
        
        // TODO Handle when the training step is complete and signal to the semaphore
        
        // TODO Commit job
        
        // TODO Return created command buffer
        return nil
    }
    
    func validate(withDataLoader dataLoader:DataLoader) -> Float{
        // TODO Create an inference network
        
        var sampleCount : Float = 0.0
        var predictionsCorrectCount : Float = 0.0
        
        // TODO Reset dataloader
        dataLoader.reset()
        
        // TODO Iterate over all batches of the dataloader
        
        // TODO For each batch
        // 1. Perform inference
        // 2. Count how many we got right
        
        return predictionsCorrectCount/sampleCount
    }
    
    private func updateDatasources(){
        // Syncronize weights and biases from GPU to CPU
        guard let commandBuffer = self.commandQueue.makeCommandBuffer() else{
            return
        }
        
        for datasource in self.datasources{
            datasource.synchronizeParameters(on: commandBuffer)
        }
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        // Persist the weightds and bias terms to disk
        for datasource in self.datasources{
            datasource.saveParametersToDisk()
        }
    }
    
    private func createTrainingGraph() -> MPSNNGraph?{
        
        // TODO Create input node
        
        // TODO Create nodes assoicated with the forward pass (same as the createInferenceGraph)
        
        // TODO Create loss node
        
        // TODO Create nodes associated with the backwards pass (reverse of the forward pass using the gradients)
        
        // TODO Create MPSNNGraph
        
        return nil
    }
}
