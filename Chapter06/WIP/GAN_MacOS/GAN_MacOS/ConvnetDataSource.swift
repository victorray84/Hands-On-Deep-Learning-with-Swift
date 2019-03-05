import Foundation
import AppKit
import MetalKit
import MetalPerformanceShaders

public class ConvnetDataSource : NSObject, MPSCNNConvolutionDataSource, DataSource{
    
    public static let FolderName = "gan_weights"
    
    let name : String
    let kernelSize : KernelSize
    let strideSize : KernelSize
    let inputFeatureChannels : Int
    let outputFeatureChannels : Int
    let weightsPathURL : URL
    
    var trainable : Bool = true
    
    var optimizer : MPSNNOptimizerAdam?
    var weightsAndBiasesState : MPSCNNConvolutionWeightsAndBiasesState?
    
    var weightsData : Data?
    var biasTermsData : Data?
    
    let useBias : Bool
    
    var momentumVectors : [MPSVector]?
    
    var velocityVectors : [MPSVector]?
    
    var weightsLength : Int{
        get{
            return self.outputFeatureChannels *
                self.kernelSize.height *
                self.kernelSize.width *
                self.inputFeatureChannels
        }
    }
    
    var biasTermsLength : Int{
        get{
            return self.outputFeatureChannels
        }
    }
    
    init(name:String,
         weightsPathURL:URL,
         kernelSize:KernelSize, strideSize:KernelSize=(width:1, height:1),
         inputFeatureChannels:Int, outputFeatureChannels:Int,
         optimizer:MPSNNOptimizerAdam? = nil,
         useBias:Bool = true){
        
        self.name = name
        self.weightsPathURL = weightsPathURL
        self.kernelSize = kernelSize
        self.strideSize = strideSize
        self.inputFeatureChannels = inputFeatureChannels
        self.outputFeatureChannels = outputFeatureChannels
        self.optimizer = optimizer
        self.useBias = useBias
    }
    
    public func label() -> String? {
        return self.name
    }
    
    public func dataType() -> MPSDataType {
        return MPSDataType.float32
    }
    
    public func descriptor() -> MPSCNNConvolutionDescriptor {
        let descriptor = MPSCNNConvolutionDescriptor(
            kernelWidth: self.kernelSize.width,
            kernelHeight: self.kernelSize.height,
            inputFeatureChannels: self.inputFeatureChannels,
            outputFeatureChannels: self.outputFeatureChannels)
        
        descriptor.strideInPixelsX = self.strideSize.width
        descriptor.strideInPixelsY = self.strideSize.height
        
        return descriptor
    }
    
    public func purge() {
        self.weightsData = nil
        self.biasTermsData = nil
    }
    
    public func weights() -> UnsafeMutableRawPointer{
        return UnsafeMutableRawPointer(mutating: (self.weightsData! as NSData).bytes)
    }
    
    public func biasTerms() -> UnsafeMutablePointer<Float>?{
        guard let biasTermsData = self.biasTermsData else{
            return nil
        }
        
        return UnsafeMutableRawPointer(
            mutating: (biasTermsData as NSData).bytes).bindMemory(
                to: Float.self,
                capacity: self.outputFeatureChannels * MemoryLayout<Float>.stride)
    }
    
    public func copy(with zone: NSZone? = nil) -> Any {
        let copy = ConvnetDataSource(
            name: self.name,
            weightsPathURL:self.weightsPathURL,
            kernelSize: self.kernelSize,
            strideSize: self.strideSize,
            inputFeatureChannels: self.inputFeatureChannels,
            outputFeatureChannels: self.outputFeatureChannels,
            optimizer: self.optimizer)
        
        copy.weightsAndBiasesState = self.weightsAndBiasesState
        return copy as Any
    }
}

// MARK: Datasource load methods

extension ConvnetDataSource{
    
    public func load() -> Bool {
        self.weightsData = self.loadWeights()
        self.biasTermsData = self.loadBiasTerms()
        
        return self.weightsData != nil
    }
    
    private func loadWeights() -> Data?{
        let url = self.weightsPathURL.appendingPathComponent("\(self.name)_conv.data")
        
        do{
            return try Data(contentsOf:url)
        } catch{
            // Generate weights
            return self.generateRandomWeights()
        }
    }
    
    private func loadBiasTerms() -> Data?{
        guard self.useBias else{
            return nil
        }
        
        let url = self.weightsPathURL.appendingPathComponent("\(self.name)_bias.data")
        
        do{
            return try Data(contentsOf:url)
        } catch{
            // Generate bias terms
            return self.generateBiasTerms()
        }
    }
    
    private func generateRandomWeights() -> Data?{
        let count = self.outputFeatureChannels
            * self.kernelSize.height
            * self.kernelSize.width
            * self.inputFeatureChannels
        
        var randomWeights = Array<Float32>(repeating: 0, count: count)
        
        /*
         He normal initializer: draws samples from a truncated normal distribution centered on 0
         with stddev = sqrt(2 / fan_in) where fan_in is the number of input units in the weight tensor.
         Reference: https://keras.io/initializers/
        */
        //let std : Float = Float(sqrt(2.0 / Double(self.inputFeatureChannels)))
        
        /*
         Glorot uniform initializer, also called Xavier uniform initializer: Draws samples from a uniform distribution
         within [-limit, limit] where limit is
         sqrt(6 / (fan_in + fan_out)) where fan_in is the number of input units in
         the weight tensor and fan_out is the number of output units in the weight tensor.
         Reference: https://keras.io/initializers/
        */
        let limit = sqrt(6.0 / (Double(self.inputFeatureChannels + self.outputFeatureChannels)))
        
        for index in 0..<count{
            // He normal initializer
            //randomWeights[index] = Float.truncatedRandomNormal(mean: 0.0, std: std)
            
            // Glorot uniform initializer (default in Keras)
            randomWeights[index] = Float(Double.random(in: -limit...limit))
        }
        
        return Data(fromArray:randomWeights)
    }
    
    private func generateBiasTerms() -> Data?{
        let weightsCount = self.outputFeatureChannels
        
        let biasTerms = Array<Float>(repeating: 0.0, count: weightsCount)
        return Data(fromArray:biasTerms)
    }
}

// MARK: Datasource update methods

extension ConvnetDataSource{
    
    // Update called when training on the CPU
    func update(with gradientState: MPSCNNConvolutionGradientState,
                sourceState: MPSCNNConvolutionWeightsAndBiasesState) -> Bool {
        return false
    }
    
    // Update called when training on the GPU
    public func update(with commandBuffer: MTLCommandBuffer,
                gradientState: MPSCNNConvolutionGradientState,
                sourceState: MPSCNNConvolutionWeightsAndBiasesState) -> MPSCNNConvolutionWeightsAndBiasesState? {
        
        // Get reference tot he weights and bias
        guard let weightsAndBiasesState = self.weightsAndBiasesState,
            let optimizer = self.optimizer else{
            return nil
        }
        
        guard self.trainable else{
            gradientState.readCount -= 1
            sourceState.readCount -= 1
            // this should reflect the updated weights for this datasource (via a different network)
            return weightsAndBiasesState
        }
        
        optimizer.encode(
            commandBuffer: commandBuffer,
            convolutionGradientState: gradientState,
            convolutionSourceState: sourceState,
            inputMomentumVectors: self.momentumVectors,
            inputVelocityVectors: self.velocityVectors,
            resultState: weightsAndBiasesState)
        
        return weightsAndBiasesState
    }        
}

// MARK: Syncronization 

extension ConvnetDataSource{
    
    func synchronizeParameters(on commandBuffer:MTLCommandBuffer){
        // Syncronise the weights so we can access them on the CPU (and save to disk)
        self.weightsAndBiasesState?.synchronize(on: commandBuffer)
    }
    
}

// MARK: Datasource save

extension ConvnetDataSource{
    
    public func saveParametersToDisk() -> Bool{
        guard let weightsAndBiasesState = self.weightsAndBiasesState else{
            fatalError("Dependent variable weightsAndBiasesState is null")
        }
        
        self.weightsData = Data(fromArray:weightsAndBiasesState.weights.toArray(type: Float32.self))
        
        if let biasData = weightsAndBiasesState.biases {
            let biasDataArray = biasData.toArray(type: Float.self)
            self.biasTermsData = Data(fromArray:biasDataArray)
        }
        
        return self.saveToDisk()
    }
    
    @discardableResult
    func saveToDisk() -> Bool{
        return self.saveWeightsToDisk() && self.saveBiasTermsToDisk()
    }
    
    @discardableResult
    func saveWeightsToDisk() -> Bool{
        guard let data = self.weightsData else{
            return false
        }
        
        // check the folder exists
        self.checkFolderExists(atPath: self.weightsPathURL)
        
        let url = self.weightsPathURL.appendingPathComponent("\(self.name)_conv.data")
        
        do{
            try data.write(to: url, options: NSData.WritingOptions.atomicWrite)
            return true
        } catch{
            print("Failed to save weights to disk \(error)")
            return false
        }
    }
    
    @discardableResult
    func saveBiasTermsToDisk() -> Bool{
        guard self.useBias,
            let data = self.biasTermsData else{
                return true
        }
        
        // check the folder exists
        self.checkFolderExists(atPath: self.weightsPathURL)
        
        let url = self.weightsPathURL.appendingPathComponent("\(self.name)_bias.data")
        
        do{
            try data.write(to: url, options: NSData.WritingOptions.atomicWrite)
            return true
        } catch{
            print("Failed to save bias terms to disk \(error)")
            return false
        }
    }
}
