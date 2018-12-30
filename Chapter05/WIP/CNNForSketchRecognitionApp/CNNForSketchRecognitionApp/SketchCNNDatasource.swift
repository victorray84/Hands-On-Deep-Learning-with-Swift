import Foundation
import AppKit
import MetalKit
import MetalPerformanceShaders

class SketchCNNDatasource : NSObject, MPSCNNConvolutionDataSource{
    
    static let FolderName = "sketch_cnn_weights"
    
    let name : String
    let kernelSize : KernelSize
    let strideSize : KernelSize
    let inputFeatureChannels : Int
    let outputFeatureChannels : Int
    let weightsPathURL : URL
    
    var optimizer : MPSNNOptimizerStochasticGradientDescent?
    var weightsAndBiasesState : MPSCNNConvolutionWeightsAndBiasesState?
    
    var weightsData : Data?
    var biasTermsData : Data?
    
    let useBias : Bool
    
    // Reference to the underlying MPSCNNConvolution; useful for debugging 
    var cnnConvolution : MPSCNNConvolution?
    
    /// DEV
    var momentumVectors : [MPSVector]?
    
    var velocityVectors : [MPSVector]?
    
    var weightsLength : Int{
        get{
            return self.outputFeatureChannels
                * self.kernelSize.height
                * self.kernelSize.width
                * self.inputFeatureChannels
        }
    }
    
    var biasTermsLength : Int{
        get{
            return self.outputFeatureChannels
        }
    }
    
    ///
    
    
    lazy var cnnDescriptor : MPSCNNConvolutionDescriptor = {
        var descriptor = MPSCNNConvolutionDescriptor(
            kernelWidth: self.kernelSize.width,
            kernelHeight: self.kernelSize.height,
            inputFeatureChannels: self.inputFeatureChannels,
            outputFeatureChannels: self.outputFeatureChannels)
        
        descriptor.strideInPixelsX = self.strideSize.width
        descriptor.strideInPixelsY = self.strideSize.height
        
        return descriptor
    }()
    
    init(name:String,
         weightsPathURL:URL,
         kernelSize:KernelSize, strideSize:KernelSize=(width:1, height:1),
         inputFeatureChannels:Int, outputFeatureChannels:Int,
         optimizer:MPSNNOptimizerStochasticGradientDescent? = nil,
         useBias:Bool = true) {
        
        self.name = name
        self.weightsPathURL = weightsPathURL
        self.kernelSize = kernelSize
        self.strideSize = strideSize
        self.inputFeatureChannels = inputFeatureChannels
        self.outputFeatureChannels = outputFeatureChannels
        self.optimizer = optimizer
        self.useBias = useBias
    }
    
    func label() -> String? {
        return self.name
    }
    
    func dataType() -> MPSDataType {
        return MPSDataType.float32
    }
    
    func descriptor() -> MPSCNNConvolutionDescriptor {
        return self.cnnDescriptor
    }
    
    func purge() {
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
    
    func copy(with zone: NSZone? = nil) -> Any {
        let copy = SketchCNNDatasource(
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

extension SketchCNNDatasource{
    
    func load() -> Bool {
        self.weightsData = self.loadWeights()
        self.biasTermsData = self.loadBiasTerms()
        
//        /// DEV
//        if let weights = self.weightsData?.toArray(type: Float32.self){
//            print("Weights loaded \(self.name) ... \(weights.count) ... \(weights[0...10])")
//        }
//
//        if let bias = self.biasTermsData?.toArray(type: Float.self){
//            print("Bias Weights loaded \(self.name) ... \(bias.count) ... \(bias[0...10])")
//        }
//        /// 
        
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
        
        for index in 0..<count{
            randomWeights[index] = Float32.random(in: -0.1...0.1)
        }
        
        return Data(fromArray:randomWeights)
    }
    
    private func generateBiasTerms() -> Data?{
        let weightsCount = self.outputFeatureChannels
        
        var biasTerms = Array<Float>(repeating: 0.0, count: weightsCount)
        
        for index in 0..<biasTerms.count{
            biasTerms[index] = Float.random(in: -0.001...0.001)
        }
        
        return Data(fromArray:biasTerms)
    }
}

// MARK: Datasource update methods

extension SketchCNNDatasource{
    
    // Update called when training on the CPU
    func update(with gradientState: MPSCNNConvolutionGradientState,
                sourceState: MPSCNNConvolutionWeightsAndBiasesState) -> Bool {
        
        self.cnnConvolution = gradientState.convolution
        
        /// DEV
        if self.name == "l1"{
            let tmp = sourceState.weights.toArray(type: Float32.self)
            print("Weights \(self.name) ... \(tmp.count) ... \(tmp[0...10])")

            let tmp2 = gradientState.gradientForWeights.toArray(type: Float32.self)
            print("Gradient weights \(self.name) ... \(tmp2.count) ... \(tmp2[0...10])")
            
            let tmp3 = gradientState.gradientForBiases.toArray(type: Float.self)
            print("Gradient for bias terms \(self.name) ... \(tmp3.count) ... \(tmp3[0...10])")
        }

        if self.name == "l6"{
            let tmp = sourceState.weights.toArray(type: Float32.self)
            print("Weights \(self.name) ... \(tmp.count) ... \(tmp[0...10])")

            let tmp2 = gradientState.gradientForWeights.toArray(type: Float32.self)
            print("Gradient weights \(self.name) ... \(tmp2.count) ... \(tmp2[0...10])")
            
            let tmp3 = gradientState.gradientForBiases.toArray(type: Float.self)
            print("Gradient for bias terms \(self.name) ... \(tmp3.count) ... \(tmp3[0...10])")
        }
        ///
        
        return true
    }
    
    // Update called when training on the GPU
    func update(with commandBuffer: MTLCommandBuffer,
                gradientState: MPSCNNConvolutionGradientState,
                sourceState: MPSCNNConvolutionWeightsAndBiasesState) -> MPSCNNConvolutionWeightsAndBiasesState? {
        
        guard let optimizer = self.optimizer,
            let weightsAndBiasesState = self.weightsAndBiasesState else{
                return nil
        }
        
        // You can get reference to the underlying MPSCNNConvolution via gradientState.convolution;
        // having access to the convolution layer exposes more properties of the controlutional layer. For
        // example; you can obtain the current layers weights via the function MPSCNNConvolution.exportWeightsAndBiases
        // (instead of our approach of retaining explicit reference via the DataSources weightsAndBiasesState property)
        self.cnnConvolution = gradientState.convolution
        
//        /// DEV
//        sourceState.readCount += 1
//
//        optimizer.encode(
//            commandBuffer: commandBuffer,
//            convolutionGradientState: gradientState,
//            convolutionSourceState: sourceState,
//            inputMomentumVectors: nil,
//            resultState: sourceState)
//        ///
        
//        sourceState.readCount += 1
        
//        optimizer.encode(
//            commandBuffer: commandBuffer,
//            convolutionGradientState: gradientState,
//            convolutionSourceState: sourceState,
//            inputMomentumVectors: nil,
//            resultState: weightsAndBiasesState)
        
        optimizer.encode(
            commandBuffer: commandBuffer,
            convolutionGradientState: gradientState,
            convolutionSourceState: sourceState,
            inputMomentumVectors: self.momentumVectors,
            resultState: weightsAndBiasesState)
        
//        optimizer.encode(
//            commandBuffer: commandBuffer,
//            convolutionGradientState: gradientState,
//            convolutionSourceState: sourceState,
//            inputMomentumVectors: nil,
//            resultState: sourceState)
        
//        return weightsAndBiasesState
        
        //gradientState.readCount -= 1
        
//        return sourceState
        
//        return weightsAndBiasesState
        
        return weightsAndBiasesState
    }
    
    func synchronizeParameters(on commandBuffer:MTLCommandBuffer){
        // Get reference to the weights and biases state
//        self.weightsAndBiasesState = self.cnnConvolution?.exportWeightsAndBiases(
//            with: commandBuffer, resultStateCanBeTemporary: false)
        
        // Syncronise the weights so we can access them on the CPU (and save to disk)
        self.weightsAndBiasesState?.synchronize(on: commandBuffer)
    }
}

// MARK: Datasource save

extension SketchCNNDatasource{
    
    @discardableResult
    private func checkFolderExists(atPath:URL) -> Bool{
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
    
    func saveParametersToDisk(){
        guard let weightsAndBiasesState = self.weightsAndBiasesState else{
            fatalError("Dependent variable weightsAndBiasesState is null")
        }

//        /// DEV
//        let data = weightsAndBiasesState.weights.toArray(type: Float.self)
//        if data[0] == Float.nan{
//            print("Weights are NAN for \(self.name)")
//        }
//        ///
        
        /// DEV
        if self.name == "l1"{
            let wtsArray = weightsAndBiasesState.weights.toArray(type: Float32.self)
            print("weights")
            print(wtsArray[0..<10])
            
            if let biasArray = weightsAndBiasesState.biases?.toArray(type: Float.self){
                print("bias terms")
                print(biasArray[0..<10])
            }
            
            if let velArray = self.velocityVectors?[0].data.toArray(type: Float32.self){
                print("velcoity")
                print(velArray[0..<10])
            }
            
            if let momentumArray = self.momentumVectors?[0].data.toArray(type: Float32.self){
                print("momentum")
                print(momentumArray[0..<10])
            }
        }
        ///
        
        self.weightsData = Data(fromArray:weightsAndBiasesState.weights.toArray(type: Float32.self))

        if let biasData = weightsAndBiasesState.biases {
            let biasDataArray = biasData.toArray(type: Float.self)
            self.biasTermsData = Data(fromArray:biasDataArray)
        }
        
        self.saveToDisk()                
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
