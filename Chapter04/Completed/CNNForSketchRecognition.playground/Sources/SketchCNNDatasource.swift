import Foundation
import AppKit
import MetalKit
import MetalPerformanceShaders
import PlaygroundSupport

class SketchCNNDatasource : NSObject, MPSCNNConvolutionDataSource{
    
    static let FolderName = "sketch_cnn_weights"
    
    let name : String
    let kernelSize : KernelSize
    let strideSize : KernelSize
    let inputFeatureChannels : Int
    let outputFeatureChannels : Int
    
    var optimizer : MPSNNOptimizerStochasticGradientDescent?
    var weightsAndBiasesState : MPSCNNConvolutionWeightsAndBiasesState?
    
    var weightsData : Data?
    var biasTermsData : Data?
    
    let useBias : Bool
    
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
         kernelSize:KernelSize, strideSize:KernelSize=(width:1, height:1),
         inputFeatureChannels:Int, outputFeatureChannels:Int,
         optimizer:MPSNNOptimizerStochasticGradientDescent? = nil,
         useBias:Bool = true) {
        
        self.name = name
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
        print("purge")
        
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
                capacity: self.outputFeatureChannels * MemoryLayout<Float>.size)
    }
    
    func copy(with zone: NSZone? = nil) -> Any {
        let copy = SketchCNNDatasource(
            name: self.name,
            kernelSize: self.kernelSize,
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
        print("load")
        
        self.weightsData = self.loadWeights()
        self.biasTermsData = self.loadBiasTerms()
        
        return self.weightsData != nil
    }
    
    private func loadWeights() -> Data?{
        let url = playgroundSharedDataDirectory.appendingPathComponent("\(SketchCNNDatasource.FolderName)/\(self.name)_conv.data")
        
        do{
            print("loading weights \(url.absoluteString)")
            return try Data(contentsOf:url)
        } catch{
            print("Generating weights \(error)")
            return self.generateRandomWeights()
        }
    }
    
    private func loadBiasTerms() -> Data?{
        guard self.useBias else{
            return nil
        }
        
        let url = playgroundSharedDataDirectory.appendingPathComponent("\(SketchCNNDatasource.FolderName)/\(self.name)_bias.data")
        
        do{
            print("loading bias terms \(url.absoluteString)")
            return try Data(contentsOf:url)
        } catch{
            print("Generating bias \(error)")
            return self.generateBiasTerms()
        }
    }
    
    private func generateRandomWeights() -> Data?{
        let count = self.outputFeatureChannels
            * self.kernelSize.height
            * self.kernelSize.width
            * self.inputFeatureChannels
        
        var randomWeights = Array<Float>(repeating: 0, count: count)
        
        for o in 0..<self.outputFeatureChannels{
            for ky in 0..<self.kernelSize.height{
                for kx in 0..<self.kernelSize.width{
                    for i in 0..<self.inputFeatureChannels{
                        let index = ((o * self.kernelSize.height + ky)
                            * self.kernelSize.width + kx)
                            * self.inputFeatureChannels + i
                        randomWeights[index] = Float.getRandom(mean: 0.0, std: 0.01)
                    }
                }
            }
        }
        
        return Data(fromArray:randomWeights)
    }
    
    private func generateBiasTerms() -> Data?{
        let weightsCount = self.outputFeatureChannels
        let biasTerms = Array<Float>(repeating: 0.001, count: weightsCount)
        
        return Data(fromArray:biasTerms)
    }
}

// MARK: Datasource update methods

extension SketchCNNDatasource{
    
    // Update called when training on the CPU
    func update(with gradientState: MPSCNNConvolutionGradientState,
                sourceState: MPSCNNConvolutionWeightsAndBiasesState) -> Bool {
        
        return false
    }
    
    // Update called when training on the GPU
    func update(with commandBuffer: MTLCommandBuffer, gradientState: MPSCNNConvolutionGradientState, sourceState: MPSCNNConvolutionWeightsAndBiasesState) -> MPSCNNConvolutionWeightsAndBiasesState? {
        
        guard let optimizer = self.optimizer,
            let weightsAndBiasesState = self.weightsAndBiasesState else{
                return nil
        }
        
        optimizer.encode(
            commandBuffer: commandBuffer,
            convolutionGradientState: gradientState,
            convolutionSourceState: sourceState,
            inputMomentumVectors: nil,
            resultState: weightsAndBiasesState)
        
        return weightsAndBiasesState
    }
    
    func synchronizeParameters(on commandBuffer:MTLCommandBuffer){
        guard let weightsAndBiasesState = self.weightsAndBiasesState else{
            return
        }
        
        weightsAndBiasesState.synchronize(on: commandBuffer)
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
    
    func updateAndSaveParametersToDisk(){
        print("updateAndSaveParametersToDisk")
        guard let weightsAndBiasesState = self.weightsAndBiasesState else{
            return
        }
        
        self.weightsData = Data(fromArray:weightsAndBiasesState.weights.toArray(type: Float.self))
        
        if let biasData = weightsAndBiasesState.biases {
            self.biasTermsData = Data(
                fromArray:biasData.toArray(type: Float.self))
        }
        
        self.saveToDisk()
    }
    
    @discardableResult
    func saveToDisk() -> Bool{
        print("saveToDisk")
        return self.saveWeightsToDisk() && self.saveBiasTermsToDisk()
    }
    
    @discardableResult
    func saveWeightsToDisk() -> Bool{
        print("saveWeightsToDisk")
        guard let data = self.weightsData else{
            return false
        }
        
        // check the folder exists
        self.checkFolderExists(atPath: playgroundSharedDataDirectory.appendingPathComponent("\(SketchCNNDatasource.FolderName)"))
        
        let url = playgroundSharedDataDirectory.appendingPathComponent("\(SketchCNNDatasource.FolderName)/\(self.name)_conv.data")
        
        do{
            try data.write(to: url, options: NSData.WritingOptions.atomicWrite)
            print("Saved weights to \(url.absoluteString)")
            return true
        } catch{
            print("Failed to save weights to disk \(error)")
            return false
        }
    }
    
    @discardableResult
    func saveBiasTermsToDisk() -> Bool{
        print("saveBiasTermsToDisk")
        guard let data = self.biasTermsData else{
            return true
        }
        
        // check the folder exists
        self.checkFolderExists(atPath: playgroundSharedDataDirectory.appendingPathComponent("\(SketchCNNDatasource.FolderName)"))
        
        let url = playgroundSharedDataDirectory.appendingPathComponent("\(SketchCNNDatasource.FolderName)/\(self.name)_bias.data")
        
        do{
            try data.write(to: url, options: NSData.WritingOptions.atomicWrite)
            print("Saved bias terms to \(url.absoluteString)")
            return true
        } catch{
            print("Failed to save bias terms to disk \(error)")
            return false
        }
    }
}
