// TODO 

////
////  BatchNormDataSource.swift
////  GAN_MacOS
////
////  Created by joshua.newnham on 19/02/2019.
////  Copyright © 2019 Joshua Newnham. All rights reserved.
////
//
//import Foundation
//import AppKit
//import MetalKit
//import MetalPerformanceShaders
//
//class BatchNormDataSource : NSObject, MPSCNNBatchNormalizationDataSource{
//
//    public static let FolderName = "gan_weights"
//
//    let name : String
//    let featureChannels : Int
//    let weightsPathURL : URL
//
//    var optimizer : MPSNNOptimizerStochasticGradientDescent?
//    var gammaAndBetaState :  MPSCNNNormalizationGammaAndBetaState?
//    var meanAndVarianceState : MPSCNNNormalizationMeanAndVarianceState?
//
//    var batchNormValues = [Float]([0.0, 1.0, 0.0, 0.0])
//    var meanData : Data?
//    var
//
//    var momentumVectors : [MPSVector]?
//
//
//    init(name:String,
//         weightsPathURL:URL,
//         numberOfFeatureChannels:Int,
//         optimizer:MPSNNOptimizerStochasticGradientDescent? = nil){
//
//        self.name = name
//        self.weightsPathURL = weightsPathURL
//        self.featureChannels = numberOfFeatureChannels
//        self.optimizer = optimizer
//    }
//
//    /*! @abstract   Returns the number of feature channels within images to be normalized
//     *              using the supplied parameters.
//     */
//    public func numberOfFeatureChannels() -> Int{
//        return featureChannels
//    }
//
//
//    /*! @abstract   Returns a pointer to the scale factors for the batch normalization.
//     */
//    public func gamma() -> UnsafeMutablePointer<Float>?{
//
//    }
//
//
//    /*! @abstract   Returns a pointer to the bias terms for the batch normalization.
//     *              If NULL then no bias is to be applied.
//     */
//    public func beta() -> UnsafeMutablePointer<Float>?{
//
//    }
//
//
//    /*! @abstract   Returns a pointer to batch mean values with which to initialize
//     *              the state for a subsequent batch normalization.
//     */
//    public func mean() -> UnsafeMutablePointer<Float>?{
//        // index 0
//
//    }
//
//
//    /*! @abstract   Returns a pointer to batch variance values with which to initialize
//     *              the state for a subsequent batch normalization.
//     */
//    public func variance() -> UnsafeMutablePointer<Float>?{
//
//    }
//
//
//    /*! @abstract   Alerts the data source that the data will be needed soon
//     *  @discussion Each load alert will be balanced by a purge later, when MPS
//     *              no longer needs the data from this object.
//     *              Load will always be called atleast once after initial construction
//     *              or each purge of the object before anything else is called.
//     *  @return     Returns YES on success.  If NO is returned, expect MPS
//     *              object construction to fail.
//     */
//    public func load() -> Bool{
//
//    }
//
//
//    /*! @abstract   Alerts the data source that the data is no longer needed
//     *  @discussion Each load alert will be balanced by a purge later, when MPS
//     *              no longer needs the data from this object.
//     */
//    public func purge(){
//
//    }
//
//
//    /*! @abstract   A label that is transferred to the batch normalization filter at init time
//     *  @discussion Overridden by a MPSCNNBatchNormalizationNode.label if it is non-nil.
//     */
//    public func label() -> String?{
//        return self.name
//    }
//
//    /*! @abstract       An optional tiny number to use to maintain numerical stability.
//     *  @discussion     output_image = (input_image - mean[c]) * gamma[c] / sqrt(variance[c] + epsilon) + beta[c];
//     *                  Defalt value if method unavailable: FLT_MIN   */
//    public func epsilon() -> Float{
//        return Float.leastNormalMagnitude
//    }
//
//}
//
//// MARK: GPU Update functions
//
//extension BatchNormDataSource{
//
//    /*! @abstract       Compute new gamma and beta values using current values and gradients contained within a
//     *                  MPSCNNBatchNormalizationState.  Perform the update using a GPU.
//     *  @discussion     This operation is expected to also decrement the read count of batchNormalizationState by 1.
//     *
//     *  @param          commandBuffer               The command buffer on which to encode the update.
//     *
//     *  @param          batchNormalizationState     The MPSCNNBatchNormalizationState object containing the current gamma and
//     *                                              beta values and the gradient values.
//     *
//     *  @return         A MPSCNNNormalizationMeanAndVarianceState object containing updated mean and variance values.  If NULL, the MPSNNGraph
//     *                  batch normalization filter gamma and beta values will remain unmodified.
//     */
//    public func updateGammaAndBeta(with commandBuffer: MTLCommandBuffer, batchNormalizationState: MPSCNNBatchNormalizationState) -> MPSCNNNormalizationGammaAndBetaState?{
//
//        guard let optimizer = self.optimizer,
//            let gammaAndBetaState = self.gammaAndBetaState else{
//                return nil
//        }
//
//        optimizer.encode(
//            commandBuffer: commandBuffer,
//            batchNormalizationState: batchNormalizationState,
//            inputMomentumVectors: self.momentumVectors,
//            resultState: gammaAndBetaState)
//
//        return gammaAndBetaState
//    }
//
//
//    /*! @abstract       Compute new mean and variance values using current batch statistics contained within a
//     *                  MPSCNNBatchNormalizationState.  Perform the update using a GPU.
//     *  @discussion     This operation is expected to also decrement the read count of batchNormalizationState by 1.
//     *
//     *  @param          commandBuffer               The command buffer on which to encode the update.
//     *
//     *  @param          batchNormalizationState     The MPSCNNBatchNormalizationState object containing the current batch statistics.
//     *
//     *  @return         A MPSCNNNormalizationMeanAndVarianceState object containing updated mean and variance values.  If NULL, the MPSNNGraph
//     *                  batch normalization filter mean and variance values will remain unmodified.
//     */
//    @available(OSX 10.14, *)
//    public func updateMeanAndVariance(with commandBuffer: MTLCommandBuffer, batchNormalizationState: MPSCNNBatchNormalizationState) -> MPSCNNNormalizationMeanAndVarianceState?{
//
//        guard let optimizer = self.optimizer,
//            let meanAndVarianceState = self.meanAndVarianceState else{
//                return nil
//        }
//
//        optimizer.encode(commandBuffer: <#T##MTLCommandBuffer#>, batchNormalizationState: <#T##MPSCNNBatchNormalizationState#>, inputMomentumVectors: <#T##[MPSVector]?#>, resultState: <#T##MPSCNNNormalizationGammaAndBetaState#>)
//
//        optimizer.encode(commandBuffer: <#T##MTLCommandBuffer#>, batchNormalizationState: <#T##MPSCNNBatchNormalizationState#>, inputMomentumVectors: <#T##[MPSVector]?#>, resultState: <#T##MPSCNNNormalizationGammaAndBetaState#>)
//
////        optimizer.encode(
////            commandBuffer: commandBuffer,
////            convolutionGradientState: gradientState,
////            convolutionSourceState: sourceState,
////            inputMomentumVectors: self.momentumVectors,
////            resultState: weightsAndBiasesState)
////
////        return weightsAndBiasesState
//
//
//    }
//
//}
//
//// MARK: DataSource extension
//
//extension BatchNormDataSource{
//
//    func synchronizeParameters(on commandBuffer:MTLCommandBuffer){
//        self.gammaAndBetaState?.synchronize(on: commandBuffer)
//        self.meanAndVarianceState?.synchronize(on: commandBuffer)
//    }
//
//    func saveToDisk() -> Bool{
//
//    }
//
//    func saveWeightsToDisk() -> Bool{
//
//    }
//
//    func saveBiasTermsToDisk() -> Bool{
//
//    }
//}
