import AppKit
import MetalKit
import MetalPerformanceShaders

/** A single training instance consisting of a image and the corresponding label */
public typealias Sample = (image:MPSImage, label:MPSCNNLossLabels)

/** A batch of samples that returned from the DataLoader and feed into the network */
public typealias Batch = (images:[MPSImage], labels:[MPSCNNLossLabels])

/** Kernel window size (used for convolutional and pooling layers */
public typealias KernelSize = (width:Int, height:Int)

public typealias StrideSize = (width:Int, height:Int)

/** 3D Shape of an input */
public typealias Shape = (width:Int, height:Int, channels:Int)

/** Used by the DataLoader to manage data retrieval */
public typealias SampleLookup = (label:String, index:Int)
