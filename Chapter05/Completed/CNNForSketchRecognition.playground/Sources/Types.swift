import AppKit
import MetalKit
import MetalPerformanceShaders
import PlaygroundSupport

public typealias Sample = (image:MPSImage, label:MPSCNNLossLabels)

public typealias Batch = (images:[MPSImage], labels:[MPSCNNLossLabels])

public typealias KernelSize = (width:Int, height:Int)

public typealias Shape = (width:Int, height:Int, channels:Int)
