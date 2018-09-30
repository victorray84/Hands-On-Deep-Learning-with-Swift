/*:
 
 # [Hands-On Deep Learning with Swift]()
 ### Chapter 3 - Metal for Machine Learning
 *Writen by [Joshua Newnham](https://www.linkedin.com/in/joshuanewnham) and published by [Packt Publishing](https://www.packtpub.com/big-data-and-business-intelligence/machine-learning-core-ml)*
 */
import Foundation
import AppKit
import MetalKit
import MetalPerformanceShaders
import Accelerate
import AVFoundation
import PlaygroundSupport
import CoreGraphics

// Required to run tasks in the background
PlaygroundPage.current.needsIndefiniteExecution = true

let MNIST_IMAGE_WIDTH = 28
let MNIST_IMAGE_HEIGHT = 28
let MNIST_FEATURE_CHANNELS = 1 // grayscale
let MNIST_NUMBER_OF_CLASSES = 10 // 0 - 9


