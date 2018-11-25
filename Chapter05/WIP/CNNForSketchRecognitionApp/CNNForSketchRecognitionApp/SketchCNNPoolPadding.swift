//
//  SketchCNNPoolPadding.swift
//  CNNForSketchRecognitionApp
//
//  Created by joshua.newnham on 13/11/2018.
//  Copyright © 2018 Joshua Newnham. All rights reserved.
//

import Foundation
import MetalPerformanceShaders

@objc
class SketchCNNPoolPadding : NSObject, MPSNNPadding{
    
    var mode : String = "forward"
    
    static var supportsSecureCoding: Bool = true
    
    override init() {
        super.init()
    }
    
    // MARK: NSCoding
    
    required init?(coder aDecoder: NSCoder) {
        super.init()
    }
    
    func encode(with aCoder: NSCoder) {
        // Ignore
    }
    
    // MARK: MPSNNPadding
    
    /*! @abstract   Get the preferred padding method for the node */
    public func paddingMethod() -> MPSNNPaddingMethod{
        return [ MPSNNPaddingMethod.custom, MPSNNPaddingMethod.validOnly ]
    }
    
    
    /*! A human readable string that describes the padding policy. Useful for verbose debugging support. */
    public func label() -> String{
        return self.className
    }
    
    /*! @abstract       Determine padding and sizing of result images
     *  @discussion     A MPSNNPaddingMethod must both return a valid MPSImageDescriptor
     *                  and set the MPSKernel.offset to the correct value.  This is a
     *                  required feature if the MPSNNPaddingMethodCustom bit is set in
     *                  the paddingMethod.
     *
     *                  Some code that may prove helpful:
     *
     *                  @code
     *                  const int centeringPolicy = 0;  // When kernelSize is even: 0 pad bottom right. 1 pad top left.    Centers the kernel for even sized kernels.
     *
     *                  typedef enum Style{
     *                      StyleValidOnly = -1,
     *                      StyleSame = 0,
     *                      StyleFull = 1
     *                  }Style;
     *
     *                  // Typical destination size in one dimension for forward filters (most filters)
     *                  static int DestSize( int sourceSize, int stride, int filterWindowSize, Style style ){
     *                      sourceSize += style * (filterWindowSize - 1);       // adjust how many pixels we are allowed to read
     *                      return (sourceSize + stride - 1) / stride;          // sourceSize / stride, round up
     *                  }
     *
     *                  // Typical destination size in one dimension for reverse filters (e.g. convolution transpose)
     *                  static int DestSizeReverse( int sourceSize, int stride, int filterWindowSize, Style style ){
     *                      return (sourceSize-1) * stride +        // center tap for the last N-1 results. Take stride into account
     *                              1 +                             // center tap for the first result
     *                              style * (filterWindowSize-1);   // add or subtract (or ignore) the filter extent
     *                  }
     *
     *                  // Find the MPSOffset in one dimension
     *                  static int Offset( int sourceSize, int stride, int filterWindowSize, Style style ){
     *                      // The correction needed to adjust from position of left edge to center per MPSOffset definition
     *                      int correction = filterWindowSize / 2;
     *
     *                      // exit if all we want is to start consuming pixels at the left edge of the image.
     *                      if( 0 )
     *                          return correction;
     *
     *                      // Center the area consumed in the source image:
     *                      // Calculate the size of the destination image
     *                      int destSize = DestSize( sourceSize, stride, filterWindowSize, style ); // use DestSizeReverse here instead as appropriate
     *
     *                      // calculate extent of pixels we need to read in source to populate the destination
     *                      int readSize = (destSize-1) * stride + filterWindowSize;
     *
     *                      // calculate number of missing pixels in source
     *                      int extraSize = readSize - sourceSize;
     *
     *                      // number of missing pixels on left side
     *                      int leftExtraPixels = (extraSize + centeringPolicy) / 2;
     *
     *                      // account for the fact that the offset is based on the center pixel, not the left edge
     *                      return correction - leftExtraPixels;
     *                  }
     *                  @endcode
     *
     *  @param          sourceImages        The list of source images to be used
     *  @param          sourceStates        The list of source states to be used
     *  @param          kernel              The MPSKernel the padding method will be applied to. Set the kernel.offset
     *  @param          inDescriptor        MPS will prepare a starting guess based on the padding policy (exclusive of
     *                                      MPSNNPaddingMethodCustom) set for the object. You should adjust the offset
     *                                      and image size accordingly. It is on an autoreleasepool.
     *
     *  @return         The MPSImageDescriptor to use to make a MPSImage to capture the results from the filter.
     *                  The MPSImageDescriptor is assumed to be on an autoreleasepool. Your method must also set the
     *                  kernel.offset property.
     */
    @available(OSX 10.13, *)
    public func destinationImageDescriptor(forSourceImages sourceImages: [MPSImage],
                                           sourceStates: [MPSState]?,
                                           for kernel: MPSKernel,
                                           suggestedDescriptor inDescriptor: MPSImageDescriptor) -> MPSImageDescriptor{
        
        if let kernel = kernel as? MPSCNNPooling {
            kernel.offset = MPSOffset(x: 1, y: 1, z: 0)
            kernel.edgeMode = .clamp
        }
                
        print("Pool padding policy \(self.mode)")
        for i in 0..<sourceImages.count{
            print("Source image \(i) \(sourceImages[i].width)x\(sourceImages[i].height)x\(sourceImages[i].featureChannels)")
        }
        print(inDescriptor.width)
        print(inDescriptor.height)
        print(inDescriptor.featureChannels)
        
        return inDescriptor
    }
}
