import Foundation
import CoreGraphics

public extension CGImage{
    
    public static func fromByteArray(bytes:[UInt8], width:Int, height:Int, channels:Int) -> CGImage?{
        
        let bytesPerRow = width * channels
        
        var bytesR = bytes
        
        if let context = CGContext(
            data: &bytesR, width: width, height: height,
            bitsPerComponent: 8, bytesPerRow: bytesPerRow,
            space: channels == 1 ? CGColorSpaceCreateDeviceGray() : CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.none.rawValue){
            
            return context.makeImage()
        }
        
        return nil
    }
    
    public func toByteArray() -> [UInt8]?{
        guard let provider = self.dataProvider,
            let providerData = provider.data,
            let data = CFDataGetBytePtr(providerData) else{
                return nil
        }
        
        let count = self.bytesPerRow * self.height
        
        var pixels = Array<UInt8>(repeating: 0, count: count)
        
        for i in 0..<count{
            pixels[i] = data[i]
        }
        
        return pixels
    }
    
}

