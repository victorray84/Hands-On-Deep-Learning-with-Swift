import AppKit

public extension NSImage{
    
    public var cgImage : CGImage?{
        get{
            var imageRect = CGRect(
                x: 0, y: 0,
                width: self.size.width, height: self.size.height)
            
            return self.cgImage(forProposedRect: &imageRect, context: nil, hints: nil)
        }
    }
}
