import AppKit

public extension NSImage{
    
    var pngData: Data? {
        get{
            guard let tiffRepresentation = self.tiffRepresentation,
                let bitmapImage = NSBitmapImageRep(data: tiffRepresentation) else {
                    return nil
                    
            }
            
            return bitmapImage.representation(using: .png, properties: [:])
        }
    }
    
    public var cgImage : CGImage?{
        get{
            var imageRect = CGRect(
                x: 0, y: 0,
                width: self.size.width, height: self.size.height)
            
            return self.cgImage(forProposedRect: &imageRect, context: nil, hints: nil)
        }
    }
    
    @discardableResult
    func pngWrite(to url: URL, options: Data.WritingOptions = .atomic) -> Bool {
        do {
            try pngData?.write(to: url, options: options)
            return true
        } catch {
            print(error)
            return false
        }
    }
}
