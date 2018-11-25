import AppKit
import MetalPerformanceShaders
import Accelerate

public class WeightsView : NSView{
    
    public var cols : Int = 28{
        didSet{
            refresh()
        }
    }
    
    public var rows : Int = 28{
        didSet{
            refresh()
        }
    }
    
    public var weights : [Float]?{
        didSet{
            refresh()
        }
    }
    
    public override init(frame frameRect: NSRect) {
        
        super.init(frame:frameRect)
    }
    
    required init?(coder decoder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }
    
    public override func draw(_ dirtyRect: NSRect) {
        super.draw(dirtyRect)
        
        guard let context = NSGraphicsContext.current?.cgContext else{
            return
        }
        
        self.clear(usingContext:context, inRect:dirtyRect)
        self.drawWeights(usingContext:context, inRect:dirtyRect)
    }
}

// MARK: - Rendering

extension WeightsView{
    
    func clear(usingContext context:CGContext, inRect rect:NSRect){
        context.setFillColor(NSColor.clear.cgColor)
        context.fill(rect)
    }
    
    func drawWeights(usingContext context:CGContext, inRect rect:NSRect){
        guard let weights = self.weights else{
            return
        }
        
        context.setStrokeColor(NSColor.white.cgColor)
        context.setLineWidth(30.0)
        
        // calculate rect size
        let pixelSize = CGSize(width: rect.width / CGFloat(cols),
                               height: rect.height / CGFloat(rows))
        
        let padding = CGSize(width: (pixelSize.width * 0.05),
                             height: (pixelSize.height * 0.05))
        
        for row in 0..<rows{
            for col in 0..<cols{
                let pixelRect = CGRect(x: CGFloat(col) * pixelSize.width,
                                       y: CGFloat(row) * pixelSize.height,
                                       width: pixelSize.width,
                                       height: pixelSize.height).insetBy(
                                        dx: padding.width,
                                        dy: padding.height)
                
                let weightIndex = col + row * cols
                if(weightIndex < weights.count){
                    let weight = CGFloat(weights[weightIndex])
                    //let colourIntensity = 1.0 - pixel/255.0 // invert
                    let colourIntensity = weight // invert
                    
                    if colourIntensity < 0{
                        let pixelColor = NSColor(
                            red: abs(colourIntensity),
                            green: 0.0,
                            blue: 0.0,
                            alpha: 1.0)
                        
                        context.setFillColor(pixelColor.cgColor)
                    } else{
                        let pixelColor = NSColor(
                            red: 0.0,
                            green: 0.0,
                            blue: colourIntensity,
                            alpha: 1.0)
                        
                        context.setFillColor(pixelColor.cgColor)
                    }
                    
                    
                } else{
                    let pixelColor = NSColor(red: 1.0, green: 1.0, blue: 1.0, alpha: 1.0)
                    context.setFillColor(pixelColor.cgColor)
                }
                
                context.fill(pixelRect)
            }
        }
    }
}

// MARK: - Util

extension WeightsView{
    
    func getViewContext() -> CGContext? {
        guard let layer = self.layer else{
            return nil
        }
        
        // our network takes in only grayscale images as input
        let colorSpace:CGColorSpace = CGColorSpaceCreateDeviceGray()
        
        // we have 3 channels no alpha value put in the network
        let bitmapInfo = CGImageAlphaInfo.none.rawValue
        
        // this is where our view pixel data will go in once we make the render call
        guard let context = CGContext(
            data: nil,
            width: self.cols,
            height: self.rows,
            bitsPerComponent: 8,
            bytesPerRow: self.cols * 1,
            space: colorSpace,
            bitmapInfo: bitmapInfo) else{
                return nil
        }
        
        // scale and translate so we have the full digit and in MNIST standard size 28x28context
        context.translateBy(x: 0 , y: 0)
        context.scaleBy(
            x: CGFloat(self.cols)/self.frame.size.width,
            y: CGFloat(self.rows)/self.frame.size.height)
        
        // put view pixel data in context
        layer.render(in: context)
        
        return context
    }
    
    func refresh(){
        self.setNeedsDisplay(self.frame)
        self.displayIfNeeded()
    }
}
