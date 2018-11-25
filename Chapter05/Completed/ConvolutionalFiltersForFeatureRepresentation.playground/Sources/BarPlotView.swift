import AppKit
import MetalPerformanceShaders

public struct PlotItem{
    public var label : String
    public var count : Float
}

public class BarPlotView : NSView{
    
    public var padding = CGSize(width: 0.05, height: 0.1)
    
    public var barColor = NSColor(red: 151/255,
                                   green: 151/255,
                                   blue: 151/255,
                                   alpha: 1.0)
    
    public var axisColor = NSColor(red: 151/255,
                                  green: 151/255,
                                  blue: 151/255,
                                  alpha: 1.0)
    
    public var textColor = NSColor(red: 151/255,
                                  green: 151/255,
                                  blue: 151/255,
                                  alpha: 1.0)
    
    var plotItems = [PlotItem]()
    
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
        self.drawPlot(usingContext:context, inRect:dirtyRect)
    }
}

// MARK: - Data item methods

extension BarPlotView{
    
    public func removeAll(){
        plotItems.removeAll()
        self.refresh()
    }
    
    @discardableResult
    public func add(label:String, image:CGImage) -> Bool{
        guard let imageData = image.toByteArray() else{
            return false
        }
        
        let pixelCount = Float(image.width * image.height)
        let pixelSummation = imageData.map { (pixel) -> Float in
            return pixel > 30 ? 1.0 : 0.0
            }.reduce(0) { (acc, pixel) -> Float in
                return acc + pixel
        }
        
        let imageIntensity = pixelSummation / pixelCount
        
        return self.add(label: label, imageIntensity: imageIntensity)
    }
    
    @discardableResult
    public func add(label:String, imageIntensity:Float) -> Bool{
        self.plotItems.append(PlotItem(label: label, count: imageIntensity))
        self.refresh()
        
        return true
    }
    
}

// MARK: - Rendering

extension BarPlotView{
    
    func refresh(){
        self.setNeedsDisplay(self.frame)
        self.displayIfNeeded()
    }
    
    func clear(usingContext context:CGContext, inRect rect:NSRect){
        context.setFillColor(NSColor.clear.cgColor)
        context.fill(rect)
    }
    
    func drawPlot(usingContext context:CGContext, inRect rect:NSRect){
        
        let paddingX : CGFloat = self.padding.width * rect.width
        let paddingY : CGFloat = self.padding.height * rect.height
        
        let barX : CGFloat = 0.15 * rect.width
        let barSpacing : CGFloat = barX/2.0
        
        let plotRect = NSRect(x: rect.origin.x + paddingX,
                              y: rect.origin.y + paddingY,
                              width: rect.size.width - (paddingX * 2),
                              height: rect.size.height - (paddingY * 2))
        
        self.drawAxis(
            usingContext:context,
            inRect:NSRect(x: (plotRect.minX - paddingX * 0.3),
                          y: plotRect.minY,
                          width: plotRect.maxX - (plotRect.minX - paddingX * 0.3),
                          height: plotRect.maxY - plotRect.minY))
        
        var currentX : CGFloat = plotRect.origin.x
        
        // font details
        let paragraphStyle = NSMutableParagraphStyle()
        paragraphStyle.alignment = .center
        
        let attrs = [
            NSAttributedString.Key.font: NSFont(name: "HelveticaNeue-Thin", size: 12)!,
            NSAttributedString.Key.paragraphStyle: paragraphStyle,
            NSAttributedString.Key.foregroundColor: self.textColor as Any
        ]
        
        guard let maxPlotItem = plotItems.sorted(by:{ (pi1, pi2) -> Bool in
            return pi1.count > pi2.count
        }).first else{ return }
        
        for plotItem in self.plotItems{
            context.setFillColor(self.barColor.cgColor)
            
            let barHeight = CGFloat(plotRect.height) * (CGFloat(plotItem.count) / CGFloat(maxPlotItem.count))
            
            let barRect = NSRect(x: currentX,
                              y: plotRect.minY,
                              width: barX,
                              height: barHeight)
            
            let textRect = NSRect(x: currentX,
                                  y: plotRect.minY - 20,
                                  width: barX,
                                  height: 20)
            
            context.fill(barRect)
            
            currentX += barX + barSpacing
            
            let text = plotItem.label as NSString
            text.draw(with: textRect,
                      options: .usesLineFragmentOrigin,
                      attributes: attrs)
        }
    }
    
    func drawAxis(usingContext context:CGContext, inRect rect:NSRect){
        context.setStrokeColor(self.axisColor.cgColor)
        context.setLineWidth(1.0)
        
        // y-axis
        context.beginPath()
        context.move(to: CGPoint(x: rect.minX, y: rect.minY))
        context.addLine(to: CGPoint(x: rect.minX, y: rect.maxY))
        context.strokePath()
        
        // x-axis
        context.beginPath()
        context.move(to: CGPoint(x: rect.minX, y: rect.minY))
        context.addLine(to: CGPoint(x: rect.maxX, y: rect.minY))
        context.strokePath()
        
        // add y-axis ticks
        let yTicks = 10
        let tickYLength = rect.height / CGFloat(yTicks)
        for i in 0..<yTicks{
            let oy = CGFloat(i) * tickYLength;
            context.beginPath()
            context.move(to: CGPoint(x: rect.minX, y: rect.minY + oy))
            context.addLine(to: CGPoint(x: rect.minX-3.0, y: rect.minY + oy))
            context.strokePath()
        }
    }
}
