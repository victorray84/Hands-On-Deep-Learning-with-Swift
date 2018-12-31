import AppKit
import MetalPerformanceShaders
import Accelerate

public class SketchCanvasView : NSView{
    
    var strokes = [[CGPoint]]()
    
    var exportSize : CGSize = CGSize(width: 128.0, height: 128.0)
    
    public var text : String?{
        didSet{
            if let text = self.text{
                self.label?.stringValue = text
            } else{
                self.label?.stringValue = ""
            }
        }
    }
    
    var clearButton : NSButton?
    
    var submitButton : NSButton?
    
    var label : NSTextField?
    
    var submitHandler : ((_ view:SketchCanvasView, _ context:CGContext?)  -> Void)?
    
    /// Padding added to the bounding box; used when clipping the context for export
    var padding : CGFloat{
        get{
            return min(self.frame.width * 0.05, self.frame.height * 0.05)
        }
    }
    
    /// Returns the bounding box containing the strokes (with padding)
    var boundingBox : NSRect{
        get{
            if self.strokes.count == 0{
                return self.frame
            }
            
            var minX = self.strokes.flatMap { $0 }.map { $0.x }.min() ?? 0
            var maxX = self.strokes.flatMap { $0 }.map { $0.x }.max() ?? (self.frame.size.width - 1)
            
            var minY = self.strokes.flatMap { $0 }.map { $0.y }.min() ?? 0
            var maxY = self.strokes.flatMap { $0 }.map { $0.y }.max() ?? (self.frame.size.height - 1)
            
            // Add padding
            minX = max(0, minX - self.padding)
            minY = max(0, minY - self.padding)
            
            maxX = min((self.frame.size.width - 1 - minX), maxX + self.padding)
            maxY = min((self.frame.size.height - 1 - minY), maxY + self.padding)
            
            let width = maxX - minX
            let height = maxY - minY
            
            return NSRect(
                x: minX,
                y: minY,
                width: width,
                height: height)
        }
    }
    
    public init(frame frameRect: NSRect,
                submitHandler:((_ view:SketchCanvasView, _ context:CGContext?)  -> Void)? = nil) {
        
        super.init(frame:frameRect)
        
        self.submitHandler = submitHandler
        
        self.initUI()
    }
    
    private func initUI(){
        
        // Clear button
        let clearButton = NSButton(
            frame: NSRect(
                x: 10,
                y: 10,
                width: 30,
                height: 30))
        clearButton.image = NSImage(named: "crossIcon")
        clearButton.imageScaling = .scaleProportionallyUpOrDown
        clearButton.isBordered = false
        clearButton.imagePosition = .imageOnly
        clearButton.target = self
        clearButton.action = #selector(SketchCanvasView.onClearClicked)
        
        self.addSubview(clearButton)
        self.clearButton = clearButton
        
        // Submit button
        let submitButton = NSButton(
            frame: NSRect(
                x: self.frame.width - (10 + 30),
                y: 10,
                width: 30,
                height: 30))
        submitButton.image = NSImage(named: "tickIcon")
        submitButton.imageScaling = .scaleProportionallyUpOrDown
        submitButton.isBordered = false
        submitButton.imagePosition = .imageOnly
        submitButton.target = self
        submitButton.action = #selector(SketchCanvasView.onSubmitClicked)
        
        self.addSubview(submitButton)
        self.submitButton = submitButton
        
        // Label
        let label = NSTextField(frame:
            NSRect(
                x: 10 + 30,
                y: 10,
                width: self.frame.width - (80),
                height: 28))
        label.textColor = NSColor(
            deviceRed: 0.0,
            green: 0.49,
            blue: 1.0,
            alpha: 1.0)
        label.isEditable = false
        label.isBordered = false
        label.isSelectable = false
        label.isBezeled = false
        label.alignment = .center
        label.font = NSFont.labelFont(ofSize: 12)
        label.backgroundColor = NSColor.clear
        self.addSubview(label)
        self.label = label
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
        self.drawPath(usingContext:context, inRect:dirtyRect)
    }
}

// MARK: - Callback methods

extension SketchCanvasView{
    
    @objc func onSubmitClicked(){
        if let submitHandler = self.submitHandler{
            submitHandler(self, self.getViewContext())
        }
    }
    
    @objc func onClearClicked(){
        self.reset()
    }
}

// MARK: - Rendering

extension SketchCanvasView{
    
    func clear(usingContext context:CGContext, inRect rect:NSRect){
        context.setFillColor(NSColor.black.cgColor)
        context.fill(rect)
    }
    
    func drawPath(usingContext context:CGContext, inRect rect:NSRect){
        guard strokes.count > 0 else{
            return
        }
        
        context.setStrokeColor(NSColor.white.cgColor)
        context.setLineWidth(10.0)
        
        for stroke in self.strokes{
            context.beginPath()
            
            if stroke.count < 1{
                continue
            }
            
            context.move(to: stroke[0])
            
            for i in 1..<stroke.count{
                context.addLine(to: stroke[i])
            }
            
            context.strokePath()
        }
    }
}

// MARK: - Util

extension SketchCanvasView{
    
    public func reset(){
        self.strokes.removeAll()
        self.text = nil
        self.refresh()
    }
    
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
            width: Int(self.exportSize.width),
            height: Int(self.exportSize.height),
            bitsPerComponent: 8,
            bytesPerRow: Int(exportSize.width),
            space: colorSpace,
            bitmapInfo: bitmapInfo) else{
                return nil
        }
        
//        // Get bounding box that encapsulates the sketch
//        let bbox = self.boundingBo
//
//        // Clip based on bounding box
//        context.clip(to: bbox)
        
        // scale and translate so we have the sketch in the standard size target size
        context.translateBy(x: 0 , y: 0)
        context.scaleBy(
            x: self.exportSize.width/self.frame.size.width,
            y: self.exportSize.height/self.frame.size.height)
        
        // put view pixel data in context
        layer.render(in: context)
        
        return context
    }
    
    func refresh(){
        self.setNeedsDisplay(self.frame)
        self.displayIfNeeded()
    }
}

// MARK: - User interaction

extension SketchCanvasView{
    
    override public func mouseDown(with event: NSEvent) {
        self.strokes.append([CGPoint]())
        self.strokes[self.strokes.count-1].append(event.locationInWindow)
        self.refresh()
    }
    
    public override func mouseDragged(with event: NSEvent) {
        self.strokes[self.strokes.count-1].append(event.locationInWindow)
        self.refresh()
    }
    
    public override func mouseUp(with event: NSEvent) {
        self.strokes[self.strokes.count-1].append(event.locationInWindow)
        self.refresh()
    }
}
