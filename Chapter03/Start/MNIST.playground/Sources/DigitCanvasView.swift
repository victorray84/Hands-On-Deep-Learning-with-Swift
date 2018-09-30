import AppKit
import MetalPerformanceShaders
import Accelerate

public class DigitCanvasView : NSView{
    
    var strokes = [[CGPoint]]()
    
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
    
    var submitHandler : ((_ view:DigitCanvasView, _ context:CGContext?)  -> Void)?
    
    public init(frame frameRect: NSRect,
                device:MTLDevice,
                submitHandler:((_ view:DigitCanvasView, _ context:CGContext?)  -> Void)? = nil) {
        
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
        clearButton.action = #selector(DigitCanvasView.onClearClicked)
        
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
        submitButton.action = #selector(DigitCanvasView.onSubmitClicked)
        
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
        label.font = NSFont.labelFont(ofSize: 20)
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

extension DigitCanvasView{
    
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

extension DigitCanvasView{
    
    func clear(usingContext context:CGContext, inRect rect:NSRect){
        context.setFillColor(NSColor.black.cgColor)
        context.fill(rect)
    }
    
    func drawPath(usingContext context:CGContext, inRect rect:NSRect){
        guard strokes.count > 0 else{
            return
        }
        
        context.setStrokeColor(NSColor.white.cgColor)
        context.setLineWidth(25.0)
        
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

extension DigitCanvasView{
    
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
            width: 28,
            height: 28,
            bitsPerComponent: 8,
            bytesPerRow: 28,
            space: colorSpace,
            bitmapInfo: bitmapInfo) else{
                return nil
        }
        
        // scale and translate so we have the full digit and in MNIST standard size 28x28context
        context.translateBy(x: 0 , y: 0)
        context.scaleBy(
            x: 28/self.frame.size.width,
            y: 28/self.frame.size.height)
        
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

extension DigitCanvasView{
    
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


