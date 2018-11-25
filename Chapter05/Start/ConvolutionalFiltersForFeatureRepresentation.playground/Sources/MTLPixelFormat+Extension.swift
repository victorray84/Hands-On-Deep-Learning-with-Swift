import MetalPerformanceShaders

public extension MTLPixelFormat{
    
    public var friendlyName : String{
        get{
            switch self{
                case .r16Float: return "r16Float"
                case .rg16Float: return "rg16Float"
                case .rgba16Float: return "rgba16Float"
                case .r32Float: return "r32Float"
                case .rg32Float: return "rg32Float"
                case .rgba32Float: return "rgba32Float"
                case .rgba8Unorm: return "rgba8Unorm"
                case .bgra8Unorm: return "bgra8Unorm"
                case .r8Unorm: return "r8Unorm"
                default: return "unknown"
            }
        }
    }
}
