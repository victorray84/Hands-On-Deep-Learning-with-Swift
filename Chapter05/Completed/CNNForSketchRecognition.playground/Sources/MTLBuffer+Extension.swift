import Foundation
import MetalPerformanceShaders

extension MTLBuffer{
    
    public func toArray<T>(type: T.Type) -> [T] {
        let count = self.length / MemoryLayout<T>.stride
        let result = self.contents().bindMemory(to: type, capacity: count)
        var data = [T]()
        for i in 0..<count{
            data.append(result[i])
        }
        return data
    }    
}
