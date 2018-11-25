import Foundation
import GameplayKit

public extension Float{
    
    public static func getRandom(mean:Float, std:Float) -> Float{
        let randomSource = GKRandomSource()
        let x1 = randomSource.nextUniform() // a random number between 0 and 1
        let x2 = randomSource.nextUniform() // a random number between 0 and 1
        let z1 = sqrt(-2 * log(x1)) * cos(2 * Float.pi * x2) // z1 is normally distributed
        
        // Convert z1 from the Standard Normal Distribution to our Normal Distribution
        return z1 * std + mean
    }
    
}
