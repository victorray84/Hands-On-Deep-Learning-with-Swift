//
//  Float+Extension.swift
//  GAN_MacOS
//
//  Created by joshua.newnham on 03/03/2019.
//  Copyright © 2019 Joshua Newnham. All rights reserved.
//

import Foundation
import GameKit

extension Float{
    
    public static func uniformRandom(mean:Float, std:Float) -> Float{
        let randomSource = GKRandomSource()
        let x1 = randomSource.nextUniform() // a random number between 0 and 1
        let x2 = randomSource.nextUniform() // a random number between 0 and 1
        let z1 = sqrt(-2 * log(x1)) * cos(2 * Float.pi * x2) // z1 is normally distributed

        // Convert z1 from the Standard Normal Distribution to our Normal Distribution
        return z1 * std + mean
    }
    
//    public static func randomNormal(mean:Float, std:Float) -> Float{
//        let r2 = -2.0 * log(Double.random(in: 0...1))
//        let theta = 2.0 * Double.pi * Double.random(in: 0...1)
//
//        let rand = (sqrt(r2) * cos(theta)) + Double(mean)
//
//        return Float(rand)
//    }
    
    public static func randomNormal(mean:Float, deviation:Float, randomSource:GKRandomSource?=nil) -> Float{
        guard deviation > 0 else { return mean }
        
        let randomSource = randomSource ?? GKRandomSource.sharedRandom()
        
        let x1 = randomSource.nextUniform() // a random number between 0 and 1
        let x2 = randomSource.nextUniform() // a random number between 0 and 1
        let z1 = sqrt(-2 * log(x1)) * cos(2 * Float.pi * x2) // z1 is normally distributed
        
        // Convert z1 from the Standard Normal Distribution to our Normal Distribution
        return z1 * deviation + mean
    }
    
    /*:
     These values are similar to values from a `randomNormal`
     except that values more than two standard deviations from the mean
     are discarded and redrawn. This is the recommended initializer for
     neural network weights and filters.
     */
    public static func truncatedRandomNormal(mean:Float, std:Float) -> Float{
        let twoStd = Double(std) * Double(std)
        var rand : Double = Double(std) * 3.0
        
        var iterations = 0 // keep track of attempts to avoid being 'stuck'
        while iterations < 100 && rand > twoStd{
            let r2 = -2.0 * log(Double.random(in: 0...1))
            let theta = 2.0 * Double.pi * Double.random(in: 0...1)
            
            rand = (sqrt(r2) * cos(theta)) + Double(mean)
            
            iterations += 1
        }
        return Float(rand)
        
    }
    
}
