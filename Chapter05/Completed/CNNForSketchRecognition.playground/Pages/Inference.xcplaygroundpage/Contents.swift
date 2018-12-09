//: [Previous](@previous)

import Foundation

var str = "Hello, playground"

print(randomWeights(count:10, seed:123))

print(randomWeights(count:10, seed:456))

print(random(count:10))

print(random(count:10))

func randomWeights(count: Int, seed: Int) -> [Float] {
    srand48(seed)
    var a = [Float](repeating: 0, count: count)
    for i in 0..<count {
        a[i] = Float(drand48() - 0.5) * 0.3
    }
    return a
}

func random(count:Int) -> [Float]{
    var a = [Float](repeating: 0, count: count)
    for i in 0..<count {
        a[i] = Float.random(in: 0...0.01)
    }
    return a
}

//: [Next](@next)
