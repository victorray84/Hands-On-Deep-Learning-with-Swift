//
//  Data+Extension.swift
//  Hands-On Deep Learning with Swift - GAN
//
//  Created by joshua.newnham on 19/02/2019.
//  Copyright © 2019 Joshua Newnham. All rights reserved.
//

import Foundation

extension Data {
    
    public init<T>(fromArray values: [T]) {
        var values = values
        self.init(buffer: UnsafeBufferPointer(start: &values, count: values.count))
    }
    
    public func toArray<T>(type: T.Type) -> [T] {
        return self.withUnsafeBytes {
            [T](UnsafeBufferPointer(start: $0, count: self.count/MemoryLayout<T>    .stride))
        }
    }
}
