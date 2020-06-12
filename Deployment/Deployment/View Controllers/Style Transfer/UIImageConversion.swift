//
//  UIImageConversion.swift
//  Deployment
//
//  Created by 曹书恒 on 2020/6/12.
//  Copyright © 2020 220 284. All rights reserved.
//

import Foundation
import UIKit

extension UIImage {
    convenience init(pixelBuffer: CVPixelBuffer) {
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        let frame = CGRect.init(x: 0, y: 0, width: width, height: height)
        
        let ciImage = CIImage(cvImageBuffer: pixelBuffer)
        let ciContext = CIContext.init()
        let cgImage = ciContext.createCGImage(ciImage, from: frame)
        self.init(cgImage: cgImage!)
    }
    func resized(to size: CGSize) -> UIImage {
        return UIGraphicsImageRenderer(size: size).image { _ in
            draw(in: CGRect(origin: .zero, size: size))
        }
    }
    
    func pixelBuffer(to size: CGSize) -> CVPixelBuffer? {
        UIGraphicsBeginImageContextWithOptions(size, true, 2.0)
        self.draw(in: CGRect(x: 0, y: 0, width: size.width, height: size.height))
        UIGraphicsEndImageContext()
     
        let attrs = [kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue, kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue] as CFDictionary
        var pixelBuffer : CVPixelBuffer?
        let status = CVPixelBufferCreate(kCFAllocatorDefault, Int(size.width), Int(size.height), kCVPixelFormatType_32ARGB, attrs, &pixelBuffer)
        guard (status == kCVReturnSuccess) else {
            return nil
        }
           
        CVPixelBufferLockBaseAddress(pixelBuffer!, CVPixelBufferLockFlags(rawValue: 0))
        let pixelData = CVPixelBufferGetBaseAddress(pixelBuffer!)
           
        let rgbColorSpace = CGColorSpaceCreateDeviceRGB()
        let context = CGContext(data: pixelData, width: Int(size.width), height: Int(size.height), bitsPerComponent: 8, bytesPerRow: CVPixelBufferGetBytesPerRow(pixelBuffer!), space: rgbColorSpace, bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue)
           
        context?.translateBy(x: 0, y: size.width)
        context?.scaleBy(x: 1.0, y: -1.0)
        
        UIGraphicsPushContext(context!)
        self.draw(in: CGRect(x: 0, y: 0, width: Int(size.width), height: Int(size.height)))
        UIGraphicsPopContext()
        CVPixelBufferUnlockBaseAddress(pixelBuffer!, CVPixelBufferLockFlags(rawValue: 0))
            
        return pixelBuffer
    }
}
