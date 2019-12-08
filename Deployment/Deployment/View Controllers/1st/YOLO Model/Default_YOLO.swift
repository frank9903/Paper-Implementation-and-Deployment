//
//  Default_YOLO.swift
//  Deployment
//
//  Created by 220 284 on 12/7/19.
//  Copyright © 2019 220 284. All rights reserved.
//

import Foundation
import Vision
import UIKit

class Default_YOLO {
    let yolo = YOLO()
    var requests = [VNCoreMLRequest]()
    var boundingBoxes = [BoundingBox]()
    var colors: [UIColor] = []
    let ciContext = CIContext()
    var resizedPixelBuffers: [CVPixelBuffer?] = []
    
    init() {
        setUpBoundingBoxes()
    }
    
    func setUpBoundingBoxes() {
        for _ in 0..<YOLO.maxBoundingBoxes {
            boundingBoxes.append(BoundingBox())
        }
        
        // Make colors for the bounding boxes. There is one color for each class,
        // 20 classes in total.
        for r: CGFloat in [0.2, 0.4, 0.6, 0.8, 1.0] {
            for g: CGFloat in [0.3, 0.7] {
                for b: CGFloat in [0.4, 0.8] {
                    let color = UIColor(red: r, green: g, blue: b, alpha: 1)
                    colors.append(color)
                }
            }
        }
    }
    
    func predict(image: UIImage) -> [YOLO.Prediction] {
        if let pixelBuffer = image.pixelBuffer(width: YOLO.inputWidth, height: YOLO.inputHeight) {
            return predict(pixelBuffer: pixelBuffer, inflightIndex: 0)
        }
        return []
    }
    
    func predict(pixelBuffer: CVPixelBuffer, inflightIndex: Int) -> [YOLO.Prediction] {
        // Resize the image (using vImage):
        if let resizedPixelBuffer = resizePixelBuffer(pixelBuffer,
                                                      width: YOLO.inputWidth,
                                                      height: YOLO.inputHeight) {
            let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
            let sx = CGFloat(YOLO.inputWidth) / CGFloat(CVPixelBufferGetWidth(pixelBuffer))
            let sy = CGFloat(YOLO.inputHeight) / CGFloat(CVPixelBufferGetHeight(pixelBuffer))
            let scaleTransform = CGAffineTransform(scaleX: sx, y: sy)
            let scaledImage = ciImage.transformed(by: scaleTransform)
            ciContext.render(scaledImage, to: resizedPixelBuffer)
            
            // Give the resized input to our model.
            if let boundingBoxes = yolo.predict(image: resizedPixelBuffer) {
                return boundingBoxes
            } else {
                print("BOGUS")
            }
        }
        return []
    }
    
    func textToImage(drawText text: String, textColor: UIColor, inImage image: UIImage, atPoint point: CGPoint) -> UIImage {
        let textColor = textColor
        let textFont = UIFont(name: "American Typewriter", size: 20)!
        
        UIGraphicsBeginImageContextWithOptions(image.size, false, 1.0)
        
        let textFontAttributes = [
            NSAttributedString.Key.font: textFont,
            NSAttributedString.Key.foregroundColor: textColor,
            ] as [NSAttributedString.Key : Any]
        image.draw(in: CGRect(origin: CGPoint.zero, size: image.size))
        
        let rect = CGRect(origin: point, size: image.size)
        text.draw(in: rect, withAttributes: textFontAttributes)
        
        let newImage = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        
        return newImage!
    }
    
    func rectangleToImage(image: UIImage, rect:CGRect, color:UIColor) -> UIImage {
        let imageSize = image.size
        UIGraphicsBeginImageContextWithOptions(imageSize, false, 1)
        
        image.draw(at: CGPoint.zero)
        
//        color.setFill()
//        color.withAlphaComponent(0.5)
//        color.setStroke()
//        UIRectFill(rect)
        let c=UIGraphicsGetCurrentContext()!
        c.setStrokeColor(color.cgColor)
        c.setLineWidth(0.005*image.size.width)
        c.setAlpha(0.5)
        c.stroke(rect)
        
        let newImage = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        return newImage!
    }
    
    func show(image:UIImage, predictions: [YOLO.Prediction]) -> UIImage {
        var final_image=image
        for i in 0..<predictions.count {
            let prediction = predictions[i]

            if (prediction.score > 0.333) {
                // The predicted bounding box is in the coordinate space of the input
                // image, which is a square image of 416x416 pixels. We want to show it
                // on the video preview, which is as wide as the screen and has a 16:9
                // aspect ratio. The video preview also may be letterboxed at the top
                // and bottom.
                let width = image.size.width
                let height = image.size.height
                let scaleX = width / CGFloat(YOLO.inputWidth)
                let scaleY = height / CGFloat(YOLO.inputHeight)

                // Translate and scale the rectangle to our own coordinate system.
                var rect = prediction.rect
                rect.origin.x *= scaleX
                rect.origin.y *= scaleY
                rect.size.width *= scaleX
                rect.size.height *= scaleY

                let label = String(format: "%@ %.1f", labels[prediction.classIndex], prediction.score * 100)
                let color = colors[Int.random(in: 0..<20)]
                
                final_image = textToImage(drawText: label, textColor: color, inImage: final_image, atPoint: rect.origin)
                final_image = rectangleToImage(image: final_image, rect: rect, color: color)
    //            UIGraphicsBeginImageContextWithOptions(final_image.size, false, 1.0)
    //            final_image.draw(in: CGRect(x: 0, y: 0, width: final_image.size.width, height: final_image.size.height))
    //
    //            //rect.origin.y=1-rect.origin.y
    //            let conv_rect=CGRect(x: rect.origin.x*final_image.size.width, y: rect.origin.y*final_image.size.height, width: rect.width*final_image.size.width, height: rect.height*final_image.size.height)
    //
    //            let c=UIGraphicsGetCurrentContext()!
    //            c.setStrokeColor(UIColor.red.cgColor)
    //            c.setLineWidth(0.01*final_image.size.width)
    //            c.stroke(conv_rect)
    //
    //            let result=UIGraphicsGetImageFromCurrentImageContext()
    //            UIGraphicsEndImageContext()
    //
    //            final_image=result!
            }
        }
        return final_image
    }
}
