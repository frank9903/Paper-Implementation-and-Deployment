//
//  Image Caption Controller.swift
//  Deployment
//
//  Created by 曹书恒 on 2020/5/28.
//  Copyright © 2020 220 284. All rights reserved.
//

import UIKit
import CoreML
import Vision

class ImageCaptionController: UIViewController {
    // UI related variables
    @IBOutlet weak var input: UIImageView!
    @IBOutlet weak var result: UITextField!
    var imagePicker: ImagePicker!
    
    // ML Model variables
    var ENCODER:VNCoreMLModel = try! VNCoreMLModel(for: encoder().model)
    var MAP = map()
    var GRU = gru()
    var idx2word:[Int:String] = [:]
    var word2idx:[String:Int] = [:]
    let MAX_LEN = 30 // max number of words to be generated
    
    override func viewDidLoad() {
        super.viewDidLoad()
        self.input.contentMode = .scaleAspectFit
        self.result.font = UIFont(name: "American Typewriter", size: 24)
        self.result.textColor = .white
        self.imagePicker = ImagePicker(presentationController: self, delegate: self)
        // load dictionary
        let dict:[String:String] = json2dict(name: "idx2word")!
        for (idxStr, wordStr) in dict {
            idx2word[Int(idxStr)!] = wordStr
            word2idx[wordStr] = Int(idxStr)!
        }
    }
    
    @IBAction func pickImage(_ sender: Any) {
        self.imagePicker.present(from: sender as! UIView)
    }

    func json2dict(name:String) -> [String:String]? {
        var jsonResult:[String:String]?
        if let path = Bundle.main.path(forResource: name, ofType: "json") {
            do {
                let data = try Data(contentsOf: URL(fileURLWithPath: path))
                jsonResult = try JSONSerialization.jsonObject(with: data, options: []) as? [String:String]
                print("Successfully load dictionary \(name).json")
            } catch {
                print("Failed to load dictionary \(name).json")
            }
        }
        return jsonResult
    }
    

    func resultHandler(request: VNRequest, error: Error?) {
        let results = request.results![0] as! VNCoreMLFeatureValueObservation
        let transferResults = results.featureValue.multiArrayValue!
        
        guard let mapResults = try? MAP.prediction(encoder_out: transferResults) else {
            print("map failure")
            return
        }
        let initStates = mapResults.featureValue(for: "init_states")?.multiArrayValue
        

        var maxVal = 0.0, maxInd = word2idx["<start>"]
        var hiddenStates = Array(repeating: initStates, count: 3)
        var sentence = ""
        
        for _ in 1...MAX_LEN {
            let mlArray = try? MLMultiArray(shape: [1], dataType: MLMultiArrayDataType.int32)
            mlArray?[0] = NSNumber(value: maxInd!)
            
            guard let results = try? GRU.prediction(decoder_input: mlArray!, decoder_gru1_h_in: hiddenStates[0], decoder_gru2_h_in: hiddenStates[1], decoder_gru3_h_in: hiddenStates[2]) else {
                print("decoder failure")
                return
            }
            
            hiddenStates[0] = results.featureValue(for: "decoder_gru1_h_out")?.multiArrayValue!
            hiddenStates[1] = results.featureValue(for: "decoder_gru2_h_out")?.multiArrayValue!
            hiddenStates[2] = results.featureValue(for: "decoder_gru3_h_out")?.multiArrayValue!
            
            let one_hot = results.featureValue(for: "decoder_output")?.multiArrayValue!
            let length = one_hot!.count
            maxVal = 0.0
            for i in 0..<length {
                let x = Double(truncating: one_hot![i])
                if x > maxVal {
                    maxVal = x
                    maxInd = i
                }
            }
            
            if maxInd == word2idx["<end>"] {
                break
            }
            sentence += idx2word[maxInd!]! + " "
            
        }
        result.text = sentence
        print("Result sentence: \(sentence)")
    }
    
    func generateCaption() {
        let request = VNCoreMLRequest(model: ENCODER, completionHandler: resultHandler)
        let handler = VNImageRequestHandler(cgImage: (input.image?.cgImage)!, options: [:])
        try! handler.perform([request])
    }
    
}

// image picker
extension ImageCaptionController: ImagePickerDelegate {
    
    func didSelect(image: UIImage?) {
        if image != nil {
            self.input.image = image
            generateCaption()
        }
    }
}
