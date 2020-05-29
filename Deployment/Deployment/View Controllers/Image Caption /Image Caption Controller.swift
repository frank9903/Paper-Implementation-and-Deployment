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
        self.imagePicker = ImagePicker(presentationController: self, delegate: self)

        self.result.delegate = self
        self.result.font = UIFont(name: "American Typewriter", size: 24)
        self.result.textColor = .white
        self.result.isUserInteractionEnabled = false // disable user interaction if no image present

        // load dictionary
        let dict:[String:String] = json2dict(name: "idx2word")!
        for (idxStr, wordStr) in dict {
            idx2word[Int(idxStr)!] = wordStr
            word2idx[wordStr] = Int(idxStr)!
        }
        // keyboard management: tap to dismiss
        let tap: UITapGestureRecognizer = UITapGestureRecognizer(target: self, action: #selector(self.dismissKeyboard))
        view.addGestureRecognizer(tap)
    }
    
    @IBAction func pickImage(_ sender: Any) {
        self.imagePicker.present(from: sender as! UIView)
        self.result.isUserInteractionEnabled = true // enable user correction for caption
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
        
        let initStates = mapResults.featureValue(for: "init_states")!.multiArrayValue!
        var hiddenStates = Array(repeating: initStates, count: 3)
        let prefix = result.text!.lowercased() // user input prefix
        var sentence = prefix
        let wordsList = ["<start>"] + prefix.words
        
        var curIdx = -1
        for word in wordsList {
            let idx = word2idx[word] ?? word2idx["<end>"]!
            curIdx = forward1step(input: idx, hiddenStates: &hiddenStates)
        }
        
        for _ in 1...MAX_LEN {
            if curIdx == word2idx["<end>"] {
                break
            }
            sentence += " " + idx2word[curIdx]!
            curIdx = forward1step(input: curIdx, hiddenStates: &hiddenStates)
        }
        result.text = sentence
        print("Result sentence: \(sentence)")
    }
    
    func forward1step(input:Int, hiddenStates:inout [MLMultiArray]) -> Int {
        let decoderInput = try? MLMultiArray(shape: [1], dataType: MLMultiArrayDataType.int32)
        decoderInput?[0] = NSNumber(value: input)
        
        guard let results = try? GRU.prediction(decoder_input: decoderInput!, decoder_gru1_h_in: hiddenStates[0], decoder_gru2_h_in: hiddenStates[1], decoder_gru3_h_in: hiddenStates[2]) else {
            print("decoder failure")
            return -1
        }
        
        hiddenStates[0] = results.featureValue(for: "decoder_gru1_h_out")!.multiArrayValue!
        hiddenStates[1] = results.featureValue(for: "decoder_gru2_h_out")!.multiArrayValue!
        hiddenStates[2] = results.featureValue(for: "decoder_gru3_h_out")!.multiArrayValue!
        let one_hot = results.featureValue(for: "decoder_output")!.multiArrayValue!
        let length = one_hot.count
        
        var maxVal = 0.0, maxInd = -1
        for i in 0..<length {
            let x = Double(truncating: one_hot[i])
            if x > maxVal {
                maxVal = x
                maxInd = i
            }
        }
        
        return maxInd
    }
    
    func generateCaption() {
        let request = VNCoreMLRequest(model: ENCODER, completionHandler: resultHandler)
        let handler = VNImageRequestHandler(cgImage: (input.image?.cgImage)!, options: [:])
        try! handler.perform([request])
    }
    
    @objc func dismissKeyboard() {
        view.endEditing(true)
        generateCaption()
    }
}

// image picker
extension ImageCaptionController: ImagePickerDelegate {
    
    func didSelect(image: UIImage?) {
        if image != nil {
            self.input.image = image
            self.result.text = "" // clear last results
            generateCaption()
        }
    }
}

// keyboard management: return to dismiss
extension ImageCaptionController: UITextFieldDelegate {
    func textFieldShouldReturn(_ textField: UITextField) -> Bool {
        self.dismissKeyboard()
        return false
    }
}

extension StringProtocol {
    var words: [String] {
        return split{ !$0.isLetter }.map { String($0) }
    }
}
