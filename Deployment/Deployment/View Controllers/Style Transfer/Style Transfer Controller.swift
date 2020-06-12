//
//  Style Transfer Controller.swift
//  Deployment
//
//  Created by 曹书恒 on 2020/6/10.
//  Copyright © 2020 220 284. All rights reserved.
//

import UIKit
import CoreML
import Vision

class StyleTransferController: UIViewController {
    @IBOutlet weak var input: UIImageView!
    @IBOutlet weak var modelPicker: UIPickerView!
    @IBOutlet weak var stylePicker: UIPickerView!
    @IBOutlet weak var intensityPicker: UIPickerView!
    
    var torchModel: [Int: [Int: VNCoreMLModel]] = [:]
    var turiModel = turicreate_model()
    // expect input size of mlmodel
    var expectSize = CGSize(width: 512, height: 512)
    // record the size of original image
    var originSize: CGSize!
    // check if user has uploaded image
    var imageExist = false
    // check if user has read the alert
    var alertRead = false
    
    // MARK: Slider Variables
    fileprivate lazy var resultView: UIImageView = {
        let iv = UIImageView()
        iv.translatesAutoresizingMaskIntoConstraints = false
        iv.contentMode = .scaleAspectFit
        iv.clipsToBounds = true
        return iv
    }()
    fileprivate lazy var resultViewWrapper: UIView = {
        let v = UIView()
        v.translatesAutoresizingMaskIntoConstraints = false
        v.clipsToBounds = true
        return v
    }()
    fileprivate lazy var sliderView: UIView = {
        let v = UIView()
        v.translatesAutoresizingMaskIntoConstraints = false
        v.clipsToBounds = true
        return v
    }()
    fileprivate lazy var sliderViewWrapper: UIView = {
        let v = UIView()
        v.translatesAutoresizingMaskIntoConstraints = false
        v.clipsToBounds = true
        return v
    }()
    fileprivate var leading: NSLayoutConstraint!
    fileprivate var originRect: CGRect!

    var imagePicker: ImagePicker!
    let modelPickerDelegate = ModelPickerDelegate()
    let stylePickerDelegate = StylePickerDelegate()
    let intensityPickerDelegate = IntensityPickerDelegate()
    override func viewDidLoad() {
        super.viewDidLoad()
        input.contentMode = .scaleAspectFit
        imagePicker = ImagePicker(presentationController: self, delegate: self)

        modelPicker.delegate = modelPickerDelegate
        modelPicker.dataSource = modelPickerDelegate
        stylePicker.delegate = stylePickerDelegate
        stylePicker.dataSource = stylePickerDelegate
        stylePicker.selectRow(3, inComponent: 0, animated: false)
        intensityPicker.delegate = intensityPickerDelegate
        intensityPicker.dataSource = intensityPickerDelegate
        intensityPicker.selectRow(2, inComponent: 0, animated: false)
        
        modelPickerDelegate.inference = self
        stylePickerDelegate.inference = self
        intensityPickerDelegate.inference = self
        
        initSlider()
        initTorchModels()
    }
    
    // MARK: Load and Save images
    @IBAction func pickImage(_ sender: Any) {
        imagePicker.present(from: sender as! UIView)
    }

    @IBAction func saveImage(_ sender: Any) {
        update()
        if let output = resultView.image {
            UIImageWriteToSavedPhotosAlbum(output, self, #selector(image(_:didFinishSavingWithError:contextInfo:)), nil)
        } else {
            let ac = UIAlertController(title: "Save error", message: "Please select an image for style transfer.", preferredStyle: .alert)
            ac.addAction(UIAlertAction(title: "OK", style: .default))
            present(ac, animated: true)
        }
    }

    @objc func image(_ image: UIImage, didFinishSavingWithError error: NSError?, contextInfo: UnsafeRawPointer) {
        if let error = error {
            let ac = UIAlertController(title: "Save error", message: error.localizedDescription, preferredStyle: .alert)
            ac.addAction(UIAlertAction(title: "OK", style: .default))
            present(ac, animated: true)
        } else {
            let ac = UIAlertController(title: "Image Successfully Saved", message: "", preferredStyle: .alert)
            ac.addAction(UIAlertAction(title: "OK", style: .default))
            present(ac, animated: true)
        }
    }
}

// MARK: ML Inference
protocol InferenceDelegate {
    func update()
}

extension StyleTransferController: InferenceDelegate {
    func readAlert(alert: UIAlertAction!) {alertRead = true}
    
    func update() {
        let model = modelPicker.selectedRow(inComponent: 0)
        let style = stylePicker.selectedRow(inComponent: 0)
        let intensity = intensityPicker.selectedRow(inComponent: 0)
        
        if model == 0 {
            // pytorch model
            // enable intensity selection
            intensityPicker.isUserInteractionEnabled = true
            intensityPicker.alpha = 1.0
            
            // model inference
            if (!imageExist) {
                return
            }
            let curModel = torchModel[style]![intensity]!
            let image = input.image?.resized(to: expectSize)
            let request = VNCoreMLRequest(model: curModel, completionHandler: trochResultHandler)
            let handler = VNImageRequestHandler(cgImage: (image?.cgImage)!, options: [:])
            try! handler.perform([request])
        } else {
            // turi create model
            // disable intensity selection
            intensityPicker.isUserInteractionEnabled = false
            intensityPicker.alpha = 0.0
            if !alertRead {
                let ac = UIAlertController(title: "Version Info", message: "Intensity Selection is not available for Turi Create Model", preferredStyle: .alert)
                ac.addAction(UIAlertAction(title: "Don't Show", style: .destructive, handler:readAlert))
                ac.addAction(UIAlertAction(title: "OK", style: .default))
                present(ac, animated: true)
            }
            
            // model inference
            if (!imageExist) {
                return
            }
            let styleArray = try? MLMultiArray(shape: [stylePicker.numberOfRows(inComponent: 0)] as [NSNumber], dataType: MLMultiArrayDataType.double)
            for i in 0...((styleArray?.count)!-1) {
                styleArray?[i] = 0.0
            }
            styleArray?[style] = 1.0
            let result = try? turiModel.prediction(image: input.image!.pixelBuffer(to: expectSize)!, index: styleArray!)
            resultView.image = UIImage(pixelBuffer: (result?.stylizedImage)!).resized(to: originSize)
        }
    }
    
    func trochResultHandler(request: VNRequest, error: Error?) {
        let results = request.results![0] as! VNPixelBufferObservation
        resultView.image = UIImage(pixelBuffer: results.pixelBuffer).resized(to: originSize)
    }
}

// MARK: Slider Implementation
extension StyleTransferController {
    fileprivate func initSlider() {
        resultViewWrapper.addSubview(resultView)
        sliderViewWrapper.addSubview(sliderView)
        view.addSubview(resultViewWrapper)
        view.addSubview(sliderViewWrapper)
        
        leading = resultViewWrapper.leadingAnchor.constraint(equalTo: input.leadingAnchor, constant: 0)
        NSLayoutConstraint.activate([
            resultViewWrapper.topAnchor.constraint(equalTo: input.topAnchor, constant: 0),
            resultViewWrapper.bottomAnchor.constraint(equalTo: input.bottomAnchor, constant: 0),
            resultViewWrapper.trailingAnchor.constraint(equalTo: input.trailingAnchor, constant: 0),
            leading
        ])
        NSLayoutConstraint.activate([
            resultView.topAnchor.constraint(equalTo: resultViewWrapper.topAnchor, constant: 0),
            resultView.bottomAnchor.constraint(equalTo: resultViewWrapper.bottomAnchor, constant: 0),
            resultView.trailingAnchor.constraint(equalTo: resultViewWrapper.trailingAnchor, constant: 0)
        ])
        
        NSLayoutConstraint.activate([
            sliderViewWrapper.topAnchor.constraint(equalTo: resultViewWrapper.topAnchor, constant: 0),
            sliderViewWrapper.bottomAnchor.constraint(equalTo: resultViewWrapper.bottomAnchor, constant: 0),
            sliderViewWrapper.leadingAnchor.constraint(equalTo: resultViewWrapper.leadingAnchor, constant: -50),
            sliderViewWrapper.widthAnchor.constraint(equalToConstant: 100)
        ])
        NSLayoutConstraint.activate([
            sliderView.centerXAnchor.constraint(equalTo: sliderViewWrapper.centerXAnchor, constant: 0),
            sliderView.centerYAnchor.constraint(equalTo: sliderViewWrapper.centerYAnchor, constant: 0),
            sliderView.widthAnchor.constraint(equalTo: sliderViewWrapper.widthAnchor, multiplier: 1),
            sliderView.heightAnchor.constraint(equalTo: sliderViewWrapper.widthAnchor, multiplier: 1)
        ])
        
        leading.constant = input.frame.width / 2
        resultView.widthAnchor.constraint(equalTo: input.widthAnchor, multiplier: 1).isActive = true
        resultView.heightAnchor.constraint(equalTo: input.heightAnchor, multiplier: 1).isActive = true
        view.layoutIfNeeded()
        originRect = resultViewWrapper.frame
        
        let tap = UIPanGestureRecognizer(target: self, action: #selector(gesture(sender:)))
        sliderViewWrapper.isUserInteractionEnabled = true
        sliderViewWrapper.addGestureRecognizer(tap)
    }
    
    @objc func gesture(sender: UIPanGestureRecognizer) {
        let translation = sender.translation(in: self.view)
        switch sender.state {
        case .began, .changed:
            var newLeading = originRect.origin.x + translation.x
            newLeading = max(newLeading, 0)
            newLeading = min(input.frame.width, newLeading)
            leading.constant = newLeading
            view.layoutIfNeeded()
        case .ended, .cancelled:
            originRect = resultViewWrapper.frame
        default: break
        }
    }
}

extension StyleTransferController: ImagePickerDelegate {
    func didSelect(image: UIImage?) {
        if image != nil {
            input.image = image
            originSize = image?.size
            imageExist = true
            update()
        }
    }
}

// MARK: Model Library
extension StyleTransferController {
    // FIXME: HARD CODING
    func initTorchModels() {
        var styleDict:[Int:VNCoreMLModel] = [:]
        // the muse
        styleDict[0] = try! VNCoreMLModel(for: muse_40().model)
        styleDict[1] = try! VNCoreMLModel(for: muse_30().model)
        styleDict[2] = try! VNCoreMLModel(for: muse_20().model)
        torchModel[0] = styleDict
        
        // starry nights
        styleDict[0] = try! VNCoreMLModel(for: night_50().model)
        styleDict[1] = try! VNCoreMLModel(for: night_40().model)
        styleDict[2] = try! VNCoreMLModel(for: night_20().model)
        torchModel[1] = styleDict
        
        // scream
        styleDict[0] = try! VNCoreMLModel(for: scream_100().model)
        styleDict[1] = try! VNCoreMLModel(for: scream_50().model)
        styleDict[2] = try! VNCoreMLModel(for: scream_30().model)
        torchModel[2] = styleDict
        
        // sketch
        styleDict[0] = try! VNCoreMLModel(for: sketch_70().model)
        styleDict[1] = try! VNCoreMLModel(for: sketch_30().model)
        styleDict[2] = try! VNCoreMLModel(for: sketch_20().model)
        torchModel[3] = styleDict
    }
}
