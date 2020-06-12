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
    
    var testModel = try! VNCoreMLModel(for: sketch_70().model)
    var image_size:CGSize!
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
        intensityPicker.delegate = intensityPickerDelegate
        intensityPicker.dataSource = intensityPickerDelegate
        
        initSlider()
    }
    
    // MARK: Load and Save images
    @IBAction func pickImage(_ sender: Any) {
        self.imagePicker.present(from: sender as! UIView)
    }

    @IBAction func saveImage(_ sender: Any) {
        UIImageWriteToSavedPhotosAlbum(resultView.image!, self, #selector(image(_:didFinishSavingWithError:contextInfo:)), nil)
    }

    @objc func image(_ image: UIImage, didFinishSavingWithError error: NSError?, contextInfo: UnsafeRawPointer) {
        if let error = error {
            // we got back an error!
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
            sliderViewWrapper.leadingAnchor.constraint(equalTo: resultViewWrapper.leadingAnchor, constant: -20),
            sliderViewWrapper.widthAnchor.constraint(equalToConstant: 40)
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
    func transformResultsMethod(request: VNRequest, error: Error?) {
        let results = request.results![0] as! VNPixelBufferObservation
        resultView.image = UIImage(pixelBuffer: results.pixelBuffer).resized(to: image_size!)
    }
    
    func didSelect(image: UIImage?) {
        if image != nil {
            input.image = image
            image_size = image?.size
            let myInput = input.image?.resized(to: CGSize(width: 512,height: 512))
            let request = VNCoreMLRequest(model: testModel, completionHandler: transformResultsMethod)
            let handler = VNImageRequestHandler(cgImage: (myInput?.cgImage)!, options: [:])
            try! handler.perform([request])
        }
    }
}

class MLModelPicker {
    
}
