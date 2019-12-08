//
//  FirstViewController.swift
//  Deployment
//
//  Created by 220 284 on 12/4/19.
//  Copyright © 2019 220 284. All rights reserved.
//

import UIKit
import AFNetworking
import CoreML
import Vision

class FirstViewController: UIViewController {
    @IBOutlet weak var result: UIImageView!
    @IBOutlet weak var label: UILabel!
    @IBOutlet weak var button: UIButton!
    @IBOutlet weak var selector: UIPickerView!
    
    var defaultModel = Default_YOLO()
    // WARNING: change address to your own IP address (run `ipconfig getifaddr en0` in terminal)
    let address = "17.230.186.33"
    
    var selectorData:[String] = [String]()
    var imagePicker: ImagePicker!
    var isNewImage = true
    
    var maskView: UIView!
    var progressView: UIProgressView!
    var loadingTextView: UITextView!
    
    var countDown: Float!
    var totalTime: Float!
    var timer: Timer!
    var inferenceSteps: [String]!
    
    override func viewDidLoad() {
        super.viewDidLoad()
        selectorData = ["Apple Default", "Customized"]
        self.selector.delegate = self
        self.selector.dataSource = self
        self.button.layer.cornerRadius = 20
        self.button.layer.borderWidth = 3
        self.button.setTitleColor(UIColor.white, for: .normal)
        self.result.contentMode = .scaleAspectFit
        self.imagePicker = ImagePicker(presentationController: self, delegate: self)
        
        initLoadingView()
    }
    
    func getData(from url: URL, completion: @escaping (Data?, URLResponse?, Error?) -> ()) {
        URLSession.shared.dataTask(with: url, completionHandler: completion).resume()
    }
    
    
    @IBAction func update(_ sender: Any) {
        if isNewImage {
            if (selector.selectedRow(inComponent: 0) == 0) {
                defaultPredict()
            } else {
                // WARNING: change this time for your own computer
                startUpdating(timeInterval: 20)
                customizedPredict()
            }
        } else {
            let ac = UIAlertController(title: "Image already been processed", message: "", preferredStyle: .alert)
            ac.addAction(UIAlertAction(title: "OK", style: .default))
            present(ac, animated: true)
        }
        isNewImage = false
    }
    
    // show image picker
    @IBAction func pickImage(_ sender: Any) {
        self.imagePicker.present(from: sender as! UIView)
        isNewImage = true
    }
    
    // Save the image to local library
    @IBAction func saveImage(_ sender: Any) {
        UIImageWriteToSavedPhotosAlbum(result.image!, self, #selector(image(_:didFinishSavingWithError:contextInfo:)), nil)
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


// for method selector
extension FirstViewController: UIPickerViewDelegate, UIPickerViewDataSource {
    // UIPickerViewDelegate
    func numberOfComponents(in pickerView: UIPickerView) -> Int {
        return 1
    }
    
    func pickerView(_ pickerView: UIPickerView, numberOfRowsInComponent component: Int) -> Int {
        return selectorData.count
    }
    
    // UIPickerViewDataSource
    func pickerView(_ pickerView: UIPickerView, attributedTitleForRow row: Int, forComponent component: Int) -> NSAttributedString? {
        
        return NSAttributedString(string: selectorData[row], attributes: [NSAttributedString.Key.foregroundColor:UIColor.white])
    }
}

// image picker
extension FirstViewController: ImagePickerDelegate {
    
    func didSelect(image: UIImage?) {
        if image != nil {
            self.result.image = image
        }
    }
}

// loading view
extension FirstViewController {
    func initLoadingView() {
        // set up the constraint for loading view programmatically
        maskView = UIView()
        maskView.backgroundColor = .white
        view.addSubview(maskView)
        
        maskView.translatesAutoresizingMaskIntoConstraints = false
        let horizontalConstraint = NSLayoutConstraint(item: maskView!, attribute: NSLayoutConstraint.Attribute.centerX, relatedBy: NSLayoutConstraint.Relation.equal, toItem: view, attribute: NSLayoutConstraint.Attribute.centerX, multiplier: 1, constant: 0)
        let verticalConstraint = NSLayoutConstraint(item: maskView!, attribute: NSLayoutConstraint.Attribute.centerY, relatedBy: NSLayoutConstraint.Relation.equal, toItem: view, attribute: NSLayoutConstraint.Attribute.centerY, multiplier: 1, constant: 0)
        let widthConstraint = NSLayoutConstraint(item: maskView!, attribute: NSLayoutConstraint.Attribute.width, relatedBy: NSLayoutConstraint.Relation.equal, toItem: view, attribute: NSLayoutConstraint.Attribute.width, multiplier: 1, constant: 0)
        let heightConstraint = NSLayoutConstraint(item: maskView!, attribute: NSLayoutConstraint.Attribute.height, relatedBy: NSLayoutConstraint.Relation.equal, toItem: view, attribute: NSLayoutConstraint.Attribute.height, multiplier: 1, constant: 0)
        view.addConstraints([horizontalConstraint, verticalConstraint, widthConstraint, heightConstraint])
        
        //        view.sendSubviewToBack(maskView)
        progressView = UIProgressView(progressViewStyle: .bar)
        progressView.center = CGPoint(x: view.center.x, y: view.center.y+290)
        progressView.transform = progressView.transform.scaledBy(x: 5, y: 3)
        progressView.setProgress(0, animated: true)
        progressView.trackTintColor = UIColor.lightGray
        progressView.tintColor = UIColor.cyan
        
        view.addSubview(progressView)
        
        loadingTextView = UITextView(frame: CGRect(x: 20.0, y: 90.0, width: 1000.0, height: 40.0))
        loadingTextView.translatesAutoresizingMaskIntoConstraints = false
        loadingTextView.center = CGPoint(x: view.center.x, y: view.center.y+260)
        loadingTextView.textColor = .white
        loadingTextView.textAlignment = .center
        loadingTextView.font = UIFont(name: "Cochin", size:24)
        loadingTextView.backgroundColor = .clear
        loadingTextView.isUserInteractionEnabled = false
        
        view.addSubview(loadingTextView)
        
        dismissMaskView()
//
        inferenceSteps = [
            "Uploading image to local server",
            "Preprocessing the image",
            "Generating anchors for image",
            "Extracting feature using ResNet50",
            "Polishing feature maps using FPN",
            "Generating region proposals and scores",
            "Refining bounding boxes",
            "Cropping corresponding feature maps",
            "Wrapping feature maps using RoI Align",
            "Predicting classes and regressing bounding box",
            "Generating mask for corresponding class",
            "Fetching image from local server"
        ]
    }
    
    func startUpdating(timeInterval: Float) {
        maskView.alpha = 0.33
        progressView.alpha = 1
        loadingTextView.alpha = 1
        view.bringSubviewToFront(maskView)
        view.bringSubviewToFront(progressView)
        view.bringSubviewToFront(loadingTextView)
        // timeInterval seconds
        totalTime = timeInterval / 0.01
        countDown = 0
        timer = Timer.scheduledTimer(timeInterval: 0.01,
                                     target: self,
                                     selector: #selector(updateTimer),
                                     userInfo: nil,
                                     repeats: true)
        
        timer.fire()
    }

    @objc func updateTimer() {
        if (countDown < totalTime) {
            // update the mock info
            // Note: be careful with overflow
            let tmp = Int(ceil(totalTime / Float(inferenceSteps.count)))
            if (Int(countDown) % tmp == 0) {
                if (totalTime == 1 / 0.01) {
                    // for accelaration
                    loadingTextView.text = "Fetching image from local serve"
                } else {
                    loadingTextView.text = inferenceSteps[Int(countDown)/tmp]
                }
            }
            
            countDown += 1
            progressView.progress += 1.0 / totalTime
        } else {
            timer.invalidate()
            self.dismissMaskView()
        }
    }
    
    func dismissMaskView() {
        maskView.alpha = 0
        progressView.alpha = 0
        loadingTextView.alpha = 0
        self.view.sendSubviewToBack(maskView)
        self.view.sendSubviewToBack(progressView)
        self.view.sendSubviewToBack(loadingTextView)
    }
}

// predictions
extension FirstViewController {
    // default prediction
    func defaultPredict() {
        let predictions = defaultModel.predict(image: result.image!)
        result.image = defaultModel.show(image: result.image!, predictions: predictions)
    }
    
    // send request to local server and receive inference result
    func customizedPredict() {
        
        let manager = AFHTTPSessionManager(baseURL: URL(string: "http://server.url"))
        let image = result.image
        let imageData = image!.jpegData(compressionQuality:0.5)
        
        // check data
        let defaultImageData = UIImage(named: "placehold.png")?.jpegData(compressionQuality: 0.5)
        if (defaultImageData == imageData) {
            let alert = UIAlertController(title: "No Image Uploaded", message: "", preferredStyle: UIAlertController.Style.alert)
            alert.addAction(UIAlertAction(title: "OK", style: UIAlertAction.Style.default, handler: nil))
            self.present(alert, animated: true, completion: nil)
            return
        }
        
        let parameters = ["name":"Shuheng"]
        
        manager.responseSerializer = AFHTTPResponseSerializer()
        //      TODO: find the proper time out interval
        manager.requestSerializer.timeoutInterval = 300
        manager.post("http://\(self.address):8000/myapp/list/",
            parameters: parameters,
            constructingBodyWith: { (formData: AFMultipartFormData) in
                formData.appendPart(withFileData: imageData!, name: "docfile", fileName: "photo.jpg", mimeType: "image/jpg")
        },
            success:
            { (operation:URLSessionDataTask, responseObject:Any?) in
                NSLog("SUCCESS: \(operation.response!)")
                // TODO: this is really hacky way of fetching image, try to extract url from responseObject
                let url:URL = URL(string: "http://\(self.address):8000/media/result.png")!
                self.getData(from: url) { (data, response, error) in
                    guard let data = data, error == nil else { return }
                    print(response?.suggestedFilename ?? url.lastPathComponent)
                    print("Download Finished")
                    DispatchQueue.main.async() {
                        self.result.image = UIImage(data: data)!
                    }
                }
                self.accelerate()
        },
            failure:
            { (operation:URLSessionDataTask?, error:Error) in
                self.dismissMaskView()
                let ac = UIAlertController(title: "Inference Failure", message: "\(error)", preferredStyle: .alert)
                ac.addAction(UIAlertAction(title: "OK", style: .default))
                self.present(ac, animated: true)
        })
        print("Hello World from Shuhneng")
    }
    
    func accelerate() {
        let currentProgress = progressView.progress
        if (currentProgress < 1.0) {
            // 1 seconds
            totalTime = 1 / 0.01
            countDown = 0
            timer = Timer.scheduledTimer(timeInterval: 0.01,
                                         target: self,
                                         selector: #selector(updateTimer),
                                         userInfo: nil,
                                         repeats: true)
            
            timer.fire()
        }
    }
}
