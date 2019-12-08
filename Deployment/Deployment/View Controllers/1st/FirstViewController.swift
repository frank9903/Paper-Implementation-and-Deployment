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
    var selectorData:[String] = [String]()
    var imagePicker: ImagePicker!
    var defaultModel = Default_YOLO()
    var isNewImage = true
    
    // WARNING: change me to your own IP address (run `ipconfig getifaddr en0` in terminal)
    let address = "17.230.186.33"
    
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
    }
    
    func getData(from url: URL, completion: @escaping (Data?, URLResponse?, Error?) -> ()) {
        URLSession.shared.dataTask(with: url, completionHandler: completion).resume()
    }
    
    
    @IBAction func update(_ sender: Any) {
        if isNewImage {
            if (selector.selectedRow(inComponent: 0) == 0) {
                defaultPredict()
            } else {
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
                //              TODO: this is really hacky way of fetching image, try to extract url from responseObject
                let url:URL = URL(string: "http://\(self.address):8000/media/result.png")!
                self.getData(from: url) { (data, response, error) in
                    guard let data = data, error == nil else { return }
                    print(response?.suggestedFilename ?? url.lastPathComponent)
                    print("Download Finished")
                    DispatchQueue.main.async() {
                        self.result.image = UIImage(data: data)!
                    }
                }
        },
            failure:
            { (operation:URLSessionDataTask?, error:Error) in
                NSLog("FAILURE: \(error)")
        })
    }
}
