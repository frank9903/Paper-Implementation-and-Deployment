//
//  FirstViewController.swift
//  Deployment
//
//  Created by 220 284 on 12/4/19.
//  Copyright © 2019 220 284. All rights reserved.
//

import UIKit
import AFNetworking

class FirstViewController: UIViewController {
    @IBOutlet weak var result: UIImageView!
    @IBOutlet weak var label: UILabel!
    @IBOutlet weak var button: UIButton!
    @IBOutlet weak var selector: UIPickerView!
    var selectorData:[String] = [String]()
    var imagePicker: ImagePicker!
    
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

    // send request to local server and receive inference result
    @IBAction func update(_ sender: Any) {
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
    
    // show image picker
    @IBAction func pickImage(_ sender: Any) {
        self.imagePicker.present(from: sender as! UIView)
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
        self.result.image = image
    }
}
