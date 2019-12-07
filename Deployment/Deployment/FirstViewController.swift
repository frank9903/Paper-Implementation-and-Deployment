//
//  FirstViewController.swift
//  Deployment
//
//  Created by 220 284 on 12/4/19.
//  Copyright © 2019 220 284. All rights reserved.
//

import UIKit
import AFNetworking

class FirstViewController: UIViewController, UIPickerViewDelegate, UIPickerViewDataSource {
    @IBOutlet weak var result: UIImageView!
    @IBOutlet weak var label: UILabel!
    @IBOutlet weak var button: UIButton!
    @IBOutlet weak var selector: UIPickerView!
    var selectorData:[String] = [String]()
    
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
        // Do any additional setup after loading the view.
    }
    
    // UIPickerViewDelegate
    func numberOfComponents(in pickerView: UIPickerView) -> Int {
        return 1
    }
    
    func pickerView(_ pickerView: UIPickerView, numberOfRowsInComponent component: Int) -> Int {
        return selectorData.count
    }
    
    // UIPickerViewDataSource
//    func pickerView(_ pickerView: UIPickerView, titleForRow row: Int, forComponent component: Int) -> String? {
//        return selectorData[row]
//    }

    func pickerView(_ pickerView: UIPickerView, attributedTitleForRow row: Int, forComponent component: Int) -> NSAttributedString? {
        
        return NSAttributedString(string: selectorData[row], attributes: [NSAttributedString.Key.foregroundColor:UIColor.white])
    }
    
    func getData(from url: URL, completion: @escaping (Data?, URLResponse?, Error?) -> ()) {
        URLSession.shared.dataTask(with: url, completionHandler: completion).resume()
    }

    // send request to local server and receive inference result
    @IBAction func update(_ sender: Any) {
        let manager = AFHTTPSessionManager(baseURL: URL(string: "http://server.url"))
        let image = UIImage(named: "1.jpg")
        let imageData = image!.jpegData(compressionQuality:0.5)
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

