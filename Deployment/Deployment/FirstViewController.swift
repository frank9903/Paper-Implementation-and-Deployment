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
    // WARNING: change me to your own IP address (run `ipconfig getifaddr en0` in terminal)
    let address = "17.230.186.33"
    
    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view.
    }
    
    func getData(from url: URL, completion: @escaping (Data?, URLResponse?, Error?) -> ()) {
        URLSession.shared.dataTask(with: url, completionHandler: completion).resume()
    }

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

