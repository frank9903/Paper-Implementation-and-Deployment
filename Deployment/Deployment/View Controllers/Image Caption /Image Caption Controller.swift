//
//  Image Caption Controller.swift
//  Deployment
//
//  Created by 曹书恒 on 2020/5/28.
//  Copyright © 2020 220 284. All rights reserved.
//

import UIKit

class ImageCaptionController: UIViewController {
    @IBOutlet weak var result: UIImageView!
    var imagePicker: ImagePicker!
    
    override func viewDidLoad() {
        super.viewDidLoad()
        self.result.contentMode = .scaleAspectFit
        self.imagePicker = ImagePicker(presentationController: self, delegate: self)
    }
    
    @IBAction func pickImage(_ sender: Any) {
        self.imagePicker.present(from: sender as! UIView)
    }

}

// image picker
extension ImageCaptionController: ImagePickerDelegate {
    
    func didSelect(image: UIImage?) {
        if image != nil {
            self.result.image = image
        }
    }
}
