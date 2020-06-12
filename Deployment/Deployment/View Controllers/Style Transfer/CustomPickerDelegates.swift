//
//  CustomPickerDelegates.swift
//  Deployment
//
//  Created by 曹书恒 on 2020/6/12.
//  Copyright © 2020 220 284. All rights reserved.
//

import Foundation
import UIKit

class ModelPickerDelegate: NSObject, UIPickerViewDataSource, UIPickerViewDelegate {
    let models = ["Pytorch Model", "Turi Create Model"]
    func numberOfComponents(in pickerView: UIPickerView) -> Int {
        return 1
    }
    
    func pickerView(_ pickerView: UIPickerView, numberOfRowsInComponent component: Int) -> Int {
        return models.count
    }
    
    // UIPickerViewDataSource
    func pickerView(_ pickerView: UIPickerView, attributedTitleForRow row: Int, forComponent component: Int) -> NSAttributedString? {
        
        return NSAttributedString(string: models[row], attributes: [NSAttributedString.Key.foregroundColor:UIColor.white])
    }
}

class StylePickerDelegate: NSObject, UIPickerViewDataSource, UIPickerViewDelegate {
    let styles = ["Starry Night", "Scream", "Sketch", "The Muse"]
    func numberOfComponents(in pickerView: UIPickerView) -> Int {
        return 1
    }
    
    func pickerView(_ pickerView: UIPickerView, numberOfRowsInComponent component: Int) -> Int {
        return styles.count
    }
    
    // UIPickerViewDataSource
    func pickerView(_ pickerView: UIPickerView, attributedTitleForRow row: Int, forComponent component: Int) -> NSAttributedString? {
        
        return NSAttributedString(string: styles[row], attributes: [NSAttributedString.Key.foregroundColor:UIColor.white])
    }
}

class IntensityPickerDelegate: NSObject, UIPickerViewDataSource, UIPickerViewDelegate {
    let intensities = ["High", "Medium", "Low"]
    func numberOfComponents(in pickerView: UIPickerView) -> Int {
        return 1
    }
    
    func pickerView(_ pickerView: UIPickerView, numberOfRowsInComponent component: Int) -> Int {
        return intensities.count
    }
    
    // UIPickerViewDataSource
    func pickerView(_ pickerView: UIPickerView, attributedTitleForRow row: Int, forComponent component: Int) -> NSAttributedString? {
        
        return NSAttributedString(string: intensities[row], attributes: [NSAttributedString.Key.foregroundColor:UIColor.white])
    }
}

