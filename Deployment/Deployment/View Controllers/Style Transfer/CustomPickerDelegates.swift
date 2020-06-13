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
    let models = ["Turi Create Model", "Pytorch Model"]
    var inference: InferenceDelegate? = nil
    
    func numberOfComponents(in pickerView: UIPickerView) -> Int {
        return 1
    }
    
    func pickerView(_ pickerView: UIPickerView, numberOfRowsInComponent component: Int) -> Int {
        return models.count
    }
    
    func pickerView(_ pickerView: UIPickerView, attributedTitleForRow row: Int, forComponent component: Int) -> NSAttributedString? {
        return NSAttributedString(string: models[row], attributes: [NSAttributedString.Key.foregroundColor:UIColor.white])
    }
    
    func pickerView(_ pickerView: UIPickerView, didSelectRow row: Int, inComponent component: Int) {
        inference?.update()
    }
}

class StylePickerDelegate: NSObject, UIPickerViewDataSource, UIPickerViewDelegate {
    let styles = ["The Muse", "Starry Night", "Scream", "Sketch"]
    var inference: InferenceDelegate? = nil
    
    func numberOfComponents(in pickerView: UIPickerView) -> Int {
        return 1
    }
    
    func pickerView(_ pickerView: UIPickerView, numberOfRowsInComponent component: Int) -> Int {
        return styles.count
    }
    
    func pickerView(_ pickerView: UIPickerView, attributedTitleForRow row: Int, forComponent component: Int) -> NSAttributedString? {
        
        return NSAttributedString(string: styles[row], attributes: [NSAttributedString.Key.foregroundColor:UIColor.white])
    }
    
    func pickerView(_ pickerView: UIPickerView, didSelectRow row: Int, inComponent component: Int) {
        inference?.update()
    }
}

class IntensityPickerDelegate: NSObject, UIPickerViewDataSource, UIPickerViewDelegate {
    let intensities = ["High", "Medium", "Low"]
    var inference: InferenceDelegate? = nil
    
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
    
    func pickerView(_ pickerView: UIPickerView, didSelectRow row: Int, inComponent component: Int) {
        inference?.update()
    }
}

