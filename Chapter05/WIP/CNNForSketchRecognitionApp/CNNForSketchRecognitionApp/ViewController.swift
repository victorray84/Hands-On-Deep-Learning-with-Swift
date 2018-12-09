//
//  ViewController.swift
//  CNNForSketchRecognitionApp
//
//  Created by joshua.newnham on 12/11/2018.
//  Copyright © 2018 Joshua Newnham. All rights reserved.
//

import Cocoa

class ViewController: NSViewController {

    override func viewDidLoad() {
        super.viewDidLoad()

        // Do any additional setup after loading the view.
//        Trainer.train()
        Trainer.validate()
    }

    override var representedObject: Any? {
        didSet {
        // Update the view, if already loaded.
        }
    }


}

