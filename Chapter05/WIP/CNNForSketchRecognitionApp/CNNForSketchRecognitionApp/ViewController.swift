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
        let trainer = Trainer()
        Trainer.train() 
    }

    override var representedObject: Any? {
        didSet {
        // Update the view, if already loaded.
        }
    }


}

