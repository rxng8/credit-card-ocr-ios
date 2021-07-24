//
//  OCRDataHandler.swift
//  credit-card-ocr-demo
//
//  Created by Alex Nguyen on 24/07/2021.
//

import CoreImage
import TensorFlowLite
import UIKit
import Accelerate

/// Stores results for a particular frame that was successfully run through the `Interpreter`.
struct OCRResult {
  let inferenceTime: Double
  let inferences: [Inference]
}

/// Stores one formatted inference.
struct OCRInference {
  
}

/// Information about the MobileNet SSD model.
enum OCRModel {
  static let modelInfo: FileInfo = (name: "", extension: "tflite")
}

/// This class handles all data preprocessing and makes calls to run inference on a given frame
/// by invoking the `Interpreter`. It then formats the inferences obtained and returns the top N
/// results for a successful inference.
class OCRDataHandler: NSObject {

  // MARK: - Internal Properties
  /// The current thread count used by the TensorFlow Lite Interpreter.
  let threadCount: Int
  let threadCountLimit = 10
    
  // MARK: Model parameters
  let batchSize = 1
  let inputChannels = 3
  let inputWidth = 320 //300
  let inputHeight = 320 //300

  // image mean and std for floating model, should be consistent with parameters used in model training
  let imageMean: Float = 127.5
  let imageStd:  Float = 127.5

  // MARK: Private properties
  private var labels: [String] = []

  /// TensorFlow Lite `Interpreter` object for performing inference on a given model.
  private var interpreter: Interpreter

  private let bgraPixel = (channels: 4, alphaComponent: 3, lastBgrComponent: 2)
  private let rgbPixelChannels = 3
  private let colorStrideValue = 10
  private let colors = [
    UIColor.red,
    UIColor(displayP3Red: 90.0/255.0, green: 200.0/255.0, blue: 250.0/255.0, alpha: 1.0),
//    UIColor.green,
    UIColor.orange,
    UIColor.blue,
    UIColor.purple,
    UIColor.magenta,
    UIColor.yellow,
    UIColor.cyan,
    UIColor.brown
  ]

  // MARK: - Initialization

  /// A failable initializer for `ModelDataHandler`. A new instance is created if the model and
  /// labels files are successfully loaded from the app's main bundle. Default `threadCount` is 1.
  init?(modelFileInfo: FileInfo, labelsFileInfo: FileInfo, threadCount: Int = 1) {
    return nil
  }

  /// This class handles all data preprocessing and makes calls to run inference on a given frame
  /// through the `Interpreter`. It then formats the inferences obtained and returns the top N
  /// results for a successful inference.
  func runModel(onFrame pixelBuffer: CVPixelBuffer) -> OCRResult? {
    return nil
  }

  /// Filters out all the results with confidence score < threshold and returns the top N results
  /// sorted in descending order.
  func formatResults(boundingBox: [Float], outputClasses: [Float], outputScores: [Float], outputCount: Int, width: CGFloat, height: CGFloat) -> [OCRInference]?{
    return nil
  }

  /// This assigns color for a particular class.
  private func colorForClass(withIndex index: Int) -> UIColor {

    // We have a set of colors and the depending upon a stride, it assigns variations to of the base
    // colors to each object based on its index.
    let baseColor = colors[index % colors.count]

    var colorToAssign = baseColor

    let percentage = CGFloat((colorStrideValue / 2 - index / colors.count) * colorStrideValue)

    if let modifiedColor = baseColor.getModified(byPercentage: percentage) {
      colorToAssign = modifiedColor
    }

    return colorToAssign
  }
}
