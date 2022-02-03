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
    let inference: String
}

/// Information about the MobileNet SSD model.
enum OCRModel {
  static let modelInfo: FileInfo = (name: "ocr_float16", extension: "tflite")
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
  let inputChannels = 1
  let inputWidth = 200
  let inputHeight = 31

  // image mean and std for floating model, should be consistent with parameters used in model training
  let imageMean: Float = 127.5
  let imageStd:  Float = 127.5

  // MARK: Private properties
  private var labels: [String] = []

    // MARK: Label
    private let digitList: [String] = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    
  /// TensorFlow Lite `Interpreter` object for performing inference on a given model.
  var interpreter: Interpreter

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
  init?(modelFileInfo: FileInfo, threadCount: Int = 1) {
    let modelFilename = modelFileInfo.name

    // Construct the path to the model file.
    guard let modelPath = Bundle.main.path(
      forResource: modelFilename,
      ofType: modelFileInfo.extension
    ) else {
      print("Failed to load the model file with name: \(modelFilename).")
      return nil
    }

    // Specify the options for the `Interpreter`.
    self.threadCount = threadCount
    var options = InterpreterOptions()
    options.threadCount = threadCount
    do {
      // Create the `Interpreter`.
      interpreter = try Interpreter(modelPath: modelPath, options: options)
      // Allocate memory for the model's input `Tensor`s.
      try interpreter.allocateTensors()
    } catch let error {
      print("Failed to create the interpreter with error: \(error.localizedDescription)")
      return nil
    }

    super.init()
  }

  /// This class handles all data preprocessing and makes calls to run inference on a given frame
  /// through the `Interpreter`. It then formats the inferences obtained and returns the top N
  /// results for a successful inference.
  func runModel(onFrame pixelBuffer: CVPixelBuffer) -> OCRResult? {
    let imageWidth = CVPixelBufferGetWidth(pixelBuffer)
    let imageHeight = CVPixelBufferGetHeight(pixelBuffer)
    let sourcePixelFormat = CVPixelBufferGetPixelFormatType(pixelBuffer)
    assert(sourcePixelFormat == kCVPixelFormatType_32ARGB ||
             sourcePixelFormat == kCVPixelFormatType_32BGRA ||
               sourcePixelFormat == kCVPixelFormatType_32RGBA)


    let imageChannels = 4
    assert(imageChannels >= inputChannels)

    // Crops the image to the biggest square in the center and scales it down to model dimensions.
    let scaledSize = CGSize(width: inputWidth, height: inputHeight)
    guard let scaledPixelBuffer = pixelBuffer.resized(to: scaledSize) else {
      return nil
    }
    
    // scale image to desired size without cropping anything
    
    
    let interval: TimeInterval
    let outputTensor: Tensor
    do {
      let inputTensor = try interpreter.input(at: 0)

      // Convert to grayscale data.
      guard let grayscaleData = grayscaleDataFromBuffer(
        scaledPixelBuffer,
        byteCount: batchSize * inputWidth * inputHeight * inputChannels,
        isModelQuantized: inputTensor.dataType == .uInt8
      ) else {
        print("Failed to convert the image buffer to RGB data.")
        return nil
      }

      // Copy the RGB data to the input `Tensor`.
      try interpreter.copy(grayscaleData, toInputAt: 0)

      // Run inference by invoking the `Interpreter`.
      let startDate = Date()
      try interpreter.invoke()
      interval = Date().timeIntervalSince(startDate) * 1000

        outputTensor = try interpreter.output(at: 0)
        let collectedData = [Int](unsafeData: outputTensor.data) ?? []
//        print(collectedData)
    } catch let error {
      print("Failed to invoke the interpreter with error: \(error.localizedDescription)")
      return nil
    }

    // Formats the results // Returns the inference time and inferences
    let inferenceString: String = getResult(tensorOutputData: [Int](unsafeData: outputTensor.data) ?? [])

    return OCRResult(inferenceTime: interval, inference: inferenceString)
  }

  /// Filters out all the results with confidence score < threshold and returns the top N results
  /// sorted in descending order.
    func getResult(tensorOutputData: [Int]) -> String{
        var inferenceString: String = ""
        for index in tensorOutputData {
            if index != -1 && index < self.digitList.count && index >= 0 {
                inferenceString += self.digitList[index]
            }
        }
        return inferenceString
  }

    /// Returns the RGB data representation of the given image buffer with the specified `byteCount`.
    ///
    /// - Parameters
    ///   - buffer: The ARGB pixel buffer to convert to grayscale data.
    ///   - byteCount: The expected byte count for the grayscale data calculated using the values that the
    ///       model was trained on: `batchSize * imageWidth * imageHeight * componentsCount`.
    ///   - isModelQuantized: Whether the model is quantized (i.e. fixed point values rather than
    ///       floating point values).
    /// - Returns: The grayscale data representation of the image buffer or `nil` if the buffer could not be
    ///     converted.
    func grayscaleDataFromBuffer(
      _ buffer: CVPixelBuffer,
      byteCount: Int,
      isModelQuantized: Bool
    ) -> Data? {
      CVPixelBufferLockBaseAddress(buffer, .readOnly)
      defer {
        CVPixelBufferUnlockBaseAddress(buffer, .readOnly)
      }
      guard let sourceData = CVPixelBufferGetBaseAddress(buffer) else {
        return nil
      }
        
        // grayscale matrices and constance
        let redCoefficient: Float = 0.2126
        let greenCoefficient: Float = 0.7152
        let blueCoefficient: Float = 0.0722
        let divisor: Int32 = 0x1000
        let fDivisor = Float(divisor)
        var coefficientsMatrix = [
            Int16(redCoefficient * fDivisor),
            Int16(greenCoefficient * fDivisor),
            Int16(blueCoefficient * fDivisor)
        ]
        let preBias: [Int16] = [0, 0, 0, 0]
        let postBias: Int32 = 0
        //
      let width = CVPixelBufferGetWidth(buffer)
      let height = CVPixelBufferGetHeight(buffer)
      let sourceBytesPerRow = CVPixelBufferGetBytesPerRow(buffer)
      let destinationChannelCount = 1
      let destinationBytesPerRow = destinationChannelCount * width
      
      var sourceBuffer = vImage_Buffer(data: sourceData,
                                       height: vImagePixelCount(height),
                                       width: vImagePixelCount(width),
                                       rowBytes: sourceBytesPerRow)
      
      guard let destinationData = malloc(height * destinationBytesPerRow) else {
        print("Error: out of memory")
        return nil
      }
      
      defer {
        free(destinationData)
      }

      var destinationBuffer = vImage_Buffer(data: destinationData,
                                            height: vImagePixelCount(height),
                                            width: vImagePixelCount(width),
                                            rowBytes: destinationBytesPerRow)
      
//      if (CVPixelBufferGetPixelFormatType(buffer) == kCVPixelFormatType_32BGRA){
//        vImageConvert_BGRA8888toRGB888(&sourceBuffer, &destinationBuffer, UInt32(kvImageNoFlags))
//      } else if (CVPixelBufferGetPixelFormatType(buffer) == kCVPixelFormatType_32ARGB) {
//        vImageConvert_ARGB8888toRGB888(&sourceBuffer, &destinationBuffer, UInt32(kvImageNoFlags))
//      }
        
        vImageMatrixMultiply_ARGB8888ToPlanar8(&sourceBuffer, &destinationBuffer, &coefficientsMatrix, divisor, preBias, postBias, vImage_Flags(kvImageNoFlags))
        

      let byteData = Data(bytes: destinationBuffer.data, count: destinationBuffer.rowBytes * height)
      if isModelQuantized {
        return byteData
      }

      // Not quantized, convert to floats
      let bytes = Array<UInt8>(unsafeData: byteData)!
      var floats = [Float]()
      for i in 0..<bytes.count {
        floats.append((Float(bytes[i]) - imageMean) / imageStd)
      }
      return Data(copyingBufferOf: floats)
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
