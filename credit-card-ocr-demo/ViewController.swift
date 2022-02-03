//
//  ViewController.swift
//  example-cam-2
//
//  Created by Alex Nguyen on 22/07/2021.
//

import UIKit
import AVFoundation
import Accelerate
import CoreImage

class ViewController: UIViewController {
    
    @IBOutlet weak var pixelDebugView: UIImageView!
    @IBOutlet weak var GuidelineView: UIView!
    @IBOutlet weak var overlayView: OverlayView!
    @IBOutlet weak var previewView: PreviewView!
    
    // MARK: Constants
    private let displayFont = UIFont.systemFont(ofSize: 14.0, weight: .medium)
    private let edgeOffset: CGFloat = 2.0
    private let labelOffset: CGFloat = 10.0
    private let animationDuration = 0.5
    private let collapseTransitionThreshold: CGFloat = -30.0
    private let expandTransitionThreshold: CGFloat = 30.0
    private let delayBetweenInferencesMs: Double = 100
    private let delayBetweenDigitInferencesMs: Double = 200
    
    // MARK: Instance Variables
    // Holds the results at any time
    private var detectorResult: Result?
    private var initialBottomSpace: CGFloat = 0.0
    private var previousInferenceTimeMs: TimeInterval = Date.distantPast.timeIntervalSince1970 * 1000
    
    private var previousDigitInferenceTimeMs: TimeInterval = Date.distantPast.timeIntervalSince1970 * 1000
    
    private var digitDetectorResult: DigitResult?
    
    // MARK: Handles all data preprocessing and makes calls to run inference through the `Interpreter`.
    private var modelDataHandler: ModelDataHandler? =
        ModelDataHandler(modelFileInfo: MobileNetSSD.modelInfo, labelsFileInfo: MobileNetSSD.labelsInfo)
    
    // MARK: OCR Data handler
    private var ocrDataHandler: OCRDataHandler? = OCRDataHandler(modelFileInfo: OCRModel.modelInfo)
    
    // MARK: Digit model data handler. efficiectdet
    private var digitModelDataHandler: DigitModelDataHandler? = DigitModelDataHandler(modelFileInfo: DigitModel.modelInfo, labelsFileInfo: DigitModel.labelsInfo)
    
    // Others
    let captureSession = AVCaptureSession()
    var previewLayer: CALayer!
    private let sessionQueue = DispatchQueue(label: "sessionQueue")
    var captureDevice: AVCaptureDevice!
    private lazy var videoDataOutput = AVCaptureVideoDataOutput()
    
    private var cameraConfiguration: CameraConfiguration = .failed
    private var isSessionRunning = false
    
    var croppedRect: CGRect?
    var previewViewSize: CGSize?
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        self.croppedRect = self.GuidelineView.frame
        print("[cropedRect] \(croppedRect!)")
        
        
        self.attemptToConfigureSession()
        self.setupPreviewLayer()
        self.checkCameraConfigurationAndStartSession()
        
        // MARK: Preview guideline
        self.GuidelineView.layer.borderWidth = 4
        self.GuidelineView.layer.borderColor = UIColor.green.cgColor
        
        self.previewViewSize = self.previewView.previewLayer.bounds.size
        print("[previewViewSize] \(previewViewSize!)")
        
    }
    
    
    func setupPreviewLayer() {
        self.previewLayer = AVCaptureVideoPreviewLayer(session: self.captureSession)
        //        self.view.layer.addSublayer(self.previewLayer)
//        self.view.layer.insertSublayer(self.previewLayer, at: 0)
//        self.previewLayer.frame = view.frame
        
                self.previewView.previewLayer.frame = self.previewView.bounds
//                self.previewView.previewLayer.connection?.videoOrientation = .portrait
//                self.previewView.previewLayer.videoGravity = .resizeAspectFill
                self.previewView.previewLayer.session = captureSession
    }
    
    // MARK: Session Start and End methods
    /**
     This method stops a running an AVCaptureSession.
     */
    func stopSession() {
        self.removeObservers()
        sessionQueue.async {
            if self.captureSession.isRunning {
                self.captureSession.stopRunning()
                self.isSessionRunning = self.captureSession.isRunning
            }
        }
        
    }
    
    /**
     This method starts the AVCaptureSession
     **/
    private func startSession() {
        self.captureSession.startRunning()
        self.isSessionRunning = self.captureSession.isRunning
    }
    
    
    /**
     This method starts an AVCaptureSession based on whether the camera configuration was successful.
     */
    func checkCameraConfigurationAndStartSession() {
        sessionQueue.async {
            switch self.cameraConfiguration {
            case .success:
                self.addObservers()
                self.startSession()
            case .failed:
                DispatchQueue.main.async {
                    
                }
            case .permissionDenied:
                DispatchQueue.main.async {
                    
                }
            }
        }
    }
    
    // MARK: Session Configuration Methods.
    /**
     This method requests for camera permissions and handles the configuration of the session and stores the result of configuration.
     */
    private func attemptToConfigureSession() {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            self.cameraConfiguration = .success
        case .notDetermined:
            self.sessionQueue.suspend()
            self.requestCameraAccess(completion: { (granted) in
                self.sessionQueue.resume()
            })
        case .denied:
            self.cameraConfiguration = .permissionDenied
        default:
            break
        }
        
        self.sessionQueue.async {
            self.configureSession()
        }
    }
    
    /**
     This method requests for camera permissions.
     */
    private func requestCameraAccess(completion: @escaping (Bool) -> ()) {
        AVCaptureDevice.requestAccess(for: .video) { (granted) in
            if !granted {
                self.cameraConfiguration = .permissionDenied
            }
            else {
                self.cameraConfiguration = .success
            }
            completion(granted)
        }
    }
    
    
    /**
     This method handles all the steps to configure an AVCaptureSession.
     */
    private func configureSession() {
        
        guard cameraConfiguration == .success else {
            return
        }
        captureSession.beginConfiguration()
        
        // Tries to add an AVCaptureDeviceInput.
        guard addVideoDeviceInput() == true else {
            self.captureSession.commitConfiguration()
            self.cameraConfiguration = .failed
            return
        }
        
        // Tries to add an AVCaptureVideoDataOutput.
        guard addVideoDataOutput() else {
            self.captureSession.commitConfiguration()
            self.cameraConfiguration = .failed
            return
        }
        
        captureSession.commitConfiguration()
        self.cameraConfiguration = .success
    }
    
    /**
     This method tries to add an AVCaptureDeviceInput to the current AVCaptureSession.
     */
    private func addVideoDeviceInput() -> Bool {
        
        /**Tries to get the default back camera.
         */
        guard let camera  = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back) else {
            fatalError("Cannot find camera")
        }
        
        do {
            let videoDeviceInput = try AVCaptureDeviceInput(device: camera)
            if captureSession.canAddInput(videoDeviceInput) {
                captureSession.addInput(videoDeviceInput)
                return true
            }
            else {
                return false
            }
        }
        catch {
            fatalError("Cannot create video device input")
        }
    }
    
    /**
     This method tries to add an AVCaptureVideoDataOutput to the current AVCaptureSession.
     */
    private func addVideoDataOutput() -> Bool {
        
        let sampleBufferQueue = DispatchQueue(label: "sampleBufferQueue")
        videoDataOutput.setSampleBufferDelegate(self, queue: sampleBufferQueue)
        videoDataOutput.alwaysDiscardsLateVideoFrames = true
        videoDataOutput.videoSettings = [ String(kCVPixelBufferPixelFormatTypeKey) : kCMPixelFormat_32BGRA]
        
        if captureSession.canAddOutput(videoDataOutput) {
            captureSession.addOutput(videoDataOutput)
            videoDataOutput.connection(with: .video)?.videoOrientation = .portrait
            return true
        }
        return false
    }
    
    // MARK: Notification Observer Handling
    private func addObservers() {
        
    }
    
    private func removeObservers() {
        
    }
    
}

/**
 This enum holds the state of the camera initialization.
 */
enum CameraConfiguration {
    
    case success
    case failed
    case permissionDenied
}


extension ViewController: AVCaptureVideoDataOutputSampleBufferDelegate {
    /** This method delegates the CVPixelBuffer of the frame seen by the camera currently.
     */
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        
        // Converts the CMSampleBuffer to a CVPixelBuffer.
        let pixelBuffer: CVPixelBuffer? = CMSampleBufferGetImageBuffer(sampleBuffer)
        
        guard let imagePixelBuffer = pixelBuffer else { return }
        
        //        let newPixelBuffer = resizePixelBuffer(imagePixelBuffer, rect: CGRect(x: 0, y: 500, width: 1080, height: 600))
        let croppedPixelBuffer = resizePixelBuffer(imagePixelBuffer, rect: croppedRect!, superViewSize: self.previewViewSize!)
        
//        let newciimage = CIImage(cvImageBuffer: croppedPixelBuffer!)
        
        // Put image to image debug view
//        DispatchQueue.main.async {
//            let newui = UIImage(ciImage: newciimage)
//            self.pixelDebugView.image = newui
//        }
        
        runModel(onPixelBuffer: croppedPixelBuffer!)
        
    }
}


// MARK: - Extension

extension ViewController {
    
    // Crop
    func resizePixelBuffer(_ pixelBuffer: CVPixelBuffer, rect: CGRect, superViewSize: CGSize?) -> CVPixelBuffer? {
        
        CVPixelBufferLockBaseAddress(pixelBuffer,
                                     CVPixelBufferLockFlags.readOnly)
        
        let resoHeight = CVPixelBufferGetHeight(pixelBuffer)
        let resoWidth = CVPixelBufferGetWidth(pixelBuffer)
        
        let pixelWidth: CGFloat?
        let pixelHeight: CGFloat?
        if superViewSize != nil {
            pixelWidth = rect.width / superViewSize!.width * CGFloat(resoWidth)
            let ratio = rect.height / rect.width
            pixelHeight = ratio * CGFloat(pixelWidth!)
//            pixelHeight = rect.height / superViewSize!.height * CGFloat(resoHeight)
        } else {
            pixelWidth = CGFloat(resoWidth)
            pixelHeight = CGFloat(resoHeight)
        }
        
        var error = kvImageNoError
        guard let data = CVPixelBufferGetBaseAddress(pixelBuffer) else {
            return nil
        }
        let bytesPerPixel = 4
        let rowBytes = CVPixelBufferGetBytesPerRow(pixelBuffer)
        let start = Int(rect.origin.y) * rowBytes + Int(rect.origin.x) * bytesPerPixel
        
        //        var inBuffer = vImage_Buffer(data: data.advanced(by: start), height: vImagePixelCount(rect.height), width: vImagePixelCount(rect.width), rowBytes: rowBytes)
        
        var newPixelBuffer: CVPixelBuffer?
        let addressPoint = data.assumingMemoryBound(to: UInt8.self)
        let options = [kCVPixelBufferCGImageCompatibilityKey:true,
                       kCVPixelBufferCGBitmapContextCompatibilityKey:true]
        let status = CVPixelBufferCreateWithBytes(kCFAllocatorDefault, Int(pixelWidth!), Int(pixelHeight!), kCVPixelFormatType_32BGRA, &addressPoint[start], Int(rowBytes), nil, nil, options as CFDictionary, &newPixelBuffer)
        if (status != 0) {
            return nil;
        }
        
        CVPixelBufferUnlockBaseAddress(pixelBuffer,
                                       CVPixelBufferLockFlags.readOnly)
        
        return newPixelBuffer
    }
    
    func convertToGrayscale(_ pixelBuffer: CVPixelBuffer) -> CGImage? {
        
        // constants
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
        
        // Lock
        CVPixelBufferLockBaseAddress(pixelBuffer, CVPixelBufferLockFlags.readOnly)
        
        // define source
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        guard let sourceData = CVPixelBufferGetBaseAddress(pixelBuffer) else {
          return nil
        }
        let rowBytes = CVPixelBufferGetBytesPerRow(pixelBuffer)
        var sourceBuffer = vImage_Buffer(data: sourceData, height: vImagePixelCount(height), width: vImagePixelCount(width), rowBytes: rowBytes)
        
        // define destination
        let destinationChannelCount = 1
        let destinationRowBytes = destinationChannelCount * width
        guard let destinationData = malloc(height * destinationRowBytes) else {
          print("Error: out of memory")
          return nil
        }
        defer {
          free(destinationData)
        }
        var destinationBuffer = vImage_Buffer(data: destinationData, height: vImagePixelCount(height), width: vImagePixelCount(width), rowBytes: destinationRowBytes)
        
        vImageMatrixMultiply_ARGB8888ToPlanar8(&sourceBuffer, &destinationBuffer, &coefficientsMatrix, divisor, preBias, postBias, vImage_Flags(kvImageNoFlags))
        
        
        CVPixelBufferUnlockBaseAddress(pixelBuffer, CVPixelBufferLockFlags.readOnly)
        
        // create core graphic image
        guard let monoFormat = vImage_CGImageFormat(
            bitsPerComponent: 8,
            bitsPerPixel: 8,
            colorSpace: CGColorSpaceCreateDeviceGray(),
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue),
            renderingIntent: .defaultIntent) else {
                return nil
        }
        let result = try? destinationBuffer.createCGImage(format: monoFormat)
//        if let result = result {
//            imageView.image = UIImage(cgImage: result)
//        }
        // MARK: TODO: create cv image buffer from cv buffer?
//        let status = CVPixelBufferCreateWithBytes(kCFAllocatorDefault, Int(pixelWidth!), Int(pixelHeight!), kCVPixelFormatType_32BGRA, &addressPoint[start], Int(rowBytes), nil, nil, options as CFDictionary, &newPixelBuffer)
//        if (status != 0) {
//            return nil;
//        }
        return result
    }
    
    /** This method runs the live camera pixelBuffer through tensorFlow to get the result.
     */
    func runModel(onPixelBuffer pixelBuffer: CVPixelBuffer) {
        
        // Run the live camera pixelBuffer through tensorFlow to get the result
        
        let currentTimeMs = Date().timeIntervalSince1970 * 1000
        
        guard (currentTimeMs - previousInferenceTimeMs) >= delayBetweenInferencesMs else {
            return
        }
        
        previousInferenceTimeMs = currentTimeMs
        //        print("[runModel] Start inference at \(currentTimeMs).")
        detectorResult = self.modelDataHandler?.runModel(onFrame: pixelBuffer)
        
        guard let displayResult = detectorResult else {
            return
        }
        
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        
        DispatchQueue.main.async { [self] in
            
            var inferenceTime: Double = 0
            if let resultInferenceTime = self.detectorResult?.inferenceTime {
                inferenceTime = resultInferenceTime
            }
            
            // Draws the bounding boxes and displays class names and confidence scores.
//            self.drawAfterPerformingCalculations(onInferences: displayResult.inferences, withImageSize: CGSize(width: CGFloat(width), height: CGFloat(height)))
            self.drawAfterPerformingCalculations(onInferences: displayResult.inferences, displayName: "Loading...", withImageSize: CGSize(width: CGFloat(width), height: CGFloat(height)))
            
//            print("[Result] \(self.result!)")
            
            if self.detectorResult != nil && self.detectorResult!.inferences.count > 0 {
//                print("[Result] \(self.result!.inferences[0].rect)")
                
                let meowRect = self.detectorResult!.inferences[0].rect
                
                // Translates bounding box rect to current view. The size of the number line rect
                let convertedRect = CGRect(x: meowRect.minX, y: meowRect.minY, width: meowRect.width * self.overlayView.bounds.size.width / CGFloat(width), height: meowRect.height * self.overlayView.bounds.size.height / CGFloat(height))
                    
                let croppedPixelBuffer = self.resizePixelBuffer(pixelBuffer, rect: convertedRect, superViewSize: CGSize(width: self.croppedRect!.width, height: self.croppedRect!.height))
                
                let numberLineWidth = CVPixelBufferGetWidth(croppedPixelBuffer!)
                let numberLineHeight = CVPixelBufferGetHeight(croppedPixelBuffer!)
                
                let paddedPixelBuffer = self.resizePixelBufferWithPad(croppedPixelBuffer!, cropX: 0, cropY: -(numberLineWidth - numberLineHeight) / 2, cropWidth: numberLineWidth, cropHeight: numberLineWidth, scaleWidth: 512, scaleHeight: 512)
                
                let newciimage = CIImage(cvImageBuffer: croppedPixelBuffer!)
                
                // Debug
                // grayscaleBuffer
//                let grayscaledcgimg = self.convertToGrayscale(croppedPixelBuffer!)
                
                // scaled the cropped buffer
//                let scaledSize = CGSize(width: self.ocrDataHandler!.inputWidth, height: self.ocrDataHandler!.inputHeight)
//                let scaledPixelBuffer = croppedPixelBuffer!.resized(to: scaledSize)
//                let scaledPixelBufferci = CIImage(cvPixelBuffer: scaledPixelBuffer!)
                
                // Put image to image debug view
//                DispatchQueue.main.async {
//                    let newui = UIImage(ciImage: newciimage)
////                    let newui = UIImage(ciImage: scaledPixelBufferci)
////                    print("[meowRect] \(meowRect)")
////                    print("[newuiimage size] \(newui.size)")
//                    self.pixelDebugView.image = newui
//                }
                
                // Process crnn model here!
//                let ocrResult = self.ocrDataHandler?.runModel(onFrame: croppedPixelBuffer!)
//                print(ocrResult)
                
                var resArray: [(String, CGFloat)] = []
                
                // process efficientdet
                // time
                let currentTimeMsDigit = Date().timeIntervalSince1970 * 1000
                guard (currentTimeMsDigit - previousDigitInferenceTimeMs) >= delayBetweenDigitInferencesMs else {
                    return
                }
                previousDigitInferenceTimeMs = currentTimeMsDigit
                
                digitDetectorResult = self.digitModelDataHandler?.runModel(onFrame: paddedPixelBuffer!)
                guard let digitDisplayResult = digitDetectorResult else {
                    return
                }
                for inferInstance in digitDetectorResult!.inferences {
                    resArray.append((inferInstance.className, inferInstance.rect.minX))
                }
                
                let sortedResArray = resArray.sorted(by: {$0.1 < $1.1 })
                var resultString = ""
                for obj in sortedResArray {
                    resultString += obj.0
                }
                print(resultString)
                
                DispatchQueue.main.async {
                    // Draws the bounding boxes and displays class names and confidence scores.
//                    self.drawAfterPerformingCalculations(onDigitInferences: digitDisplayResult.inferences, superViewSize: convertedRect.size, withImageSize: CGSize(width: CGFloat(numberLineWidth), height: CGFloat(numberLineHeight)))
                    self.drawAfterPerformingCalculations(onInferences: displayResult.inferences, displayName: resultString, withImageSize: CGSize(width: CGFloat(width), height: CGFloat(height)))
                }
                
                
            } else {
                // Debug
                DispatchQueue.main.async {
                    self.pixelDebugView.image = nil
                    // Draw nil on object overlay if it has not found the number line
//                    self.drawAfterPerformingCalculations(onInferences: displayResult.inferences, withImageSize: CGSize(width: CGFloat(width), height: CGFloat(height)))
                }
                // Hands off drawing to the OverlayView
                self.draw(objectOverlays: [])
            }
        }
    }
    
    /**
     This method takes the results, translates the bounding box rects to the current view, draws the bounding boxes, classNames and confidence scores of inferences.
     */
    func drawAfterPerformingCalculations(onInferences inferences: [Inference], withImageSize imageSize:CGSize) {
        
        self.overlayView.objectOverlays = []
        self.overlayView.setNeedsDisplay()
        
        guard !inferences.isEmpty else {
            return
        }
        
        var objectOverlays: [ObjectOverlay] = []
        
        for inference in inferences {
            
//            print("number rect before trans: \(inference.rect)")
            
            // Translates bounding box rect to current view.
            var convertedRect = inference.rect.applying(CGAffineTransform(scaleX: self.overlayView.bounds.size.width / imageSize.width, y: self.overlayView.bounds.size.height / imageSize.height))
            
            if convertedRect.origin.x < 0 {
                convertedRect.origin.x = self.edgeOffset
            }
            
            if convertedRect.origin.y < 0 {
                convertedRect.origin.y = self.edgeOffset
            }
            
            if convertedRect.maxY > self.overlayView.bounds.maxY {
                convertedRect.size.height = self.overlayView.bounds.maxY - convertedRect.origin.y - self.edgeOffset
            }
            
            if convertedRect.maxX > self.overlayView.bounds.maxX {
                convertedRect.size.width = self.overlayView.bounds.maxX - convertedRect.origin.x - self.edgeOffset
            }
            
//            print("number rect: \(convertedRect)")
            
            let confidenceValue = Int(inference.confidence * 100.0)
            let string = "\(inference.className)  (\(confidenceValue)%)"
            
            let size = string.size(usingFont: self.displayFont)
            
            let objectOverlay = ObjectOverlay(name: string, borderRect: convertedRect, nameStringSize: size, color: inference.displayColor, font: self.displayFont)
            
            objectOverlays.append(objectOverlay)
        }
        
        // Hands off drawing to the OverlayView
        self.draw(objectOverlays: objectOverlays)
        
    }
    
    /**
     This method takes the results, translates the bounding box rects to the current view, draws the bounding boxes, classNames and confidence scores of inferences.
     */
    func drawAfterPerformingCalculations(onDigitInferences inferences: [DigitInference], superViewSize: CGSize, withImageSize imageSize:CGSize) {
        
        self.overlayView.objectOverlays = []
        self.overlayView.setNeedsDisplay()
        
        guard !inferences.isEmpty else {
            return
        }
        
        var objectOverlays: [ObjectOverlay] = []
        
        for inference in inferences {
//            print("digit Rect before trans: \(inference.rect)")
            // Translates bounding box rect to current view.
            var convertedRect = inference.rect.applying(CGAffineTransform(scaleX: superViewSize.width / imageSize.width, y: superViewSize.height / imageSize.height))
            
            print("digit Rect: \(convertedRect)")
//            if convertedRect.origin.x < 0 {
//                convertedRect.origin.x = self.edgeOffset
//            }
//
//            if convertedRect.origin.y < 0 {
//                convertedRect.origin.y = self.edgeOffset
//            }
//
//            if convertedRect.maxY > self.overlayView.bounds.maxY {
//                convertedRect.size.height = self.overlayView.bounds.maxY - convertedRect.origin.y - self.edgeOffset
//            }
//
//            if convertedRect.maxX > self.overlayView.bounds.maxX {
//                convertedRect.size.width = self.overlayView.bounds.maxX - convertedRect.origin.x - self.edgeOffset
//            }
            
            let confidenceValue = Int(inference.confidence * 100.0)
            let string = "\(inference.className)  (\(confidenceValue)%)"
            
            let size = string.size(usingFont: self.displayFont)
            
            let objectOverlay = ObjectOverlay(name: string, borderRect: convertedRect, nameStringSize: size, color: inference.displayColor, font: self.displayFont)
            
            objectOverlays.append(objectOverlay)
        }
        
        // Hands off drawing to the OverlayView
        self.draw(objectOverlays: objectOverlays)
        
    }
    
    func drawAfterPerformingCalculations(onInferences inferences: [Inference], displayName name: String, withImageSize imageSize:CGSize) {
        
        self.overlayView.objectOverlays = []
        self.overlayView.setNeedsDisplay()
        
        guard !inferences.isEmpty else {
            return
        }
        
        var objectOverlays: [ObjectOverlay] = []
        
        for inference in inferences {
            
//            print("number rect before trans: \(inference.rect)")
            
            // Translates bounding box rect to current view.
            var convertedRect = inference.rect.applying(CGAffineTransform(scaleX: self.overlayView.bounds.size.width / imageSize.width, y: self.overlayView.bounds.size.height / imageSize.height))
            
            if convertedRect.origin.x < 0 {
                convertedRect.origin.x = self.edgeOffset
            }
            
            if convertedRect.origin.y < 0 {
                convertedRect.origin.y = self.edgeOffset
            }
            
            if convertedRect.maxY > self.overlayView.bounds.maxY {
                convertedRect.size.height = self.overlayView.bounds.maxY - convertedRect.origin.y - self.edgeOffset
            }
            
            if convertedRect.maxX > self.overlayView.bounds.maxX {
                convertedRect.size.width = self.overlayView.bounds.maxX - convertedRect.origin.x - self.edgeOffset
            }
            
//            print("number rect: \(convertedRect)")
            
            let confidenceValue = Int(inference.confidence * 100.0)
            let string = "\(name)"
            
            let size = string.size(usingFont: self.displayFont)
            
            let objectOverlay = ObjectOverlay(name: string, borderRect: convertedRect, nameStringSize: size, color: inference.displayColor, font: self.displayFont)
            
            objectOverlays.append(objectOverlay)
        }
        
        // Hands off drawing to the OverlayView
        self.draw(objectOverlays: objectOverlays)
        
    }
    
    /** Calls methods to update overlay view with detected bounding boxes and class names.
     */
    func draw(objectOverlays: [ObjectOverlay]) {
        
        self.overlayView.objectOverlays = objectOverlays
        self.overlayView.setNeedsDisplay()
    }
    
}


/**
 resize extension. https://stackoverflow.com/a/53823222
 */
extension ViewController {
    public func resizePixelBufferWithPad(_ srcPixelBuffer: CVPixelBuffer,
                                  cropX: Int,
                                  cropY: Int,
                                  cropWidth: Int,
                                  cropHeight: Int,
                                  scaleWidth: Int,
                                  scaleHeight: Int) -> CVPixelBuffer? {
        let flags = CVPixelBufferLockFlags(rawValue: 0)
        let pixelFormat = CVPixelBufferGetPixelFormatType(srcPixelBuffer)
        guard kCVReturnSuccess == CVPixelBufferLockBaseAddress(srcPixelBuffer, flags) else {
            return nil
        }
        defer { CVPixelBufferUnlockBaseAddress(srcPixelBuffer, flags) }

        guard let srcData = CVPixelBufferGetBaseAddress(srcPixelBuffer) else {
            print("Error: could not get pixel buffer base address")
            return nil
        }

        let srcHeight = CVPixelBufferGetHeight(srcPixelBuffer)
        let srcWidth = CVPixelBufferGetWidth(srcPixelBuffer)
        let srcBytesPerRow = CVPixelBufferGetBytesPerRow(srcPixelBuffer)
        let offset = cropY * srcBytesPerRow + cropX * 4

        var srcBuffer: vImage_Buffer!
        var paddedSrcPixelBuffer: CVPixelBuffer!

        if (cropX < 0 || cropY < 0 || cropX + cropWidth > srcWidth || cropY + cropHeight > srcHeight) {
            let paddingLeft = abs(min(cropX, 0))
            let paddingRight = max((cropX + cropWidth) - (srcWidth - 1), 0)
            let paddingBottom = max((cropY + cropHeight) - (srcHeight - 1), 0)
            let paddingTop = abs(min(cropY, 0))

            let paddedHeight = paddingTop + srcHeight + paddingBottom
            let paddedWidth = paddingLeft + srcWidth + paddingRight

            guard kCVReturnSuccess == CVPixelBufferCreate(kCFAllocatorDefault, paddedWidth, paddedHeight, pixelFormat, nil, &paddedSrcPixelBuffer) else {
                print("failed to allocate a new padded pixel buffer")
                return nil
            }

            guard kCVReturnSuccess == CVPixelBufferLockBaseAddress(paddedSrcPixelBuffer, flags) else {
                return nil
            }

            guard let paddedSrcData = CVPixelBufferGetBaseAddress(paddedSrcPixelBuffer) else {
                print("Error: could not get padded pixel buffer base address")
                return nil
            }

            let paddedBytesPerRow = CVPixelBufferGetBytesPerRow(paddedSrcPixelBuffer)
            for yIndex in paddingTop..<srcHeight+paddingTop {
                let dstRowStart = paddedSrcData.advanced(by: yIndex*paddedBytesPerRow).advanced(by: paddingLeft*4)
                let srcRowStart = srcData.advanced(by: (yIndex - paddingTop)*srcBytesPerRow)
                dstRowStart.copyMemory(from: srcRowStart, byteCount: srcBytesPerRow)
            }

            let paddedOffset = (cropY + paddingTop)*paddedBytesPerRow + (cropX + paddingLeft)*4
            srcBuffer = vImage_Buffer(data: paddedSrcData.advanced(by: paddedOffset),
                                      height: vImagePixelCount(cropHeight),
                                      width: vImagePixelCount(cropWidth),
                                      rowBytes: paddedBytesPerRow)

        } else {
            srcBuffer = vImage_Buffer(data: srcData.advanced(by: offset),
                                      height: vImagePixelCount(cropHeight),
                                      width: vImagePixelCount(cropWidth),
                                      rowBytes: srcBytesPerRow)
        }

        let destBytesPerRow = scaleWidth*4
        guard let destData = malloc(scaleHeight*destBytesPerRow) else {
            print("Error: out of memory")
            return nil
        }
        var destBuffer = vImage_Buffer(data: destData,
                                       height: vImagePixelCount(scaleHeight),
                                       width: vImagePixelCount(scaleWidth),
                                       rowBytes: destBytesPerRow)

        let vImageFlags: vImage_Flags = vImage_Flags(kvImageEdgeExtend)
        let error = vImageScale_ARGB8888(&srcBuffer, &destBuffer, nil, vImageFlags)
        if error != kvImageNoError {
            print("Error:", error)
            free(destData)
            return nil
        }

        let releaseCallback: CVPixelBufferReleaseBytesCallback = { _, ptr in
            if let ptr = ptr {
                free(UnsafeMutableRawPointer(mutating: ptr))
            }
        }

        var dstPixelBuffer: CVPixelBuffer?
        let status = CVPixelBufferCreateWithBytes(nil, scaleWidth, scaleHeight,
                                                  pixelFormat, destData,
                                                  destBytesPerRow, releaseCallback,
                                                  nil, nil, &dstPixelBuffer)
        if status != kCVReturnSuccess {
            print("Error: could not create new pixel buffer")
            free(destData)
            return nil
        }

        if paddedSrcPixelBuffer != nil {
            CVPixelBufferUnlockBaseAddress(paddedSrcPixelBuffer, flags)
        }


        return dstPixelBuffer
    }
}
