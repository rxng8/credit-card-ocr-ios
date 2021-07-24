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
    
    // MARK: Instance Variables
    // Holds the results at any time
    private var detectorResult: Result?
    private var initialBottomSpace: CGFloat = 0.0
    private var previousInferenceTimeMs: TimeInterval = Date.distantPast.timeIntervalSince1970 * 1000
    
    // MARK: Handles all data preprocessing and makes calls to run inference through the `Interpreter`.
    private var modelDataHandler: ModelDataHandler? =
        ModelDataHandler(modelFileInfo: MobileNetSSD.modelInfo, labelsFileInfo: MobileNetSSD.labelsInfo)
    
    // MARK: OCR Data handler
    private var ocrDataHandler: OCRDataHandler? = OCRDataHandler(modelFileInfo: OCRModel.modelInfo)
    
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
        
        let newciimage = CIImage(cvImageBuffer: croppedPixelBuffer!)
        
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
    
    
    /** This method runs the live camera pixelBuffer through tensorFlow to get the result.
     */
    func runModel(onPixelBuffer pixelBuffer: CVPixelBuffer) {
        
        // Run the live camera pixelBuffer through tensorFlow to get the result
        
        let currentTimeMs = Date().timeIntervalSince1970 * 1000
        
        guard  (currentTimeMs - previousInferenceTimeMs) >= delayBetweenInferencesMs else {
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
        
        DispatchQueue.main.async {
            
            var inferenceTime: Double = 0
            if let resultInferenceTime = self.detectorResult?.inferenceTime {
                inferenceTime = resultInferenceTime
            }
            
            // Draws the bounding boxes and displays class names and confidence scores.
            self.drawAfterPerformingCalculations(onInferences: displayResult.inferences, withImageSize: CGSize(width: CGFloat(width), height: CGFloat(height)))
            
//            print("[Result] \(self.result!)")
            
            if self.detectorResult != nil && self.detectorResult!.inferences.count > 0 {
//                print("[Result] \(self.result!.inferences[0].rect)")
                
                let meowRect = self.detectorResult!.inferences[0].rect
                
                // Translates bounding box rect to current view.
                var convertedRect = CGRect(x: meowRect.minX, y: meowRect.minY, width: meowRect.width * self.overlayView.bounds.size.width / CGFloat(width), height: meowRect.height * self.overlayView.bounds.size.height / CGFloat(height))
                    
                let croppedPixelBuffer = self.resizePixelBuffer(pixelBuffer, rect: convertedRect, superViewSize: CGSize(width: self.croppedRect!.width, height: self.croppedRect!.height))
                let newciimage = CIImage(cvImageBuffer: croppedPixelBuffer!)
                
                // Process crnn model here!
                let ocrResult = self.ocrDataHandler?.runModel(onFrame: croppedPixelBuffer!)
                
                // Debug
                
                // Put image to image debug view
                DispatchQueue.main.async {
                    let newui = UIImage(ciImage: newciimage)
//                    print("[meowRect] \(meowRect)")
//                    print("[newuiimage size] \(newui.size)")
                    self.pixelDebugView.image = newui
                }
            } else {
                // Debug
                DispatchQueue.main.async {
                    self.pixelDebugView.image = nil
                }
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
            
            let confidenceValue = Int(inference.confidence * 100.0)
            let string = "\(inference.className)  (\(confidenceValue)%)"
            
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
