//// Copyright (c) 2020 Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.



import UIKit
import AVFoundation

extension String {
    func stringByAppendingPathComponent(path: String) -> String {
        let s = self as NSString
        return s.appendingPathComponent(path)
    }
}

func fillBuffer(_ buffer: AVAudioPCMBuffer, _ input: [NSNumber]) {
    let data = buffer.floatChannelData?[0]
    let numberFrames = input.count
    for frame in 00..<Int(numberFrames) {
        data?[frame] = Float32(truncating: input[frame])
    }
    buffer.frameLength = AVAudioFrameCount(numberFrames)
}

class ViewController: UIViewController, AVAudioRecorderDelegate  {

    @IBOutlet weak var btnStart: UIButton!
    @IBOutlet weak var btnPlay: UIButton!
    @IBOutlet weak var tvResult: UITextView!
    
    private var audioRecorder: AVAudioRecorder!
    private var _recorderFilePath: String!
    
    private let AUDIO_LEN_IN_SECOND = 10
    private let SAMPLE_RATE = 16000
    

    private lazy var module: InferenceModule = {
        guard let filePath = Bundle.main.path(forResource:"model_67M_multitask_testing_full", ofType: "ptl") else {
            fatalError("Cannot find the module file")
        }
        guard let module = InferenceModule(fileAtPath: filePath) else {
            fatalError("Cannot build the inference module")
        }
        return module
    }()
    
    private lazy var vocoder: InferenceModule = {
        guard let filePath = Bundle.main.path(forResource:"mhubert", ofType: "ptl") else {
            fatalError("Cannot find the vocoder file")
        }
        guard let vocoder = InferenceModule(fileAtPath: filePath) else {
            fatalError("Cannot build the inference vocoder")
        }
        return vocoder
    }()
    
    @IBAction func startTapped(_ sender: Any) {
        AVAudioSession.sharedInstance().requestRecordPermission ({(granted: Bool)-> Void in
            if granted {
                self.btnStart.setTitle("Listening...", for: .normal)
            } else{
                self.tvResult.text = "Record premission needs to be granted"
            }
         })
        
        let audioSession = AVAudioSession.sharedInstance()
        
        do {
            try audioSession.setCategory(AVAudioSession.Category.record)
            try audioSession.setActive(true)
        } catch {
            tvResult.text = "recording exception"
            return
        }

        let settings = [
            AVFormatIDKey: Int(kAudioFormatLinearPCM),
            AVSampleRateKey: SAMPLE_RATE,
            AVNumberOfChannelsKey: 1,
            AVLinearPCMBitDepthKey: 16,
            AVLinearPCMIsBigEndianKey: false,
            AVLinearPCMIsFloatKey: false,
            AVEncoderAudioQualityKey: AVAudioQuality.high.rawValue
            ] as [String : Any]
        
        do {
            _recorderFilePath = NSHomeDirectory().stringByAppendingPathComponent(path: "tmp").stringByAppendingPathComponent(path: "recorded_file.wav")
            audioRecorder = try AVAudioRecorder(url: NSURL.fileURL(withPath: _recorderFilePath), settings: settings)
            audioRecorder.delegate = self
            audioRecorder.record(forDuration: TimeInterval(AUDIO_LEN_IN_SECOND))
        } catch let error {
            tvResult.text = "error:" + error.localizedDescription
        }
    }
    
    @IBAction func playTapped(_ send: Any) {
        let audioSession = AVAudioSession.sharedInstance()
        
        do {
            try audioSession.setCategory(AVAudioSession.Category.playback)
            try audioSession.setActive(true)
        } catch {
            tvResult.text = "playback exception"
            return
        }
        
        // TODO: Replace input tensor with the real units from unitY inference module's output
        let fakeInput = UnsafeMutablePointer<Float>.init(mutating: [Float(0.0)])
        if let result = self.vocoder.vocode(fakeInput, lang: "en") {
            let format = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: Double(16_000), channels: 1, interleaved: false)
            let buf = AVAudioPCMBuffer(pcmFormat: format!, frameCapacity: AVAudioFrameCount(result.count))
            fillBuffer(buf!, result)
            let engine = AVAudioEngine()
            let player = AVAudioPlayerNode()
            engine.attach(player)
            let mixer = engine.mainMixerNode
            engine.connect(player, to: mixer, format: engine.inputNode.outputFormat(forBus: 0))
            do {
                try engine.start()
                player.scheduleBuffer(buf!, completionHandler: nil)
                player.play()

            }
            catch let error {
                tvResult.text = "error:" + error.localizedDescription
            }
            
        }
        else {
            fatalError("Error in vocoding units")
        }

    }

    func audioRecorderDidFinishRecording(_ recorder: AVAudioRecorder, successfully flag: Bool) {
        btnStart.setTitle("Recognizing...", for: .normal)
        
        if flag {
            let url = NSURL.fileURL(withPath: _recorderFilePath)
            let file = try! AVAudioFile(forReading: url)
            let format = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: file.fileFormat.sampleRate, channels: 1, interleaved: false)
            let buf = AVAudioPCMBuffer(pcmFormat: format!, frameCapacity: AVAudioFrameCount(file.length))
            try! file.read(into: buf!)

            var floatArray = Array(UnsafeBufferPointer(start: buf?.floatChannelData![0], count:Int(buf!.frameLength)))

            DispatchQueue.global().async {
                floatArray.withUnsafeMutableBytes {
                    let result = self.module.recognize($0.baseAddress!, bufLength: Int32(self.AUDIO_LEN_IN_SECOND * self.SAMPLE_RATE))
                    DispatchQueue.main.async {
                        self.tvResult.text = result
                        self.btnStart.setTitle("Start", for: .normal)
                    }
                }
            }
        }
        else {
            tvResult.text = "Recording error"
        }
    }
}

